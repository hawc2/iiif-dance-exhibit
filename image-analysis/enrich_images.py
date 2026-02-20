#!/usr/bin/env python3
"""
enrich_images.py — CLIP-based visual tagging and similarity pipeline
for IIIF-based digital collection exhibits.

Adds to each item in the data JSON:
  id             str   URL-safe slug from title
  image_url      str   Primary image URL resolved from IIIF manifest
  visual_tags    list  Top-4 descriptive labels via CLIP zero-shot classification
  face_detected  bool  Whether faces/portraits are prominent (CLIP zero-shot)
  similar_items  list  IDs of top-3 visually similar items (cosine similarity)

Usage:
  # Enrich iiif-dance-exhibit (flat items list):
  python3 scripts/enrich_images.py --input data/dance.json --output data/dance.json

  # Enrich tu-digital-collections (nested collections, cross-collection similarity):
  python3 scripts/enrich_images.py \\
      --input data/collections.json \\
      --output data/collections.json \\
      --mode collections

  # Preview without writing (dry run):
  python3 scripts/enrich_images.py --input data/dance.json --output data/dance.json --dry-run

Manifest JSON files are cached in .manifest_cache/ to avoid re-fetching on re-runs.
Delete that directory to force a full refresh.
"""

import argparse
import json
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# ---------------------------------------------------------------------------
# Tag schema — grouped by analytical dimension.
# Keys are short display labels; values are longer CLIP prompts.
# Grouped dimensions let viewers filter and compare within a category.
# ---------------------------------------------------------------------------
TAG_PROMPTS = {
    # -- People count --
    "one person":          "exactly one person alone in the frame",
    "two people":          "exactly two people together in the frame",
    "group (3+)":          "three or more people together in the frame",

    # -- Physical activity --
    "airborne":            "a dancer jumping or leaping off the ground",
    "partnering":          "two dancers in physical contact, one lifting or supporting the other",
    "floor work":          "a dancer kneeling, crouching, or lying on the floor",
    "static or posed":     "a person standing still or posed for the camera, not in motion",

    # -- Setting --
    "on stage":            "a performance on a theatrical stage with stage lighting and backdrop",
    "studio / rehearsal":  "a rehearsal or practice in a studio or practice room, not on stage",
    "portrait session":    "a formal portrait session with a plain or neutral background",

    # -- Attire --
    "ballet costume":      "a dancer wearing a ballet tutu, pointe shoes, or classical ballet dress",
    "contemporary costume":"a dancer in a modern or contemporary dance costume",
    "practice clothes":    "a person wearing casual or athletic practice clothing",

    # -- Photo characteristics --
    "black and white":     "a black and white photograph with no color",
    "color photo":         "a color photograph",
    "high contrast":       "dramatic high-contrast or theatrical stage lighting",
}

# Face detection: first label = face present
FACE_LABELS = [
    "a portrait where one or more faces are clearly visible",
    "a photo where no faces are clearly visible",
]

TOP_TAGS = 4       # visual_tags per item
TOP_SIMILAR = 3    # similar_items per item
REQUEST_DELAY = 0.5  # seconds between image downloads

# Embedding dimension for openai/clip-vit-base-patch32
EMBEDDING_DIM = 512


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s.strip())
    return s[:80]


def fetch_manifest(url: str, cache_dir: Path) -> Optional[dict]:
    """Fetch a IIIF Presentation manifest, caching locally as JSON."""
    slug = slugify(url[-60:])
    cache_path = cache_dir / f"{slug}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        cache_path.write_text(json.dumps(data))
        return data
    except Exception as e:
        print(f"    [WARN] manifest fetch failed: {url} — {e}")
        return None


def extract_image_url(manifest: dict) -> str:
    """Extract the primary image URL from a IIIF Presentation 2 manifest."""
    try:
        seq = manifest.get("sequences", [])
        if seq:
            canvas = seq[0].get("canvases", [])[0]
            img = canvas.get("images", [])[0]
            res = img.get("resource", {})
            url = res.get("@id", "")
            if url:
                return url
            svc = res.get("service", {})
            svc_id = svc.get("@id", "") if isinstance(svc, dict) else ""
            if svc_id:
                return f"{svc_id}/full/800,/0/default.jpg"
    except (IndexError, KeyError, TypeError):
        pass
    # Thumbnail fallback
    thumb = manifest.get("thumbnail")
    if thumb:
        if isinstance(thumb, list):
            thumb = thumb[0]
        return thumb.get("@id", "")
    return ""


def fetch_image(url: str) -> Optional[Image.Image]:
    """Download an image and return a PIL Image (RGB), or None on failure."""
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"    [WARN] image fetch failed: {url} — {e}")
        return None


# ---------------------------------------------------------------------------
# CLIP inference
# ---------------------------------------------------------------------------

def get_embedding(model: CLIPModel, processor: CLIPProcessor, image: Image.Image) -> np.ndarray:
    """Return a unit-normalized 512-d CLIP image embedding."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy()


def zero_shot(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    labels: list,
    top_k: int,
) -> list:
    """Return the top_k labels from a candidate list via CLIP zero-shot."""
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image[0]
    probs = logits.softmax(dim=-1).cpu().numpy()
    ranked = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])
    return [lbl for lbl, _ in ranked[:top_k]]


def cosine_sim_matrix(embeddings: list) -> np.ndarray:
    """Compute n×n cosine similarity matrix. Assumes unit-normalized embeddings."""
    mat = np.array(embeddings)
    return mat @ mat.T


def save_embedding(cache_dir: Path, item_id: str, emb: np.ndarray) -> None:
    np.save(str(cache_dir / f"{item_id}_emb.npy"), emb)


def load_embedding(cache_dir: Path, item_id: str) -> Optional[np.ndarray]:
    path = cache_dir / f"{item_id}_emb.npy"
    if path.exists():
        return np.load(str(path))
    return None


def zero_shot_from_embedding(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_emb: np.ndarray,
    labels: list,
    top_k: int,
) -> list:
    """Zero-shot classify using a cached image embedding (no image download needed).

    Computes text embeddings on the fly and finds the best-matching labels.
    """
    inputs = processor(text=labels, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_feats = model.get_text_features(**inputs)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    img_tensor = torch.tensor(image_emb, dtype=torch.float32).unsqueeze(0)
    logits = (img_tensor @ text_feats.T) * model.logit_scale.exp()
    probs = logits[0].softmax(dim=-1).detach().cpu().numpy()

    ranked = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])
    return [lbl for lbl, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------

def phase1_fetch_and_tag(
    items: list,
    model: CLIPModel,
    processor: CLIPProcessor,
    cache_dir: Path,
    force_retag: bool = False,
) -> list:
    """
    Phase 1: resolve image URLs, download images, compute visual tags and embeddings.
    Mutates items in-place. Returns parallel list of embeddings (None for failures).

    If a cached embedding exists for an item, the image is not re-downloaded —
    tags are recomputed from the cached embedding (fast). Pass force_retag=True
    to always recompute tags even for cached items.
    """
    prompts = list(TAG_PROMPTS.values())
    prompt_to_label = {v: k for k, v in TAG_PROMPTS.items()}
    embeddings = []

    for item in items:
        item["id"] = slugify(item["title"])
        print(f"  {item['title']}")

        # --- Try loading a cached embedding first ---
        cached_emb = load_embedding(cache_dir, item["id"])
        if cached_emb is not None:
            if force_retag or "visual_tags" not in item:
                top_prompts = zero_shot_from_embedding(model, processor, cached_emb, prompts, TOP_TAGS)
                item["visual_tags"] = [prompt_to_label.get(p, p) for p in top_prompts]
                top_face = zero_shot_from_embedding(model, processor, cached_emb, FACE_LABELS, 1)
                item["face_detected"] = top_face[0] == FACE_LABELS[0]
                print("    [cached embedding, tags updated]")
            else:
                print("    [cached, skipping]")
            embeddings.append(cached_emb)
            continue

        # --- No cache: fetch manifest and download image ---
        manifest = fetch_manifest(item["manifest"], cache_dir)
        if not manifest:
            item["image_url"] = ""
            embeddings.append(None)
            continue

        image_url = extract_image_url(manifest)
        item["image_url"] = image_url

        if not image_url:
            print("    [WARN] no image URL in manifest")
            embeddings.append(None)
            continue

        image = fetch_image(image_url)
        if image is None:
            embeddings.append(None)
            continue

        # Visual tags
        top_prompts = zero_shot(model, processor, image, prompts, TOP_TAGS)
        item["visual_tags"] = [prompt_to_label.get(p, p) for p in top_prompts]

        # Face detection
        top_face = zero_shot(model, processor, image, FACE_LABELS, 1)
        item["face_detected"] = top_face[0] == FACE_LABELS[0]

        emb = get_embedding(model, processor, image)
        save_embedding(cache_dir, item["id"], emb)
        embeddings.append(emb)
        time.sleep(REQUEST_DELAY)

    return embeddings


def phase2_assign_similar(
    items: list,
    embeddings: list,
    ids: list,
) -> None:
    """
    Phase 2: compute pairwise cosine similarity and assign similar_items.
    items/embeddings/ids must be parallel lists of the same length.
    """
    valid = [(i, e) for i, e in enumerate(embeddings) if e is not None]
    if len(valid) < 2:
        return

    valid_indices, valid_embs = zip(*valid)
    sim = cosine_sim_matrix(list(valid_embs))

    for rank, i in enumerate(valid_indices):
        row = sim[rank]
        sorted_j = sorted(enumerate(row), key=lambda x: -x[1])
        similar = [
            ids[valid_indices[j]]
            for j, _ in sorted_j
            if valid_indices[j] != i
        ][:TOP_SIMILAR]
        items[i]["similar_items"] = similar


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def enrich_dance(data: dict, model, processor, cache_dir: Path, force_retag: bool = False) -> dict:
    """Handle dance.json — flat list of items."""
    items = data["items"]
    print(f"Processing {len(items)} items...\n")
    embeddings = phase1_fetch_and_tag(items, model, processor, cache_dir, force_retag)
    ids = [item["id"] for item in items]
    phase2_assign_similar(items, embeddings, ids)
    data["items"] = items
    return data


def enrich_collections(data: dict, model, processor, cache_dir: Path) -> dict:
    """
    Handle collections.json — nested collections.
    Phase 1 runs per-collection; Phase 2 runs across all items for
    cross-collection similarity (dancers from different collections can match).
    """
    all_items = []
    all_embeddings = []

    for coll in data["collections"]:
        print(f"\n=== {coll['title']} ({len(coll['items'])} items) ===")
        embs = phase1_fetch_and_tag(coll["items"], model, processor, cache_dir)
        all_items.extend(coll["items"])
        all_embeddings.extend(embs)

    all_ids = [item["id"] for item in all_items]

    print(f"\nComputing cross-collection similarity for {len(all_items)} items...")
    phase2_assign_similar(all_items, all_embeddings, all_ids)

    return data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global TOP_TAGS, TOP_SIMILAR
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",   required=True, help="Input JSON data file")
    parser.add_argument("--output",  required=True, help="Output JSON file (can equal --input)")
    parser.add_argument(
        "--mode",
        choices=["dance", "collections"],
        default="dance",
        help="'dance' = flat items list (dance.json); 'collections' = nested (collections.json)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".manifest_cache",
        help="Directory for caching manifest JSON files (default: .manifest_cache/)",
    )
    parser.add_argument("--top-tags",    type=int, default=TOP_TAGS,    help="Visual tags per item")
    parser.add_argument("--top-similar", type=int, default=TOP_SIMILAR, help="Similar items per item")
    parser.add_argument("--force-retag", action="store_true",
                        help="Recompute visual_tags for all items, even cached ones")
    parser.add_argument("--dry-run", action="store_true", help="Process but do not write output")
    args = parser.parse_args()

    TOP_TAGS = args.top_tags
    TOP_SIMILAR = args.top_similar

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    print(f"Loading CLIP ({CLIP_MODEL_ID})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    print("Model ready.\n")

    with open(args.input) as f:
        data = json.load(f)

    force_retag = args.force_retag

    if args.mode == "dance":
        data = enrich_dance(data, model, processor, cache_dir, force_retag)
    else:
        data = enrich_collections(data, model, processor, cache_dir)

    if args.dry_run:
        print("\nDry run — no file written.")
        return

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone. Enriched data written to {args.output}")


if __name__ == "__main__":
    main()
