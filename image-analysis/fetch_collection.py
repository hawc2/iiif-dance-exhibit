#!/usr/bin/env python3
"""
fetch_collection.py — Fetch all items from a ContentDM IIIF collection
and merge them into the exhibit's data JSON.

Existing items (matched by pointer/ID) are preserved with their manually
curated fields (featured, themes, company, etc.) and any existing CLIP
enrichment (visual_tags, similar_items, etc.).

New items are added with fields populated from the ContentDM metadata API.

Usage:
  python3 image-analysis/fetch_collection.py \\
      --collection p16002coll17 \\
      --output data/dance.json

  # Also pull from a second collection (Philadanco):
  python3 image-analysis/fetch_collection.py \\
      --collection p16002coll17 \\
      --collection p15037coll3 \\
      --pointers 11514,11516,11529 \\
      --output data/dance.json
"""

import argparse
import json
import time
from pathlib import Path

import requests


CONTENTDM_BASE = "https://cdm16002.contentdm.oclc.org"
TEMPLE_DIGITAL  = "https://digital.library.temple.edu/digital"
REQUEST_DELAY   = 0.4  # seconds between API calls


def get_all_pointers(collection: str) -> list:
    """Return all item pointers in a ContentDM collection.

    The dmQuery 'pointer' field sometimes returns empty strings depending on
    the environment; we derive the pointer from the 'find' filename instead,
    where the convention is: pointer = int(stem) - 1  (e.g. "21.jp2" → 20).
    """
    url = (
        f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
        f"?q=dmQuery/{collection}/0/pointer!descri/pointer/1000/0/1/0/0/0/json"
    )
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        d = r.json()
        recs = d.get("records", [])
        pointers = []
        for rec in recs:
            if rec.get("filetype") != "jp2" or rec.get("parentobject", -1) != -1:
                continue
            # Prefer the explicit pointer value if present
            ptr = rec.get("pointer")
            if ptr != "" and ptr is not None:
                pointers.append(int(ptr))
                continue
            # Fall back to deriving from the find filename ("21.jp2" → 20)
            find = rec.get("find", "")
            if find:
                try:
                    pointers.append(int(find.split(".")[0]) - 1)
                except ValueError:
                    pass
        return pointers
    except Exception as e:
        print(f"  [WARN] failed to list {collection}: {e}")
        return []


def fetch_item_info(collection: str, pointer: int) -> dict:
    """Fetch full metadata for a single item via dmGetItemInfo."""
    url = (
        f"{CONTENTDM_BASE}/digital/bl/dmwebservices/index.php"
        f"?q=dmGetItemInfo/{collection}/{pointer}/json"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    [WARN] dmGetItemInfo failed for {collection}/{pointer}: {e}")
        return {}


def parse_subjects(raw: str) -> list:
    """Split a ContentDM semicolon-delimited subjects string into a list."""
    if not raw or not isinstance(raw, str):
        return []
    return [s.strip().rstrip(";").strip() for s in raw.split(";") if s.strip().rstrip(";").strip()]


def build_item(collection: str, pointer: int, info: dict) -> dict:
    """Build an exhibit item dict from ContentDM metadata."""
    title = info.get("title", "").strip() or f"Untitled ({collection}/{pointer})"
    return {
        "title":       title,
        "featured":    False,
        "record":      f"{TEMPLE_DIGITAL}/collection/{collection}/id/{pointer}",
        "manifest":    f"{CONTENTDM_BASE}/iiif/{collection}:{pointer}/manifest.json",
        "date":        info.get("date", "Undated").strip() or "Undated",
        "photographer": (info.get("creato", "") or "").strip() or None,
        "subjects":    parse_subjects(info.get("subjec", "")),
        "description": (info.get("descri", "") or "").strip(),
        "_pointer":    pointer,         # keep for dedup; strip before shipping
        "_collection": collection,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", action="append", default=[],
                        help="ContentDM collection alias (repeatable)")
    parser.add_argument("--pointers", default="",
                        help="Comma-separated specific pointers to fetch (for partial pulls)")
    parser.add_argument("--output", required=True,
                        help="Path to dance.json to merge into")
    args = parser.parse_args()

    collections = args.collection or ["p16002coll17"]

    # Load existing data
    out_path = Path(args.output)
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
    else:
        data = {"collection": {}, "items": []}

    existing_items = data.get("items", [])

    # Build an index of existing items by (collection, pointer) if available,
    # or by manifest URL as fallback
    existing_by_manifest = {item["manifest"]: item for item in existing_items}

    new_items = []
    total_fetched = 0

    for coll in collections:
        print(f"\nCollection: {coll}")

        if args.pointers:
            pointers = [int(p.strip()) for p in args.pointers.split(",") if p.strip()]
        else:
            print("  Fetching pointer list...")
            pointers = get_all_pointers(coll)
            print(f"  Found {len(pointers)} items")

        for ptr in pointers:
            manifest_url = f"{CONTENTDM_BASE}/iiif/{coll}:{ptr}/manifest.json"

            # Skip if already in the data file
            if manifest_url in existing_by_manifest:
                continue

            print(f"  Fetching {coll}/{ptr}...")
            info = fetch_item_info(coll, ptr)
            time.sleep(REQUEST_DELAY)

            if not info or info.get("code") == -2:
                print(f"    [SKIP] item not found or access denied")
                continue

            # Only keep photographs (filter out posters, clippings, newsletters)
            item_type = (info.get("type", "") or "").lower()
            if item_type and "photograph" not in item_type:
                print(f"    [SKIP] type={info.get('type', '?')!r}")
                continue

            item = build_item(coll, ptr, info)
            new_items.append(item)
            total_fetched += 1

    # Strip internal tracking fields before writing
    for item in new_items:
        item.pop("_pointer", None)
        item.pop("_collection", None)
        # Remove None-valued keys so JSON stays clean
        for k in list(item.keys()):
            if item[k] is None:
                del item[k]

    print(f"\nExisting items: {len(existing_items)}")
    print(f"New items fetched: {total_fetched}")

    data["items"] = existing_items + new_items

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Total items now: {len(data['items'])}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
