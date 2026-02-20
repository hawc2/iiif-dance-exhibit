# Image Analysis Pipeline

Uses [CLIP](https://openai.com/research/clip) (via HuggingFace `transformers`) to enrich each item in the
collection data with machine-detected visual tags, face detection, and visual similarity groupings.

## What it produces

Each item in `data/dance.json` gains four new fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | URL-safe slug derived from the title |
| `image_url` | string | Primary image URL resolved from the IIIF manifest |
| `visual_tags` | list | Top-4 CLIP zero-shot classification labels |
| `face_detected` | bool | Whether faces are prominent in the image |
| `similar_items` | list | IDs of the 3 most visually similar items |

## Requirements

Python 3.9+ with the following packages (all available in the project venv):

- `torch`
- `transformers`
- `Pillow`
- `numpy`
- `requests`

## Running

From the repo root:

```bash
# Enrich the dance exhibit (flat items list):
/path/to/.venv/bin/python3 image-analysis/enrich_images.py \
  --input data/dance.json \
  --output data/dance.json

# Enrich tu-digital-collections (nested collections, cross-collection similarity):
/path/to/.venv/bin/python3 image-analysis/enrich_images.py \
  --input ../tu-digital-collections/data/collections.json \
  --output ../tu-digital-collections/data/collections.json \
  --mode collections
```

IIIF manifests are cached in `.manifest_cache/` so re-runs skip network fetches.
Delete that directory to force a full refresh.

## Options

| Flag | Default | Description |
|---|---|---|
| `--top-tags` | 4 | Number of visual tags per item |
| `--top-similar` | 3 | Number of similar items per item |
| `--cache-dir` | `.manifest_cache` | Path for manifest cache |
| `--dry-run` | false | Process images without writing output |

## How it works

1. **Manifest fetch** — downloads each item's IIIF Presentation manifest and extracts the primary image URL
2. **CLIP zero-shot tagging** — scores each image against 14 descriptive prompts (e.g. "classical ballet with pointe shoes and tutus") and keeps the top 4
3. **Face detection** — zero-shot comparison between "a portrait where faces are clearly visible" and "a photo where no faces are visible"
4. **Embedding + similarity** — extracts a 512-d CLIP image embedding, builds a cosine similarity matrix, and assigns the 3 nearest neighbors per item

In `--mode collections`, Phase 2 runs across all collections so items can match across collection boundaries.
