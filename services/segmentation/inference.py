"""
Phase 2A — SegFormer batch inference on orthophoto tiles.

INPUTS (from Julie's inference_prep.py):
  data/raw/orthophoto_tiles/*.png              Preprocessed RGB tiles
  data/raw/orthophoto_tiles/tile_index.json    Spatial bounds per tile in EPSG:3347

  tile_index.json format Julie must produce:
  {
    "tile_001_002.png": { "bounds": [minx, miny, maxx, maxy], "crs": "EPSG:3347" },
    ...
  }

OUTPUTS:
  data/processed/segmentation_masks/*.npy       Per-tile class mask (H×W uint8)
  data/processed/segmentation_masks/mask_index.json

  mask pixel values are raw model class indices.
  mask_index.json["class_map"] maps index -> schema field name.

MODEL: jgerbscheid/segformer_b1-nlver_finetuned-1024-1024
  Primary classes: buildings, road/pavement, vegetation, water
  seg_land_pct and seg_unlabeled_pct are derived as residual in aggregate.py

USAGE:
  python -m services.segmentation.inference            # full run
  python -m services.segmentation.inference --test     # dry run with synthetic tile (no tiles needed)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TILE_DIR = REPO_ROOT / "data" / "raw" / "orthophoto_tiles"
MASK_DIR = REPO_ROOT / "data" / "processed" / "segmentation_masks"
MODEL_ID = "jgerbscheid/segformer_b1-nlver_finetuned-1024-1024"

# Maps model label strings (lowercased) -> our schema field names
# Model uses Dutch labels (jgerbscheid trained on Dutch aerial imagery):
#   pand=building, wegdeel=road, overbruggingsdeel=bridge/overpass, waterdeel=water, vegetatie=vegetation
#   background -> unmapped, counts as residual (seg_land_pct)
LABEL_TO_FIELD: dict[str, str] = {
    # Dutch labels (this model)
    "pand": "building",
    "wegdeel": "road",
    "overbruggingsdeel": "road",
    "waterdeel": "water",
    "vegetatie": "vegetation",
    # English fallbacks
    "building": "building",
    "buildings": "building",
    "road": "road",
    "road/pavement": "road",
    "pavement": "road",
    "roads": "road",
    "impervious surface": "road",
    "impervious": "road",
    "vegetation": "vegetation",
    "tree": "vegetation",
    "trees": "vegetation",
    "low vegetation": "vegetation",
    "grass": "vegetation",
    "water": "water",
}


def load_model(
    device: str,
) -> tuple[SegformerForSemanticSegmentation, SegformerImageProcessor, dict[int, str]]:
    log.info("Loading model: %s", MODEL_ID)
    # Some HuggingFace fine-tuned models omit preprocessor_config.json.
    # Fall back to a standard SegFormer processor with sensible defaults.
    try:
        processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    except OSError:
        log.warning("No processor config found for %s — using default SegformerImageProcessor", MODEL_ID)
        processor = SegformerImageProcessor(
            do_resize=True,
            size={"height": 1024, "width": 1024},
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.eval()
    model.to(device)

    id2label: dict[int, str] = model.config.id2label
    log.info("Model id2label: %s", id2label)

    class_map: dict[int, str] = {}
    for idx, label in id2label.items():
        field = LABEL_TO_FIELD.get(label.lower())
        if field:
            class_map[int(idx)] = field
        else:
            log.warning(
                "Unmapped model class %d=%r — pixels will count as residual (seg_land_pct)",
                idx,
                label,
            )

    log.info("Final class map (model_idx -> schema_field): %s", class_map)
    if not class_map:
        raise RuntimeError(
            "No model classes mapped to schema fields. Check LABEL_TO_FIELD against model id2label above."
        )
    return model, processor, class_map


def infer_tile(
    image: Image.Image,
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    device: str,
) -> np.ndarray:
    """Run inference on a single PIL image. Returns H×W uint8 mask of model class indices."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, num_classes, H/4, W/4)
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )
    mask = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask


def run_test(model, processor, class_map, device):
    """
    CHECKPOINT 1 — runs without any real tiles.
    Verifies model loads and produces correct output shape/dtype.
    """
    log.info("=== CHECKPOINT 1: Synthetic tile inference ===")
    synthetic = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
    mask = infer_tile(synthetic, model, processor, device)

    assert mask.shape == (1024, 1024), f"Bad shape: {mask.shape}"
    assert mask.dtype == np.uint8, f"Bad dtype: {mask.dtype}"

    unique, counts = np.unique(mask, return_counts=True)
    pcts = counts / counts.sum() * 100
    log.info("Class distribution on synthetic tile:")
    for cls, pct in zip(unique, pcts):
        field = class_map.get(int(cls), f"unmapped({cls})")
        log.info("  class %d (%s): %.1f%%", cls, field, pct)

    log.info("CHECKPOINT 1 PASSED — model output shape and dtype correct")
    log.info(
        ">>> USER CHECK: Does the class_map above make sense? "
        "Expected keys: building, road, vegetation, water. "
        "If any are missing or mislabeled, update LABEL_TO_FIELD in inference.py."
    )


def main(test_mode: bool = False) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    model, processor, class_map = load_model(device)

    if test_mode:
        run_test(model, processor, class_map, device)
        return

    # --- Full run ---
    tile_index_path = TILE_DIR / "tile_index.json"
    if not tile_index_path.exists():
        raise FileNotFoundError(
            f"tile_index.json not found at {tile_index_path}\n"
            "Julie's inference_prep.py must produce this file before inference can run."
        )
    with open(tile_index_path) as f:
        tile_index: dict = json.load(f)

    tiles = sorted(TILE_DIR.glob("*.png"))
    if not tiles:
        raise FileNotFoundError(f"No PNG tiles found in {TILE_DIR}")

    log.info("Found %d tiles", len(tiles))
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    mask_index: dict = {
        "class_map": {str(k): v for k, v in class_map.items()},
        "tiles": {},
    }

    for i, tile_path in enumerate(tiles):
        tile_name = tile_path.name
        if tile_name not in tile_index:
            log.warning("Tile %r not in tile_index.json — skipping", tile_name)
            continue

        log.info("[%d/%d] %s", i + 1, len(tiles), tile_name)
        image = Image.open(tile_path).convert("RGB")
        mask = infer_tile(image, model, processor, device)

        mask_name = tile_path.stem + ".npy"
        np.save(MASK_DIR / mask_name, mask)

        meta = tile_index[tile_name]
        mask_index["tiles"][mask_name] = {
            "source_tile": tile_name,
            "bounds": meta["bounds"],  # [minx, miny, maxx, maxy] EPSG:3347
            "crs": meta.get("crs", "EPSG:3347"),
            "shape": list(mask.shape),
        }

        # Log sample distribution every 10 tiles
        if (i + 1) % 10 == 0 or i == 0:
            unique, counts = np.unique(mask, return_counts=True)
            pcts = counts / counts.sum() * 100
            dist = {class_map.get(int(c), f"unmapped({c})"): f"{p:.1f}%" for c, p in zip(unique, pcts)}
            log.info("  Sample class distribution: %s", dist)

    mask_index_path = MASK_DIR / "mask_index.json"
    with open(mask_index_path, "w") as f:
        json.dump(mask_index, f, indent=2)

    log.info("Done. %d masks saved to %s", len(mask_index["tiles"]), MASK_DIR)
    log.info(
        "=== CHECKPOINT 2: All tiles processed ===\n"
        ">>> USER CHECK: Open a few masks in data/processed/segmentation_masks/ "
        "and visually compare against the source tiles in orthophoto_tiles/. "
        "Buildings, roads, vegetation, and water should be plausibly segmented."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Dry run with synthetic tile")
    args = parser.parse_args()
    main(test_mode=args.test)
