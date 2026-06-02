#!/usr/bin/env python3
"""
Reclassify photos using embeddings already stored in PostgreSQL.

Loads each stored 512-d CLIP vector, computes cosine similarity against
9 Italian category descriptions (embedded in English), and writes the
winning category back to the DB.  Photos with no confident match are
labelled "Other".  category_2 / category_3 are cleared (single-label output).

Usage:
    python3 reclassify_photos.py                  # defaults
    python3 reclassify_photos.py --min-score 0.18 # stricter Other threshold
    python3 reclassify_photos.py --dry-run         # preview without DB writes
    python3 reclassify_photos.py --device cuda     # GPU
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import psycopg
import torch
from pgvector.psycopg import register_vector
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
FETCH_BATCH = 500   # rows fetched per SELECT
COMMIT_EVERY = 200  # rows updated per COMMIT

# Display labels paired with English CLIP prompts.
CATEGORIES: list[tuple[str, str]] = [
    (
        "Portrait (1-3 persons)",
        "a portrait photograph showing one, two, or three people with visible faces or upper bodies",
    ),
    (
        "Photo by night",
        "a photograph taken at nighttime with dark sky, artificial lights, street lights, or stars",
    ),
    (
        "Summer beach landscape",
        "a sunny summer beach scene with sea or ocean water, sand, and coastline",
    ),
    (
        "Summer mountain landscape",
        "a summer mountain landscape with green grass, trees, and rocky peaks under blue sky",
    ),
    (
        "Winter mountain landscape",
        "a winter mountain landscape with snow-covered slopes or frozen alpine scenery",
    ),
    (
        "Autumn",
        "an autumn or fall scene with orange, red, and yellow leaves, foliage, or seasonal colors",
    ),
    (
        "Flowers",
        "a close-up or garden photo of flowers, petals, blossoms, or a flower garden",
    ),
    (
        "Macrophotography",
        "an extreme close-up macro photograph revealing fine details of tiny objects like insects, water drops, or textures",
    ),
    (
        "Nature",
        "a natural landscape with forests, meadows, rivers, lakes, or open countryside without people",
    ),
    (
        "Trees",
        "a photo centered on trees, a forest, woodland, or tall trees against the sky",
    ),
    (
        "Urban photos of cities and small towns",
        "urban photography of a city or small town with streets, buildings, shops, and contemporary architecture",
    ),
    (
        "Ancient Italian cities",
        "an ancient Italian city with historic architecture, old piazzas, churches, monuments, or medieval streets",
    ),
    (
        "Car, motorcycles, bikes",
        "a photo featuring a car, motorcycle, bicycle, or other motor vehicle on a road",
    ),
]
CATEGORY_OTHER = "Other"
LABELS = [label for label, _ in CATEGORIES]
PROMPTS = [prompt for _, prompt in CATEGORIES]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_db_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5433/photos",
    )


def _norm(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True)


def _to_tensor(output) -> torch.Tensor:
    """Extract a plain tensor from a model output (handles BaseModelOutputWithPooling)."""
    return output.pooler_output if hasattr(output, "pooler_output") else output


def load_model(device: str):
    logger.info("Loading CLIP model %s on %s ...", CLIP_MODEL_ID, device)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def build_text_features(model, processor, device: str) -> tuple[np.ndarray, float]:
    """Return ((9, 512) normalised text embeddings, CLIP logit_scale temperature)."""
    inputs = processor(text=PROMPTS, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_out = model.text_projection(model.text_model(**inputs).pooler_output)

    feats = _norm(text_out).cpu().numpy().astype("float32")
    logit_scale = float(model.logit_scale.exp().item())
    logger.info("CLIP logit_scale (temperature): %.2f", logit_scale)
    return feats, logit_scale


Row = tuple[
    str | None, float | None,   # category_1, score_1
    str | None, float | None,   # category_2, score_2
    str | None, float | None,   # category_3, score_3
]


def classify_batch(
    embeddings: np.ndarray,     # (N, 512), already normalised
    text_features: np.ndarray,  # (9, 512)
    logit_scale: float,         # CLIP temperature (model.logit_scale.exp())
    min_score: float,
    category_margin: float,
) -> list[Row]:
    """Return up to 3 (label, score) pairs per image using min_score and margin."""
    sims = embeddings @ text_features.T          # (N, 9) cosine similarities
    sims = sims * logit_scale                    # scale as CLIP was trained to use
    sims -= sims.max(axis=1, keepdims=True)      # numerical stability
    exp = np.exp(sims)
    probs = exp / exp.sum(axis=1, keepdims=True)  # (N, 9)

    results: list[Row] = []
    for prob_row in probs.tolist():
        ranked = sorted(enumerate(prob_row), key=lambda x: -x[1])
        best_score = ranked[0][1]

        selected: list[tuple[str, float]] = []
        for idx, score in ranked:
            if score < min_score:
                break
            if score < best_score - category_margin:
                break
            selected.append((LABELS[idx], score))
            if len(selected) == 3:
                break

        if not selected:
            selected.append((CATEGORY_OTHER, best_score))

        while len(selected) < 3:
            selected.append((None, None))

        results.append((
            selected[0][0], selected[0][1],
            selected[1][0], selected[1][1],
            selected[2][0], selected[2][1],
        ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reclassify photos from stored CLIP embeddings using new Italian categories"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for CLIP text encoding (default: cpu)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.15,
        help=(
            "Minimum softmax probability to keep a category (default: 0.15). "
            "Scores below this become 'Other'."
        ),
    )
    parser.add_argument(
        "--category-margin",
        type=float,
        default=0.10,
        help=(
            "Keep extra categories whose score is within this margin of the best "
            "(default: 0.10). Increase to assign more secondary categories."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print classifications without writing to the database",
    )
    parser.add_argument(
        "--diagnose",
        type=int,
        default=0,
        metavar="N",
        help="Print full score breakdown for the first N embeddings, then exit",
    )
    args = parser.parse_args()

    model, processor = load_model(args.device)
    text_features, logit_scale = build_text_features(model, processor, args.device)

    db_url = get_db_url()
    updated = 0
    other_count = 0

    with psycopg.connect(db_url) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM photo_embeddings")
            total = int(cur.fetchone()[0])
        logger.info("Rows to reclassify: %d", total)

        if args.diagnose > 0:
            from pathlib import Path
            from PIL import Image as PILImage

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, file_path, embedding FROM photo_embeddings ORDER BY id LIMIT %s",
                    (args.diagnose,),
                )
                sample_rows = cur.fetchall()

            sample_embs = np.stack([np.asarray(r[2], dtype="float32") for r in sample_rows])

            # ── Text-feature similarity table ──────────────────────────────
            sims = sample_embs @ text_features.T
            sims_scaled = sims * logit_scale
            exp = np.exp(sims_scaled - sims_scaled.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
            print(f"\n{'id':>8}  {'norm':>6}  " + "  ".join(f"{l[:12]:>12}" for l in LABELS))
            for (row_id, _, _), emb, prob_row in zip(sample_rows, sample_embs, probs):
                norm = float(np.linalg.norm(emb))
                scores = "  ".join(f"{p:>12.3f}" for p in prob_row)
                print(f"{row_id:>8}  {norm:>6.3f}  {scores}")
            print(f"\nRaw cosine sim range: [{sims.min():.4f}, {sims.max():.4f}]")

            # ── Image embedding consistency check ──────────────────────────
            # Re-compute image features for these photos with the current model
            # and compare against stored embeddings. Cosine ≈ 1.0 = model unchanged.
            print("\n--- Image embedding consistency (stored vs current model) ---")
            for row_id, file_path, stored_emb in sample_rows:
                p = Path(file_path)
                if not p.exists():
                    print(f"  id={row_id}: file missing ({file_path})")
                    continue
                try:
                    img = PILImage.open(p).convert("RGB")
                except Exception as e:
                    print(f"  id={row_id}: cannot open image: {e}")
                    continue
                img_inputs = processor(images=img, return_tensors="pt")
                img_inputs = {k: v.to(args.device) for k, v in img_inputs.items()}
                with torch.no_grad():
                    raw = model.get_image_features(**img_inputs)
                fresh = _norm(_to_tensor(raw)).cpu().numpy()[0]
                stored = np.asarray(stored_emb, dtype="float32")
                cos = float(stored @ fresh)
                dim = fresh.shape[0]
                print(f"  id={row_id}: dim={dim}  cos(stored, fresh)={cos:.4f}  {'OK' if cos > 0.99 else 'MISMATCH'}")
            return

        offset = 0
        while True:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, embedding FROM photo_embeddings ORDER BY id LIMIT %s OFFSET %s",
                    (FETCH_BATCH, offset),
                )
                rows = cur.fetchall()

            if not rows:
                break

            ids = [int(r[0]) for r in rows]
            embeddings = np.stack(
                [np.asarray(r[1], dtype="float32") for r in rows]
            )

            classifications = classify_batch(
                embeddings, text_features, logit_scale, args.min_score, args.category_margin
            )

            if args.dry_run:
                for photo_id, row in zip(ids, classifications):
                    c1, s1, c2, s2, c3, s3 = row
                    cats = f"{c1}({s1:.2f})"
                    if c2:
                        cats += f" | {c2}({s2:.2f})"
                    if c3:
                        cats += f" | {c3}({s3:.2f})"
                    logger.info("[dry-run] id=%-8d -> %s", photo_id, cats)
            else:
                with conn.cursor() as cur:
                    for photo_id, row in zip(ids, classifications):
                        c1, s1, c2, s2, c3, s3 = row
                        cur.execute(
                            """
                            UPDATE photo_embeddings SET
                                category_1       = %s,
                                category_1_score = %s,
                                category_2       = %s,
                                category_2_score = %s,
                                category_3       = %s,
                                category_3_score = %s,
                                classified_at    = NOW()
                            WHERE id = %s
                            """,
                            (c1, s1, c2, s2, c3, s3, photo_id),
                        )
                        updated += 1
                        if c1 == CATEGORY_OTHER:
                            other_count += 1

                        if updated % COMMIT_EVERY == 0:
                            conn.commit()
                            logger.info(
                                "Progress: %d / %d committed ...", updated, total
                            )

            other_count += sum(1 for row in classifications if row[0] == CATEGORY_OTHER) if args.dry_run else 0
            offset += len(rows)

        if not args.dry_run:
            conn.commit()

    if args.dry_run:
        logger.info("Dry-run complete. %d rows previewed, %d would be 'Other'.", total, other_count)
    else:
        logger.info(
            "Done. Updated: %d, classified as 'Other': %d (%.1f%%)",
            updated,
            other_count,
            100.0 * other_count / updated if updated else 0,
        )


if __name__ == "__main__":
    main()
