#!/usr/bin/env python3
"""
Embed photos from a configurable folder using CLIP (openai/clip-vit-base-patch32)
and store vectors in PostgreSQL with pgvector.

Parallelism strategy:
  - A ThreadPoolExecutor pre-loads and decodes the *next* super-batch of images
    from disk while CLIP runs inference on the *current* super-batch (pipeline
    overlap).
  - Each loader thread reads --read-batch-size images at once (default 8).
    A super-batch has loader_threads × read_batch_size images in total.
  - CLIP processes the collected super-batch in --batch-size chunks per forward
    pass, which saturates CPU/GPU far better than one image at a time.
  - DB writes stay on the main thread; psycopg connections are not thread-safe.
"""

import argparse
import logging
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import psycopg
import torch
from pgvector.psycopg import register_vector
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# WSL: map Windows drives as /mnt/<letter>/... (e.g. D:\_Scanner -> /mnt/d/_Scanner).
# Override without editing: export PHOTOS_DIR=/mnt/d/_Scanner
PHOTOS_DIR = Path(os.environ.get("PHOTOS_DIR", "/mnt/e/_Scanner"))

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_COMMIT_SIZE = 50
CATEGORY_LABELS = [
    "beach landscapes",
    "other landscapes",
    "macro photography",
    "flowers",
    "portrait of one person",
    "portrait of more than one person",
    "trees",
    "cities",
    "vehicles",
    "other",
]
CATEGORY_PROMPTS = [
    "a photo of a beach, sea, or coastline landscape",
    "a photo of a natural mountain, hill, countryside, or lake landscape, not beach",
    "a macro close-up photograph",
    "a close-up photo of flowers, petals, or blossoms",
    "a portrait photo focused on one person's face",
    "a group portrait photo with two or more people faces",
    "a photo of trees or a forest",
    "a photo of a city or urban scene",
    "a photo of a vehicle like a car, bike, bus, train, boat, or airplane",
    "a photo that does not match the previous categories",
]

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


def init_db(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        logger.info("Extension vector created/enabled")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS photo_embeddings (
                id BIGSERIAL PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                embedding vector(512) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("ALTER TABLE photo_embeddings ADD COLUMN IF NOT EXISTS category_1 TEXT")
        cur.execute("ALTER TABLE photo_embeddings ADD COLUMN IF NOT EXISTS category_1_score REAL")
        cur.execute("ALTER TABLE photo_embeddings ADD COLUMN IF NOT EXISTS category_2 TEXT")
        cur.execute("ALTER TABLE photo_embeddings ADD COLUMN IF NOT EXISTS category_2_score REAL")
        cur.execute("ALTER TABLE photo_embeddings ADD COLUMN IF NOT EXISTS category_3 TEXT")
        cur.execute("ALTER TABLE photo_embeddings ADD COLUMN IF NOT EXISTS category_3_score REAL")
        cur.execute("ALTER TABLE photo_embeddings ADD COLUMN IF NOT EXISTS classified_at TIMESTAMPTZ")
        conn.commit()
        logger.info("Table photo_embeddings created/verified")

        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS photo_embeddings_embedding_idx
                ON photo_embeddings USING hnsw (embedding vector_cosine_ops)
            """)
            conn.commit()
            logger.info("HNSW index created/verified")
        except Exception as e:
            conn.rollback()
            logger.warning("Index creation skipped (may already exist): %s", e)


def collect_image_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(path)
    return sorted(paths)


def load_existing_paths(conn) -> set[str]:
    """Return the set of file_path values already stored in the DB."""
    with conn.cursor() as cur:
        cur.execute("SELECT file_path FROM photo_embeddings")
        return {row[0] for row in cur.fetchall()}


def to_feature_tensor(output: torch.Tensor) -> torch.Tensor:
    return output.pooler_output if hasattr(output, "pooler_output") else output


def select_categories(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    min_category_score: float,
    category_margin: float,
) -> tuple[str | None, float | None, str | None, float | None, str | None, float | None]:
    """Return up to top-3 category labels/scores for a single image feature row."""
    similarities = torch.matmul(image_features, text_features.T).squeeze(0)
    probabilities = torch.softmax(similarities, dim=0)
    ranked_indices = torch.argsort(probabilities, descending=True).tolist()

    best_score = float(probabilities[ranked_indices[0]].item())
    selected: list[tuple[str, float]] = []

    for idx in ranked_indices:
        score = float(probabilities[idx].item())
        if score < min_category_score:
            continue
        if score < (best_score - category_margin):
            continue
        selected.append((CATEGORY_LABELS[idx], score))
        if len(selected) == 3:
            break

    if not selected:
        best_idx = ranked_indices[0]
        selected.append((CATEGORY_LABELS[best_idx], float(probabilities[best_idx].item())))

    while len(selected) < 3:
        selected.append((None, None))

    return (
        selected[0][0], selected[0][1],
        selected[1][0], selected[1][1],
        selected[2][0], selected[2][1],
    )


# ── Image loading (runs in thread pool) ───────────────────────────────────────

def _load_images(paths: list[Path]) -> list[tuple[Path, Image.Image | None, str | None]]:
    """Open and decode a batch of images in a single thread."""
    results = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
            results.append((path, img, None))
        except Exception as e:
            results.append((path, None, str(e)))
    return results


def _chunked(lst: list, size: int):
    """Yield successive chunks of `size` from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _submit_super_batch(
    executor: ThreadPoolExecutor,
    paths: list[Path],
    read_batch_size: int,
) -> list[Future]:
    """Submit one future per loader thread, each reading read_batch_size images."""
    return [
        executor.submit(_load_images, sub_batch)
        for sub_batch in _chunked(paths, read_batch_size)
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed photos with CLIP and store in PostgreSQL")
    parser.add_argument("--init-db", action="store_true",
                        help="Initialize database and exit")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Device for CLIP model (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Images per CLIP forward pass (default: 8). "
                             "Increase for GPU (16–32), decrease if RAM is tight.")
    parser.add_argument("--read-batch-size", type=int, default=8,
                        help="Images each loader thread reads at once (default: 8). "
                             "Total pre-loaded = loader_threads × read_batch_size.")
    parser.add_argument("--loader-threads", type=int, default=4,
                        help="Threads for parallel image loading (default: 4)")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip photos already in the DB (default: True). "
                             "Use --no-skip-existing to re-embed everything.")
    parser.add_argument("--min-category-score", type=float, default=0.08,
                        help="Minimum score to keep a predicted category (default: 0.08)")
    parser.add_argument("--category-margin", type=float, default=0.05,
                        help="Keep categories within this margin of the best score (default: 0.05)")
    parser.add_argument("--debug-topn", type=int, default=0,
                        help="Log top-3 category scores for first N processed images (default: 0)")
    parser.add_argument("--calibration-limit", type=int, default=0,
                        help="Process only first N images for threshold calibration (default: 0 = all)")
    args = parser.parse_args()

    db_url = get_db_url()

    with psycopg.connect(db_url) as conn:
        init_db(conn)
        register_vector(conn)
        if args.init_db:
            logger.info("Database initialized. Run without --init-db to embed photos.")
            return

    if not PHOTOS_DIR.exists():
        logger.error("Photos directory does not exist: %s", PHOTOS_DIR)
        sys.exit(1)

    logger.info("Loading CLIP model %s on %s...", CLIP_MODEL_ID, args.device)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(args.device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    text_inputs = processor(text=CATEGORY_PROMPTS, return_tensors="pt", padding=True, truncation=True)
    text_inputs = {k: v.to(args.device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_output = model.get_text_features(**text_inputs)
    text_features = to_feature_tensor(text_output)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image_paths = collect_image_paths(PHOTOS_DIR)
    if args.calibration_limit > 0:
        image_paths = image_paths[: args.calibration_limit]
        logger.info("Calibration mode: processing first %d images", len(image_paths))

    if args.skip_existing:
        with psycopg.connect(db_url) as conn:
            register_vector(conn)
            existing = load_existing_paths(conn)
        before = len(image_paths)
        image_paths = [p for p in image_paths if str(p.resolve()) not in existing]
        logger.info(
            "Skip-existing: %d already in DB, %d remaining to process",
            before - len(image_paths), len(image_paths),
        )

    super_chunk_size = args.loader_threads * args.read_batch_size
    logger.info(
        "Found %d images in %s — read_batch_size=%d, loader_threads=%d, "
        "super_chunk=%d, clip_batch=%d",
        len(image_paths), PHOTOS_DIR,
        args.read_batch_size, args.loader_threads,
        super_chunk_size, args.batch_size,
    )

    if not image_paths:
        logger.info("No images to process. Exiting.")
        return

    processed = 0
    skipped = 0
    total = len(image_paths)
    super_chunks = list(_chunked(image_paths, super_chunk_size))

    with psycopg.connect(db_url) as conn:
        register_vector(conn)

        with conn.cursor() as cur, ThreadPoolExecutor(max_workers=args.loader_threads) as executor:

            # Pre-load the first super-batch immediately.
            pending_futures: list[Future] = (
                _submit_super_batch(executor, super_chunks[0], args.read_batch_size)
                if super_chunks else []
            )

            for chunk_idx, super_chunk in enumerate(super_chunks):
                # While we run CLIP on this super-batch, pre-load the next one.
                next_futures: list[Future] = []
                if chunk_idx + 1 < len(super_chunks):
                    next_futures = _submit_super_batch(
                        executor, super_chunks[chunk_idx + 1], args.read_batch_size
                    )

                # Collect all images loaded by the current super-batch threads.
                loaded: list[tuple[Path, Image.Image]] = []
                for fut in pending_futures:
                    for path, img, err in fut.result():
                        if err:
                            logger.warning("Skipping corrupt/invalid image %s: %s", path, err)
                            skipped += 1
                        else:
                            loaded.append((path, img))

                pending_futures = next_futures

                if not loaded:
                    continue

                # ── CLIP inference in batch_size sub-chunks ────────────────
                for clip_chunk in _chunked(loaded, args.batch_size):
                    paths_ok, images_ok = zip(*clip_chunk)
                    try:
                        inputs = processor(images=list(images_ok), return_tensors="pt", padding=True)
                        inputs = {k: v.to(args.device) for k, v in inputs.items()}
                        with torch.no_grad():
                            output = model.get_image_features(**inputs)
                        batch_features = to_feature_tensor(output)
                        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                    except Exception as e:
                        logger.warning(
                            "Batch inference failed (super-chunk %d): %s — skipping %d images",
                            chunk_idx, e, len(clip_chunk),
                        )
                        skipped += len(clip_chunk)
                        continue

                    # ── Per-image results → DB ─────────────────────────────
                    for j, (path, _img) in enumerate(zip(paths_ok, images_ok)):
                        features = batch_features[j].unsqueeze(0)  # shape [1, 512]
                        embedding = features.cpu().numpy().astype("float32")[0]

                        (
                            category_1, category_1_score,
                            category_2, category_2_score,
                            category_3, category_3_score,
                        ) = select_categories(features, text_features,
                                              args.min_category_score, args.category_margin)

                        if args.debug_topn > 0 and processed < args.debug_topn:
                            similarities = torch.matmul(features, text_features.T).squeeze(0)
                            probabilities = torch.softmax(similarities, dim=0)
                            ranked = torch.argsort(probabilities, descending=True).tolist()[:3]
                            preview = [
                                f"{CATEGORY_LABELS[i]}={float(probabilities[i].item()):.3f}"
                                for i in ranked
                            ]
                            logger.info("Top-3 for %s -> %s", path.name, ", ".join(preview))

                        file_path = str(path.resolve())
                        cur.execute(
                            """
                            INSERT INTO photo_embeddings (
                                file_path, embedding,
                                category_1, category_1_score,
                                category_2, category_2_score,
                                category_3, category_3_score,
                                classified_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                            ON CONFLICT (file_path) DO UPDATE SET
                                embedding        = EXCLUDED.embedding,
                                category_1       = EXCLUDED.category_1,
                                category_1_score = EXCLUDED.category_1_score,
                                category_2       = EXCLUDED.category_2,
                                category_2_score = EXCLUDED.category_2_score,
                                category_3       = EXCLUDED.category_3,
                                category_3_score = EXCLUDED.category_3_score,
                                classified_at    = NOW(),
                                created_at       = NOW()
                            """,
                            (
                                file_path, embedding,
                                category_1, category_1_score,
                                category_2, category_2_score,
                                category_3, category_3_score,
                            ),
                        )
                        processed += 1
                        logger.info("[%d/%d] %s", processed + skipped, total, path.name)

                        if processed % BATCH_COMMIT_SIZE == 0:
                            conn.commit()
                            logger.info("Committed batch of %d", BATCH_COMMIT_SIZE)

            conn.commit()

    logger.info("Done. Processed: %d, Skipped: %d", processed, skipped)


if __name__ == "__main__":
    main()
