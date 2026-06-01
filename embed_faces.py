#!/usr/bin/env python3
"""
Face embedding — pipeline step 2.

For every photo that contains faces, runs SCRFD (detection) + ArcFace (recognition)
to produce a 512-dimensional embedding per face, stored in the face_embeddings table.

Pilot mode (--pilot):
    Reads photos tagged 'yes' from pilot_report.txt (first --limit entries),
    computes embeddings, and writes a summary TXT report.

Full mode (default):
    Processes all photos in photo_embeddings where has_faces = TRUE and
    face_embeddings rows are not yet present.

Pipeline:
    Photo → SCRFD → aligned crop + landmarks → ArcFace → vector(512)

Usage:
    python3 embed_faces.py --init-db                 # create face_embeddings table (once)
    python3 embed_faces.py --pilot                   # pilot from pilot_report.txt
    python3 embed_faces.py --pilot --limit 500 --report my_report.txt
    python3 embed_faces.py                           # full run
    python3 embed_faces.py --device cpu              # force CPU
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_cuda_libs() -> None:
    import site
    lib_dirs: list[str] = []
    for sp in site.getsitepackages():
        nv = os.path.join(sp, "nvidia")
        if os.path.isdir(nv):
            for pkg in os.listdir(nv):
                lib = os.path.join(nv, pkg, "lib")
                if os.path.isdir(lib):
                    lib_dirs.append(lib)
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    needed   = ":".join(lib_dirs)
    if needed not in existing:
        os.environ["LD_LIBRARY_PATH"] = needed + (":" + existing if existing else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_libs()

import cv2
import numpy as np
import psycopg
from PIL import Image, ImageOps

DEFAULT_DB_URL       = "postgresql://postgres:postgres@localhost:5433/photos"
DEFAULT_MODEL        = "buffalo_l"    # ResNet50 ArcFace — better quality than buffalo_sc
DEFAULT_DET_SIZE     = 640
DEFAULT_LIMIT        = 2000
DEFAULT_REPORT       = "pilot_report.txt"
DEFAULT_OUTPUT       = "embed_faces_report.txt"
DEFAULT_MIN_SCORE    = 0.5            # minimum SCRFD detection confidence
BATCH_SIZE           = 50


def get_db_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


def get_conn() -> psycopg.Connection:
    return psycopg.connect(get_db_url())


# ── Schema ─────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id          SERIAL PRIMARY KEY,
                photo_id    INTEGER NOT NULL
                                REFERENCES photo_embeddings(id) ON DELETE CASCADE,
                face_index  SMALLINT NOT NULL,
                embedding   vector(512),
                bbox_x1     REAL,
                bbox_y1     REAL,
                bbox_x2     REAL,
                bbox_y2     REAL,
                det_score   REAL,
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(photo_id, face_index)
            )
        """)
        # HNSW index for fast similarity search in the matching step
        cur.execute("""
            CREATE INDEX IF NOT EXISTS face_embeddings_hnsw
            ON face_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)
        conn.commit()
    print("Table face_embeddings and HNSW index created (or already existed).")


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(model: str, det_size: int, device: str, min_score: float):
    from insightface.app import FaceAnalysis
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(name=model, providers=providers)
    app.prepare(
        ctx_id=0 if device == "cuda" else -1,
        det_size=(det_size, det_size),
        det_thresh=min_score,
    )
    return app


# ── Image loading ──────────────────────────────────────────────────────────────

def _imread(file_path: str):
    """BGR numpy array; Pillow fallback for non-standard JPEGs."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved   = os.dup(2)
    os.dup2(devnull, 2)
    try:
        img = cv2.imread(file_path)
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)
    if img is not None:
        return img
    try:
        pil = ImageOps.exif_transpose(Image.open(file_path).convert("RGB"))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# ── Per-photo processing ───────────────────────────────────────────────────────

def process_photo(app, file_path: str) -> list[dict] | None:
    """
    Returns a list of face dicts (one per detected face), or None on read error.
    Each dict has: face_index, embedding, bbox, det_score.
    """
    img = _imread(file_path)
    if img is None:
        return None
    try:
        faces = app.get(img)
    except Exception:
        return None

    results = []
    for i, face in enumerate(faces):
        emb = face.embedding
        if emb is None:
            continue
        # Normalise to unit vector (ArcFace embeddings should already be, but be safe)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        bbox = face.bbox.tolist() if face.bbox is not None else [None] * 4
        results.append({
            "face_index": i,
            "embedding":  emb.tolist(),
            "bbox_x1":    bbox[0],
            "bbox_y1":    bbox[1],
            "bbox_x2":    bbox[2],
            "bbox_y2":    bbox[3],
            "det_score":  float(face.det_score) if face.det_score is not None else None,
        })
    return results


# ── Pilot mode ─────────────────────────────────────────────────────────────────

def read_pilot_report(report_path: str, limit: int) -> list[tuple[str, str]]:
    """Return list of (folder_name, photo_name) for the first `limit` 'yes' entries."""
    entries: list[tuple[str, str]] = []
    with open(report_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            if len(entries) >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[2] == "yes":
                entries.append((parts[0], parts[1]))
    return entries


def run_pilot(app, report: str, limit: int, output: str) -> None:
    entries = read_pilot_report(report, limit)
    total   = len(entries)
    print(f"Pilot: {total} photos with faces from {report}")

    # Build a lookup: (folder_name, filename) → (photo_id, file_path) from DB
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, file_path FROM photo_embeddings ORDER BY id LIMIT %s",
            (limit * 5,),
        )
        rows = cur.fetchall()

    path_map: dict[tuple[str, str], tuple[int, str]] = {}
    for photo_id, fp in rows:
        p = Path(fp)
        path_map[(p.parent.name, p.name)] = (photo_id, fp)

    photos_ok = faces_found = errors = 0
    lines   = ["folder\tphoto\tfaces_detected"]
    pending: list[tuple] = []

    def flush(conn: psycopg.Connection) -> None:
        if not pending:
            return
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO face_embeddings
                    (photo_id, face_index, embedding,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_score)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s)
                ON CONFLICT (photo_id, face_index) DO NOTHING
                """,
                pending,
            )
        conn.commit()
        pending.clear()

    with get_conn() as conn:
        for folder, photo in entries:
            hit = path_map.get((folder, photo))
            if hit is None:
                lines.append(f"{folder}\t{photo}\terror")
                errors += 1
                continue

            photo_id, fp = hit
            faces = process_photo(app, fp)
            if faces is None:
                lines.append(f"{folder}\t{photo}\terror")
                errors += 1
                continue

            photos_ok   += 1
            faces_found += len(faces)
            lines.append(f"{folder}\t{photo}\t{len(faces)}")

            for f in faces:
                emb_str = "[" + ",".join(f"{x:.6f}" for x in f["embedding"]) + "]"
                pending.append((
                    photo_id, f["face_index"], emb_str,
                    f["bbox_x1"], f["bbox_y1"], f["bbox_x2"], f["bbox_y2"],
                    f["det_score"],
                ))

            if len(pending) >= BATCH_SIZE:
                flush(conn)

            if photos_ok % 200 == 0 or photos_ok == total:
                print(f"  {photos_ok}/{total}  faces={faces_found}  errors={errors}")

        flush(conn)

    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {output}")
    print(f"  Photos processed : {photos_ok}")
    print(f"  Faces embedded   : {faces_found}")
    print(f"  Errors           : {errors}")


# ── Full mode ──────────────────────────────────────────────────────────────────

def run_full(app) -> None:
    # Photos with faces but no embeddings yet
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT pe.id, pe.file_path
            FROM photo_embeddings pe
            WHERE pe.has_faces = TRUE
              AND NOT EXISTS (
                  SELECT 1 FROM face_embeddings fe WHERE fe.photo_id = pe.id
              )
            ORDER BY pe.id
        """)
        rows = cur.fetchall()

    total = len(rows)
    if total == 0:
        print("Nothing to process — all photos with has_faces=TRUE already have embeddings.")
        return

    print(f"Full run: {total} photos to embed")
    photos_ok = faces_total = errors = 0
    pending: list[tuple] = []

    def flush(conn: psycopg.Connection) -> None:
        if not pending:
            return
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO face_embeddings
                    (photo_id, face_index, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_score)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s)
                ON CONFLICT (photo_id, face_index) DO NOTHING
                """,
                pending,
            )
        conn.commit()
        pending.clear()

    with get_conn() as conn:
        for i, (photo_id, file_path) in enumerate(rows, 1):
            faces = process_photo(app, file_path)
            if faces is None:
                errors += 1
            else:
                photos_ok   += 1
                faces_total += len(faces)
                for f in faces:
                    emb_str = "[" + ",".join(f"{x:.6f}" for x in f["embedding"]) + "]"
                    pending.append((
                        photo_id, f["face_index"], emb_str,
                        f["bbox_x1"], f["bbox_y1"], f["bbox_x2"], f["bbox_y2"],
                        f["det_score"],
                    ))

            if len(pending) >= BATCH_SIZE:
                flush(conn)

            if i % 200 == 0 or i == total:
                print(f"  {i}/{total}  faces={faces_total}  errors={errors}")

        flush(conn)

    print(f"\nDone.")
    print(f"  Photos processed : {photos_ok}")
    print(f"  Faces embedded   : {faces_total}")
    print(f"  Errors           : {errors}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed faces with ArcFace (InsightFace) — pipeline step 2"
    )
    parser.add_argument("--init-db",   action="store_true",
                        help="Create face_embeddings table + HNSW index and exit")
    parser.add_argument("--pilot",     action="store_true",
                        help="Pilot mode: process 'yes' photos from the report")
    parser.add_argument("--limit",     type=int, default=DEFAULT_LIMIT,
                        help=f"Max photos to process in pilot mode (default {DEFAULT_LIMIT})")
    parser.add_argument("--report",    default=DEFAULT_REPORT,
                        help=f"Pilot report to read from (default {DEFAULT_REPORT})")
    parser.add_argument("--output",    default=DEFAULT_OUTPUT,
                        help=f"Pilot output report (default {DEFAULT_OUTPUT})")
    parser.add_argument("--model",     default=DEFAULT_MODEL,
                        help=f"InsightFace model pack (default {DEFAULT_MODEL})")
    parser.add_argument("--det-size",  type=int, default=DEFAULT_DET_SIZE,
                        help=f"Detection input size (default {DEFAULT_DET_SIZE})")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"Minimum face detection confidence (default {DEFAULT_MIN_SCORE})")
    parser.add_argument("--device",    choices=["cpu", "cuda"], default="cuda",
                        help="Inference device (default cuda)")
    args = parser.parse_args()

    if args.init_db:
        init_db()
        return

    print(f"Loading model={args.model}  device={args.device}  det-size={args.det_size}  min-score={args.min_score}")
    app = load_model(args.model, args.det_size, args.device, args.min_score)
    print("Model ready.\n")

    if args.pilot:
        run_pilot(app, args.report, args.limit, args.output)
    else:
        run_full(app)


if __name__ == "__main__":
    main()
