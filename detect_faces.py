#!/usr/bin/env python3
"""
Face detection — pipeline step 1.

Uses SCRFD (InsightFace buffalo_sc) to find which photos contain at least one face.

Pilot mode (--pilot):
    Processes the first N photos (default 2000), writes a tab-separated TXT report:
        folder  photo  faces
        Vacanze IMG_001.jpg  yes
        Vacanze IMG_002.jpg  no
        ...

Full mode (default):
    Processes every photo in photo_embeddings where has_faces IS NULL and stores
    the boolean result in that column.  Run --init-db once first to add it.

Usage:
    python3 detect_faces.py --init-db                # add has_faces column (once)
    python3 detect_faces.py --pilot                  # pilot: first 2000 → pilot_report.txt
    python3 detect_faces.py --pilot --limit 500 --output report.txt
    python3 detect_faces.py                          # full run
    python3 detect_faces.py --device cuda            # GPU inference
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_cuda_libs() -> None:
    """
    On WSL2 / venv setups PyTorch ships CUDA as nvidia-* packages whose .so
    files are NOT on LD_LIBRARY_PATH.  onnxruntime-gpu needs them at process
    start, so we patch the env and re-exec once if necessary.
    """
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

DEFAULT_DB_URL    = "postgresql://postgres:postgres@localhost:5433/photos"
DEFAULT_MODEL     = "buffalo_sc"
DEFAULT_DET_SIZE  = 640
DEFAULT_LIMIT     = 2000
DEFAULT_OUTPUT    = "pilot_report.txt"
BATCH_SIZE        = 50   # DB commit interval for full run


def get_db_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


def get_conn() -> psycopg.Connection:
    return psycopg.connect(get_db_url())


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "ALTER TABLE photo_embeddings ADD COLUMN IF NOT EXISTS has_faces BOOLEAN"
        )
        conn.commit()
    print("Column has_faces added to photo_embeddings (or already existed).")


# ── Detector setup ─────────────────────────────────────────────────────────────

def load_detector(model: str, det_size: int, device: str):
    from insightface.app import FaceAnalysis

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(
        name=model,
        allowed_modules=["detection"],
        providers=providers,
    )
    ctx_id = 0 if device == "cuda" else -1
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def _imread(file_path: str):
    """
    Load an image as a BGR numpy array.
    cv2 is tried first; falls back to Pillow for non-standard JPEGs that
    cause libjpeg to print 'Invalid SOS parameters for sequential JPEG'.
    stderr is suppressed during the cv2 call to keep output clean.
    """
    # Suppress libjpeg warnings printed directly to stderr by cv2
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

    # Pillow fallback — handles progressive/non-standard JPEGs gracefully
    try:
        pil = ImageOps.exif_transpose(Image.open(file_path).convert("RGB"))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def detect(detector, file_path: str) -> bool | None:
    """Return True/False if face detected, None on read/decode error."""
    try:
        img = _imread(file_path)
        if img is None:
            return None
        faces = detector.get(img)
        return len(faces) > 0
    except Exception:
        return None


# ── Pilot mode ─────────────────────────────────────────────────────────────────

def run_pilot(detector, limit: int, output: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT file_path FROM photo_embeddings ORDER BY id LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()

    total = len(rows)
    print(f"Pilot: processing {total} photos → {output}")

    yes = no = errors = 0
    lines: list[str] = ["folder\tphoto\tfaces"]

    for i, (file_path,) in enumerate(rows, 1):
        p = Path(file_path)
        result = detect(detector, file_path)

        if result is None:
            tag = "error"
            errors += 1
        elif result:
            tag = "yes"
            yes += 1
        else:
            tag = "no"
            no += 1

        lines.append(f"{p.parent.name}\t{p.name}\t{tag}")

        if i % 200 == 0 or i == total:
            print(f"  {i}/{total}  yes={yes}  no={no}  errors={errors}")

    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {output}")
    print(f"  Faces detected : {yes}")
    print(f"  No faces       : {no}")
    print(f"  Errors         : {errors}")


# ── Full mode ─────────────────────────────────────────────────────────────────

def run_full(detector) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, file_path FROM photo_embeddings WHERE has_faces IS NULL ORDER BY id"
        )
        rows = cur.fetchall()

    total = len(rows)
    if total == 0:
        print("Nothing to process — all photos already have has_faces set.")
        return

    print(f"Full run: processing {total} photos with has_faces IS NULL")
    yes = no = errors = 0
    pending: list[tuple] = []

    def flush(conn: psycopg.Connection) -> None:
        if not pending:
            return
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE photo_embeddings SET has_faces = %s WHERE id = %s",
                pending,
            )
        conn.commit()
        pending.clear()

    with get_conn() as conn:
        for i, (photo_id, file_path) in enumerate(rows, 1):
            result = detect(detector, file_path)
            if result is None:
                errors += 1
            else:
                pending.append((result, photo_id))
                if result:
                    yes += 1
                else:
                    no += 1

            if len(pending) >= BATCH_SIZE:
                flush(conn)

            if i % 200 == 0 or i == total:
                print(f"  {i}/{total}  yes={yes}  no={no}  errors={errors}")

        flush(conn)

    print(f"\nDone.")
    print(f"  Faces detected : {yes}")
    print(f"  No faces       : {no}")
    print(f"  Errors         : {errors}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect faces with SCRFD (InsightFace) — pipeline step 1"
    )
    parser.add_argument("--init-db",  action="store_true",
                        help="Add has_faces column to photo_embeddings and exit")
    parser.add_argument("--pilot",    action="store_true",
                        help=f"Pilot mode: process first N photos and write TXT report")
    parser.add_argument("--limit",    type=int, default=DEFAULT_LIMIT,
                        help=f"Number of photos for pilot mode (default {DEFAULT_LIMIT})")
    parser.add_argument("--output",   default=DEFAULT_OUTPUT,
                        help=f"Pilot report file (default {DEFAULT_OUTPUT})")
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help=f"InsightFace model pack (default {DEFAULT_MODEL})")
    parser.add_argument("--det-size", type=int, default=DEFAULT_DET_SIZE,
                        help=f"Detection input size in pixels (default {DEFAULT_DET_SIZE})")
    parser.add_argument("--device",   choices=["cpu", "cuda"], default="cuda",
                        help="Inference device (default cuda)")
    args = parser.parse_args()

    if args.init_db:
        init_db()
        return

    print(f"Loading SCRFD detector  model={args.model}  device={args.device}  det-size={args.det_size}")
    detector = load_detector(args.model, args.det_size, args.device)
    print("Detector ready.\n")

    if args.pilot:
        run_pilot(detector, args.limit, args.output)
    else:
        run_full(detector)


if __name__ == "__main__":
    main()
