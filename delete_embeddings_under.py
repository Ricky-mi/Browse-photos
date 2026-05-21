#!/usr/bin/env python3
"""
One-off utility: delete rows in photo_embeddings whose file_path is a given directory
or any path under it (e.g. stale WSL paths after moving the library).

Uses DATABASE_URL if set; otherwise the same default as embed_photos.py.

Examples:
  .venv/bin/python delete_embeddings_under.py
  .venv/bin/python delete_embeddings_under.py /mnt/e/_Scanner
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg

DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5433/photos"


def get_db_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete embedding rows whose file_path equals DIR or starts with DIR/"
    )
    parser.add_argument(
        "dir_prefix",
        nargs="?",
        default="/mnt/e/_Scanner",
        help="Directory path prefix to remove (default: %(default)s)",
    )
    args = parser.parse_args()
    root = args.dir_prefix.strip().rstrip("/")
    if not root:
        print("dir_prefix must be non-empty", file=sys.stderr)
        return 1

    with psycopg.connect(get_db_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM photo_embeddings
                WHERE file_path = %s OR file_path LIKE %s
                """,
                (root, root + "/%"),
            )
            deleted = cur.rowcount
        conn.commit()

    print(f"Deleted {deleted} row(s) matching {root!r} or under it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
