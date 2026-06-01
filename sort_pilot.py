#!/usr/bin/env python3
"""
Read pilot_report.txt and copy photos into two folders:
  faces/     — photos where faces were detected
  no_faces/  — photos where no faces were detected

Subfolder structure is preserved (one level) to avoid filename collisions:
  faces/Vacanze/IMG_001.jpg
  no_faces/Lavoro/scan_042.jpg

Usage:
    python3 sort_pilot.py
    python3 sort_pilot.py --report pilot_report.txt --dest ./sorted
    python3 sort_pilot.py --link          # symlink instead of copy (saves disk space)
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import psycopg

DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5433/photos"
DEFAULT_REPORT = "pilot_report.txt"
DEFAULT_DEST   = "."


def get_db_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


def read_report(report_path: str) -> dict[tuple[str, str], str]:
    """Return {(folder_name, photo_name): tag} from the report file."""
    records: dict[tuple[str, str], str] = {}
    with open(report_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                folder, photo, tag = parts
                if tag in ("yes", "no"):
                    records[(folder, photo)] = tag
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort pilot photos into faces / no_faces folders")
    parser.add_argument("--report", default=DEFAULT_REPORT,
                        help=f"Pilot report file (default {DEFAULT_REPORT})")
    parser.add_argument("--dest",   default=DEFAULT_DEST,
                        help=f"Parent directory for output folders (default current dir)")
    parser.add_argument("--link",   action="store_true",
                        help="Create symlinks instead of copying files (saves disk space)")
    args = parser.parse_args()

    report = read_report(args.report)
    if not report:
        print(f"No usable rows found in {args.report}")
        return

    yes_count = sum(1 for t in report.values() if t == "yes")
    no_count  = sum(1 for t in report.values() if t == "no")
    print(f"Report: {len(report)} photos  ({yes_count} with faces, {no_count} without)")

    dest       = Path(args.dest)
    faces_dir  = dest / "faces"
    noface_dir = dest / "no_faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    noface_dir.mkdir(parents=True, exist_ok=True)

    # Query DB for full file paths, ordered the same way as the pilot run
    with psycopg.connect(get_db_url()) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT file_path FROM photo_embeddings ORDER BY id LIMIT %s",
            (len(report),),
        )
        rows = cur.fetchall()

    copied = skipped = errors = 0

    for (file_path,) in rows:
        p   = Path(file_path)
        key = (p.parent.name, p.name)
        tag = report.get(key)

        if tag is None:
            skipped += 1
            continue

        out_dir  = faces_dir if tag == "yes" else noface_dir
        out_sub  = out_dir / p.parent.name
        out_sub.mkdir(exist_ok=True)
        out_file = out_sub / p.name

        if out_file.exists():
            skipped += 1
            continue

        try:
            if args.link:
                out_file.symlink_to(p.resolve())
            else:
                shutil.copy2(file_path, out_file)
            copied += 1
        except Exception as e:
            print(f"  ERROR {file_path}: {e}")
            errors += 1

    action = "linked" if args.link else "copied"
    print(f"\nDone — {action} {copied} photos, skipped {skipped}, errors {errors}")
    print(f"  {faces_dir}/")
    print(f"  {noface_dir}/")


if __name__ == "__main__":
    main()
