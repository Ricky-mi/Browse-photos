#!/usr/bin/env python3
"""
Cluster face embeddings — pipeline step 3.

Groups ArcFace embeddings using Mini-Batch K-means. Each cluster ideally
represents one person or a group of visually similar faces.

Usage:
    python3 cluster_faces.py --init-db             # add schema (once)
    python3 cluster_faces.py                       # auto cluster count
    python3 cluster_faces.py --num-clusters 20
    python3 cluster_faces.py --num-clusters 20 --random-state 7
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter

import numpy as np
import psycopg
from sklearn.cluster import MiniBatchKMeans

DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5433/photos"


def get_db_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


def get_conn() -> psycopg.Connection:
    return psycopg.connect(get_db_url())


# ── Schema ─────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_clusters (
                id          INTEGER PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                face_count  INTEGER DEFAULT 0
            )
        """)
        cur.execute(
            "ALTER TABLE face_embeddings "
            "ADD COLUMN IF NOT EXISTS face_cluster_id INTEGER REFERENCES face_clusters(id)"
        )
        conn.commit()
    print("Table face_clusters and column face_cluster_id ready (or already existed).")


# ── Clustering ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster face embeddings with Mini-Batch K-means"
    )
    parser.add_argument("--init-db",      action="store_true",
                        help="Create face_clusters table and face_cluster_id column, then exit")
    parser.add_argument("--num-clusters", type=int, default=0,
                        help="Number of clusters (default: auto = max(5, min(50, √n)))")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility (default 42)")
    args = parser.parse_args()

    if args.init_db:
        init_db()
        return

    # Load embeddings
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, embedding FROM face_embeddings WHERE embedding IS NOT NULL"
        )
        rows = cur.fetchall()

    if not rows:
        print("No face embeddings found — run embed_faces.py first.")
        return

    ids = [r[0] for r in rows]
    # pgvector returns embeddings as strings "[x,y,...]" — parse them
    X   = np.array(
        [[float(v) for v in r[1].strip("[]").split(",")] for r in rows],
        dtype=np.float32,
    )
    n   = len(X)

    k = args.num_clusters if args.num_clusters > 0 else max(5, min(50, int(math.sqrt(n))))
    print(f"Clustering {n} face embeddings into {k} clusters  (random_state={args.random_state})")

    km     = MiniBatchKMeans(n_clusters=k, random_state=args.random_state, n_init=3)
    labels = (km.fit_predict(X) + 1).tolist()   # 1-based IDs

    counts = Counter(labels)

    with get_conn() as conn, conn.cursor() as cur:
        # Clear previous clusters (null FK first to satisfy constraint)
        cur.execute("UPDATE face_embeddings SET face_cluster_id = NULL")
        cur.execute("DELETE FROM face_clusters")

        # Insert new clusters (sorted by face count descending so #1 is the largest)
        ranked = sorted(counts.items(), key=lambda x: -x[1])
        new_id_map = {old_id: new_id for new_id, (old_id, _) in enumerate(ranked, 1)}

        cur.executemany(
            "INSERT INTO face_clusters (id, description, face_count) VALUES (%s, %s, %s)",
            [
                (new_id_map[old_id], f"Face cluster {new_id_map[old_id]}", cnt)
                for old_id, cnt in ranked
            ],
        )

        # Update face_embeddings with remapped cluster IDs
        cur.executemany(
            "UPDATE face_embeddings SET face_cluster_id = %s WHERE id = %s",
            [(new_id_map[labels[i]], ids[i]) for i in range(n)],
        )
        conn.commit()

    print(f"\nDone — {k} face clusters created:")
    for old_id, cnt in ranked[:20]:
        new_id = new_id_map[old_id]
        print(f"  Cluster {new_id:3d}: {cnt:4d} faces")
    if len(ranked) > 20:
        print(f"  ... and {len(ranked) - 20} more")


if __name__ == "__main__":
    main()
