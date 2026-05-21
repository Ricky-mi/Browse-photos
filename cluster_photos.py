#!/usr/bin/env python3
"""
Cluster photos using existing CLIP embeddings stored in PostgreSQL.

Adds a `cluster_id` column to `photo_embeddings` and creates a `clusters`
table.  Runs K-means on the stored 512-d vectors — no images re-processed.

Usage:
    python3 cluster_photos.py                       # 50 clusters (default)
    python3 cluster_photos.py --num-clusters 80
    python3 cluster_photos.py --init-db             # schema only, no clustering
    python3 cluster_photos.py --num-clusters 40 --random-state 7
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sklearn.cluster import MiniBatchKMeans

FETCH_BATCH = 2000

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
        cur.execute(
            "ALTER TABLE photo_embeddings ADD COLUMN IF NOT EXISTS cluster_id INTEGER"
        )
        cur.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id          INTEGER PRIMARY KEY,
                description TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS photo_embeddings_cluster_id_idx "
            "ON photo_embeddings (cluster_id)"
        )
    conn.commit()
    logger.info("Schema ready (cluster_id column + clusters table + index)")


def clear_clusters(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE photo_embeddings SET cluster_id = NULL")
        cur.execute("TRUNCATE TABLE clusters")
    conn.commit()
    logger.info("Previous cluster assignments cleared")


def fetch_all_embeddings(conn) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids int64 array, embeddings float32 matrix) for all rows."""
    ids_list: list[int] = []
    embs_list: list[np.ndarray] = []
    offset = 0
    while True:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, embedding FROM photo_embeddings ORDER BY id "
                "LIMIT %s OFFSET %s",
                (FETCH_BATCH, offset),
            )
            rows = cur.fetchall()
        if not rows:
            break
        for row_id, emb in rows:
            ids_list.append(int(row_id))
            embs_list.append(np.asarray(emb, dtype="float32"))
        offset += len(rows)
        logger.info("Fetched %d embeddings ...", offset)

    ids = np.array(ids_list, dtype=np.int64)
    embeddings = np.stack(embs_list)
    logger.info("Loaded %d embeddings, shape %s", len(ids), embeddings.shape)
    return ids, embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster photos from stored CLIP embeddings using K-means"
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=50,
        help="Number of K-means clusters (default: 50)",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Create schema only (cluster_id column + clusters table) and exit",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for K-means reproducibility (default: 42)",
    )
    args = parser.parse_args()

    db_url = get_db_url()

    with psycopg.connect(db_url) as conn:
        register_vector(conn)
        init_db(conn)

        if args.init_db:
            logger.info("Schema initialised. Run without --init-db to cluster.")
            return

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM photo_embeddings")
            total = int(cur.fetchone()[0])

        if total == 0:
            logger.info("No embeddings found. Exiting.")
            return

        if args.num_clusters > total:
            logger.error(
                "--num-clusters %d exceeds number of photos %d",
                args.num_clusters, total,
            )
            sys.exit(1)

        clear_clusters(conn)

        ids, embeddings = fetch_all_embeddings(conn)

        logger.info(
            "Running MiniBatchKMeans: %d clusters on %d photos ...",
            args.num_clusters, len(ids),
        )
        kmeans = MiniBatchKMeans(
            n_clusters=args.num_clusters,
            random_state=args.random_state,
            n_init=3,
            batch_size=min(10 * args.num_clusters, len(ids)),
            verbose=0,
        )
        raw_labels = kmeans.fit_predict(embeddings)
        logger.info("Clustering done. Inertia: %.2f", float(kmeans.inertia_))

        # Map 0-based K-means labels to 1-based cluster IDs
        unique = sorted(set(raw_labels.tolist()))
        remap = {lbl: i + 1 for i, lbl in enumerate(unique)}
        cluster_ids = np.array([remap[lbl] for lbl in raw_labels.tolist()], dtype=np.int32)
        n_clusters = len(unique)

        # Insert cluster records
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO clusters (id, description) VALUES (%s, %s)",
                [(i, f"cluster {i}") for i in range(1, n_clusters + 1)],
            )
        conn.commit()
        logger.info("Inserted %d cluster records", n_clusters)

        # Bulk-update photo_embeddings using unnest for a single round-trip
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE photo_embeddings pe
                SET cluster_id = v.cluster_id
                FROM unnest(%s::bigint[], %s::integer[]) AS v(photo_id, cluster_id)
                WHERE pe.id = v.photo_id
                """,
                (ids.tolist(), cluster_ids.tolist()),
            )
        conn.commit()
        logger.info("Updated cluster_id for all %d photos", len(ids))

    sizes = np.bincount(cluster_ids)[1:]   # index 0 unused (1-based IDs)
    logger.info(
        "Done. %d clusters — sizes: min=%d  max=%d  mean=%.1f",
        n_clusters,
        int(sizes.min()),
        int(sizes.max()),
        float(sizes.mean()),
    )


if __name__ == "__main__":
    main()
