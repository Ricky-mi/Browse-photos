#!/usr/bin/env python3
"""
Simple frontend/backend to browse categorized photos from PostgreSQL.
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from typing import Any

import psycopg
from flask import Flask, abort, jsonify, request, send_file
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5433/photos"
CATEGORY_LABELS = [
    "Ritratti & Gruppi",
    "Panorami Naturali",
    "Mare & Coste",
    "Montagne & Rocce",
    "Urban & Architettura",
    "Natura da vicino",
    "Animali",
    "Food & Drink",
    "Interni & Musei",
    "Other",
]

app = Flask(__name__, static_folder="web", static_url_path="")

# ── Lazy CLIP loader ──────────────────────────────────────────────────────────
_clip_lock      = threading.Lock()
_clip_model     = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    with _clip_lock:
        if _clip_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415
                _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                _clip_model.eval()
            except Exception as exc:
                raise RuntimeError(f"Failed to load CLIP model: {exc}") from exc
    return _clip_model, _clip_processor


def _embed_text(text: str) -> list[float]:
    import torch  # noqa: PLC0415
    model, processor = _load_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model.get_text_features(**inputs)
    features = output.pooler_output if hasattr(output, "pooler_output") else output
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].tolist()


def get_db_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


def get_conn() -> psycopg.Connection:
    return psycopg.connect(get_db_url())


def init_db() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS playlists (
                id         SERIAL PRIMARY KEY,
                name       TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS playlist_photos (
                id          SERIAL PRIMARY KEY,
                playlist_id INTEGER NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
                photo_id    INTEGER NOT NULL REFERENCES photo_embeddings(id) ON DELETE CASCADE,
                prev_id     INTEGER,
                next_id     INTEGER,
                position    INTEGER NOT NULL DEFAULT 0,
                UNIQUE(playlist_id, photo_id)
            )
        """)
        conn.commit()

    # Face recognition tables — may fail if photo_embeddings doesn't exist yet
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS known_people (
                    id            SERIAL PRIMARY KEY,
                    name          TEXT NOT NULL UNIQUE,
                    reference_dir TEXT,
                    photo_count   INTEGER DEFAULT 0
                )
            """)
            cur.execute(
                "ALTER TABLE photo_embeddings "
                "ADD COLUMN IF NOT EXISTS person_names TEXT[]"
            )
            conn.commit()
    except Exception as _fe:
        import sys
        print(f"Face recognition schema init (non-fatal): {_fe}", file=sys.stderr)


try:
    init_db()
except Exception as _e:
    import sys
    print(f"DB init warning: {_e}", file=sys.stderr)


def _build_items(rows: list) -> list[dict]:
    items = []
    for row in rows:
        photo_id = int(row[0])
        file_path = row[1]
        cats = [
            {"name": row[2], "score": float(row[3]) if row[3] is not None else None},
            {"name": row[4], "score": float(row[5]) if row[5] is not None else None},
            {"name": row[6], "score": float(row[7]) if row[7] is not None else None},
        ]
        person_names = list(row[8]) if len(row) > 8 and row[8] else []
        items.append({
            "id": photo_id,
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "categories": [c for c in cats if c["name"]],
            "person_names": person_names,
            "thumbnail_url": f"/image?id={photo_id}&thumb=1",
            "image_url": f"/image?id={photo_id}",
        })
    return items


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/categories")
def categories() -> Any:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT category, COUNT(DISTINCT photo_id) AS cnt
            FROM (
                SELECT id AS photo_id, category_1 AS category
                FROM photo_embeddings WHERE category_1 IS NOT NULL
                UNION ALL
                SELECT id, category_2 FROM photo_embeddings WHERE category_2 IS NOT NULL
                UNION ALL
                SELECT id, category_3 FROM photo_embeddings WHERE category_3 IS NOT NULL
            ) t
            GROUP BY category
            ORDER BY cnt DESC, category ASC
            """
        )
        db_counts = {row[0]: int(row[1]) for row in cur.fetchall()}

    payload = [{"name": name, "count": db_counts.get(name, 0)} for name in CATEGORY_LABELS]
    return jsonify(payload)


@app.get("/api/clusters")
def clusters() -> Any:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.description, COUNT(pe.id) AS cnt
                FROM clusters c
                LEFT JOIN photo_embeddings pe ON pe.cluster_id = c.id
                GROUP BY c.id, c.description
                ORDER BY c.id
                """
            )
            rows = cur.fetchall()
    except Exception:
        return jsonify([])
    return jsonify([{"id": int(r[0]), "description": r[1], "count": int(r[2])} for r in rows])


@app.get("/api/people")
def people() -> Any:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT name, photo_count FROM known_people ORDER BY name"
            )
            rows = cur.fetchall()
    except Exception:
        return jsonify([])
    return jsonify([{"name": r[0], "count": int(r[1])} for r in rows])


@app.get("/api/face_clusters")
def face_clusters() -> Any:
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT fc.id, fc.description, fc.face_count,
                       COUNT(DISTINCT fe.photo_id) AS photo_count
                FROM face_clusters fc
                LEFT JOIN face_embeddings fe ON fe.face_cluster_id = fc.id
                GROUP BY fc.id, fc.description, fc.face_count
                ORDER BY fc.face_count DESC
                """
            )
            rows = cur.fetchall()
    except Exception:
        return jsonify([])
    return jsonify([
        {"id": int(r[0]), "description": r[1], "face_count": int(r[2]), "photo_count": int(r[3])}
        for r in rows
    ])


@app.patch("/api/face_clusters/<int:cluster_id>")
def update_face_cluster(cluster_id: int) -> Any:
    data = request.get_json(silent=True) or {}
    description = str(data.get("description", "")).strip()
    if not description:
        return jsonify({"error": "description is required"}), 400
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE face_clusters SET description = %s WHERE id = %s RETURNING id",
            (description, cluster_id),
        )
        if cur.fetchone() is None:
            return jsonify({"error": "Face cluster not found"}), 404
        conn.commit()
    return jsonify({"id": cluster_id, "description": description})


@app.patch("/api/clusters/<int:cluster_id>")
def update_cluster(cluster_id: int) -> Any:
    data = request.get_json(silent=True) or {}
    description = str(data.get("description", "")).strip()
    if not description:
        return jsonify({"error": "description is required"}), 400
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE clusters SET description = %s WHERE id = %s RETURNING id",
            (description, cluster_id),
        )
        if cur.fetchone() is None:
            return jsonify({"error": "Cluster not found"}), 404
        conn.commit()
    return jsonify({"id": cluster_id, "description": description})


def _photo_where(
    cluster_id_raw: str,
    cat_list: list[str],
    folder: str,
    person: str = "",
) -> tuple[str, list]:
    """Build a (WHERE clause, params list) for the photos query."""
    clauses: list[str] = []
    params: list = []

    if cluster_id_raw:
        clauses.append("cluster_id = %s")
        params.append(int(cluster_id_raw))
    elif person:
        clauses.append("%s = ANY(person_names)")
        params.append(person)
    else:
        for cat in cat_list:
            clauses.append("(%s = ANY(ARRAY[category_1, category_2, category_3]))")
            params.append(cat)

    if folder:
        clauses.append("regexp_replace(file_path, '/[^/]+$', '') = %s")
        params.append(folder)

    return " AND ".join(clauses), params


@app.get("/api/folders")
def folders() -> Any:
    """Return distinct parent folders, optionally scoped to the current filter."""
    cluster_id_raw      = request.args.get("cluster_id", "").strip()
    face_cluster_id_raw = request.args.get("face_cluster_id", "").strip()
    cat_list = [c.strip() for c in request.args.getlist("categories") if c.strip()]
    cat_list = [c for c in cat_list if c in CATEGORY_LABELS]
    person = request.args.get("person", "").strip()

    if face_cluster_id_raw and face_cluster_id_raw.isdigit():
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT regexp_replace(pe.file_path, '/[^/]+$', '') AS folder,
                       COUNT(DISTINCT pe.id) AS cnt
                FROM photo_embeddings pe
                JOIN face_embeddings fe ON fe.photo_id = pe.id
                WHERE fe.face_cluster_id = %s
                GROUP BY folder
                ORDER BY folder
                """,
                (int(face_cluster_id_raw),),
            )
            rows = cur.fetchall()
    else:
        where, params = _photo_where(cluster_id_raw, cat_list, "", person)
        where_sql = f"WHERE {where}" if where else ""
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT regexp_replace(file_path, '/[^/]+$', '') AS folder,
                       COUNT(*) AS cnt
                FROM photo_embeddings
                {where_sql}
                GROUP BY folder
                ORDER BY folder
                """,
                params,
            )
            rows = cur.fetchall()
    return jsonify([{"path": r[0], "count": int(r[1])} for r in rows])


@app.get("/api/photos")
def photos() -> Any:
    limit  = min(max(int(request.args.get("limit",  50)), 1), 200)
    offset = max(int(request.args.get("offset",  0)), 0)
    folder = request.args.get("folder", "").strip()

    face_cluster_id_raw = request.args.get("face_cluster_id", "").strip()
    cluster_id_raw      = request.args.get("cluster_id", "").strip()
    cat_list = [c.strip() for c in request.args.getlist("categories") if c.strip()]
    person   = request.args.get("person", "").strip()

    # ── Face cluster filter (requires JOIN on face_embeddings) ────────────────
    if face_cluster_id_raw:
        if not face_cluster_id_raw.isdigit():
            return jsonify({"error": "Invalid face_cluster_id"}), 400
        where_parts = ["fe.face_cluster_id = %s"]
        params: list = [int(face_cluster_id_raw)]
        if folder:
            where_parts.append("regexp_replace(pe.file_path, '/[^/]+$', '') = %s")
            params.append(folder)
        where_sql = " AND ".join(where_parts)
        params.extend([limit + 1, offset])
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT pe.id, pe.file_path,
                       pe.category_1, pe.category_1_score,
                       pe.category_2, pe.category_2_score,
                       pe.category_3, pe.category_3_score,
                       pe.person_names
                FROM photo_embeddings pe
                JOIN face_embeddings fe ON fe.photo_id = pe.id
                WHERE {where_sql}
                ORDER BY pe.id
                LIMIT %s OFFSET %s
                """,
                params,
            )
            rows = cur.fetchall()
        items = _build_items(rows[:limit])
        return jsonify({"items": items, "next_offset": offset + len(items), "has_more": len(rows) > limit})

    # ── Standard filters ──────────────────────────────────────────────────────
    if cluster_id_raw:
        if not cluster_id_raw.isdigit():
            return jsonify({"error": "Invalid cluster_id"}), 400
    elif person:
        pass
    elif cat_list:
        if any(c not in CATEGORY_LABELS for c in cat_list):
            return jsonify({"error": "Unknown category"}), 400
    else:
        if not folder:
            return jsonify({"error": "Provide 'categories', 'cluster_id', 'face_cluster_id', or 'folder'"}), 400

    where, params = _photo_where(cluster_id_raw, cat_list, folder, person)
    params.extend([limit, offset])

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, file_path,
                   category_1, category_1_score,
                   category_2, category_2_score,
                   category_3, category_3_score,
                   person_names
            FROM photo_embeddings
            WHERE {where}
            ORDER BY id
            LIMIT %s OFFSET %s
            """,
            params,
        )
        rows = cur.fetchall()

    items = _build_items(rows)
    return jsonify({"items": items, "next_offset": offset + len(items), "has_more": len(items) == limit})


# ── Playlist endpoints ────────────────────────────────────────────────────────

@app.get("/api/playlists")
def get_playlists() -> Any:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT p.id, p.name, COUNT(pp.id) AS cnt
            FROM playlists p
            LEFT JOIN playlist_photos pp ON pp.playlist_id = p.id
            GROUP BY p.id, p.name
            ORDER BY p.id
        """)
        rows = cur.fetchall()
    return jsonify([{"id": int(r[0]), "name": r[1], "count": int(r[2])} for r in rows])


@app.post("/api/playlists")
def create_playlist() -> Any:
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO playlists (name) VALUES (%s) RETURNING id, name",
            (name,),
        )
        row = cur.fetchone()
        conn.commit()
    return jsonify({"id": int(row[0]), "name": row[1]}), 201


@app.delete("/api/playlists/<int:playlist_id>")
def delete_playlist(playlist_id: int) -> Any:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM playlists WHERE id = %s RETURNING id", (playlist_id,))
        if cur.fetchone() is None:
            return jsonify({"error": "Playlist not found"}), 404
        conn.commit()
    return jsonify({"ok": True})


@app.get("/api/playlists/<int:playlist_id>/photos")
def get_playlist_photos(playlist_id: int) -> Any:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT pe.id, pe.file_path,
                   pe.category_1, pe.category_1_score,
                   pe.category_2, pe.category_2_score,
                   pe.category_3, pe.category_3_score,
                   pe.person_names,
                   pp.position
            FROM playlist_photos pp
            JOIN photo_embeddings pe ON pe.id = pp.photo_id
            WHERE pp.playlist_id = %s
            ORDER BY pp.position, pp.id
        """, (playlist_id,))
        rows = cur.fetchall()

    items = []
    for row in rows:
        photo_id = int(row[0])
        file_path = row[1]
        cats = [
            {"name": row[2], "score": float(row[3]) if row[3] is not None else None},
            {"name": row[4], "score": float(row[5]) if row[5] is not None else None},
            {"name": row[6], "score": float(row[7]) if row[7] is not None else None},
        ]
        items.append({
            "id": photo_id,
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "categories": [c for c in cats if c["name"]],
            "person_names": list(row[8]) if row[8] else [],
            "thumbnail_url": f"/image?id={photo_id}&thumb=1",
            "image_url": f"/image?id={photo_id}",
            "position": int(row[9]),
        })
    return jsonify(items)


@app.post("/api/playlists/<int:playlist_id>/photos")
def add_photo_to_playlist(playlist_id: int) -> Any:
    data = request.get_json(silent=True) or {}
    raw_photo_id = data.get("photo_id")
    if not isinstance(raw_photo_id, int):
        return jsonify({"error": "photo_id (integer) is required"}), 400
    photo_id = raw_photo_id

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM playlists WHERE id = %s", (playlist_id,))
        if cur.fetchone() is None:
            return jsonify({"error": "Playlist not found"}), 404

        cur.execute("SELECT id FROM photo_embeddings WHERE id = %s", (photo_id,))
        if cur.fetchone() is None:
            return jsonify({"error": "Photo not found"}), 404

        cur.execute(
            "SELECT id FROM playlist_photos WHERE playlist_id = %s AND photo_id = %s",
            (playlist_id, photo_id),
        )
        if cur.fetchone():
            return jsonify({"error": "Photo already in playlist"}), 409

        # Find current tail (next_id IS NULL) to append after it
        cur.execute(
            "SELECT id, position FROM playlist_photos WHERE playlist_id = %s AND next_id IS NULL",
            (playlist_id,),
        )
        tail = cur.fetchone()
        new_position = (int(tail[1]) + 1) if tail else 0
        prev_id = int(tail[0]) if tail else None

        cur.execute(
            """
            INSERT INTO playlist_photos (playlist_id, photo_id, prev_id, next_id, position)
            VALUES (%s, %s, %s, NULL, %s) RETURNING id
            """,
            (playlist_id, photo_id, prev_id, new_position),
        )
        new_id = int(cur.fetchone()[0])

        if prev_id is not None:
            cur.execute(
                "UPDATE playlist_photos SET next_id = %s WHERE id = %s",
                (new_id, prev_id),
            )

        conn.commit()
    return jsonify({"id": new_id, "position": new_position}), 201


@app.delete("/api/playlists/<int:playlist_id>/photos/<int:photo_id>")
def remove_photo_from_playlist(playlist_id: int, photo_id: int) -> Any:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, prev_id, next_id FROM playlist_photos WHERE playlist_id = %s AND photo_id = %s",
            (playlist_id, photo_id),
        )
        row = cur.fetchone()
        if row is None:
            return jsonify({"error": "Photo not in playlist"}), 404

        item_id, prev_id, next_id = int(row[0]), row[1], row[2]

        if prev_id is not None:
            cur.execute("UPDATE playlist_photos SET next_id = %s WHERE id = %s", (next_id, prev_id))
        if next_id is not None:
            cur.execute("UPDATE playlist_photos SET prev_id = %s WHERE id = %s", (prev_id, next_id))

        cur.execute("DELETE FROM playlist_photos WHERE id = %s", (item_id,))
        conn.commit()
    return jsonify({"ok": True})


@app.put("/api/playlists/<int:playlist_id>/photos/order")
def reorder_playlist_photos(playlist_id: int) -> Any:
    data = request.get_json(silent=True) or {}
    photo_ids = data.get("photo_ids")
    if not isinstance(photo_ids, list):
        return jsonify({"error": "photo_ids must be a list"}), 400

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, photo_id FROM playlist_photos WHERE playlist_id = %s",
            (playlist_id,),
        )
        existing = {int(r[1]): int(r[0]) for r in cur.fetchall()}

        incoming = set(int(p) for p in photo_ids)
        if incoming != set(existing.keys()):
            return jsonify({"error": "photo_ids must match all photos currently in the playlist"}), 400

        # Clear all pointers first to avoid constraint conflicts
        cur.execute(
            "UPDATE playlist_photos SET prev_id = NULL, next_id = NULL WHERE playlist_id = %s",
            (playlist_id,),
        )

        pp_ids = [existing[int(pid)] for pid in photo_ids]
        for i, pp_id in enumerate(pp_ids):
            prev_pp = pp_ids[i - 1] if i > 0 else None
            next_pp = pp_ids[i + 1] if i < len(pp_ids) - 1 else None
            cur.execute(
                "UPDATE playlist_photos SET prev_id = %s, next_id = %s, position = %s WHERE id = %s",
                (prev_pp, next_pp, i, pp_id),
            )

        conn.commit()
    return jsonify({"ok": True})


# ── Same-person face search ───────────────────────────────────────────────────

@app.get("/api/same_person")
def same_person_photos() -> Any:
    photo_id_raw = request.args.get("photo_id", "").strip()
    if not photo_id_raw.isdigit():
        return jsonify({"error": "photo_id non valido"}), 400
    photo_id = int(photo_id_raw)

    limit  = min(max(int(request.args.get("limit",  50)), 1), 200)
    offset = max(int(request.args.get("offset",  0)), 0)
    folder = request.args.get("folder", "").strip()

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT embedding FROM face_embeddings WHERE photo_id = %s ORDER BY face_index",
            (photo_id,),
        )
        faces = cur.fetchall()

    if len(faces) != 1:
        return jsonify({"error": "La foto non ha una sola persona"}), 400

    emb_raw = faces[0][0]
    if isinstance(emb_raw, str):
        vec_str = emb_raw
    else:
        vec_str = "[" + ",".join(str(float(x)) for x in emb_raw) + "]"

    # Cosine distance threshold — lower = more restrictive (ArcFace buffalo_l)
    THRESHOLD = 0.35

    where_parts = ["pe.id != %s", "(fe.embedding <=> %s::vector) < %s"]
    params: list = [photo_id, vec_str, THRESHOLD]

    if folder:
        where_parts.append("regexp_replace(pe.file_path, '/[^/]+$', '') = %s")
        params.append(folder)

    where_sql = " AND ".join(where_parts)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT pe.id, pe.file_path,
                   pe.category_1, pe.category_1_score,
                   pe.category_2, pe.category_2_score,
                   pe.category_3, pe.category_3_score,
                   pe.person_names
            FROM photo_embeddings pe
            JOIN face_embeddings fe ON fe.photo_id = pe.id
            WHERE {where_sql}
            GROUP BY pe.id, pe.file_path,
                     pe.category_1, pe.category_1_score,
                     pe.category_2, pe.category_2_score,
                     pe.category_3, pe.category_3_score,
                     pe.person_names
            ORDER BY MIN(fe.embedding <=> %s::vector) ASC
            LIMIT %s OFFSET %s
            """,
            [*params, vec_str, limit + 1, offset],
        )
        rows = cur.fetchall()

    items = _build_items(rows[:limit])
    return jsonify({
        "items": items,
        "next_offset": offset + len(items),
        "has_more": len(rows) > limit,
    })


@app.get("/api/same_person_similar")
def same_person_similar_photos() -> Any:
    photo_id_raw = request.args.get("photo_id", "").strip()
    if not photo_id_raw.isdigit():
        return jsonify({"error": "photo_id non valido"}), 400
    photo_id = int(photo_id_raw)

    limit  = min(max(int(request.args.get("limit",  50)), 1), 200)
    offset = max(int(request.args.get("offset",  0)), 0)
    folder = request.args.get("folder", "").strip()

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT embedding FROM face_embeddings WHERE photo_id = %s ORDER BY face_index",
            (photo_id,),
        )
        faces = cur.fetchall()

    if len(faces) != 1:
        return jsonify({"error": "La foto non ha una sola persona"}), 400

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT embedding FROM photo_embeddings WHERE id = %s",
            (photo_id,),
        )
        clip_row = cur.fetchone()

    if not clip_row or clip_row[0] is None:
        return jsonify({"error": "La foto non ha un embedding visivo"}), 400

    def _to_vec(raw) -> str:
        if isinstance(raw, str):
            return raw
        return "[" + ",".join(str(float(x)) for x in raw) + "]"

    face_vec = _to_vec(faces[0][0])
    clip_vec = _to_vec(clip_row[0])

    FACE_THRESHOLD  = 0.35   # ArcFace cosine distance — very restrictive
    PHOTO_THRESHOLD = 0.40   # CLIP cosine distance — visually similar

    where_parts = ["pe.id != %s", "pe.embedding IS NOT NULL", "(pe.embedding <=> %s::vector) < %s"]
    where_params: list = [photo_id, clip_vec, PHOTO_THRESHOLD]

    if folder:
        where_parts.append("regexp_replace(pe.file_path, '/[^/]+$', '') = %s")
        where_params.append(folder)

    where_sql = " AND ".join(where_parts)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            WITH face_matches AS (
                SELECT photo_id, MIN(embedding <=> %s::vector) AS face_dist
                FROM face_embeddings
                WHERE (embedding <=> %s::vector) < %s
                GROUP BY photo_id
            )
            SELECT pe.id, pe.file_path,
                   pe.category_1, pe.category_1_score,
                   pe.category_2, pe.category_2_score,
                   pe.category_3, pe.category_3_score,
                   pe.person_names
            FROM photo_embeddings pe
            JOIN face_matches fm ON fm.photo_id = pe.id
            WHERE {where_sql}
            ORDER BY fm.face_dist + (pe.embedding <=> %s::vector) ASC
            LIMIT %s OFFSET %s
            """,
            [face_vec, face_vec, FACE_THRESHOLD, *where_params, clip_vec, limit + 1, offset],
        )
        rows = cur.fetchall()

    items = _build_items(rows[:limit])
    return jsonify({
        "items": items,
        "next_offset": offset + len(items),
        "has_more": len(rows) > limit,
    })


# ── Semantic search ───────────────────────────────────────────────────────────

@app.get("/api/search")
def search_photos() -> Any:
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    if len(query) > 600:
        return jsonify({"error": "query too long (max 600 characters)"}), 400

    limit  = min(max(int(request.args.get("limit",  50)), 1), 200)
    offset = max(int(request.args.get("offset",  0)), 0)
    folder = request.args.get("folder", "").strip()

    try:
        embedding = _embed_text(query)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"

    where_clauses: list[str] = ["embedding IS NOT NULL"]
    base_params: list = []
    if folder:
        where_clauses.append("regexp_replace(file_path, '/[^/]+$', '') = %s")
        base_params.append(folder)
    base_params.append(vec_str)  # ORDER BY placeholder

    where_sql = " AND ".join(where_clauses)

    # ef_search must cover offset + all rows we want to inspect (limit+1 for has_more).
    ef_search = max(100, offset + limit + 1)

    with get_conn() as conn, conn.cursor() as cur:
        try:
            cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
        except Exception:
            pass  # older pgvector / no HNSW index — proceed with default
        cur.execute(
            f"""
            SELECT id, file_path,
                   category_1, category_1_score,
                   category_2, category_2_score,
                   category_3, category_3_score,
                   person_names
            FROM photo_embeddings
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s OFFSET %s
            """,
            [*base_params, limit + 1, offset],
        )
        rows = cur.fetchall()

    items = _build_items(rows[:limit])
    return jsonify({
        "items": items,
        "next_offset": offset + len(items),
        "has_more": len(rows) > limit,
    })


# ── Image serving ─────────────────────────────────────────────────────────────

def _resolve_photo_path_from_id(photo_id: int) -> Path:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT file_path FROM photo_embeddings WHERE id = %s", (photo_id,))
        row = cur.fetchone()
    if not row:
        abort(404, "Photo not found")
    image_path = Path(row[0])
    if not image_path.exists() or not image_path.is_file():
        abort(404, "Image file missing on disk")
    return image_path


@app.get("/image")
def image() -> Any:
    raw_id = request.args.get("id", "").strip()
    if not raw_id.isdigit():
        abort(400, "Missing or invalid photo id")

    image_path = _resolve_photo_path_from_id(int(raw_id))
    is_thumb = request.args.get("thumb", "0") == "1"

    if not is_thumb:
        return send_file(image_path)

    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((320, 320))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
    except Exception:
        abort(500, "Failed to render thumbnail")

    return send_file(buf, mimetype="image/jpeg")


if __name__ == "__main__":
    host = os.environ.get("PHOTO_BROWSER_HOST", "127.0.0.1")
    port = int(os.environ.get("PHOTO_BROWSER_PORT", "8080"))
    app.run(host=host, port=port, debug=False)
