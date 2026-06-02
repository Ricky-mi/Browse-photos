# Photo Browser — CLIP + ArcFace

> A self-hosted photo library with AI-powered semantic search, face recognition, and automatic categorisation. Runs entirely on your own machine — no cloud, no subscriptions.

## Screenshots

![Browse by category](Screenshots/Browse-categories.jpeg)
![Semantic search](Screenshots/Semantic-search.jpeg)
![Face clusters](Screenshots/Face-clusters.jpeg)
![Slide Show](Screenshots/Slide-Show-All.jpeg)

---

## Features

- **Semantic search** — describe what you're looking for in plain English ("sunset over the mountains", "birthday party with kids") and get visually matching photos, powered by [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- **Auto-categorisation** — 14 categories assigned automatically at embedding time (portraits, landscapes, night shots, flowers, cities, …)
- **Visual clustering** — K-means grouping of CLIP vectors groups visually similar photos together; clusters are renameable in the browser
- **Face recognition** — [InsightFace ArcFace](https://github.com/deepinsight/insightface) embeds every detected face; DBSCAN clustering groups faces by identity so you can browse by person
- **Person search** — right-click any portrait to find all other photos of the same person across the whole library
- **Playlists** — create named playlists, add photos with a right-click or keyboard shortcut, reorder with drag-and-drop
- **Slideshow** — play any playlist or the full result of any active filter as a fullscreen slideshow; adjustable speed, keyboard controls
- **Folder filter** — always-visible dropdown scopes any view to a subfolder
- **No cloud** — all vectors stored in PostgreSQL with [pgvector](https://github.com/pgvector/pgvector); images never leave your machine

---

## Quick Start

Minimum steps to get the browser running with category browse and semantic search.

**Requirements:** Python 3.9+, Docker (for PostgreSQL)

```bash
# 1 — start PostgreSQL + pgvector
docker compose up -d

# 2 — create Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3 — initialise the full database schema (all tables, indexes, columns)
export PHOTOS_DIR=/path/to/your/photos
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/photos
python3 embed_photos.py --init-db

# 4 — embed your photos (CLIP — runs on CPU, ~1–4 s/photo)
python3 embed_photos.py

# 5 — start the browser
python3 photo_browser.py
```

Open **http://127.0.0.1:8080** — browse by category or use the Search tab straight away.

> **GPU users:** add `--device cuda --batch-size 32` to `embed_photos.py` to cut embedding time by ~10×.

---

## Full Pipeline (Optional Steps)

The quick start gives you categories and semantic search. Run the steps below to unlock clustering, face recognition, and person search.

### Visual clustering

Groups photos by visual similarity using K-means on CLIP vectors.

```bash
python3 cluster_photos.py
```

### Face detection

Marks which photos contain at least one face (SCRFD detector).

```bash
python3 detect_faces.py
```

### Face embedding + clustering

Embeds each face with ArcFace, then clusters faces by identity (one cluster ≈ one person).

```bash
python3 embed_faces.py
python3 cluster_faces.py
```

After these steps restart `photo_browser.py` and the **Face Clusters** tab will be populated.

> **Note:** `--init-db` on individual scripts is still supported but no longer required — `embed_photos.py --init-db` creates the complete schema in one shot.

---

## Browser Guide

### Browse modes

| Tab | What it shows |
|---|---|
| **By Category** | Select up to 3 categories (AND logic); chips show counts |
| **By Cluster** | Visual K-means clusters; rename clusters inline |
| **Face Clusters** | One cluster per person; rename with the person's name |
| **Search** | Free-text semantic search via CLIP (up to 600 chars) |

All modes support the **Folder** dropdown to scope results to a subdirectory.

### Lightbox

Click any thumbnail to open full-resolution.

| Key / Action | Effect |
|---|---|
| **← →** | Previous / next photo |
| **Esc** | Close |
| **Ctrl + scroll** | Zoom in / out |
| **Drag** (zoomed) | Pan |
| **P** | Add to selected playlist |
| **Right-click** | Context menu |

### Right-click context menu

| Option | Effect |
|---|---|
| Add to "[playlist]" | Adds photo to the active playlist |
| Same person | Finds all photos of the same face (ArcFace cosine dist < 0.35) |
| Same person — similar photos | Same person + visually similar (CLIP dist < 0.40), ranked by combined score |

### Slideshow

Two ways to start a slideshow:

- **Playlist slideshow** — select a playlist and click **Slideshow** in the playlist panel
- **Filter slideshow** — click **▶ Slideshow all** below the photo grid to play every photo matching the current filter (no 50-photo limit — fetches the full result set first)

| Key | Effect |
|---|---|
| **Space** | Pause / Resume |
| **← →** | Step manually |
| **Esc** | Exit |

Speed slider adjusts seconds per photo (1–10 s).

### Playlists

| Action | How |
|---|---|
| Create | **New** → enter name → **Create** |
| Add a photo | Right-click thumbnail or press **P** in the lightbox |
| Reorder | Drag thumbnails in the **Show** modal |
| Remove a photo | Hover → ✕, or right-click → Remove |
| Delete playlist | Select → **Delete** (confirmation required) |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PHOTOS_DIR` | `/mnt/e/_Scanner` | Root directory scanned for images |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5433/photos` | PostgreSQL connection string |
| `PHOTO_BROWSER_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on the LAN) |
| `PHOTO_BROWSER_PORT` | `8080` | TCP port |

---

## Scripts Reference

### `embed_photos.py`

Scans `PHOTOS_DIR`, runs CLIP inference, stores 512-d vectors and up to 3 category labels.

| Flag | Default | Description |
|---|---|---|
| `--init-db` | — | Create the **full schema** (all tables, indexes, columns) and exit |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--batch-size` | `8` | Images per CLIP forward pass (try 32 on GPU) |
| `--loader-threads` | `4` | Parallel image-loading threads |
| `--skip-existing` / `--no-skip-existing` | `True` | Skip photos already in the database |
| `--min-category-score` | `0.08` | Minimum softmax probability for a category |
| `--category-margin` | `0.05` | Keep categories within this margin of the best score |
| `--calibration-limit` | `0` | Process only the first N images (0 = all) |

### `reclassify_photos.py`

Re-runs classification from stored vectors (no images re-read). Use after editing category labels or thresholds.

| Flag | Default | Description |
|---|---|---|
| `--min-score` | `0.15` | Minimum probability to keep a category |
| `--category-margin` | `0.10` | Secondary/tertiary category margin |
| `--dry-run` | — | Preview without writing to the database |
| `--diagnose N` | `0` | Print score breakdown for the first N embeddings |

### `cluster_photos.py`

Mini-Batch K-means on CLIP vectors. Re-running replaces all previous assignments.

| Flag | Default | Description |
|---|---|---|
| `--init-db` | — | Add `cluster_id` column + `clusters` table (idempotent) |
| `--num-clusters` | `50` | Number of clusters |
| `--random-state` | `42` | Random seed |

### `detect_faces.py`

SCRFD face detection — sets `has_faces` boolean on each photo.

| Flag | Default | Description |
|---|---|---|
| `--init-db` | — | Add `has_faces` column (idempotent) |
| `--model` | `buffalo_sc` | InsightFace model pack |
| `--device` | `cuda` | `cpu` or `cuda` |
| `--det-size` | `640` | Detection input resolution |

### `embed_faces.py`

ArcFace embeddings for photos where `has_faces = TRUE`.

| Flag | Default | Description |
|---|---|---|
| `--init-db` | — | Create `face_embeddings` table + HNSW index (idempotent) |
| `--model` | `buffalo_l` | InsightFace model (ResNet50 ArcFace) |
| `--device` | `cuda` | `cpu` or `cuda` |
| `--min-score` | `0.5` | Minimum face detection confidence |

### `cluster_faces.py`

DBSCAN / HDBSCAN clustering of ArcFace embeddings → one cluster per person.

| Flag | Default | Description |
|---|---|---|
| `--init-db` | — | Create `face_clusters` table (idempotent) |

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/categories` | All categories with photo counts |
| `GET` | `/api/photos` | Paginated photos — filter by `categories`, `cluster_id`, `face_cluster_id`, `folder` |
| `GET` | `/api/search?query=…` | CLIP semantic search |
| `GET` | `/api/same_person?photo_id=N` | Photos of the same person (face cosine dist < 0.35) |
| `GET` | `/api/same_person_similar?photo_id=N` | Same person + visually similar (CLIP dist < 0.40) |
| `GET` | `/api/folders` | Distinct subfolders for the current filter |
| `GET` | `/api/clusters` | Visual clusters |
| `PATCH` | `/api/clusters/:id` | Rename a visual cluster |
| `GET` | `/api/face_clusters` | Face clusters (people) |
| `PATCH` | `/api/face_clusters/:id` | Rename a face cluster |
| `GET` | `/api/playlists` | All playlists |
| `POST` | `/api/playlists` | Create a playlist |
| `DELETE` | `/api/playlists/:id` | Delete a playlist |
| `GET` | `/api/playlists/:id/photos` | Photos in a playlist |
| `POST` | `/api/playlists/:id/photos` | Add a photo to a playlist |
| `DELETE` | `/api/playlists/:id/photos/:photoId` | Remove a photo from a playlist |
| `PUT` | `/api/playlists/:id/photos/order` | Reorder playlist photos |
| `GET` | `/image?id=N` | Full-resolution image |
| `GET` | `/image?id=N&thumb=1` | 320×320 JPEG thumbnail |

---

## Database Schema

### `photo_embeddings`
| Column | Type | Notes |
|---|---|---|
| `id` | BIGSERIAL | Primary key |
| `file_path` | TEXT UNIQUE | Absolute path on disk |
| `embedding` | vector(512) | Normalised CLIP image embedding · HNSW index |
| `category_1/2/3` | TEXT | Up to 3 category labels |
| `category_1/2/3_score` | REAL | Softmax confidence per label |
| `cluster_id` | INTEGER | FK → clusters.id |
| `has_faces` | BOOLEAN | SCRFD face detection result |
| `person_names` | TEXT[] | Named people (from face pipeline) |
| `classified_at` | TIMESTAMPTZ | Last classification time |

### `face_embeddings`
| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL | Primary key |
| `photo_id` | INTEGER | FK → photo_embeddings.id |
| `face_index` | SMALLINT | Per-photo 0-based face index |
| `embedding` | vector(512) | Normalised ArcFace embedding · HNSW index |
| `bbox_x1/y1/x2/y2` | REAL | Face bounding box |
| `det_score` | REAL | Detection confidence |
| `face_cluster_id` | INTEGER | FK → face_clusters.id |

### Other tables
| Table | Purpose |
|---|---|
| `clusters` | Visual cluster descriptions (renameable) |
| `face_clusters` | Face cluster descriptions / person names |
| `known_people` | Named person registry |
| `playlists` | Playlist metadata |
| `playlist_photos` | Ordered photo–playlist join with doubly-linked list |

---

## Supported Formats

`jpg` · `jpeg` · `png` · `webp` · `bmp` · `gif`

---

## Useful SQL

```sql
-- Photo counts per category
SELECT category_1, COUNT(*) AS n
FROM photo_embeddings
GROUP BY category_1 ORDER BY n DESC;

-- Photos with detected faces
SELECT COUNT(*) FROM photo_embeddings WHERE has_faces = TRUE;

-- Top face clusters (people) by photo count
SELECT fc.description, fc.face_count,
       COUNT(DISTINCT fe.photo_id) AS photos
FROM face_clusters fc
JOIN face_embeddings fe ON fe.face_cluster_id = fc.id
GROUP BY fc.id, fc.description, fc.face_count
ORDER BY fc.face_count DESC;

-- Visual cluster sizes
SELECT c.description, COUNT(pe.id) AS n
FROM clusters c
JOIN photo_embeddings pe ON pe.cluster_id = c.id
GROUP BY c.id, c.description ORDER BY n DESC;
```
