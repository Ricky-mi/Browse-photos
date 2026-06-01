# CLIP + ArcFace Photo Browser

A personal photo library browser that uses [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) for semantic search and classification, and [InsightFace (ArcFace)](https://github.com/deepinsight/insightface) for face detection, embedding, and person search. All vectors are stored in PostgreSQL with pgvector.

---

## Architecture

```
Photos on disk
      │
      ├──► embed_photos.py ──────► photo_embeddings  (512-d CLIP vector, categories, cluster_id)
      │      (CLIP model)
      │
      ├──► detect_faces.py ───────► photo_embeddings.has_faces  (boolean per photo)
      │      (SCRFD detector)
      │
      └──► embed_faces.py ────────► face_embeddings  (512-d ArcFace vector per face)
             (ArcFace model)               │
                                           ▼
                                   cluster_faces.py ──► face_clusters  (face cluster_id per embedding)
                                    (DBSCAN / HDBSCAN)

photo_embeddings + face_embeddings + clusters + face_clusters + playlists
      │
      ▼
photo_browser.py  (Flask web server + REST API)
      │
      ▼
Browser SPA — web/index.html + app.js
```

### How it works

1. **CLIP embedding** — `embed_photos.py` scans photos on disk, runs each image through the CLIP vision encoder, and stores the resulting 512-dimensional vector plus up to 3 semantic category labels. An HNSW index enables fast approximate nearest-neighbour search.

2. **Classification** — Categories are assigned by computing cosine similarity between stored image vectors and CLIP text embeddings (one per category prompt). `reclassify_photos.py` can re-run classification at any time from the stored vectors — no images need to be re-read.

3. **Visual clustering** — `cluster_photos.py` runs Mini-Batch K-means on the CLIP vectors. Each photo gets a `cluster_id`. Clusters can be renamed interactively in the browser.

4. **Face detection** — `detect_faces.py` runs SCRFD on each photo and records a `has_faces` boolean in `photo_embeddings`.

5. **Face embedding** — `embed_faces.py` runs ArcFace on photos where `has_faces = TRUE` and stores one 512-dimensional embedding per detected face in `face_embeddings`. An HNSW index on the face vectors enables fast person search.

6. **Face clustering** — `cluster_faces.py` groups all face embeddings with DBSCAN or HDBSCAN, creating face clusters that correspond to individual people. Clusters can be browsed and renamed in the browser.

7. **Web frontend** — `photo_browser.py` is a Flask server serving the SPA and a REST API. The browser supports browsing by category, visual cluster, face cluster, free-text semantic search, and face-based person search.

---

## Prerequisites

- **Python 3.9+**
- **PostgreSQL 14+** with the [pgvector](https://github.com/pgvector/pgvector) extension
  - Easiest via Docker (see below)
  - Native install on Ubuntu/WSL: `sudo apt install postgresql-16-pgvector`
- **Docker & Docker Compose** (recommended)

---

## Setup Flow

### Step 1 — Start the database

```bash
docker compose up -d
```

Starts PostgreSQL 16 + pgvector on port **5433**. Data persists in a named Docker volume.

### Step 2 — Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Initialise the database schema

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/photos

python3 embed_photos.py --init-db
```

Creates the `photo_embeddings` table, enables the `vector` extension, and builds the HNSW index.

### Step 4 — Embed photos (CLIP)

```bash
python3 embed_photos.py
```

Scans `PHOTOS_DIR` (default `/mnt/e/_Scanner`), runs CLIP inference, and stores 512-d vectors and initial category labels. Re-running is safe: existing photos are skipped.

### Step 5 — Reclassify with Italian categories

```bash
python3 reclassify_photos.py
```

Replaces initial labels with refined Italian categories using the stored vectors (no images re-read).

### Step 6 — Visual clustering (optional but recommended)

```bash
python3 cluster_photos.py --init-db   # add cluster_id column (first time only)
python3 cluster_photos.py             # run K-means and assign clusters
```

### Step 7 — Face detection

```bash
python3 detect_faces.py --init-db     # add has_faces column (first time only)
python3 detect_faces.py               # detect faces in all photos
```

### Step 8 — Face embedding (ArcFace)

```bash
python3 embed_faces.py --init-db      # create face_embeddings table + HNSW index (first time only)
python3 embed_faces.py                # embed faces for all photos with has_faces = TRUE
```

### Step 9 — Face clustering

```bash
python3 cluster_faces.py --init-db    # create face_clusters table (first time only)
python3 cluster_faces.py              # cluster face embeddings → one cluster per person
```

### Step 10 — Start the web server

```bash
python3 photo_browser.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080).

---

## Scripts Reference

### `embed_photos.py`

Scans a directory of photos, generates CLIP image embeddings, and classifies each photo into up to 3 semantic categories.

| Parameter | Default | Description |
|---|---|---|
| `--init-db` | — | Initialise schema and exit |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--batch-size` | `8` | Images per CLIP forward pass |
| `--read-batch-size` | `8` | Images each loader thread reads at once |
| `--loader-threads` | `4` | Parallel image-loading threads |
| `--skip-existing` / `--no-skip-existing` | `True` | Skip photos already in the database |
| `--min-category-score` | `0.08` | Minimum softmax probability for a category |
| `--category-margin` | `0.05` | Keep categories within this margin of the best score |
| `--debug-topn` | `0` | Log top-3 scores for the first N images |
| `--calibration-limit` | `0` | Process only the first N images and exit |

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `PHOTOS_DIR` | `/mnt/e/_Scanner` | Root directory to scan |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5433/photos` | Connection string |

```bash
python3 embed_photos.py --init-db
python3 embed_photos.py
python3 embed_photos.py --device cuda --batch-size 32
```

---

### `reclassify_photos.py`

Re-classifies all photos using stored CLIP vectors — applies nine Italian-labelled categories. No images re-read.

| Parameter | Default | Description |
|---|---|---|
| `--device` | `cpu` | Device for CLIP text encoding |
| `--min-score` | `0.15` | Minimum softmax probability to keep a category |
| `--category-margin` | `0.10` | Secondary/tertiary category margin |
| `--dry-run` | — | Preview without writing |
| `--diagnose N` | `0` | Print score breakdown for the first N embeddings |

**Categories**

| Label | Matches |
|---|---|
| Ritratti & Gruppi | Portraits, selfies, group photos |
| Panorami Naturali | Natural landscapes, forests, sunsets |
| Mare & Coste | Sea, beaches, cliffs, boats |
| Montagne & Rocce | Mountains, peaks, snow |
| Urban & Architettura | Cities, monuments, streets |
| Natura da vicino | Flowers, plants, macro details |
| Animali | Dogs, cats, wildlife, insects |
| Food & Drink | Dishes, coffee, table settings |
| Interni & Musei | Rooms, interior design, exhibitions |
| Other | Does not match any of the above |

```bash
python3 reclassify_photos.py
python3 reclassify_photos.py --dry-run
python3 reclassify_photos.py --diagnose 10
```

---

### `cluster_photos.py`

Groups photos into clusters using Mini-Batch K-means on stored CLIP vectors.

| Parameter | Default | Description |
|---|---|---|
| `--init-db` | — | Add `cluster_id` column and create `clusters` table |
| `--num-clusters` | `50` | Number of K-means clusters |
| `--random-state` | `42` | Random seed |

```bash
python3 cluster_photos.py --init-db
python3 cluster_photos.py
python3 cluster_photos.py --num-clusters 100
```

Re-running replaces all previous cluster assignments.

---

### `detect_faces.py`

Runs SCRFD (InsightFace) on every photo to detect whether it contains at least one face. Results are stored as a `has_faces` boolean.

| Parameter | Default | Description |
|---|---|---|
| `--init-db` | — | Add `has_faces` column and exit |
| `--pilot` | — | Process the first N photos and write a TXT report |
| `--limit` | `2000` | Photos to process in pilot mode |
| `--output` | `pilot_report.txt` | Pilot report path |
| `--model` | `buffalo_sc` | InsightFace model pack |
| `--det-size` | `640` | Detection input size in pixels |
| `--device` | `cuda` | `cpu` or `cuda` |

```bash
python3 detect_faces.py --init-db
python3 detect_faces.py
python3 detect_faces.py --device cpu
```

---

### `embed_faces.py`

Runs ArcFace (InsightFace) on photos where `has_faces = TRUE` to produce a 512-dimensional face embedding per detected face.

| Parameter | Default | Description |
|---|---|---|
| `--init-db` | — | Create `face_embeddings` table and HNSW index |
| `--pilot` | — | Process photos listed in a pilot report |
| `--limit` | `2000` | Max photos in pilot mode |
| `--report` | `pilot_report.txt` | Pilot report to read |
| `--output` | `embed_faces_report.txt` | Pilot output report |
| `--model` | `buffalo_l` | InsightFace model pack (buffalo_l = ResNet50 ArcFace) |
| `--det-size` | `640` | Detection input size |
| `--min-score` | `0.5` | Minimum face detection confidence |
| `--device` | `cuda` | `cpu` or `cuda` |

```bash
python3 embed_faces.py --init-db
python3 embed_faces.py
python3 embed_faces.py --device cpu
```

---

### `cluster_faces.py`

Clusters all ArcFace embeddings to group faces by person identity (one cluster ≈ one person).

| Parameter | Default | Description |
|---|---|---|
| `--init-db` | — | Create `face_clusters` table |
| (other params) | — | See `--help` for clustering algorithm options |

```bash
python3 cluster_faces.py --init-db
python3 cluster_faces.py
```

---

### `photo_browser.py`

Flask web server that serves the SPA and exposes the REST API. Lazy-loads the CLIP model on the first semantic search.

| Environment variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5433/photos` | Connection string |
| `PHOTO_BROWSER_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on the network) |
| `PHOTO_BROWSER_PORT` | `8080` | TCP port |

**REST API**

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/categories` | List all categories with photo counts |
| `GET` | `/api/photos` | Paginated photos filtered by `categories`, `cluster_id`, `face_cluster_id`, or `folder` |
| `GET` | `/api/search` | Semantic search: embed `query` with CLIP, return nearest photos |
| `GET` | `/api/same_person?photo_id=N` | Photos of the same person (face similarity only, cosine dist < 0.35) |
| `GET` | `/api/same_person_similar?photo_id=N` | Photos of the same person AND visually similar (face dist < 0.35 + CLIP dist < 0.40) |
| `GET` | `/api/folders` | Distinct parent folders for the current filter |
| `GET` | `/api/clusters` | List visual clusters with descriptions and counts |
| `PATCH` | `/api/clusters/:id` | Rename a visual cluster |
| `GET` | `/api/face_clusters` | List face clusters (people) with face and photo counts |
| `PATCH` | `/api/face_clusters/:id` | Rename a face cluster |
| `GET` | `/api/playlists` | List playlists |
| `POST` | `/api/playlists` | Create a playlist |
| `DELETE` | `/api/playlists/:id` | Delete a playlist |
| `GET` | `/api/playlists/:id/photos` | Photos in a playlist (ordered) |
| `POST` | `/api/playlists/:id/photos` | Add a photo to a playlist |
| `DELETE` | `/api/playlists/:id/photos/:photoId` | Remove a photo from a playlist |
| `PUT` | `/api/playlists/:id/photos/order` | Reorder photos in a playlist |
| `GET` | `/image?id=N` | Serve full-resolution image |
| `GET` | `/image?id=N&thumb=1` | Serve 320×320 JPEG thumbnail |

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/photos
python3 photo_browser.py
```

---

## Web Frontend

Single-page application at [http://127.0.0.1:8080](http://127.0.0.1:8080).

### Browse by Category

Select up to **3 categories** simultaneously (AND logic). Chips show photo counts and disable when the limit is reached.

### Browse by Cluster

Choose a visual cluster from the dropdown (K-means grouping of CLIP vectors). Cluster descriptions can be edited and saved directly in the browser.

### Browse by Face Cluster

Choose a face cluster from the dropdown — each cluster corresponds to a person detected across the library. Face cluster descriptions (person names) can be edited and saved in the browser.

### Search

Free-text semantic search (up to 600 characters). The server embeds the query with CLIP and returns photos ordered by cosine similarity. The CLIP model is loaded on the first search and cached. **Enter** to search, **Shift+Enter** for a new line.

### Folder filter

Always-visible **Folder** dropdown scopes the current results to a subdirectory.

### Photo grid

50 photos per page. Click **Load 50 more** to fetch the next page. Each card shows the file name and category labels with confidence scores.

### Lightbox

Click any thumbnail to open the full-resolution image.

| Action | Effect |
|---|---|
| Click outside image | Close |
| **Esc** | Close |
| **←** / **→** | Previous / next photo |
| **Ctrl + scroll** | Zoom in / out |
| **Drag** (when zoomed) | Pan the image |
| **P** | Add to the selected playlist |
| Right-click on image | Context menu (see below) |

### Right-click context menu

Right-clicking any thumbnail (in the grid or in the lightbox) opens a context menu. Available actions depend on context:

| Option | Condition | Effect |
|---|---|---|
| Add to "[playlist]" | A playlist is selected | Adds the photo to the active playlist |
| **Foto della stessa persona** | Always shown | Searches for photos containing the same person using ArcFace face embeddings (cosine dist < 0.35). Shows a warning toast if the photo does not contain exactly one face. |
| **Foto simili della stessa persona** | Always shown | Same person search (face dist < 0.35) **and** visual similarity (CLIP dist < 0.40). Results ranked by the combined score — best face + photo match first. Shows a warning toast if the photo does not contain exactly one face or has no CLIP embedding. |

Both face-search modes enter an ephemeral result view (no tab). Clicking any mode tab returns to normal browsing.

### Playlists

| Action | How |
|---|---|
| Create | Click **New**, enter a name, press **Create** |
| Select | Choose from the **Playlist** dropdown |
| Add a photo | Right-click a thumbnail or press **P** in the lightbox |
| Delete | Select and click **Delete** (confirmation required) |
| View | Click **Show** — opens the playlist modal |
| Reorder | Drag thumbnails in the playlist modal |
| Remove a photo | Hover in the modal and click ✕, or right-click → Remove |

### Slideshow

With a playlist selected, click **Slideshow** to play all photos in order.

| Control | Effect |
|---|---|
| **Pause / Resume** | Toggle auto-advance |
| Speed slider | Seconds per photo (1–10 s) |
| **←** / **→** | Manual step |
| **Space** | Pause / Resume |
| **Esc** or ✕ | Exit |

---

## Database Schema

### `photo_embeddings`

| Column | Type | Description |
|---|---|---|
| `id` | BIGSERIAL | Primary key |
| `file_path` | TEXT UNIQUE | Absolute path on disk |
| `embedding` | vector(512) | Normalised CLIP image embedding |
| `category_1` | TEXT | Best matching category label |
| `category_1_score` | REAL | Softmax score for category_1 |
| `category_2` | TEXT | Second category (NULL if not assigned) |
| `category_2_score` | REAL | Score for category_2 |
| `category_3` | TEXT | Third category (NULL if not assigned) |
| `category_3_score` | REAL | Score for category_3 |
| `cluster_id` | INTEGER | FK → clusters.id |
| `has_faces` | BOOLEAN | Whether SCRFD detected at least one face |
| `person_names` | TEXT[] | Person names (populated by face recognition pipeline) |
| `classified_at` | TIMESTAMPTZ | When categories were last written |
| `created_at` | TIMESTAMPTZ | When the row was first inserted |

HNSW index on `embedding` (cosine ops) for semantic search.

### `face_embeddings`

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL | Primary key |
| `photo_id` | INTEGER | FK → photo_embeddings.id |
| `face_index` | SMALLINT | Per-photo face index (0-based) |
| `embedding` | vector(512) | Normalised ArcFace face embedding |
| `bbox_x1/y1/x2/y2` | REAL | Face bounding box pixels |
| `det_score` | REAL | SCRFD detection confidence |
| `face_cluster_id` | INTEGER | FK → face_clusters.id (assigned by cluster_faces.py) |
| `created_at` | TIMESTAMPTZ | Insertion time |

UNIQUE on `(photo_id, face_index)`. HNSW index on `embedding` (cosine ops) for fast person search.

### `face_clusters`

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL | Primary key |
| `description` | TEXT | User-editable cluster name (person name) |
| `face_count` | INTEGER | Number of face embeddings in this cluster |

### `clusters`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Cluster number |
| `description` | TEXT | User-editable cluster name |

### `playlists`

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL | Primary key |
| `name` | TEXT | Playlist name |
| `created_at` | TIMESTAMPTZ | Creation time |

### `playlist_photos`

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL | Primary key |
| `playlist_id` | INTEGER | FK → playlists.id |
| `photo_id` | INTEGER | FK → photo_embeddings.id |
| `position` | INTEGER | Sort order |
| `prev_id` | INTEGER | Doubly-linked list prev pointer |
| `next_id` | INTEGER | Doubly-linked list next pointer |

---

## Supported Image Formats

`jpg`, `jpeg`, `png`, `webp`, `bmp`, `gif`

---

## Quick SQL Checks

```sql
-- Sample of classified photos
SELECT file_path, category_1, category_1_score, category_2
FROM photo_embeddings ORDER BY id LIMIT 20;

-- Count per category
SELECT category_1, COUNT(*) AS n
FROM photo_embeddings
GROUP BY category_1
ORDER BY n DESC;

-- Photos with faces
SELECT COUNT(*) FROM photo_embeddings WHERE has_faces = TRUE;

-- Face cluster sizes (people)
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
GROUP BY c.id, c.description
ORDER BY n DESC;

-- Photos not yet classified
SELECT COUNT(*) FROM photo_embeddings WHERE category_1 IS NULL;
```
