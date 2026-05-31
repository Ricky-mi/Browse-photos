# CLIP Photo Browser

A personal photo library browser that uses [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (Contrastive Language–Image Pre-training) to embed, classify, cluster, and semantically search a photo collection stored in PostgreSQL with pgvector.

---

## Architecture

```
Photos on disk
      │
      ▼
embed_photos.py ──► PostgreSQL / pgvector
  (CLIP model)       table: photo_embeddings
                      ├─ 512-d vector per photo
                      ├─ up to 3 category labels + scores
                      └─ cluster_id (after clustering)
                           │
          ┌────────────────┼───────────────┐
          ▼                ▼               ▼
reclassify_photos.py  cluster_photos.py  photo_browser.py
  (re-label with         (K-means on       (Flask web server)
   Italian categories)    stored vectors)        │
                                                 ▼
                                         Browser (SPA)
                                    web/index.html + app.js
```

### How it works

1. **Embedding** — `embed_photos.py` scans photos on disk, runs each image through the CLIP vision encoder, and stores the resulting 512-dimensional vector in PostgreSQL along with up to 3 semantic category labels. An HNSW index on the `embedding` column enables fast approximate nearest-neighbour search.

2. **Classification** — Categories are assigned by computing cosine similarity between each stored image vector and a set of CLIP text embeddings (one per category prompt). The top-scoring categories whose scores are close enough to the best are stored. `reclassify_photos.py` can re-run classification at any time using the already-stored vectors — no images need to be re-read.

3. **Clustering** — `cluster_photos.py` loads all stored vectors and runs Mini-Batch K-means. Each photo is assigned a `cluster_id`. Clusters are automatically visualisable in the browser and can be renamed interactively.

4. **Web frontend** — `photo_browser.py` is a Flask server that serves the single-page app from the `web/` folder and exposes a REST API. The browser queries the API to browse photos by category, cluster, or free-text semantic search.

### Key design choices

- All expensive CLIP inference (image encoding) happens once during `embed_photos.py`. Every subsequent operation (classification, clustering, search) works on the stored vectors.
- Semantic text search at query time embeds the user's text with the same CLIP model used for images, so both live in the same vector space and are directly comparable with cosine similarity.
- pgvector's `<=>` cosine-distance operator and HNSW index make semantic search fast even on large collections.

---

## Prerequisites

- **Python 3.9+**
- **PostgreSQL 14+** with the [pgvector](https://github.com/pgvector/pgvector) extension
  - Easiest via Docker (see below)
  - Native install on Ubuntu/WSL: `sudo apt install postgresql-16-pgvector`
- **Docker & Docker Compose** (recommended)

---

## Setup Flow

Follow these steps in order the first time you set up the project.

### Step 1 — Start the database

```bash
docker compose up -d
```

This starts a PostgreSQL 16 + pgvector container on port **5433** (avoiding conflicts with any local PostgreSQL on 5432). The `photos` database is created automatically. Data is persisted in a named Docker volume.

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

### Step 4 — Embed photos

```bash
python3 embed_photos.py
```

Scans `PHOTOS_DIR` (default `/mnt/e/_Scanner`), runs CLIP inference on every image, and stores 512-d vectors and initial category labels in the database. This step is the slowest — typically a few seconds per photo on CPU.

Re-running is safe: existing photos are skipped (`--skip-existing` is on by default). Only new files are processed.

### Step 5 — Reclassify with Italian categories

```bash
python3 reclassify_photos.py
```

Replaces the initial English category labels with refined Italian categories (e.g. *Ritratti & Gruppi*, *Mare & Coste*). Works entirely from the stored vectors — no images are re-read. Runs in seconds even for large collections.

### Step 6 — Cluster photos (optional but recommended)

```bash
python3 cluster_photos.py --init-db   # add cluster_id column (first time only)
python3 cluster_photos.py             # run K-means and assign clusters
```

Groups visually similar photos into 50 clusters (configurable). Clusters can be browsed and renamed in the web frontend.

### Step 7 — Start the web server

```bash
python3 photo_browser.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in a browser.

---

## Scripts Reference

### `embed_photos.py`

Scans a directory of photos, generates CLIP image embeddings, and classifies each photo into up to 3 semantic categories. Results are stored in the `photo_embeddings` table.

| Parameter | Default | Description |
|---|---|---|
| `--init-db` | — | Initialise the database schema (extension, table, HNSW index) and exit without processing images |
| `--device` | `cpu` | Device for CLIP inference: `cpu` or `cuda` |
| `--batch-size` | `8` | Number of images per CLIP forward pass. Increase for GPU (16–32), decrease if RAM is tight |
| `--read-batch-size` | `8` | Images each loader thread reads at once |
| `--loader-threads` | `4` | Number of parallel image-loading threads (pipeline overlap with CLIP inference) |
| `--skip-existing` / `--no-skip-existing` | `True` | Skip photos already in the database. Use `--no-skip-existing` to re-embed everything |
| `--min-category-score` | `0.08` | Minimum softmax probability for a category to be stored. Lower = more secondary labels |
| `--category-margin` | `0.05` | Keep categories within this margin of the best score. Lower = stricter, often single label |
| `--debug-topn` | `0` | Log top-3 category scores for the first N processed images (useful for calibration) |
| `--calibration-limit` | `0` | Process only the first N images and exit. Use with `--debug-topn` to tune thresholds |

**Environment variables**

| Variable | Default | Description |
|---|---|---|
| `PHOTOS_DIR` | `/mnt/e/_Scanner` | Root directory to scan for photos |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5433/photos` | PostgreSQL connection string |

**Examples**

```bash
# First-time setup
python3 embed_photos.py --init-db

# Standard run on CPU
python3 embed_photos.py

# GPU with larger batches
python3 embed_photos.py --device cuda --batch-size 32

# Calibrate category thresholds on a small sample
python3 embed_photos.py --calibration-limit 100 --debug-topn 100

# Stricter labelling (fewer secondary categories)
python3 embed_photos.py --min-category-score 0.20 --category-margin 0.05
```

---

### `reclassify_photos.py`

Re-classifies all photos using the 512-d vectors already stored in the database. Applies nine Italian-labelled categories (with English CLIP prompts) and optionally an *Other* fallback. No images are re-read from disk.

Run this after `embed_photos.py` to apply the refined Italian category set.

| Parameter | Default | Description |
|---|---|---|
| `--device` | `cpu` | Device for CLIP text encoding: `cpu` or `cuda` |
| `--min-score` | `0.15` | Minimum softmax probability to keep a category. Photos below this threshold for all categories are labelled *Other* |
| `--category-margin` | `0.10` | Assign secondary/tertiary categories when their score is within this margin of the best score |
| `--dry-run` | — | Preview classifications without writing to the database |
| `--diagnose N` | `0` | Print the full score breakdown for the first N embeddings and exit. Also checks stored vectors against the current model for consistency |

**Categories**

| Label | Matches |
|---|---|
| Ritratti & Gruppi | Portraits, selfies, group photos, people close-ups |
| Panorami Naturali | Natural landscapes, forests, lakes, sunsets |
| Mare & Coste | Sea, beaches, cliffs, boats |
| Montagne & Rocce | Mountains, peaks, snow, high-altitude scenery |
| Urban & Architettura | Cities, monuments, streets, building facades |
| Natura da vicino | Flowers, plants, leaves, macro details |
| Animali | Dogs, cats, wildlife, insects |
| Food & Drink | Dishes, coffee, table settings |
| Interni & Musei | Rooms, interior design, exhibitions |
| Other | Does not match any of the above |

**Examples**

```bash
# Standard reclassification
python3 reclassify_photos.py

# Preview without writing
python3 reclassify_photos.py --dry-run

# Stricter Other threshold
python3 reclassify_photos.py --min-score 0.20

# Diagnose scoring on first 10 embeddings
python3 reclassify_photos.py --diagnose 10
```

---

### `cluster_photos.py`

Groups photos into clusters using Mini-Batch K-means on the stored 512-d CLIP vectors. Adds a `cluster_id` to every photo and creates a `clusters` table. Clusters can be browsed and renamed in the web frontend.

| Parameter | Default | Description |
|---|---|---|
| `--init-db` | — | Add the `cluster_id` column and create the `clusters` table, then exit (run once before first clustering) |
| `--num-clusters` | `50` | Number of K-means clusters. Increase for larger collections (rule of thumb: √N where N is number of photos) |
| `--random-state` | `42` | Random seed for reproducible clustering |

**Examples**

```bash
# First-time schema setup
python3 cluster_photos.py --init-db

# Default 50 clusters
python3 cluster_photos.py

# More clusters for a large library
python3 cluster_photos.py --num-clusters 100

# Reproducible alternative clustering
python3 cluster_photos.py --num-clusters 80 --random-state 7
```

Re-running `cluster_photos.py` replaces all previous cluster assignments.

---

### `photo_browser.py`

Flask web server that serves the single-page photo browser and exposes the REST API. Also lazy-loads the CLIP model on the first semantic search request.

| Environment variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5433/photos` | PostgreSQL connection string |
| `PHOTO_BROWSER_HOST` | `127.0.0.1` | Bind address (`0.0.0.0` to expose on the network) |
| `PHOTO_BROWSER_PORT` | `8080` | TCP port |

**REST API endpoints**

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/categories` | List all categories with photo counts |
| `GET` | `/api/photos` | Paginated photos filtered by `categories`, `cluster_id`, or `folder` |
| `GET` | `/api/search` | Semantic search: embed `query` text with CLIP and return nearest photos |
| `GET` | `/api/folders` | Distinct parent folders for the current filter |
| `GET` | `/api/clusters` | List clusters with descriptions and counts |
| `PATCH` | `/api/clusters/:id` | Rename a cluster |
| `GET` | `/api/playlists` | List playlists |
| `POST` | `/api/playlists` | Create a playlist |
| `DELETE` | `/api/playlists/:id` | Delete a playlist |
| `GET` | `/api/playlists/:id/photos` | Photos in a playlist (ordered) |
| `POST` | `/api/playlists/:id/photos` | Add a photo to a playlist |
| `DELETE` | `/api/playlists/:id/photos/:photoId` | Remove a photo from a playlist |
| `PUT` | `/api/playlists/:id/photos/order` | Reorder photos in a playlist |
| `GET` | `/image?id=N` | Serve full-resolution image |
| `GET` | `/image?id=N&thumb=1` | Serve 320×320 JPEG thumbnail |

**Run**

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/photos
python3 photo_browser.py
```

---

### `delete_embeddings_under.py`

Utility that deletes all rows in `photo_embeddings` whose `file_path` matches a given directory or any path under it. Useful after moving the photo library to a different path.

| Argument | Default | Description |
|---|---|---|
| `dir_prefix` (positional) | `/mnt/e/_Scanner` | Directory path prefix — all rows whose path equals this or starts with `this/` are deleted |

**Examples**

```bash
# Delete all rows for the default directory
python3 delete_embeddings_under.py

# Delete rows for a specific path
python3 delete_embeddings_under.py /mnt/d/Photos/Old
```

After deleting, re-run `embed_photos.py` with the new path to re-embed.

---

## Web Frontend

The browser is a single-page application served by Flask from the `web/` folder. Open [http://127.0.0.1:8080](http://127.0.0.1:8080).

### Browse by Category

Select up to **3 categories** simultaneously. Photos must belong to **all** selected categories (AND logic). Categories are shown as chips with their photo count.

- Click a chip to select/deselect it.
- When 3 are already selected, unselected chips are disabled until one is deselected.
- The photo grid and folder list update automatically on each change.

### Browse by Cluster

Choose a cluster from the dropdown. Each cluster groups visually similar photos identified by K-means.

- The **Name** field lets you edit the cluster description. Click **Save** to persist the name to the database — the new name appears in the dropdown immediately.
- Re-running `cluster_photos.py` replaces all clusters, so save meaningful names before re-clustering.

### Search

Type a free-text description (up to 600 characters) and press **Search** or **Enter**. The server embeds the text with CLIP and returns photos ordered by cosine similarity to your description.

- This works because CLIP images and text share the same vector space: you can describe *what you want to see* in natural language.
- The **CLIP model is loaded on the first search** and cached for subsequent requests. The first search may take several seconds.
- **Shift+Enter** inserts a newline without triggering the search.
- The character counter shows how many of the 600 characters have been used.
- Results are paginated: click **Load 50 more** to continue.

### Folder filter

A **Folder** dropdown is always visible below the mode tabs. It lists all subdirectories in the current result set. Select one to restrict the displayed photos to that folder. Select *All folders* to remove the filter.

### Photo grid

Photos are displayed in a responsive grid of thumbnails (50 per page). Click **Load 50 more** at the bottom to fetch the next page.

Each card shows the file name and the assigned category labels with their confidence scores.

### Lightbox

Click any thumbnail to open the full-resolution image in a lightbox overlay.

| Action | Effect |
|---|---|
| Click outside image | Close |
| **Esc** | Close |
| **←** / **→** arrows | Previous / next photo |
| **Ctrl + scroll wheel** | Zoom in / out |
| **Drag** (when zoomed) | Pan the image |
| **P** | Add current photo to the selected playlist |
| Right-click on image | Context menu to add to playlist |

### Playlists

Create named playlists to collect photos of interest.

| Action | How |
|---|---|
| Create | Click **New**, enter a name, press **Create** |
| Select | Choose from the **Playlist** dropdown |
| Add a photo | Right-click a thumbnail or press **P** in the lightbox |
| Modify | Click **Modify** to display a reminder of how to add photos to the selected playlist |
| Rename | Not yet supported — delete and recreate |
| Delete | Select it and click **Delete** (confirmation required) |
| View | Click **Show** — opens the playlist modal |
| Reorder | Drag thumbnails in the playlist modal |
| Remove a photo | Hover over a thumbnail in the modal and click ✕, or right-click → Remove |

### Slideshow

With a playlist selected, click **Slideshow** to play all photos in order.

| Control | Effect |
|---|---|
| **Pause / Resume** | Toggle auto-advance |
| Speed slider | Set seconds per photo (1–10 s) |
| **←** / **→** | Manual step |
| **Space** | Pause / Resume |
| **Esc** or ✕ | Exit slideshow |

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
| `cluster_id` | INTEGER | FK → clusters.id (NULL before clustering) |
| `classified_at` | TIMESTAMPTZ | When categories were last written |
| `created_at` | TIMESTAMPTZ | When the row was first inserted |

An HNSW index on `embedding` (cosine ops) enables fast semantic search.

### `clusters`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Cluster number (1-based) |
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
SELECT file_path, category_1, category_1_score, category_2, category_2_score
FROM photo_embeddings
ORDER BY id
LIMIT 20;

-- Count per category
SELECT category_1, COUNT(*) AS n
FROM photo_embeddings
GROUP BY category_1
ORDER BY n DESC;

-- Cluster sizes
SELECT c.description, COUNT(pe.id) AS n
FROM clusters c
JOIN photo_embeddings pe ON pe.cluster_id = c.id
GROUP BY c.id, c.description
ORDER BY n DESC;

-- Photos not yet classified
SELECT COUNT(*) FROM photo_embeddings WHERE category_1 IS NULL;
```
