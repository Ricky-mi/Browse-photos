# CLIP Photo Embeddings to PostgreSQL

Embed photos from `/mnt/e/_Scanner` using [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (openai/clip-vit-base-patch32), store 512-dimensional vectors in PostgreSQL with pgvector, and classify each photo into up to 3 semantic categories.

## Prerequisites

- **Python 3.9+**
- **PostgreSQL** with [pgvector](https://github.com/pgvector/pgvector) extension
  - Ubuntu/WSL: `sudo apt install postgresql-*-pgvector` (match your Postgres version)
  - Or compile from source

## Docker

To run PostgreSQL with pgvector in a container:

```bash
docker compose up -d
```

The container uses port **5433** (to avoid conflict with a local PostgreSQL on 5432). Set `PGPASSWORD=postgres` and `DATABASE_URL=postgresql://postgres:postgres@localhost:5433/photos` when running the app.

## Setup

**Docker users**: The `photos` database is created automatically when the container starts. Skip the `createdb` step.

**Non-Docker users** (local PostgreSQL on 5432):

```bash
PGPASSWORD=postgres createdb -h localhost -U postgres photos
```

**Create database manually** (e.g. Docker on port 5433, if needed):

```bash
PGPASSWORD=postgres createdb -h localhost -p 5433 -U postgres photos
```

**Virtual environment**:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Activate venv first (if using one)
source .venv/bin/activate

# Initialize database (extension, table, index)
python3 embed_photos.py --init-db

# Run embedding pipeline (CPU)
python3 embed_photos.py

# Use GPU if available
python3 embed_photos.py --device cuda

# Tune category selection strictness
python3 embed_photos.py --min-category-score 0.25 --category-margin 0.08

# Quick calibration run on first 200 photos with top-3 debug logs
python3 embed_photos.py --calibration-limit 200 --debug-topn 20
```

## Configuration

- **Photos directory**: Hardcoded to `/mnt/e/_Scanner` (WSL mapping of `E:\_Scanner`). Edit `PHOTOS_DIR` in `embed_photos.py` to change.
- **Database**: Defaults to `postgresql://postgres@localhost/photos`. Override with `DATABASE_URL` environment variable. Use `PGPASSWORD` for password authentication.
- **Categories**: The script evaluates 10 fixed CLIP categories and stores up to 3 labels per photo when scores are close enough to the best match.
- **Category thresholds**:
  - `--min-category-score` (default `0.08`): minimum score for a category to be kept.
  - `--category-margin` (default `0.05`): keep categories with score within this margin from the best score.
  - If nothing passes filtering, the script now keeps the top-1 predicted category (instead of forcing `other`).
  - `--debug-topn`: logs top-3 category scores for first N images.
  - `--calibration-limit`: process only first N images for fast threshold tuning.

## Supported image formats

jpg, jpeg, png, webp, bmp, gif

## Database schema

| Column           | Type         |
|------------------|--------------|
| id               | BIGSERIAL    |
| file_path        | TEXT (unique)|
| embedding        | vector(512)  |
| category_1       | TEXT         |
| category_1_score | REAL         |
| category_2       | TEXT         |
| category_2_score | REAL         |
| category_3       | TEXT         |
| category_3_score | REAL         |
| classified_at    | TIMESTAMPTZ  |
| created_at       | TIMESTAMPTZ  |

Re-running the script updates existing embeddings via upsert on `file_path`.

## Category labels

- beach landscapes
- other landscapes
- macro photography
- flowers
- portrait of one person
- portrait of more than one person
- trees
- cities
- vehicles
- other

## Quick SQL check

```sql
SELECT
  file_path,
  category_1, category_1_score,
  category_2, category_2_score,
  category_3, category_3_score
FROM photo_embeddings
LIMIT 20;
```

## Category browser frontend

A lightweight JavaScript frontend is included to browse categories and open full-size photos.

### Run

```bash
source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/photos
python3 photo_browser.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080).

### Features

- Select a category from a dropdown
- View first 20 photos for that category
- Click `Load 20 more` to continue browsing
- Click a thumbnail to open full screen
- Switch categories anytime
