// ── DOM refs ──────────────────────────────────────────────────────────────────
const modeCategoryBtn      = document.getElementById("modeCategory");
const modeClusterBtn       = document.getElementById("modeCluster");
const modeFaceClusterBtn   = document.getElementById("modeFaceCluster");
const modeSearchBtn        = document.getElementById("modeSearch");
const categoryControls     = document.getElementById("categoryControls");
const clusterControls      = document.getElementById("clusterControls");
const faceClusterControls  = document.getElementById("faceClusterControls");
const searchControls       = document.getElementById("searchControls");
const searchQueryEl        = document.getElementById("searchQuery");
const searchBtn            = document.getElementById("searchBtn");
const searchCharCount      = document.getElementById("searchCharCount");
const categoryList         = document.getElementById("categoryList");
const selectedCount        = document.getElementById("selectedCount");
const clusterSelect        = document.getElementById("clusterSelect");
const clusterDesc          = document.getElementById("clusterDesc");
const saveClusterBtn       = document.getElementById("saveClusterBtn");
const saveStatus           = document.getElementById("saveStatus");
const faceClusterSelect    = document.getElementById("faceClusterSelect");
const faceClusterDesc      = document.getElementById("faceClusterDesc");
const saveFaceClusterBtn   = document.getElementById("saveFaceClusterBtn");
const saveFaceClusterStatus= document.getElementById("saveFaceClusterStatus");
const folderSelect         = document.getElementById("folderSelect");
const folderCount          = document.getElementById("folderCount");
const grid                 = document.getElementById("grid");
const statusEl             = document.getElementById("status");
const loadMoreBtn          = document.getElementById("loadMoreBtn");
const slideshowViewBtn     = document.getElementById("slideshowViewBtn");
const lightbox             = document.getElementById("lightbox");
const lightboxImage        = document.getElementById("lightboxImage");
const lightboxCaption      = document.getElementById("lightboxCaption");
const prevBtn              = document.getElementById("prevPhoto");
const nextBtn              = document.getElementById("nextPhoto");
const closeLightbox        = document.getElementById("closeLightbox");
const contextMenu          = document.getElementById("contextMenu");

// Playlist panel
const playlistSelect       = document.getElementById("playlistSelect");
const playlistPhotoCnt     = document.getElementById("playlistPhotoCnt");
const newPlaylistBtn       = document.getElementById("newPlaylistBtn");
const modifyPlaylistBtn    = document.getElementById("modifyPlaylistBtn");
const deletePlaylistBtn    = document.getElementById("deletePlaylistBtn");
const showPlaylistBtn      = document.getElementById("showPlaylistBtn");
const slideshowPlaylistBtn = document.getElementById("slideshowPlaylistBtn");
const playlistHint         = document.getElementById("playlistHint");

// New playlist dialog
const newPlaylistOverlay       = document.getElementById("newPlaylistOverlay");
const newPlaylistName          = document.getElementById("newPlaylistName");
const closeNewPlaylistDialog   = document.getElementById("closeNewPlaylistDialog");
const cancelNewPlaylist        = document.getElementById("cancelNewPlaylist");
const confirmNewPlaylist       = document.getElementById("confirmNewPlaylist");

// Playlist show modal
const playlistModalOverlay = document.getElementById("playlistModalOverlay");
const playlistModalTitle   = document.getElementById("playlistModalTitle");
const closePlaylistModal   = document.getElementById("closePlaylistModal");
const playlistEmpty        = document.getElementById("playlistEmpty");
const playlistGrid         = document.getElementById("playlistGrid");

// Slideshow
const slideshowOverlay = document.getElementById("slideshowOverlay");
const slideshowImage   = document.getElementById("slideshowImage");
const closeSlideshowBtn= document.getElementById("closeSlideshowBtn");
const ssPrevBtn        = document.getElementById("ssPrevBtn");
const ssNextBtn        = document.getElementById("ssNextBtn");
const ssPauseBtn       = document.getElementById("ssPauseBtn");
const ssSpeedSlider    = document.getElementById("ssSpeedSlider");
const ssSpeedVal       = document.getElementById("ssSpeedVal");
const ssCounter        = document.getElementById("ssCounter");

// ── State ─────────────────────────────────────────────────────────────────────
let mode               = "category";
let selectedCategories = new Set();
let selectedFolder     = "";
let nextOffset         = 0;
let hasMore            = false;
let photoList          = [];
let currentPhotoIndex  = -1;
let lightboxScale      = 1;
let lightboxTranslateX = 0;
let lightboxTranslateY = 0;
let lbDragging         = false;
let lbDragStartX       = 0;
let lbDragStartY       = 0;
let lbDragStartTX      = 0;
let lbDragStartTY      = 0;
let lbDragMoved        = false;
let saveTimer          = null;

// Playlist state
let selectedPlaylistId   = null;
let selectedPlaylistName = "";
let playlistPhotos       = [];

// Same-person search state
let samePersonSourceId   = null;
let samePersonSourceName = "";

// Slideshow state
let ssPhotos  = [];
let ssIndex   = 0;
let ssTimer   = null;
let ssPaused  = false;

// Drag & drop state (playlist modal)
let dragSrcIndex = -1;

const MAX_CATS     = 3;
const SCALE_FACTOR = 1.18;
const SCALE_MIN    = 0.25;
const SCALE_MAX    = 8;

// ── Toast notifications ────────────────────────────────────────────────────────
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, 2500);
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function fmtScore(s) { return s != null ? s.toFixed(2) : "-"; }
function getClusterId() { return clusterSelect.value ? parseInt(clusterSelect.value, 10) : null; }
function lightboxOpen() { return !lightbox.classList.contains("hidden"); }

function getFaceClusterId() { return faceClusterSelect.value ? parseInt(faceClusterSelect.value, 10) : null; }

function stripPrefix(paths) {
  if (paths.length === 0) return [];
  const parts = paths.map((p) => p.split("/").filter(Boolean));
  const minLen = Math.min(...parts.map((p) => p.length));
  let commonLen = 0;
  outer: for (let i = 0; i < minLen; i++) {
    const seg = parts[0][i];
    for (const p of parts) { if (p[i] !== seg) break outer; }
    commonLen = i + 1;
  }
  return paths.map((p, idx) => {
    const rel = parts[idx].slice(commonLen);
    return rel.length > 0 ? rel.join("/") : parts[idx][parts[idx].length - 1] ?? p;
  });
}

function primaryQS() {
  const qs = new URLSearchParams();
  if (mode === "category") {
    for (const cat of selectedCategories) qs.append("categories", cat);
  } else if (mode === "face_cluster") {
    const fcid = getFaceClusterId();
    if (fcid) qs.set("face_cluster_id", String(fcid));
  } else {
    const cid = getClusterId();
    if (cid) qs.set("cluster_id", String(cid));
  }
  return qs;
}

// ── Context menu ──────────────────────────────────────────────────────────────
function showContextMenu(e, actions) {
  contextMenu.innerHTML = "";
  for (const { label, handler } of actions) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.addEventListener("click", () => { hideContextMenu(); handler(); });
    contextMenu.appendChild(btn);
  }
  const x = Math.min(e.clientX, window.innerWidth - 200);
  const y = Math.min(e.clientY, window.innerHeight - actions.length * 44);
  contextMenu.style.left = `${x}px`;
  contextMenu.style.top  = `${y}px`;
  contextMenu.classList.remove("hidden");
}

function hideContextMenu() {
  contextMenu.classList.add("hidden");
}

document.addEventListener("click", hideContextMenu);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideContextMenu(); });

// ── Same-person search ────────────────────────────────────────────────────────
async function searchSamePerson(photo) {
  samePersonSourceId   = photo.id;
  samePersonSourceName = photo.file_name;
  mode = "same_person";

  modeCategoryBtn.classList.remove("active");
  modeClusterBtn.classList.remove("active");
  modeFaceClusterBtn.classList.remove("active");
  modeSearchBtn.classList.remove("active");

  categoryControls.classList.add("hidden");
  clusterControls.classList.add("hidden");
  faceClusterControls.classList.add("hidden");
  searchControls.classList.add("hidden");

  selectedFolder = "";
  folderSelect.value = "";
  statusEl.textContent = "Ricerca in corso…";

  await Promise.all([loadFolders(), loadPhotos({ reset: true })]);
}

async function searchSamePersonSimilar(photo) {
  samePersonSourceId   = photo.id;
  samePersonSourceName = photo.file_name;
  mode = "same_person_similar";

  modeCategoryBtn.classList.remove("active");
  modeClusterBtn.classList.remove("active");
  modeFaceClusterBtn.classList.remove("active");
  modeSearchBtn.classList.remove("active");

  categoryControls.classList.add("hidden");
  clusterControls.classList.add("hidden");
  faceClusterControls.classList.add("hidden");
  searchControls.classList.add("hidden");

  selectedFolder = "";
  folderSelect.value = "";
  statusEl.textContent = "Ricerca in corso…";

  await Promise.all([loadFolders(), loadPhotos({ reset: true })]);
}

// ── Playlist API ───────────────────────────────────────────────────────────────
async function apiGetPlaylists() {
  const r = await fetch("/api/playlists");
  return r.ok ? r.json() : [];
}

async function apiCreatePlaylist(name) {
  const r = await fetch("/api/playlists", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return r.ok ? r.json() : null;
}

async function apiDeletePlaylist(id) {
  const r = await fetch(`/api/playlists/${id}`, { method: "DELETE" });
  return r.ok;
}

async function apiGetPlaylistPhotos(id) {
  const r = await fetch(`/api/playlists/${id}/photos`);
  return r.ok ? r.json() : [];
}

async function apiAddPhotoToPlaylist(playlistId, photoId) {
  const r = await fetch(`/api/playlists/${playlistId}/photos`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ photo_id: photoId }),
  });
  if (r.status === 409) {
    showToast("Photo already in playlist", "warn");
    return null;
  }
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    showToast(err.error || "Failed to add photo", "err");
    return null;
  }
  showToast(`Added to "${selectedPlaylistName}"`, "ok");
  refreshPlaylistCount();
  return r.json();
}

async function apiRemovePhotoFromPlaylist(playlistId, photoId) {
  const r = await fetch(`/api/playlists/${playlistId}/photos/${photoId}`, { method: "DELETE" });
  if (!r.ok) { showToast("Failed to remove photo", "err"); return false; }
  refreshPlaylistCount();
  return true;
}

async function apiReorderPlaylistPhotos(playlistId, photoIds) {
  const r = await fetch(`/api/playlists/${playlistId}/photos/order`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ photo_ids: photoIds }),
  });
  if (!r.ok) { showToast("Failed to reorder", "err"); return false; }
  return true;
}

// ── Playlist UI ────────────────────────────────────────────────────────────────
async function loadPlaylists() {
  const lists = await apiGetPlaylists();
  const prev = playlistSelect.value;

  playlistSelect.innerHTML = '<option value="">— none —</option>';
  for (const pl of lists) {
    const opt = document.createElement("option");
    opt.value = String(pl.id);
    opt.textContent = `${pl.name} (${pl.count})`;
    opt.dataset.name = pl.name;
    opt.dataset.count = String(pl.count);
    playlistSelect.appendChild(opt);
  }

  // Restore previous selection if still present
  if (prev && playlistSelect.querySelector(`option[value="${CSS.escape(prev)}"]`)) {
    playlistSelect.value = prev;
  } else {
    playlistSelect.value = "";
    selectedPlaylistId   = null;
    selectedPlaylistName = "";
  }
  syncPlaylistState();
}

function syncPlaylistState() {
  const sel = playlistSelect.options[playlistSelect.selectedIndex];
  if (sel && sel.value) {
    selectedPlaylistId   = parseInt(sel.value, 10);
    selectedPlaylistName = sel.dataset.name || sel.text.replace(/\s*\(\d+\)$/, "");
    const cnt = parseInt(sel.dataset.count || "0", 10);
    playlistPhotoCnt.textContent = `${cnt} photo${cnt !== 1 ? "s" : ""}`;
    playlistPhotoCnt.classList.remove("hidden");
  } else {
    selectedPlaylistId   = null;
    selectedPlaylistName = "";
    playlistPhotoCnt.classList.add("hidden");
  }

  const hasSelection = selectedPlaylistId !== null;
  modifyPlaylistBtn.disabled    = !hasSelection;
  deletePlaylistBtn.disabled    = !hasSelection;
  showPlaylistBtn.disabled      = !hasSelection;
  slideshowPlaylistBtn.disabled = !hasSelection;
  playlistHint.classList.toggle("hidden", !hasSelection);
}

async function refreshPlaylistCount() {
  if (!selectedPlaylistId) return;
  const lists = await apiGetPlaylists();
  const me = lists.find(p => p.id === selectedPlaylistId);
  if (!me) return;
  const opt = playlistSelect.querySelector(`option[value="${selectedPlaylistId}"]`);
  if (opt) {
    const name = me.name;
    opt.textContent = `${name} (${me.count})`;
    opt.dataset.count = String(me.count);
    opt.dataset.name  = name;
    playlistPhotoCnt.textContent = `${me.count} photo${me.count !== 1 ? "s" : ""}`;
  }
}

playlistSelect.addEventListener("change", syncPlaylistState);

// New playlist dialog
newPlaylistBtn.addEventListener("click", () => {
  newPlaylistName.value = "";
  newPlaylistOverlay.classList.remove("hidden");
  newPlaylistName.focus();
});

function dismissNewPlaylistDialog() {
  newPlaylistOverlay.classList.add("hidden");
}

closeNewPlaylistDialog.addEventListener("click", dismissNewPlaylistDialog);
cancelNewPlaylist.addEventListener("click", dismissNewPlaylistDialog);

newPlaylistOverlay.addEventListener("click", (e) => {
  if (e.target === newPlaylistOverlay) dismissNewPlaylistDialog();
});

newPlaylistName.addEventListener("keydown", (e) => {
  if (e.key === "Enter") confirmNewPlaylist.click();
  if (e.key === "Escape") dismissNewPlaylistDialog();
});

confirmNewPlaylist.addEventListener("click", async () => {
  const name = newPlaylistName.value.trim();
  if (!name) { newPlaylistName.focus(); return; }
  confirmNewPlaylist.disabled = true;
  const pl = await apiCreatePlaylist(name);
  confirmNewPlaylist.disabled = false;
  if (!pl) { showToast("Failed to create playlist", "err"); return; }
  dismissNewPlaylistDialog();
  await loadPlaylists();
  playlistSelect.value = String(pl.id);
  syncPlaylistState();
  showToast(`Playlist "${pl.name}" created`, "ok");
});

// Delete playlist
deletePlaylistBtn.addEventListener("click", async () => {
  if (!selectedPlaylistId) return;
  if (!confirm(`Delete playlist "${selectedPlaylistName}"? This cannot be undone.`)) return;
  const ok = await apiDeletePlaylist(selectedPlaylistId);
  if (ok) {
    showToast(`Playlist "${selectedPlaylistName}" deleted`, "info");
    await loadPlaylists();
  } else {
    showToast("Failed to delete playlist", "err");
  }
});

// Modify playlist — highlight that the playlist is selected for adding
modifyPlaylistBtn.addEventListener("click", () => {
  if (!selectedPlaylistId) return;
  showToast(`Right-click any photo or press P in the lightbox to add to "${selectedPlaylistName}"`, "info");
});

// Show playlist modal
showPlaylistBtn.addEventListener("click", async () => {
  if (!selectedPlaylistId) return;
  await openPlaylistModal();
});

async function openPlaylistModal() {
  playlistPhotos = await apiGetPlaylistPhotos(selectedPlaylistId);
  playlistModalTitle.textContent = selectedPlaylistName;
  renderPlaylistModal(playlistPhotos);
  playlistModalOverlay.classList.remove("hidden");
}

function renderPlaylistModal(photos) {
  playlistGrid.innerHTML = "";
  playlistEmpty.classList.toggle("hidden", photos.length > 0);

  photos.forEach((photo, i) => {
    const card = document.createElement("article");
    card.className = "card playlist-card";
    card.draggable = true;
    card.dataset.photoId = String(photo.id);

    const img = document.createElement("img");
    img.className = "thumb";
    img.src = photo.thumbnail_url;
    img.alt = photo.file_name;
    img.loading = "lazy";

    const removeBtn = document.createElement("button");
    removeBtn.className = "card-remove-btn";
    removeBtn.type = "button";
    removeBtn.title = "Remove from playlist";
    removeBtn.textContent = "✕";
    removeBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const ok = await apiRemovePhotoFromPlaylist(selectedPlaylistId, photo.id);
      if (ok) {
        playlistPhotos = playlistPhotos.filter(p => p.id !== photo.id);
        renderPlaylistModal(playlistPhotos);
      }
    });

    const meta = document.createElement("div");
    meta.className = "meta";
    const name = document.createElement("div");
    name.className = "name";
    name.title = photo.file_name;
    name.textContent = photo.file_name;
    meta.appendChild(name);

    card.appendChild(img);
    card.appendChild(removeBtn);
    card.appendChild(meta);

    // Right-click to remove
    card.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      showContextMenu(e, [{
        label: "Remove from playlist",
        handler: async () => {
          const ok = await apiRemovePhotoFromPlaylist(selectedPlaylistId, photo.id);
          if (ok) {
            playlistPhotos = playlistPhotos.filter(p => p.id !== photo.id);
            renderPlaylistModal(playlistPhotos);
          }
        },
      }]);
    });

    // Drag & drop
    card.addEventListener("dragstart", (e) => {
      dragSrcIndex = i;
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", String(i));
      setTimeout(() => card.classList.add("dragging"), 0);
    });

    card.addEventListener("dragend", () => {
      card.classList.remove("dragging");
      playlistGrid.querySelectorAll(".drag-before, .drag-after")
        .forEach(c => c.classList.remove("drag-before", "drag-after"));
    });

    card.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      playlistGrid.querySelectorAll(".drag-before, .drag-after")
        .forEach(c => c.classList.remove("drag-before", "drag-after"));
      const rect = card.getBoundingClientRect();
      if (e.clientX < rect.left + rect.width / 2) {
        card.classList.add("drag-before");
      } else {
        card.classList.add("drag-after");
      }
    });

    card.addEventListener("dragleave", (e) => {
      if (!card.contains(e.relatedTarget)) {
        card.classList.remove("drag-before", "drag-after");
      }
    });

    card.addEventListener("drop", async (e) => {
      e.preventDefault();
      card.classList.remove("drag-before", "drag-after");

      const fromIndex = dragSrcIndex;
      if (fromIndex === i) return;

      const insertBefore = e.clientX < card.getBoundingClientRect().left + card.getBoundingClientRect().width / 2;

      const newPhotos = [...playlistPhotos];
      const [moved] = newPhotos.splice(fromIndex, 1);
      let insertIdx;
      if (insertBefore) {
        insertIdx = fromIndex < i ? i - 1 : i;
      } else {
        insertIdx = fromIndex < i ? i : i + 1;
      }
      newPhotos.splice(insertIdx, 0, moved);

      const ok = await apiReorderPlaylistPhotos(selectedPlaylistId, newPhotos.map(p => p.id));
      if (ok) {
        playlistPhotos = newPhotos;
        renderPlaylistModal(playlistPhotos);
      }
    });

    playlistGrid.appendChild(card);
  });
}

closePlaylistModal.addEventListener("click", () => {
  playlistModalOverlay.classList.add("hidden");
});

playlistModalOverlay.addEventListener("click", (e) => {
  if (e.target === playlistModalOverlay) playlistModalOverlay.classList.add("hidden");
});

// ── Slideshow ──────────────────────────────────────────────────────────────────
slideshowPlaylistBtn.addEventListener("click", async () => {
  if (!selectedPlaylistId) return;
  ssPhotos = await apiGetPlaylistPhotos(selectedPlaylistId);
  if (ssPhotos.length === 0) { showToast("Playlist is empty", "warn"); return; }
  startSlideshow();
});

function startSlideshow() {
  ssIndex  = 0;
  ssPaused = false;
  ssPauseBtn.textContent = "Pause";
  slideshowOverlay.classList.remove("hidden");
  showSlideshowPhoto(0);
  scheduleSlideshowAdvance();
}

function showSlideshowPhoto(index) {
  ssIndex = Math.max(0, Math.min(index, ssPhotos.length - 1));
  slideshowImage.src = ssPhotos[ssIndex].image_url;
  ssCounter.textContent = `${ssIndex + 1} / ${ssPhotos.length}`;
  ssPrevBtn.disabled = ssIndex === 0;
  ssNextBtn.disabled = ssIndex === ssPhotos.length - 1;
}

function getSlideshowDelay() {
  return parseInt(ssSpeedSlider.value, 10) * 1000;
}

function scheduleSlideshowAdvance() {
  if (ssTimer) clearTimeout(ssTimer);
  if (!ssPaused && ssPhotos.length > 1) {
    ssTimer = setTimeout(() => {
      if (ssIndex < ssPhotos.length - 1) {
        showSlideshowPhoto(ssIndex + 1);
        scheduleSlideshowAdvance();
      } else {
        // Loop back to start
        showSlideshowPhoto(0);
        scheduleSlideshowAdvance();
      }
    }, getSlideshowDelay());
  }
}

function stopSlideshow() {
  if (ssTimer) { clearTimeout(ssTimer); ssTimer = null; }
  slideshowOverlay.classList.add("hidden");
  slideshowImage.src = "";
}

closeSlideshowBtn.addEventListener("click", stopSlideshow);

ssPrevBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  if (ssIndex > 0) { showSlideshowPhoto(ssIndex - 1); scheduleSlideshowAdvance(); }
});

ssNextBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  if (ssIndex < ssPhotos.length - 1) { showSlideshowPhoto(ssIndex + 1); scheduleSlideshowAdvance(); }
});

ssPauseBtn.addEventListener("click", () => {
  ssPaused = !ssPaused;
  ssPauseBtn.textContent = ssPaused ? "Resume" : "Pause";
  if (!ssPaused) scheduleSlideshowAdvance();
  else if (ssTimer) { clearTimeout(ssTimer); ssTimer = null; }
});

ssSpeedSlider.addEventListener("input", () => {
  ssSpeedVal.textContent = `${ssSpeedSlider.value}s`;
  if (!ssPaused) scheduleSlideshowAdvance();
});

slideshowOverlay.addEventListener("keydown", (e) => {
  if (e.key === "Escape") stopSlideshow();
});

// ── Slideshow from current filter view ───────────────────────────────────────
async function fetchAllPhotosForSlideshow() {
  let offset = 0;
  const BATCH = 200;
  const all = [];

  while (true) {
    const qs = new URLSearchParams();
    let apiUrl;

    if (mode === "search") {
      apiUrl = "/api/search";
      qs.set("query", searchQueryEl.value.trim());
    } else if (mode === "same_person") {
      apiUrl = "/api/same_person";
      qs.set("photo_id", String(samePersonSourceId));
    } else if (mode === "same_person_similar") {
      apiUrl = "/api/same_person_similar";
      qs.set("photo_id", String(samePersonSourceId));
    } else {
      apiUrl = "/api/photos";
      for (const [k, v] of primaryQS()) qs.append(k, v);
    }

    qs.set("offset", String(offset));
    qs.set("limit", String(BATCH));
    if (selectedFolder) qs.set("folder", selectedFolder);

    const resp = await fetch(`${apiUrl}?${qs}`);
    if (!resp.ok) break;
    const data = await resp.json();

    all.push(...data.items);
    if (!data.has_more) break;
    offset = data.next_offset;
  }

  return all;
}

slideshowViewBtn.addEventListener("click", async () => {
  slideshowViewBtn.disabled = true;
  slideshowViewBtn.textContent = "Loading…";
  try {
    const photos = await fetchAllPhotosForSlideshow();
    if (photos.length === 0) { showToast("No photos to show", "warn"); return; }
    ssPhotos = photos;
    startSlideshow();
  } finally {
    slideshowViewBtn.disabled = photoList.length === 0;
    slideshowViewBtn.textContent = "▶ Slideshow all";
  }
});

// ── Folder filter ─────────────────────────────────────────────────────────────
async function loadFolders() {
  const qs = (mode === "search" || mode === "same_person" || mode === "same_person_similar") ? new URLSearchParams() : primaryQS();
  const resp = await fetch(`/api/folders?${qs}`).catch(() => null);
  if (!resp || !resp.ok) return;
  const data = await resp.json();

  const paths  = data.map((f) => f.path);
  const labels = stripPrefix(paths);

  const inResults = !selectedFolder || data.some((f) => f.path === selectedFolder);

  folderSelect.innerHTML = '<option value="">All folders</option>';
  data.forEach((f, i) => {
    const opt = document.createElement("option");
    opt.value = f.path;
    opt.textContent = `${labels[i]} (${f.count})`;
    opt.title = f.path;
    folderSelect.appendChild(opt);
  });

  if (selectedFolder && !inResults) {
    const opt = document.createElement("option");
    opt.value = selectedFolder;
    opt.textContent = `${selectedFolder.split("/").filter(Boolean).pop() ?? selectedFolder} (0)`;
    opt.title = selectedFolder;
    folderSelect.appendChild(opt);
  }

  folderSelect.value = selectedFolder;

  if (data.length > 0) {
    folderCount.textContent = `${data.length} folder${data.length > 1 ? "s" : ""}`;
    folderCount.classList.remove("hidden");
  } else {
    folderCount.classList.add("hidden");
  }
}

folderSelect.addEventListener("change", () => {
  selectedFolder = folderSelect.value;
  loadPhotos({ reset: true });
});

// ── Photo card ────────────────────────────────────────────────────────────────
function renderCard(photo) {
  const card = document.createElement("article");
  card.className = "card";

  const img = document.createElement("img");
  img.className = "thumb";
  img.src = photo.thumbnail_url;
  img.alt = photo.file_name;
  img.loading = "lazy";

  const meta = document.createElement("div");
  meta.className = "meta";

  const name = document.createElement("div");
  name.className = "name";
  name.title = photo.file_name;
  name.textContent = photo.file_name;

  const cats = document.createElement("div");
  cats.className = "cats";
  cats.textContent = photo.categories.map((c) => `${c.name} (${fmtScore(c.score)})`).join(" | ");

  meta.appendChild(name);
  meta.appendChild(cats);
  card.appendChild(img);
  card.appendChild(meta);

  // Right-click context menu
  card.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    const actions = [];
    if (selectedPlaylistId) {
      actions.push({
        label: `Add to "${selectedPlaylistName}"`,
        handler: () => apiAddPhotoToPlaylist(selectedPlaylistId, photo.id),
      });
    }
    actions.push({
      label: "Foto della stessa persona",
      handler: () => searchSamePerson(photo),
    });
    actions.push({
      label: "Foto simili della stessa persona",
      handler: () => searchSamePersonSimilar(photo),
    });
    showContextMenu(e, actions);
  });

  return card;
}

// ── Photo loading ─────────────────────────────────────────────────────────────
async function loadPhotos({ reset = false } = {}) {
  if (reset) {
    nextOffset = 0;
    hasMore    = false;
    photoList  = [];
    grid.innerHTML = "";
    slideshowViewBtn.disabled = true;
  }

  loadMoreBtn.disabled = true;

  if (mode === "category" && selectedCategories.size === 0 && !selectedFolder) {
    statusEl.textContent = "Select at least one category.";
    return;
  }
  if (mode === "cluster" && !getClusterId()) {
    statusEl.textContent = "No clusters available.";
    return;
  }
  if (mode === "face_cluster" && !getFaceClusterId()) {
    statusEl.textContent = "No face clusters available.";
    return;
  }
  if (mode === "search" && !searchQueryEl.value.trim()) {
    statusEl.textContent = "Enter a description and click Search.";
    return;
  }

  const qs = new URLSearchParams();
  let apiUrl;

  if (mode === "search") {
    apiUrl = "/api/search";
    qs.set("query", searchQueryEl.value.trim());
  } else if (mode === "same_person") {
    apiUrl = "/api/same_person";
    qs.set("photo_id", String(samePersonSourceId));
  } else if (mode === "same_person_similar") {
    apiUrl = "/api/same_person_similar";
    qs.set("photo_id", String(samePersonSourceId));
  } else {
    apiUrl = "/api/photos";
    for (const [k, v] of primaryQS()) qs.append(k, v);
  }

  qs.set("offset", String(nextOffset));
  qs.set("limit", "50");
  if (selectedFolder) qs.set("folder", selectedFolder);

  statusEl.textContent = "Loading…";

  const resp = await fetch(`${apiUrl}?${qs}`);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    if (mode === "same_person" || mode === "same_person_similar") {
      showToast(err.error || "Errore nella ricerca", "warn");
      statusEl.textContent = "";
    } else {
      statusEl.textContent = `Error: ${err.error || resp.statusText}`;
    }
    return;
  }
  const data = await resp.json();

  for (const item of data.items) {
    const index = photoList.length;
    photoList.push(item);
    const card = renderCard(item);
    card.addEventListener("click", () => openLightbox(index));
    grid.appendChild(card);
  }

  nextOffset                = data.next_offset;
  hasMore                   = data.has_more;
  loadMoreBtn.disabled      = !hasMore;
  slideshowViewBtn.disabled = photoList.length === 0;

  const parts = [];
  if (mode === "category" && selectedCategories.size > 0) {
    parts.push([...selectedCategories].join(" + "));
  } else if (mode === "cluster") {
    parts.push(clusterSelect.options[clusterSelect.selectedIndex]?.text ?? "");
  } else if (mode === "face_cluster") {
    parts.push(faceClusterSelect.options[faceClusterSelect.selectedIndex]?.text ?? "");
  } else if (mode === "search") {
    const q = searchQueryEl.value.trim();
    parts.push(`"${q.length > 60 ? q.slice(0, 60) + "…" : q}"`);
  } else if (mode === "same_person") {
    const n = samePersonSourceName;
    parts.push(`stessa persona di "${n.length > 40 ? n.slice(0, 40) + "…" : n}"`);
  } else if (mode === "same_person_similar") {
    const n = samePersonSourceName;
    parts.push(`stessa persona · foto simili di "${n.length > 40 ? n.slice(0, 40) + "…" : n}"`);
  }
  if (selectedFolder) {
    parts.push(folderSelect.options[folderSelect.selectedIndex]?.text ?? selectedFolder);
  }
  statusEl.textContent = `${grid.childElementCount} photos — ${parts.join(" · ")}`;
}

// ── Lightbox ──────────────────────────────────────────────────────────────────
function folderName(filePath) {
  const parts = (filePath || "").split("/").filter(Boolean);
  return parts.length > 1 ? parts[parts.length - 2] : "";
}

document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && lightboxOpen()) closeLightboxFn();
});

function applyLightboxTransform() {
  lightboxImage.style.transform =
    `translate(${lightboxTranslateX}px, ${lightboxTranslateY}px) scale(${lightboxScale})`;
}

function openLightbox(index) {
  currentPhotoIndex  = index;
  lightboxScale      = 1;
  lightboxTranslateX = 0;
  lightboxTranslateY = 0;
  lightboxImage.style.cursor = "";
  applyLightboxTransform();
  lightboxImage.src = photoList[index].image_url;
  lightbox.classList.remove("hidden");
  const folder = folderName(photoList[index].file_path || "");
  if (folder) {
    lightboxCaption.textContent = folder;
    lightboxCaption.classList.remove("hidden");
  } else {
    lightboxCaption.classList.add("hidden");
  }
  if (lightbox.requestFullscreen) lightbox.requestFullscreen().catch(() => {});
  updateNavButtons();
}

function closeLightboxFn() {
  lightbox.classList.add("hidden");
  lightboxImage.src  = "";
  lightboxScale      = 1;
  lightboxTranslateX = 0;
  lightboxTranslateY = 0;
  lightboxImage.style.cursor = "";
  lightboxCaption.classList.add("hidden");
  applyLightboxTransform();
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
}

function updateNavButtons() {
  prevBtn.classList.toggle("hidden", currentPhotoIndex <= 0);
  nextBtn.classList.toggle("hidden", currentPhotoIndex >= photoList.length - 1 && !hasMore);
}

async function navigatePrev() {
  if (currentPhotoIndex > 0) openLightbox(currentPhotoIndex - 1);
}

async function navigateNext() {
  if (currentPhotoIndex < photoList.length - 1) {
    openLightbox(currentPhotoIndex + 1);
  } else if (hasMore) {
    await loadPhotos({ reset: false });
    if (currentPhotoIndex < photoList.length - 1) openLightbox(currentPhotoIndex + 1);
  }
}

prevBtn.addEventListener("click", (e) => { e.stopPropagation(); navigatePrev(); });
nextBtn.addEventListener("click", (e) => { e.stopPropagation(); navigateNext(); });

lightbox.addEventListener("click", (e) => {
  if (lbDragMoved) { lbDragMoved = false; return; }
  if (e.target === lightbox) closeLightboxFn();
});
closeLightbox.addEventListener("click", closeLightboxFn);

// Right-click on lightbox image
lightboxImage.addEventListener("contextmenu", (e) => {
  if (currentPhotoIndex < 0) return;
  e.preventDefault();
  e.stopPropagation();
  const photo = photoList[currentPhotoIndex];
  const actions = [];
  if (selectedPlaylistId) {
    actions.push({
      label: `Add to "${selectedPlaylistName}"`,
      handler: () => apiAddPhotoToPlaylist(selectedPlaylistId, photo.id),
    });
  }
  actions.push({
    label: "Foto della stessa persona",
    handler: () => { closeLightboxFn(); searchSamePerson(photo); },
  });
  actions.push({
    label: "Foto simili della stessa persona",
    handler: () => { closeLightboxFn(); searchSamePersonSimilar(photo); },
  });
  showContextMenu(e, actions);
});

window.addEventListener("keydown", async (e) => {
  if (lightboxOpen()) {
    if      (e.key === "Escape")     closeLightboxFn();
    else if (e.key === "ArrowLeft")  navigatePrev();
    else if (e.key === "ArrowRight") navigateNext();
    else if (e.key === "p" || e.key === "P") {
      if (selectedPlaylistId && currentPhotoIndex >= 0) {
        await apiAddPhotoToPlaylist(selectedPlaylistId, photoList[currentPhotoIndex].id);
      } else if (!selectedPlaylistId) {
        showToast("No playlist selected — create or select one in the Playlist panel", "warn");
      }
    }
    return;
  }
  if (!slideshowOverlay.classList.contains("hidden")) {
    if      (e.key === "Escape")     stopSlideshow();
    else if (e.key === "ArrowLeft")  ssPrevBtn.click();
    else if (e.key === "ArrowRight") ssNextBtn.click();
    else if (e.key === " ")          ssPauseBtn.click();
  }
});

lightbox.addEventListener("wheel", (e) => {
  if (!e.ctrlKey) return;
  e.preventDefault();
  lightboxScale = e.deltaY < 0
    ? Math.min(SCALE_MAX, lightboxScale * SCALE_FACTOR)
    : Math.max(SCALE_MIN, lightboxScale / SCALE_FACTOR);
  if (lightboxScale <= 1) {
    lightboxTranslateX = 0;
    lightboxTranslateY = 0;
    lightboxImage.style.cursor = "";
  } else {
    lightboxImage.style.cursor = "grab";
  }
  applyLightboxTransform();
}, { passive: false });

// Lightbox pan via drag when zoomed in
lightboxImage.addEventListener("mousedown", (e) => {
  if (lightboxScale <= 1 || e.button !== 0) return;
  e.preventDefault();
  lbDragging   = true;
  lbDragMoved  = false;
  lbDragStartX = e.clientX;
  lbDragStartY = e.clientY;
  lbDragStartTX = lightboxTranslateX;
  lbDragStartTY = lightboxTranslateY;
  lightboxImage.style.cursor = "grabbing";
});

document.addEventListener("mousemove", (e) => {
  if (!lbDragging) return;
  const dx = e.clientX - lbDragStartX;
  const dy = e.clientY - lbDragStartY;
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) lbDragMoved = true;
  lightboxTranslateX = lbDragStartTX + dx;
  lightboxTranslateY = lbDragStartTY + dy;
  applyLightboxTransform();
});

document.addEventListener("mouseup", () => {
  if (!lbDragging) return;
  lbDragging = false;
  lightboxImage.style.cursor = lightboxScale > 1 ? "grab" : "";
});

// ── Category mode ─────────────────────────────────────────────────────────────
async function loadCategories() {
  const resp = await fetch("/api/categories");
  if (!resp.ok) throw new Error("Failed to load categories");
  const cats = await resp.json();

  categoryList.innerHTML = "";
  for (const cat of cats) {
    const label = document.createElement("label");
    label.className = "cat-chip";
    label.dataset.name = cat.name;

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = cat.name;

    label.appendChild(cb);
    label.appendChild(document.createTextNode(`${cat.name} (${cat.count})`));
    categoryList.appendChild(label);

    label.addEventListener("click", (e) => { e.preventDefault(); toggleCategory(cat.name, label, cb); });
  }

  const first = cats.find((c) => c.count > 0);
  if (first) {
    const lbl = categoryList.querySelector(`[data-name="${CSS.escape(first.name)}"]`);
    if (lbl) toggleCategory(first.name, lbl, lbl.querySelector("input"));
  }
}

async function toggleCategory(name, labelEl, cbEl) {
  if (selectedCategories.has(name)) {
    selectedCategories.delete(name);
    labelEl.classList.remove("selected");
    cbEl.checked = false;
  } else {
    if (selectedCategories.size >= MAX_CATS) return;
    selectedCategories.add(name);
    labelEl.classList.add("selected");
    cbEl.checked = true;
  }
  updateCategoryChipStates();
  selectedCount.textContent = `${selectedCategories.size} selected`;
  await Promise.all([loadFolders(), loadPhotos({ reset: true })]);
}

function updateCategoryChipStates() {
  const full = selectedCategories.size >= MAX_CATS;
  for (const lbl of categoryList.querySelectorAll(".cat-chip")) {
    lbl.classList.toggle("disabled", full && !selectedCategories.has(lbl.dataset.name));
  }
}

// ── Face Cluster mode ─────────────────────────────────────────────────────────
async function loadFaceClusters() {
  const resp = await fetch("/api/face_clusters");
  if (!resp.ok) throw new Error("Failed to load face clusters");
  const clusters = await resp.json();

  faceClusterSelect.innerHTML = "";
  if (clusters.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "No face clusters — run cluster_faces.py first";
    faceClusterSelect.appendChild(opt);
    faceClusterDesc.value = "";
    return;
  }

  for (const c of clusters) {
    const opt = document.createElement("option");
    opt.value = String(c.id);
    opt.textContent = `${c.description} (${c.face_count} faces · ${c.photo_count} photos)`;
    faceClusterSelect.appendChild(opt);
  }

  syncFaceClusterDesc();
  selectedFolder = "";
  await Promise.all([loadFolders(), loadPhotos({ reset: true })]);
}

function syncFaceClusterDesc() {
  const sel = faceClusterSelect.options[faceClusterSelect.selectedIndex];
  if (!sel) return;
  faceClusterDesc.value = sel.text.replace(/\s*\(\d.*\)$/, "");
  clearFaceClusterSaveStatus();
}

function clearFaceClusterSaveStatus() {
  saveFaceClusterStatus.textContent = "";
  saveFaceClusterStatus.className = "save-status";
}

faceClusterSelect.addEventListener("change", async () => {
  syncFaceClusterDesc();
  await Promise.all([loadFolders(), loadPhotos({ reset: true })]);
});

saveFaceClusterBtn.addEventListener("click", async () => {
  const fcid = getFaceClusterId();
  if (!fcid) return;
  const desc = faceClusterDesc.value.trim();
  if (!desc) return;

  saveFaceClusterBtn.disabled = true;
  clearFaceClusterSaveStatus();

  const resp = await fetch(`/api/face_clusters/${fcid}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ description: desc }),
  });

  saveFaceClusterBtn.disabled = false;

  if (resp.ok) {
    const opt = faceClusterSelect.options[faceClusterSelect.selectedIndex];
    const m = opt.text.match(/\(.*\)$/);
    opt.text = `${desc}${m ? " " + m[0] : ""}`;
    saveFaceClusterStatus.textContent = "Saved";
    saveFaceClusterStatus.className = "save-status ok";
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(clearFaceClusterSaveStatus, 3000);
  } else {
    const err = await resp.json().catch(() => ({}));
    saveFaceClusterStatus.textContent = err.error || "Save failed";
    saveFaceClusterStatus.className = "save-status err";
  }
});

// ── Cluster mode ──────────────────────────────────────────────────────────────
async function loadClusters() {
  const resp = await fetch("/api/clusters");
  if (!resp.ok) throw new Error("Failed to load clusters");
  const clusters = await resp.json();

  clusterSelect.innerHTML = "";
  if (clusters.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "No clusters — run cluster_photos.py first";
    clusterSelect.appendChild(opt);
    clusterDesc.value = "";
    return;
  }

  for (const c of clusters) {
    const opt = document.createElement("option");
    opt.value = String(c.id);
    opt.textContent = `${c.description} (${c.count})`;
    clusterSelect.appendChild(opt);
  }

  syncClusterDesc();
  selectedFolder = "";
  await Promise.all([loadFolders(), loadPhotos({ reset: true })]);
}

function syncClusterDesc() {
  const sel = clusterSelect.options[clusterSelect.selectedIndex];
  if (!sel) return;
  clusterDesc.value = sel.text.replace(/\s*\(\d+\)$/, "");
  clearSaveStatus();
}

function clearSaveStatus() {
  saveStatus.textContent = "";
  saveStatus.className = "save-status";
}

clusterSelect.addEventListener("change", async () => {
  syncClusterDesc();
  await Promise.all([loadFolders(), loadPhotos({ reset: true })]);
});

saveClusterBtn.addEventListener("click", async () => {
  const cid = getClusterId();
  if (!cid) return;
  const desc = clusterDesc.value.trim();
  if (!desc) return;

  saveClusterBtn.disabled = true;
  clearSaveStatus();

  const resp = await fetch(`/api/clusters/${cid}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ description: desc }),
  });

  saveClusterBtn.disabled = false;

  if (resp.ok) {
    const opt = clusterSelect.options[clusterSelect.selectedIndex];
    const m = opt.text.match(/\(\d+\)$/);
    opt.text = `${desc}${m ? " " + m[0] : ""}`;
    saveStatus.textContent = "Saved";
    saveStatus.className = "save-status ok";
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(clearSaveStatus, 3000);
  } else {
    const err = await resp.json().catch(() => ({}));
    saveStatus.textContent = err.error || "Save failed";
    saveStatus.className = "save-status err";
  }
});

// ── Mode switching ────────────────────────────────────────────────────────────
async function switchMode(newMode) {
  samePersonSourceId   = null;
  samePersonSourceName = "";
  mode = newMode;
  const isCat        = newMode === "category";
  const isCluster    = newMode === "cluster";
  const isFaceCluster= newMode === "face_cluster";
  const isSearch     = newMode === "search";

  modeCategoryBtn.classList.toggle("active",    isCat);
  modeClusterBtn.classList.toggle("active",     isCluster);
  modeFaceClusterBtn.classList.toggle("active", isFaceCluster);
  modeSearchBtn.classList.toggle("active",      isSearch);
  categoryControls.classList.toggle("hidden",    !isCat);
  clusterControls.classList.toggle("hidden",     !isCluster);
  faceClusterControls.classList.toggle("hidden", !isFaceCluster);
  searchControls.classList.toggle("hidden",      !isSearch);

  selectedFolder = "";
  photoList  = [];
  nextOffset = 0;
  hasMore    = false;
  grid.innerHTML = "";
  loadMoreBtn.disabled = true;

  if (isCat) {
    statusEl.textContent = "Loading…";
    await loadFolders();
    loadPhotos({ reset: true });
  } else if (isCluster) {
    statusEl.textContent = "Loading…";
    loadClusters();
  } else if (isFaceCluster) {
    statusEl.textContent = "Loading…";
    loadFaceClusters();
  } else {
    statusEl.textContent = "Enter a description and click Search.";
    await loadFolders();
  }
}

modeCategoryBtn.addEventListener("click",    () => { if (mode !== "category")     switchMode("category"); });
modeClusterBtn.addEventListener("click",     () => { if (mode !== "cluster")      switchMode("cluster"); });
modeFaceClusterBtn.addEventListener("click", () => { if (mode !== "face_cluster") switchMode("face_cluster"); });
modeSearchBtn.addEventListener("click",      () => { if (mode !== "search")       switchMode("search"); });

// ── Search ────────────────────────────────────────────────────────────────────
searchQueryEl.addEventListener("input", () => {
  searchCharCount.textContent = `${searchQueryEl.value.length} / 600`;
});

searchQueryEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    searchBtn.click();
  }
});

searchBtn.addEventListener("click", async () => {
  const q = searchQueryEl.value.trim();
  if (!q) { searchQueryEl.focus(); return; }
  searchBtn.disabled = true;
  searchBtn.textContent = "Searching…";
  await loadPhotos({ reset: true });
  searchBtn.disabled = false;
  searchBtn.textContent = "Search";
});

// ── Pagination ────────────────────────────────────────────────────────────────
loadMoreBtn.addEventListener("click", () => { if (hasMore) loadPhotos({ reset: false }); });

// ── Boot ──────────────────────────────────────────────────────────────────────
async function main() {
  try {
    await Promise.all([
      loadCategories(),
      loadPlaylists(),
    ]);
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
}

main();
