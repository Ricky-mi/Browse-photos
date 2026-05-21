// ── DOM refs ──────────────────────────────────────────────────────────────────
const modeCategoryBtn      = document.getElementById("modeCategory");
const modeClusterBtn       = document.getElementById("modeCluster");
const categoryControls     = document.getElementById("categoryControls");
const clusterControls      = document.getElementById("clusterControls");
const categoryList         = document.getElementById("categoryList");
const selectedCount        = document.getElementById("selectedCount");
const clusterSelect        = document.getElementById("clusterSelect");
const clusterDesc          = document.getElementById("clusterDesc");
const saveClusterBtn       = document.getElementById("saveClusterBtn");
const saveStatus           = document.getElementById("saveStatus");
const folderSelect         = document.getElementById("folderSelect");
const folderCount          = document.getElementById("folderCount");
const grid                 = document.getElementById("grid");
const statusEl             = document.getElementById("status");
const loadMoreBtn          = document.getElementById("loadMoreBtn");
const lightbox             = document.getElementById("lightbox");
const lightboxImage        = document.getElementById("lightboxImage");
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
let saveTimer          = null;

// Playlist state
let selectedPlaylistId   = null;
let selectedPlaylistName = "";
let playlistPhotos       = [];

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

// ── Folder filter ─────────────────────────────────────────────────────────────
async function loadFolders() {
  const qs = primaryQS();
  const resp = await fetch(`/api/folders?${qs}`).catch(() => null);
  if (!resp || !resp.ok) return;
  const data = await resp.json();

  const paths  = data.map((f) => f.path);
  const labels = stripPrefix(paths);

  const prevFolder = selectedFolder;
  const stillValid = data.some((f) => f.path === prevFolder);
  if (!stillValid) selectedFolder = "";

  folderSelect.innerHTML = '<option value="">All folders</option>';
  data.forEach((f, i) => {
    const opt = document.createElement("option");
    opt.value = f.path;
    opt.textContent = `${labels[i]} (${f.count})`;
    opt.title = f.path;
    folderSelect.appendChild(opt);
  });
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

  // Right-click to add to selected playlist
  card.addEventListener("contextmenu", (e) => {
    if (!selectedPlaylistId) return;
    e.preventDefault();
    showContextMenu(e, [{
      label: `Add to "${selectedPlaylistName}"`,
      handler: () => apiAddPhotoToPlaylist(selectedPlaylistId, photo.id),
    }]);
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
  }

  loadMoreBtn.disabled = true;

  const qs = primaryQS();
  if (mode === "category" && selectedCategories.size === 0 && !selectedFolder) {
    statusEl.textContent = "Select at least one category.";
    return;
  }
  if (mode === "cluster" && !getClusterId()) {
    statusEl.textContent = "No clusters available.";
    return;
  }

  qs.set("offset", String(nextOffset));
  qs.set("limit", "50");
  if (selectedFolder) qs.set("folder", selectedFolder);

  statusEl.textContent = "Loading…";

  const resp = await fetch(`/api/photos?${qs}`);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    statusEl.textContent = `Error: ${err.error || resp.statusText}`;
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

  nextOffset           = data.next_offset;
  hasMore              = data.has_more;
  loadMoreBtn.disabled = !hasMore;

  const parts = [];
  if (mode === "category" && selectedCategories.size > 0) {
    parts.push([...selectedCategories].join(" + "));
  } else if (mode === "cluster") {
    parts.push(clusterSelect.options[clusterSelect.selectedIndex]?.text ?? "");
  }
  if (selectedFolder) {
    parts.push(folderSelect.options[folderSelect.selectedIndex]?.text ?? selectedFolder);
  }
  statusEl.textContent = `${grid.childElementCount} photos — ${parts.join(" · ")}`;
}

// ── Lightbox ──────────────────────────────────────────────────────────────────
function openLightbox(index) {
  currentPhotoIndex             = index;
  lightboxScale                 = 1;
  lightboxImage.style.transform = "scale(1)";
  lightboxImage.src             = photoList[index].image_url;
  lightbox.classList.remove("hidden");
  updateNavButtons();
}

function closeLightboxFn() {
  lightbox.classList.add("hidden");
  lightboxImage.src             = "";
  lightboxImage.style.transform = "scale(1)";
  lightboxScale                 = 1;
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

lightbox.addEventListener("click", (e) => { if (e.target === lightbox) closeLightboxFn(); });
closeLightbox.addEventListener("click", closeLightboxFn);

// Right-click on lightbox image → add to playlist
lightboxImage.addEventListener("contextmenu", (e) => {
  if (!selectedPlaylistId || currentPhotoIndex < 0) return;
  e.preventDefault();
  e.stopPropagation();
  showContextMenu(e, [{
    label: `Add to "${selectedPlaylistName}"`,
    handler: () => apiAddPhotoToPlaylist(selectedPlaylistId, photoList[currentPhotoIndex].id),
  }]);
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
  lightboxImage.style.transform = `scale(${lightboxScale})`;
}, { passive: false });

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
  mode = newMode;
  const isCat = mode === "category";

  modeCategoryBtn.classList.toggle("active", isCat);
  modeClusterBtn.classList.toggle("active", !isCat);
  categoryControls.classList.toggle("hidden", !isCat);
  clusterControls.classList.toggle("hidden", isCat);

  photoList  = [];
  nextOffset = 0;
  hasMore    = false;
  grid.innerHTML = "";
  loadMoreBtn.disabled = true;
  statusEl.textContent = "Loading…";

  if (isCat) {
    await loadFolders();
    loadPhotos({ reset: true });
  } else {
    loadClusters();
  }
}

modeCategoryBtn.addEventListener("click", () => { if (mode !== "category") switchMode("category"); });
modeClusterBtn.addEventListener("click",  () => { if (mode !== "cluster")  switchMode("cluster"); });

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
