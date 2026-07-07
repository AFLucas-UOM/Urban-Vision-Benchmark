#!/usr/bin/env python3
"""Build the MTSD Atlas - an interactive map of geolocated dataset captures.

Reads the image inventory produced by the EDA tooling (``mtsd_eda.inventory``),
embeds every geolocated capture into a single self-contained HTML page, and
renders it with Leaflet (canvas circle markers + clustering + a single-hue
density layer). Collection groups (GRP-1..n) are kept as navigation layers:
each can be toggled, soloed, or combined into the complete-dataset view.

Usage:
    python mtsd_mapper.py                 # cached inventory -> default output
    python mtsd_mapper.py --rescan        # force a fresh image scan first
    python mtsd_mapper.py --output custom.html
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from mtsd_eda import config, inventory

DEFAULT_OUTPUT = config.EDA_DIR / "MTSD_mapped.html"
MAP_CENTER = (35.9375, 14.3754)
MAP_ZOOM = 11

# Group palettes, fixed slot order (GRP-1 .. GRP-11). Both variants were
# validated with the Machado-2009 CVD checks against their map surface
# (all-pairs; light min dE 14.4, dark min dE 17.0). Identity is never
# colour-alone: the legend, tooltips, and popups all name the group.
GROUP_COLORS_LIGHT = [
    "#2a78d6", "#eda100", "#8c2a8c", "#00a3f0", "#8c5400", "#5b21b6",
    "#d88c41", "#b21a66", "#f26d87", "#b23542", "#00add8",
]
GROUP_COLORS_DARK = [
    "#574bd8", "#c98500", "#c23ac2", "#0c92f2", "#d86f4b", "#a5218f",
    "#a54221", "#d84b9e", "#a53a54", "#d86c6c", "#6030f2",
]


def load_points(rescan: bool) -> tuple[pd.DataFrame, dict]:
    """Load the inventory and return geolocated points plus corpus stats."""
    if rescan or not config.INVENTORY_CSV.exists():
        inv = inventory.build_inventory(force=rescan)
    else:
        inv = inventory.load_inventory()

    readable = inv[~inv["is_corrupted"]]
    points = readable[readable["has_gps"]].copy()
    points = points[
        points["gps_latitude"].between(-90, 90)
        & points["gps_longitude"].between(-180, 180)
        & ~((points["gps_latitude"] == 0) & (points["gps_longitude"] == 0))
    ]

    stats = {
        "total_images": int(len(inv)),
        "readable_images": int(len(readable)),
        "geolocated": int(len(points)),
        "coverage_pct": round(len(points) / len(readable) * 100, 1) if len(readable) else 0.0,
        "generated": datetime.now().strftime("%d %b %Y %H:%M"),
    }
    return points, stats


def relative_image_url(relative_path: str, output_html: Path) -> str:
    """URL from the output HTML to an image inside the repository."""
    path = Path(relative_path)
    absolute = config.BASE_DIR / path

    # Older inventory CSVs were built before the dataset moved under
    # Datasets/MTSD, so keep generated maps usable even with stale caches.
    if not absolute.exists():
        parts = path.parts
        if len(parts) >= 2 and parts[0] == "Datasets" and parts[1].startswith("GRP-"):
            absolute = config.BASE_DIR / "Datasets" / "MTSD" / Path(*parts[1:])
        elif parts and parts[0].startswith("GRP-"):
            absolute = config.DATASETS_ROOT / path

    try:
        rel = os.path.relpath(absolute, start=output_html.parent)
        return quote(Path(rel).as_posix(), safe="/")
    except ValueError:  # different drive
        return quote(absolute.as_posix(), safe="/:")


def build_payload(points: pd.DataFrame, output_html: Path) -> dict:
    """Compact JSON payload embedded into the page."""
    groups = sorted(points["group"].unique(), key=lambda g: int(g.split("-")[-1]))
    group_index = {group: idx for idx, group in enumerate(groups)}

    rows = []
    for row in points.itertuples(index=False):
        taken = ""
        if pd.notna(row.datetime_original):
            taken = pd.Timestamp(row.datetime_original).strftime("%d %b %Y %H:%M")
        camera = row.camera_model if isinstance(row.camera_model, str) else ""
        rows.append([
            group_index[row.group],
            round(float(row.gps_latitude), 6),
            round(float(row.gps_longitude), 6),
            row.filename,
            relative_image_url(row.relative_path, output_html),
            taken,
            camera,
        ])

    return {
        "groups": [
            {
                "name": group,
                "count": int((points["group"] == group).sum()),
                "light": GROUP_COLORS_LIGHT[idx % len(GROUP_COLORS_LIGHT)],
                "dark": GROUP_COLORS_DARK[idx % len(GROUP_COLORS_DARK)],
            }
            for idx, group in enumerate(groups)
        ],
        "points": rows,
        "center": MAP_CENTER,
        "zoom": MAP_ZOOM,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>
<style>
:root {
  --surface: #fcfcfb; --panel: #f9f9f7; --card: #ffffff;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --line: #e1e0d9; --ring: rgba(11,11,11,0.10);
  --accent: #2a78d6;
}
html.dark {
  --surface: #1a1a19; --panel: #0d0d0d; --card: #232322;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --line: #2c2c2a; --ring: rgba(255,255,255,0.10);
  --accent: #3987e5;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink); background: var(--panel);
  display: flex; overflow: hidden;
}
#sidebar {
  width: 330px; min-width: 330px; height: 100%; overflow-y: auto;
  background: var(--panel); border-right: 1px solid var(--line);
  padding: 20px 18px 14px; display: flex; flex-direction: column; gap: 16px;
}
#map { flex: 1; height: 100%; background: var(--surface); }
h1 { font-size: 17px; margin: 0; letter-spacing: 0.01em; }
.subtitle { font-size: 12px; color: var(--ink-2); margin-top: 4px; line-height: 1.45; }
.tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.tile {
  background: var(--card); border: 1px solid var(--ring); border-radius: 10px;
  padding: 10px 12px;
}
.tile .v { font-size: 20px; font-weight: 650; }
.tile .k { font-size: 10.5px; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.07em; margin-top: 2px; }
.section-label { font-size: 10.5px; color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.08em; margin: 2px 0 6px; }
.seg { display: flex; background: var(--card); border: 1px solid var(--ring);
  border-radius: 9px; padding: 3px; gap: 3px; }
.seg button {
  flex: 1; border: 0; background: transparent; color: var(--ink-2);
  font: inherit; font-size: 12px; padding: 6px 4px; border-radius: 6px; cursor: pointer;
}
.seg button.on { background: var(--accent); color: #fff; font-weight: 600; }
.rowline { display: flex; gap: 8px; align-items: center; }
.rowline .seg { flex: 1; }
input[type="search"] {
  width: 100%; padding: 8px 10px; font: inherit; font-size: 12.5px;
  color: var(--ink); background: var(--card);
  border: 1px solid var(--ring); border-radius: 9px; outline: none;
}
input[type="search"]:focus { border-color: var(--accent); }
.search-hint { font-size: 11px; color: var(--muted); margin-top: 4px; min-height: 14px; }
.group-actions { display: flex; gap: 6px; margin-bottom: 8px; }
.group-actions button {
  flex: 1; font: inherit; font-size: 11.5px; padding: 5px 0; cursor: pointer;
  background: var(--card); color: var(--ink-2);
  border: 1px solid var(--ring); border-radius: 7px;
}
.group-actions button:hover { border-color: var(--accent); color: var(--ink); }
#groups { display: flex; flex-direction: column; gap: 3px; }
.grp {
  display: flex; align-items: center; gap: 9px; padding: 6px 8px;
  border-radius: 8px; cursor: pointer; border: 1px solid transparent; user-select: none;
}
.grp:hover { background: var(--card); border-color: var(--ring); }
.grp .swatch { width: 13px; height: 13px; border-radius: 50%;
  border: 2px solid var(--surface); box-shadow: 0 0 0 1px var(--ring); flex: none; }
.grp .name { font-size: 13px; font-weight: 570; flex: 1; }
.grp .count { font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
.grp .solo {
  font-size: 10px; color: var(--muted); border: 1px solid var(--ring);
  background: transparent; border-radius: 5px; padding: 2px 6px; cursor: pointer;
  opacity: 0; transition: opacity 0.12s;
}
.grp:hover .solo { opacity: 1; }
.grp .solo:hover { color: var(--accent); border-color: var(--accent); }
.grp.off { opacity: 0.38; }
.grp.off .swatch { background: transparent !important; }
footer { margin-top: auto; font-size: 10.5px; color: var(--muted); line-height: 1.6;
  border-top: 1px solid var(--line); padding-top: 10px; }
#coords { font-variant-numeric: tabular-nums; }

/* Leaflet chrome */
.leaflet-container { background: var(--surface); font: inherit; }
.leaflet-popup-content-wrapper {
  background: var(--card); color: var(--ink); border-radius: 12px;
  box-shadow: 0 10px 32px rgba(0,0,0,0.25); border: 1px solid var(--ring);
}
.leaflet-popup-tip { background: var(--card); }
.leaflet-popup-content { margin: 12px 14px; }
.leaflet-bar a { background: var(--card); color: var(--ink); border-bottom-color: var(--line); }
.pop { width: 232px; font-size: 12px; }
.pop .t { font-weight: 650; font-size: 12.5px; word-break: break-all; margin-bottom: 6px; }
.pop .meta { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px;
  color: var(--ink-2); margin-bottom: 8px; }
.pop .meta span:nth-child(odd) { color: var(--muted); }
.pop .preview { display: block; color: var(--ink-2); text-decoration: none; }
.pop img { width: 100%; height: 140px; object-fit: cover; border-radius: 8px;
  border: 1px solid var(--ring); display: block; }
.pop .image-missing {
  display: none; align-items: center; justify-content: center; min-height: 74px;
  border: 1px dashed var(--ring); border-radius: 8px; color: var(--muted);
  padding: 10px; text-align: center;
}
.pop .preview.missing img { display: none; }
.pop .preview.missing .image-missing { display: flex; }
.pop .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: 8px; }
.pop .action {
  width: 100%; font: inherit; font-size: 11.5px; padding: 6px 0; text-align: center;
  background: transparent; color: var(--accent); border: 1px solid var(--ring);
  border-radius: 7px; cursor: pointer; text-decoration: none;
}
.pop .action:hover { border-color: var(--accent); }
.cluster {
  background: var(--card); border: 2px solid var(--accent); color: var(--ink);
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: 11.5px; font-weight: 650; box-shadow: 0 2px 10px rgba(0,0,0,0.18);
}
@media (max-width: 760px) {
  body { flex-direction: column; }
  #sidebar { width: 100%; min-width: 0; height: 45%; order: 2; }
  #map { height: 55%; }
}
</style>
</head>
<body>
<aside id="sidebar">
  <div>
    <h1>MTSD Atlas</h1>
    <div class="subtitle">Maltese Traffic Sign Dataset &mdash; geolocated captures.
      Collection groups are navigation layers only.</div>
  </div>

  <div class="tiles">
    <div class="tile"><div class="v" id="stat-visible">0</div><div class="k">Markers shown</div></div>
    <div class="tile"><div class="v">__COVERAGE__%</div><div class="k">GPS coverage</div></div>
    <div class="tile"><div class="v">__GEOLOCATED__</div><div class="k">Geolocated</div></div>
    <div class="tile"><div class="v">__READABLE__</div><div class="k">Images in corpus</div></div>
  </div>

  <div>
    <div class="section-label">Display</div>
    <div class="seg" id="mode-seg">
      <button data-mode="markers" class="on">Markers</button>
      <button data-mode="both">Both</button>
      <button data-mode="heat">Density</button>
    </div>
    <div class="rowline" style="margin-top:8px">
      <div class="seg" id="cluster-seg">
        <button data-cluster="1" class="on">Clustered</button>
        <button data-cluster="0">Raw points</button>
      </div>
      <div class="seg" id="theme-seg" style="max-width:110px">
        <button data-theme="light" class="on">Light</button>
        <button data-theme="dark">Dark</button>
      </div>
    </div>
  </div>

  <div>
    <div class="section-label">Search</div>
    <input id="search" type="search" placeholder="Filter by filename&hellip; (Enter zooms)"/>
    <div class="search-hint" id="search-hint"></div>
  </div>

  <div>
    <div class="section-label">Collection groups</div>
    <div class="group-actions">
      <button id="grp-all">Show all</button>
      <button id="grp-none">Hide all</button>
      <button id="grp-fit">Fit view</button>
    </div>
    <div id="groups"></div>
  </div>

  <footer>
    <div id="coords">&mdash;</div>
    <div>Generated __GENERATED__ &middot; single-hue density ramp &middot;
      click a group to toggle, <em>solo</em> to isolate</div>
  </footer>
</aside>
<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<script>
const DATA = __DATA__;

const state = {
  mode: "markers",            // markers | both | heat
  clustered: true,
  theme: "light",
  visible: DATA.groups.map(() => true),
  query: "",
};

// ── map & tiles ─────────────────────────────────────────────────────────────
const map = L.map("map", {
  center: DATA.center, zoom: DATA.zoom, zoomControl: true,
  preferCanvas: true, worldCopyJump: false,
});
const tiles = {
  light: L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    { attribution: "&copy; OpenStreetMap contributors &copy; CARTO", maxZoom: 20 }),
  dark: L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    { attribution: "&copy; OpenStreetMap contributors &copy; CARTO", maxZoom: 20 }),
};
tiles.light.addTo(map);
map.createPane("heatPane");
map.getPane("heatPane").style.zIndex = 350;
map.getPane("heatPane").style.pointerEvents = "none";
const canvasRenderer = L.canvas({ padding: 0.4 });

// ── layers ──────────────────────────────────────────────────────────────────
function groupColor(gi) { return DATA.groups[gi][state.theme]; }

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
  }[ch]));
}

function popupHtml(pt) {
  const [gi, lat, lon, name, url, taken, camera] = pt;
  const g = DATA.groups[gi];
  const safeName = escapeHtml(name);
  const safeUrl = escapeHtml(url);
  const copyUrl = url.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
  return `<div class="pop">
    <div class="t">${safeName}</div>
    <div class="meta">
      <span>Group</span><span>${escapeHtml(g.name)}</span>
      <span>Position</span><span>${lat.toFixed(5)}, ${lon.toFixed(5)}</span>
      ${taken ? `<span>Captured</span><span>${escapeHtml(taken)}</span>` : ""}
      ${camera ? `<span>Camera</span><span>${escapeHtml(camera)}</span>` : ""}
    </div>
    <a class="preview" href="${safeUrl}" target="_blank" rel="noopener">
      <img src="${safeUrl}" loading="lazy" alt="${safeName}"
           onerror="this.closest('.preview').classList.add('missing')"/>
      <span class="image-missing">Image not found<br>${safeUrl}</span>
    </a>
    <div class="actions">
      <a class="action" href="${safeUrl}" target="_blank" rel="noopener">Open image</a>
      <button class="action" onclick="copyText('${copyUrl}', this)">Copy path</button>
    </div>
  </div>`;
}

function copyText(text, btn) {
  const done = () => { btn.textContent = "Copied"; setTimeout(() => btn.textContent = "Copy image path", 1100); };
  if (navigator.clipboard && navigator.clipboard.writeText)
    navigator.clipboard.writeText(text).then(done).catch(() => {});
  else done();
}

function makeMarker(pt) {
  const marker = L.circleMarker([pt[1], pt[2]], {
    renderer: canvasRenderer, radius: 6,
    fillColor: groupColor(pt[0]), fillOpacity: 0.88,
    color: state.theme === "dark" ? "#1a1a19" : "#ffffff", weight: 2,
  });
  marker.bindTooltip(`${pt[3]}<br><b>${DATA.groups[pt[0]].name}</b>`, { direction: "top", offset: [0, -6] });
  marker.bindPopup(() => popupHtml(pt), { maxWidth: 260 });
  return marker;
}

const clusterLayer = L.markerClusterGroup({
  chunkedLoading: true, maxClusterRadius: 46, showCoverageOnHover: false,
  spiderfyOnMaxZoom: true,
  iconCreateFunction: (cluster) => L.divIcon({
    html: `<div class="cluster" style="width:100%;height:100%">${cluster.getChildCount()}</div>`,
    className: "", iconSize: [34, 34],
  }),
});
const flatLayer = L.layerGroup();
let heatLayer = null;

function heatGradient() {
  return state.theme === "dark"
    ? { 0.15: "#104281", 0.45: "#256abf", 0.75: "#5598e7", 1.0: "#cde2fb" }
    : { 0.15: "#cde2fb", 0.45: "#86b6ef", 0.75: "#2a78d6", 1.0: "#0d366b" };
}

function activePoints() {
  const q = state.query;
  return DATA.points.filter((pt) =>
    state.visible[pt[0]] && (!q || pt[3].toLowerCase().includes(q)));
}

function rebuildLayers() {
  const pts = activePoints();

  clusterLayer.clearLayers();
  flatLayer.clearLayers();
  if (heatLayer) { map.removeLayer(heatLayer); heatLayer = null; }

  const wantMarkers = state.mode !== "heat";
  const wantHeat = state.mode !== "markers";

  if (wantMarkers) {
    const markers = pts.map(makeMarker);
    if (state.clustered) {
      clusterLayer.addLayers(markers);
      if (!map.hasLayer(clusterLayer)) map.addLayer(clusterLayer);
      if (map.hasLayer(flatLayer)) map.removeLayer(flatLayer);
    } else {
      markers.forEach((m) => flatLayer.addLayer(m));
      if (!map.hasLayer(flatLayer)) map.addLayer(flatLayer);
      if (map.hasLayer(clusterLayer)) map.removeLayer(clusterLayer);
    }
  } else {
    if (map.hasLayer(clusterLayer)) map.removeLayer(clusterLayer);
    if (map.hasLayer(flatLayer)) map.removeLayer(flatLayer);
  }

  if (wantHeat && pts.length) {
    heatLayer = L.heatLayer(pts.map((pt) => [pt[1], pt[2], 1]), {
      pane: "heatPane", interactive: false,
      radius: 16, blur: 18, minOpacity: 0.28, gradient: heatGradient(),
    }).addTo(map);
  }

  document.getElementById("stat-visible").textContent = pts.length.toLocaleString();
  const hint = document.getElementById("search-hint");
  hint.textContent = state.query ? `${pts.length.toLocaleString()} matching capture(s)` : "";
}

function fitToActive() {
  const pts = activePoints();
  if (!pts.length) return;
  const bounds = L.latLngBounds(pts.map((pt) => [pt[1], pt[2]]));
  map.fitBounds(bounds.pad(0.06));
}

// ── sidebar: groups ─────────────────────────────────────────────────────────
const groupsBox = document.getElementById("groups");
DATA.groups.forEach((g, gi) => {
  const row = document.createElement("div");
  row.className = "grp";
  row.innerHTML = `<span class="swatch"></span><span class="name">${g.name}</span>
    <span class="count">${g.count.toLocaleString()}</span>
    <button class="solo" title="Show only this group">solo</button>`;
  row.addEventListener("click", () => { state.visible[gi] = !state.visible[gi]; syncGroups(); rebuildLayers(); });
  row.querySelector(".solo").addEventListener("click", (ev) => {
    ev.stopPropagation();
    state.visible = state.visible.map((_, i) => i === gi);
    syncGroups(); rebuildLayers(); fitToActive();
  });
  groupsBox.appendChild(row);
});

function syncGroups() {
  [...groupsBox.children].forEach((row, gi) => {
    row.classList.toggle("off", !state.visible[gi]);
    row.querySelector(".swatch").style.background = groupColor(gi);
  });
}

document.getElementById("grp-all").addEventListener("click", () => {
  state.visible = state.visible.map(() => true); syncGroups(); rebuildLayers();
});
document.getElementById("grp-none").addEventListener("click", () => {
  state.visible = state.visible.map(() => false); syncGroups(); rebuildLayers();
});
document.getElementById("grp-fit").addEventListener("click", fitToActive);

// ── sidebar: segmented controls ─────────────────────────────────────────────
function wireSeg(id, attr, onPick) {
  const seg = document.getElementById(id);
  seg.querySelectorAll("button").forEach((btn) => {
    btn.addEventListener("click", () => {
      seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === btn));
      onPick(btn.dataset[attr]);
    });
  });
}
wireSeg("mode-seg", "mode", (mode) => { state.mode = mode; rebuildLayers(); });
wireSeg("cluster-seg", "cluster", (val) => { state.clustered = val === "1"; rebuildLayers(); });
wireSeg("theme-seg", "theme", (theme) => {
  state.theme = theme;
  document.documentElement.classList.toggle("dark", theme === "dark");
  map.removeLayer(theme === "dark" ? tiles.light : tiles.dark);
  map.addLayer(theme === "dark" ? tiles.dark : tiles.light);
  syncGroups(); rebuildLayers();
});

// ── sidebar: search ─────────────────────────────────────────────────────────
const searchInput = document.getElementById("search");
let searchTimer = null;
searchInput.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.query = searchInput.value.trim().toLowerCase();
    rebuildLayers();
  }, 160);
});
searchInput.addEventListener("keydown", (ev) => { if (ev.key === "Enter") fitToActive(); });

// ── footer coordinates ──────────────────────────────────────────────────────
const coords = document.getElementById("coords");
map.on("mousemove", (ev) => {
  coords.textContent = `${ev.latlng.lat.toFixed(5)}° N, ${ev.latlng.lng.toFixed(5)}° E · z${map.getZoom()}`;
});
map.on("zoomend", () => {
  coords.textContent = `zoom ${map.getZoom()}`;
});

// ── boot ────────────────────────────────────────────────────────────────────
syncGroups();
rebuildLayers();
fitToActive();
</script>
</body>
</html>
"""


def render_html(payload: dict, stats: dict, title: str) -> str:
    data_js = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (
        HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__DATA__", data_js)
        .replace("__COVERAGE__", f"{stats['coverage_pct']:g}")
        .replace("__GEOLOCATED__", f"{stats['geolocated']:,}")
        .replace("__READABLE__", f"{stats['readable_images']:,}")
        .replace("__GENERATED__", stats["generated"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output HTML file (default: Documents/MTSD-EDA/MTSD_mapped.html)")
    parser.add_argument("--rescan", action="store_true",
                        help="Force a fresh image scan instead of using the cached inventory.")
    parser.add_argument("--title", default="MTSD Atlas - Capture Locations",
                        help="Page title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()

    points, stats = load_points(rescan=args.rescan)
    print(f"Inventory: {stats['readable_images']:,} readable images, "
          f"{stats['geolocated']:,} geolocated ({stats['coverage_pct']}%)")

    payload = build_payload(points, output)
    print(f"Groups   : {', '.join(g['name'] for g in payload['groups'])}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(payload, stats, args.title), encoding="utf-8")
    size_kb = output.stat().st_size / 1024
    print(f"Atlas    : {output} ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    main()
