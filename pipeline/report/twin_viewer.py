"""Interactive 3D digital twin for the report (three.js r147, classic build, offline).

Reads `twin.json` (embedded in the page) and replays it in sync with the camera video:
smoothed worker paths interpolated to the display rate, zones as volumes, the open edge
as a hazard barrier, the CCTV camera and its coverage. Orbit / pan / zoom, preset views,
hover details, play / scrub / speed. Classic script (not an ES module) because browsers
block module imports from file:// pages.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor"
THREE_ASSET = "assets/three.min.js"


def install_assets(run_dir: Path) -> None:
    """Copy three.js next to the report (MIT licence included)."""
    target = run_dir / "assets"
    target.mkdir(exist_ok=True)
    shutil.copyfile(VENDOR / "three.min.js", target / "three.min.js")
    shutil.copyfile(VENDOR / "THREE_LICENSE.txt", target / "THREE_LICENSE.txt")


def embed_meta(meta: dict) -> str:
    text = json.dumps(meta, separators=(",", ":")).replace("<", "\\u003c")
    return f'<script type="application/json" id="twin-meta">{text}</script>'


def embed(data: dict) -> str:
    """The twin as an inert JSON block; '<' escaped so no data can close the tag."""
    text = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    return f'<script type="application/json" id="twin-data">{text}</script>'


STYLE = """
.twin-layout { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:16px; align-items:start; }
.twin-stage { position:relative; overflow:hidden; }
#twin-canvas { position:relative; width:100%; height:min(68vh, 680px); min-height:420px; cursor:grab;
  background:radial-gradient(120% 90% at 50% 0%, var(--twin-sky-top), var(--twin-sky-bottom)); }
#twin-canvas:active { cursor:grabbing; }
#twin-canvas canvas { display:block; }
.twin-hud { position:absolute; left:14px; top:14px; display:flex; gap:8px; flex-wrap:wrap; pointer-events:none; }
.twin-hud .pill { pointer-events:auto; background:color-mix(in srgb, var(--surface) 88%, transparent);
  backdrop-filter:blur(8px); border:1px solid var(--line); border-radius:999px; padding:4px 10px; font-size:.78rem;
  color:var(--text-2); display:inline-flex; gap:6px; align-items:center; }
.twin-hud .pill b { color:var(--text); font-variant-numeric:tabular-nums; }
.twin-views { position:absolute; right:14px; top:14px; display:flex; gap:6px; }
.twin-views button, .twin-toggles button { background:color-mix(in srgb, var(--surface) 88%, transparent);
  backdrop-filter:blur(8px); border:1px solid var(--line); color:var(--text-2); border-radius:8px; padding:5px 10px;
  font-size:.78rem; cursor:pointer; }
.twin-views button[aria-pressed="true"], .twin-toggles button[aria-pressed="true"] { color:var(--accent);
  border-color:var(--accent); background:var(--accent-soft); }
.twin-bar { display:flex; flex-wrap:wrap; gap:10px 12px; align-items:center; padding:14px 16px 12px; border-top:1px solid var(--line); }
.twin-bar .twin-scrubwrap { flex:1 1 380px; min-width:240px; }
.twin-bar .twin-toggles { margin-left:auto; flex-wrap:wrap; }
.twin-bar .time { font-variant-numeric:tabular-nums; color:var(--text-2); font-size:.84rem; min-width:92px; }
.twin-bar input[type=range] { width:100%; accent-color:var(--accent); }
.twin-play { width:36px; height:36px; border-radius:50%; border:none; background:var(--accent); color:#fff; cursor:pointer;
  display:grid; place-items:center; font-size:14px; }
.twin-toggles { display:flex; gap:6px; }
.twin-legend { display:flex; gap:14px; flex-wrap:wrap; padding:0 16px 14px; color:var(--text-3); font-size:.78rem; }
.twin-legend span { display:inline-flex; align-items:center; gap:6px; }
.twin-legend i { display:inline-block; width:10px; height:10px; border-radius:50%; }
.twin-tip { position:absolute; pointer-events:none; display:none; background:var(--surface); border:1px solid var(--line);
  border-radius:10px; padding:8px 10px; font-size:.8rem; box-shadow:0 8px 28px rgba(0,0,0,.25); min-width:150px; z-index:5; }
.twin-tip b { display:block; margin-bottom:2px; }
.twin-side video { width:100%; border-radius:10px; display:block; background:#000; aspect-ratio:16/9; }
.twin-live { list-style:none; margin:0; padding:0; }
.twin-live li { display:flex; align-items:center; gap:10px; padding:8px 0; border-bottom:1px solid var(--line); font-size:.85rem; }
.twin-live li:last-child { border-bottom:none; }
.twin-live .dot { width:9px; height:9px; border-radius:50%; flex:none; }
.twin-empty { color:var(--text-3); font-size:.85rem; padding:6px 0; }
.twin-scrubwrap { position:relative; }
#twin-marks { position:absolute; left:0; right:0; top:-14px; height:10px; }
#twin-marks .mark { position:absolute; width:4px; height:10px; border-radius:2px; border:none; padding:0; cursor:pointer; transform:translateX(-2px); opacity:.9; }
#twin-marks .mark:hover { opacity:1; transform:translateX(-2px) scaleY(1.4); }
.sel-head { display:flex; align-items:center; gap:8px; margin-bottom:10px; }
.sel-head b { font-size:.95rem; }
.sel-head .btn { margin-left:auto; }
.sel-dot { width:10px; height:10px; border-radius:50%; background:#3987e5; flex:none; }
.sel-facts { display:grid; grid-template-columns:auto 1fr; gap:4px 12px; margin:0 0 10px; font-size:.84rem; }
.sel-facts dt { color:var(--text-3); } .sel-facts dd { margin:0; font-variant-numeric:tabular-nums; }
.sel-inc { display:flex; align-items:center; gap:8px; width:100%; text-align:left; background:var(--surface-2); border:1px solid var(--line);
  border-radius:8px; padding:6px 10px; margin:4px 0; cursor:pointer; color:var(--text); font-size:.82rem; }
.sel-inc:hover { border-color:var(--accent); }
.sel-inc .dot { width:8px; height:8px; border-radius:50%; flex:none; } .sel-inc .muted { margin-left:auto; }
.sel-actions { display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; }
.sel-actions .btn[aria-pressed="true"] { color:var(--accent); border-color:var(--accent); }
.twin-keys { color:var(--text-3); font-size:.76rem; padding:0 16px 12px; }
.twin-keys kbd { font:inherit; font-size:.72rem; border:1px solid var(--line); border-bottom-width:2px; border-radius:5px; padding:0 5px; background:var(--surface-2); }
:root { --twin-sky-top:#eef2f7; --twin-sky-bottom:#dfe5ec; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { --twin-sky-top:#1c232d; --twin-sky-bottom:#0f1217; } }
:root[data-theme="dark"] { --twin-sky-top:#1c232d; --twin-sky-bottom:#0f1217; }
@media (max-width:1100px) { .twin-layout { grid-template-columns:1fr; } }
@media (max-width:860px) { .twin-bar .speed { display:none; } }
"""

SCRIPT = r"""
(() => {
const dataEl = document.getElementById('twin-data');
const host = document.getElementById('twin-canvas');
if (!dataEl || !host || typeof THREE === 'undefined') return;
const data = JSON.parse(dataEl.textContent);
const metaEl = document.getElementById('twin-meta');
const meta = metaEl ? JSON.parse(metaEl.textContent) : { incidents: [], areas: {} };
const esc = v => String(v).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
// Hex colours are sRGB: let three.js convert them (r147 defaults to legacy, washed-out colour).
if (THREE.ColorManagement) THREE.ColorManagement.legacyMode = false;
const video = document.getElementById('twin-video');
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---- renderer, scene, camera
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 400);
// Twin coordinates: x right of the CCTV camera, y away from it (worker heights, WH).
// three.js: X = x, Z = -y, Y = up.
const P = (x, y, h = 0) => new THREE.Vector3(x, h, -y);

const support = (data.camera && data.camera.support && data.camera.support.length) ? data.camera.support : [[-6, 2], [6, 2], [8, 22], [-8, 22]];
const xs = support.map(p => p[0]), ys = support.map(p => p[1]);
const lo = [Math.min(...xs), Math.min(...ys)], hi = [Math.max(...xs), Math.max(...ys)];
const centre = P((lo[0] + hi[0]) / 2, lo[1] + 0.45 * (hi[1] - lo[1]));
const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1]) + 4;

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x1a1e24, 0.55));
const sun = new THREE.DirectionalLight(0xfff4e6, 0.9);
sun.position.copy(centre).add(new THREE.Vector3(-extent * 0.4, extent * 0.9, extent * 0.3));
sun.target.position.copy(centre);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
const s = extent * 0.75;
Object.assign(sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s, near: 1, far: extent * 4 });
scene.add(sun, sun.target);

// ---- ground, grid, coverage
const groundMat = new THREE.MeshStandardMaterial({ color: 0x1f242b, roughness: 1, metalness: 0 });
const ground = new THREE.Mesh(new THREE.PlaneGeometry(extent * 14, extent * 14), groundMat);
ground.rotation.x = -Math.PI / 2; ground.position.copy(centre); ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.GridHelper(Math.ceil(extent * 3), Math.ceil(extent * 3), 0x55606d, 0x3a424c);
grid.position.copy(centre); grid.position.y = 0.002;
grid.material.transparent = true; grid.material.opacity = 0.35;
scene.add(grid);
function applyTheme() {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  const sky = new THREE.Color(dark ? 0x0f1217 : 0xdfe5ec);
  scene.fog = new THREE.Fog(sky, extent * 1.6, extent * 4.5);
  groundMat.color.setHex(dark ? 0x1d2228 : 0xc9d0d8);
  grid.material.color && grid.material.color.setHex(dark ? 0x3a434e : 0x9aa5b2);
}
applyTheme();
const themeButton = document.getElementById('theme');
if (themeButton) themeButton.addEventListener('click', () => setTimeout(applyTheme, 0));
function shapeOf(points) {
  const shape = new THREE.Shape();
  points.forEach(([x, y], i) => i ? shape.lineTo(x, y) : shape.moveTo(x, y));
  shape.closePath(); return shape;
}
// ShapeGeometry lies in XY; rotate so shape-y maps to -Z (twin y away from camera).
function flat(points, material, height = 0.004) {
  const mesh = new THREE.Mesh(new THREE.ShapeGeometry(shapeOf(points)), material);
  mesh.rotation.x = -Math.PI / 2; mesh.position.y = height; return mesh;
}
const coverage = flat(support, new THREE.MeshBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.08, depthWrite: false }), 0.003);
const coverageEdge = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(support.map(([x, y]) => P(x, y, 0.01))),
  new THREE.LineDashedMaterial({ color: 0x5b9cf0, dashSize: 0.35, gapSize: 0.25, transparent: true, opacity: 0.8 }));
coverageEdge.computeLineDistances();
scene.add(coverage, coverageEdge);

// ---- CCTV camera marker
const camHeight = (data.camera && data.camera.height_wh) || 2;
const cam = new THREE.Group();
const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.05, camHeight, 12), new THREE.MeshStandardMaterial({ color: 0x9aa4b1 }));
pole.position.y = camHeight / 2; pole.castShadow = true;
const body = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.16, 0.42), new THREE.MeshStandardMaterial({ color: 0xe8edf3, roughness: 0.4 }));
body.position.set(0, camHeight, -0.1);
const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.06, 16), new THREE.MeshStandardMaterial({ color: 0x1b2430 }));
lens.rotation.x = Math.PI / 2; lens.position.set(0, camHeight, -0.34);
cam.add(pole, body, lens); scene.add(cam);

// ---- labels (sprites from canvas text)
function label(text, color = '#ffffff', bg = 'rgba(17,22,29,0.88)', scale = 0.9) {
  const c = document.createElement('canvas'); const g = c.getContext('2d');
  const font = '600 40px Inter, "Segoe UI", system-ui, sans-serif'; g.font = font;
  const w = Math.ceil(g.measureText(text).width) + 36; c.width = w; c.height = 64;
  g.font = font; g.fillStyle = bg; const r = 28;
  g.beginPath(); g.moveTo(r, 4); g.arcTo(w - 4, 4, w - 4, 60, r); g.arcTo(w - 4, 60, 4, 60, r); g.arcTo(4, 60, 4, 4, r); g.arcTo(4, 4, w - 4, 4, r); g.fill();
  g.fillStyle = color; g.textBaseline = 'middle'; g.fillText(text, 18, 34);
  const tex = new THREE.CanvasTexture(c); tex.encoding = THREE.sRGBEncoding; tex.anisotropy = 4;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true, sizeAttenuation: false }));
  const hgt = 0.038 * scale; sprite.scale.set(hgt * w / 64, hgt, 1); sprite.renderOrder = 10; return sprite;
}
const camLabel = label('CCTV', '#ffffff', 'rgba(42,120,214,0.95)'); camLabel.position.set(0, camHeight + 0.45, 0); scene.add(camLabel);
const fixedTags = [camLabel];
// Labels never overlap: each frame, worker tags are placed first (nearest the camera first),
// then zone and camera tags; a tag that would cover one already placed is lifted above it.
// The lift moves the sprite's anchor (center.y), eased so tags glide instead of jumping.
const tagPoint = new THREE.Vector3();
function declutter() {
  const W = host.clientWidth, H = host.clientHeight, e = camera.projectionMatrix.elements, placed = [];
  const shown = workers.filter(w => w.tag && w.tag.visible).map(w => w.tag);
  const boxes = [...shown, ...fixedTags.filter(t => t.visible)].map(tag => {
    tagPoint.setFromMatrixPosition(tag.matrixWorld).project(camera);
    return { tag, x: (tagPoint.x + 1) / 2 * W, y: (1 - tagPoint.y) / 2 * H, w: tag.scale.x * e[0] * W / 2,
      h: tag.scale.y * e[5] * H / 2, fixed: !shown.includes(tag), behind: tagPoint.z > 1 };
  }).filter(b => !b.behind);
  boxes.sort((a, b) => (a.fixed - b.fixed) || (b.y - a.y));
  for (const b of boxes) {
    let lift = 0;
    for (let pass = 0; pass < 8; pass++) {
      const x0 = b.x - b.w / 2, x1 = b.x + b.w / 2, y0 = b.y - b.h / 2 - lift, y1 = b.y + b.h / 2 - lift;
      const hit = placed.find(r => x0 < r.x1 + 3 && x1 > r.x0 - 3 && y0 < r.y1 + 3 && y1 > r.y0 - 3);
      if (!hit) break;
      lift = b.y + b.h / 2 - hit.y0 + 4;
    }
    placed.push({ x0: b.x - b.w / 2, x1: b.x + b.w / 2, y0: b.y - b.h / 2 - lift, y1: b.y + b.h / 2 - lift });
    const target = 0.5 - lift / Math.max(b.h, 1);
    b.tag.center.y += (target - b.tag.center.y) * (b.tag.userData.placed ? 0.25 : 1);
    b.tag.userData.placed = true;
  }
}

// ---- zones and edges
function stripes() {
  const c = document.createElement('canvas'); c.width = 64; c.height = 16; const g = c.getContext('2d');
  g.fillStyle = '#f5c518'; g.fillRect(0, 0, 64, 16); g.fillStyle = '#111';
  for (let i = -16; i < 80; i += 16) { g.beginPath(); g.moveTo(i, 16); g.lineTo(i + 8, 16); g.lineTo(i + 16, 0); g.lineTo(i + 8, 0); g.fill(); }
  const t = new THREE.CanvasTexture(c); t.wrapS = THREE.RepeatWrapping; t.encoding = THREE.sRGBEncoding; return t;
}
const zoneGroup = new THREE.Group(); scene.add(zoneGroup);
const zoneMeshes = [];
(data.zones || []).forEach(zone => {
  const pts = zone.points;
  if (zone.line) {
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = P(...pts[i]), b = P(...pts[i + 1]); const len = a.distanceTo(b);
      const tex = stripes(); tex.repeat.set(Math.max(1, len * 2), 1);
      const bar = new THREE.Mesh(new THREE.BoxGeometry(len, 0.12, 0.05), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.6 }));
      bar.position.copy(a.clone().lerp(b, 0.5)).setY(0.35); bar.lookAt(b.clone().setY(0.35)); bar.rotateY(Math.PI / 2); bar.castShadow = true;
      zoneGroup.add(bar);
      [a, b].forEach(p => { const post = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.42, 8), new THREE.MeshStandardMaterial({ color: 0xf5c518 })); post.position.copy(p).setY(0.21); zoneGroup.add(post); });
      const drop = new THREE.Mesh(new THREE.PlaneGeometry(len, 0.9), new THREE.MeshBasicMaterial({ color: 0xec835a, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
      drop.position.copy(a.clone().lerp(b, 0.5)).setY(-0.45); drop.lookAt(b.clone().setY(-0.45)); drop.rotateY(Math.PI / 2); zoneGroup.add(drop);
    }
    const tag = label('⚠ ' + zone.label, '#111', 'rgba(245,197,24,0.95)', 0.8); const m = pts[Math.floor(pts.length / 2)];
    tag.position.copy(P(m[0], m[1], 0.9)); zoneGroup.add(tag); fixedTags.push(tag);
    return;
  }
  const red = zone.type === 'restricted_zone' ? 0xd03b3b : zone.type === 'traffic_zone' ? 0xfab219 : 0x38c6e8;
  zoneGroup.add(flat(pts, new THREE.MeshBasicMaterial({ color: red, transparent: true, opacity: 0.22, depthWrite: false }), 0.006));
  const vol = new THREE.Mesh(new THREE.ExtrudeGeometry(shapeOf(pts), { depth: 1.0, bevelEnabled: false }),
    new THREE.MeshStandardMaterial({ color: red, transparent: true, opacity: 0.12, depthWrite: false, side: THREE.DoubleSide }));
  vol.rotation.x = -Math.PI / 2; vol.userData.zone = zone; zoneGroup.add(vol); zoneMeshes.push(vol);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(vol.geometry), new THREE.LineBasicMaterial({ color: red, transparent: true, opacity: 0.9 }));
  edges.rotation.x = -Math.PI / 2; zoneGroup.add(edges);
  const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length, cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
  const tag = label('⛔ ' + zone.label, '#fff', 'rgba(208,59,59,0.95)', 0.8); tag.position.copy(P(cx, cy, 1.35)); zoneGroup.add(tag); fixedTags.push(tag);
});

// ---- workers
const MAT = {
  0: new THREE.MeshStandardMaterial({ color: 0x6b819e, roughness: 0.55 }),
  1: new THREE.MeshStandardMaterial({ color: 0xfab219, roughness: 0.5, emissive: 0x3a2800 }),
  2: new THREE.MeshStandardMaterial({ color: 0xe0454a, roughness: 0.45, emissive: 0x3a0606 }),
};
const SKIN = new THREE.MeshStandardMaterial({ color: 0xd2a07c, roughness: 0.7 });
const HAT = new THREE.MeshStandardMaterial({ color: 0xf6c343, roughness: 0.35 });
const LEGS = new THREE.MeshStandardMaterial({ color: 0x39424e, roughness: 0.8 });
const torsoGeo = new THREE.CapsuleGeometry(0.13, 0.26, 4, 12);
const legGeo = new THREE.CapsuleGeometry(0.055, 0.32, 3, 8);
const headGeo = new THREE.SphereGeometry(0.095, 16, 12);
const hatGeo = new THREE.SphereGeometry(0.108, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2);
const ringGeo = new THREE.RingGeometry(0.34, 0.42, 40);
const shadowGeo = new THREE.CircleGeometry(0.2, 24);
const shadowMat = new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.28, depthWrite: false });
const RULES = data.rules || {};
const ruleNames = mask => Object.entries(RULES).filter(([, bit]) => mask & bit).map(([r]) => r);

const workers = (data.workers || []).map(w => {
  const g = new THREE.Group();
  const legs = [-0.06, 0.06].map(dx => { const l = new THREE.Mesh(legGeo, LEGS); l.position.set(dx, 0.22, 0); l.castShadow = true; return l; });
  const torso = new THREE.Mesh(torsoGeo, MAT[0]); torso.position.y = 0.6; torso.castShadow = true;
  const head = new THREE.Mesh(headGeo, SKIN); head.position.y = 0.9; head.castShadow = true;
  const hat = new THREE.Mesh(hatGeo, HAT); hat.position.y = 0.915; hat.visible = false;
  const ring = new THREE.Mesh(ringGeo, new THREE.MeshBasicMaterial({ color: 0xe0454a, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.01; ring.visible = false;
  const blob = new THREE.Mesh(shadowGeo, shadowMat); blob.rotation.x = -Math.PI / 2; blob.position.y = 0.005;
  g.add(...legs, torso, head, hat, ring, blob); g.visible = false;
  g.userData = { id: w.id, label: w.label };
  scene.add(g);
  const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x5b9cf0, transparent: true, opacity: 0.55 }));
  trail.visible = false; scene.add(trail);
  return { w, g, torso, hat, ring, trail, tag: null, tagText: '' };
});

function sampleAt(w, t) {
  for (const seg of w.segments) {
    const ts = seg.t; const n = ts.length;
    if (t < ts[0] || t > ts[n - 1]) continue;
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ts[mid] <= t) lo = mid; else hi = mid; }
    const span = Math.max(ts[hi] - ts[lo], 1e-6); const f = Math.min(1, Math.max(0, (t - ts[lo]) / span));
    const x = seg.xy[2 * lo] + (seg.xy[2 * hi] - seg.xy[2 * lo]) * f;
    const y = seg.xy[2 * lo + 1] + (seg.xy[2 * hi + 1] - seg.xy[2 * lo + 1]) * f;
    return { x, y, state: +seg.state[lo], rules: seg.rules[lo], helmet: seg.helmet[lo], ok: seg.ok[lo] === '1', seg, i: lo };
  }
  return null;
}

// ---- machines
const machines = (data.machines || []).map(m => {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0xf5a623, roughness: 0.5 }));
  mesh.castShadow = true; mesh.visible = false; scene.add(mesh); return { m, mesh };
});

// ---- orbit controls (drag rotate, right/shift-drag pan, wheel zoom)
const orbit = { target: centre.clone(), radius: extent * 1.1, theta: 0, phi: 1.0 };
function applyOrbit() {
  orbit.phi = Math.min(1.52, Math.max(0.12, orbit.phi));
  orbit.radius = Math.min(extent * 4, Math.max(2, orbit.radius));
  const sinP = Math.sin(orbit.phi);
  camera.position.set(orbit.target.x + orbit.radius * sinP * Math.sin(orbit.theta),
    orbit.target.y + orbit.radius * Math.cos(orbit.phi), orbit.target.z + orbit.radius * sinP * Math.cos(orbit.theta));
  camera.lookAt(orbit.target);
}
const views = {
  cctv: () => Object.assign(orbit, { target: centre.clone(), theta: 0, phi: 1.12, radius: extent * 0.72 }),
  iso: () => Object.assign(orbit, { target: centre.clone(), theta: -0.75, phi: 0.92, radius: extent * 0.85 }),
  top: () => Object.assign(orbit, { target: centre.clone(), theta: 0, phi: 0.12, radius: extent * 1.05 }),
};
let tween = null;
function goto(name) {
  const from = { ...orbit, target: orbit.target.clone() }; views[name](); const to = { ...orbit, target: orbit.target.clone() };
  Object.assign(orbit, from); tween = { from, to, start: performance.now() };
  document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', b.dataset.view === name));
}
let drag = null;
host.addEventListener('contextmenu', e => e.preventDefault());
let down = null;
host.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; down = { x: e.clientX, y: e.clientY, left: e.button === 0 }; host.setPointerCapture(e.pointerId); tween = null; });
host.addEventListener('pointerup', e => {
  drag = null;
  if (down && down.left && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5) handleClick(e.offsetX, e.offsetY);
  down = null;
});
host.addEventListener('pointermove', e => {
  pointer.x = e.offsetX; pointer.y = e.offsetY;
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    const right = new THREE.Vector3().subVectors(camera.position, orbit.target).cross(camera.up).normalize();
    const forward = new THREE.Vector3(-Math.sin(orbit.theta), 0, -Math.cos(orbit.theta));
    const k = orbit.radius * 0.0016;
    orbit.target.addScaledVector(right, dx * k).addScaledVector(forward, dy * k);
  } else { orbit.theta -= dx * 0.005; orbit.phi -= dy * 0.005; }
  document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', 'false'));
});
host.addEventListener('wheel', e => { e.preventDefault(); orbit.radius *= Math.exp(e.deltaY * 0.0012); tween = null; }, { passive: false });
document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => goto(b.dataset.view)));

// ---- toggles
const show = { labels: true, trails: true, coverage: true, heat: false, focus: false };
let heatMesh = null;
function buildHeat() {
  // Exposure density from the twin data itself: every sample in a hazard condition.
  const cell = 0.25, pad = 2;
  const x0 = lo[0] - pad, y0 = lo[1] - pad, nx = Math.ceil((hi[0] - lo[0] + 2 * pad) / cell), ny = Math.ceil((hi[1] - lo[1] + 2 * pad) / cell);
  const grid = new Float32Array(nx * ny);
  for (const w of data.workers || []) for (const seg of w.segments) for (let i = 0; i < seg.t.length; i++) {
    if (seg.state[i] === '0') continue;
    const gx = Math.floor((seg.xy[2 * i] - x0) / cell), gy = Math.floor((seg.xy[2 * i + 1] - y0) / cell);
    if (gx >= 0 && gy >= 0 && gx < nx && gy < ny) grid[gy * nx + gx] += 1;
  }
  let peak = 0; for (const v of grid) peak = Math.max(peak, v);
  const c = document.createElement('canvas'); c.width = nx; c.height = ny; const g = c.getContext('2d');
  const img = g.createImageData(nx, ny);
  for (let i = 0; i < grid.length; i++) { const v = peak ? Math.sqrt(grid[i] / peak) : 0; const gy = Math.floor(i / nx), gx = i % nx; const o = ((ny - 1 - gy) * nx + gx) * 4;
    img.data[o] = 236 - 28 * v; img.data[o + 1] = 131 - 72 * v; img.data[o + 2] = 90 - 31 * v; img.data[o + 3] = Math.min(230, 300 * v); }
  g.putImageData(img, 0, 0);
  const blurred = document.createElement('canvas'); blurred.width = nx * 4; blurred.height = ny * 4; const b2 = blurred.getContext('2d');
  b2.filter = 'blur(10px)'; b2.imageSmoothingEnabled = true; b2.drawImage(c, 0, 0, blurred.width, blurred.height);
  const tex = new THREE.CanvasTexture(blurred); tex.encoding = THREE.sRGBEncoding;
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(nx * cell, ny * cell), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
  plane.rotation.x = -Math.PI / 2; plane.position.copy(P(x0 + nx * cell / 2, y0 + ny * cell / 2, 0.012));
  return plane;
}
document.querySelectorAll('[data-toggle]').forEach(b => b.addEventListener('click', () => {
  const k = b.dataset.toggle; show[k] = !show[k]; b.setAttribute('aria-pressed', show[k]);
  if (k === 'coverage') { coverage.visible = coverageEdge.visible = show.coverage; }
  if (k === 'heat') { if (!heatMesh) { heatMesh = buildHeat(); scene.add(heatMesh); } heatMesh.visible = show.heat; }
}));

// ---- clock: follows the camera video; plays on its own when there is none
const duration = data.duration || (video && video.duration) || 1;
let t = 0, playing = false, speed = 1, last = performance.now();
const scrub = document.getElementById('twin-scrub'), timeEl = document.getElementById('twin-time'), playBtn = document.getElementById('twin-play');
const fmt = s => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
function setPlaying(on) {
  playing = on; if (playBtn) playBtn.textContent = on ? '❚❚' : '▶';
  if (video) { video.playbackRate = speed; if (on) video.play().catch(() => {}); else video.pause(); }
}
if (playBtn) playBtn.addEventListener('click', () => setPlaying(!playing));
if (scrub) { scrub.max = duration; scrub.addEventListener('input', () => { t = +scrub.value; if (video) video.currentTime = t; }); }
document.querySelectorAll('[data-speed]').forEach(b => b.addEventListener('click', () => {
  speed = +b.dataset.speed; if (video) video.playbackRate = speed;
  document.querySelectorAll('[data-speed]').forEach(x => x.setAttribute('aria-pressed', x === b));
}));
if (video) { video.addEventListener('play', () => { playing = true; if (playBtn) playBtn.textContent = '❚❚'; });
  video.addEventListener('pause', () => { playing = false; if (playBtn) playBtn.textContent = '▶'; }); }
window.twinSeek = s => { t = s; if (video) video.currentTime = s; };

// ---- selection: click a worker to follow them, a zone to see its exposure
const ray = new THREE.Raycaster(); const pointer = { x: -1, y: -1 }; const tip = document.getElementById('twin-tip');
const panel = document.getElementById('twin-selected');
let selected = null, follow = true, pathLines = [];
const selRing = new THREE.Mesh(new THREE.RingGeometry(0.46, 0.53, 48), new THREE.MeshBasicMaterial({ color: 0x3987e5, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthWrite: false }));
selRing.rotation.x = -Math.PI / 2; selRing.visible = false; scene.add(selRing);
const RULE_TEXT = { R1: 'Missing helmet', R2: 'Missing hi-vis', R3: 'Restricted zone', R4: 'Machinery proximity', R5: 'Open edge', R6: 'Worker down', R7: 'Suspended load', R8: 'Fire or smoke' };
function exposureOf(w) {
  let total = 0; for (const seg of w.segments) for (let i = 1; i < seg.t.length; i++) if (seg.state[i - 1] !== '0') total += seg.t[i] - seg.t[i - 1];
  return total;
}
function clearPath() { pathLines.forEach(l => { scene.remove(l); l.geometry.dispose(); }); pathLines = []; }
function select(item) {
  selected = item; follow = true; clearPath();
  if (!item) { selRing.visible = false; if (panel) panel.innerHTML = '<p class="twin-empty">Click a worker in the twin to follow them, or a zone to see its exposure.</p>'; return; }
  for (const seg of item.w.segments) {
    const pts = []; for (let i = 0; i < seg.t.length; i++) pts.push(P(seg.xy[2 * i], seg.xy[2 * i + 1], 0.015));
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x3987e5, transparent: true, opacity: 0.85 }));
    scene.add(line); pathLines.push(line);
  }
  const first = item.w.segments[0].t[0], lastSeg = item.w.segments[item.w.segments.length - 1], lastT = lastSeg.t[lastSeg.t.length - 1];
  const mine = (meta.incidents || []).filter(i => i.subject === item.w.id);
  if (panel) panel.innerHTML = `<div class="sel-head"><span class="sel-dot"></span><b>Worker ${esc(item.w.label)}</b><button class="btn sm" id="sel-close" aria-label="Clear selection">✕</button></div>
    <dl class="sel-facts"><dt>In view</dt><dd>${fmt(first)} – ${fmt(lastT)}</dd><dt>Hazard exposure</dt><dd>${exposureOf(item.w).toFixed(1)} s</dd>
    <dt>Incidents</dt><dd>${mine.length}</dd></dl>
    ${mine.map(i => `<button class="sel-inc" data-jump="${i.start}"><span class="dot" style="background:${esc(i.color)}"></span>${esc(i.rule)} · ${esc(RULE_TEXT[i.rule] || i.title)}<span class="muted">${fmt(i.start)}</span></button>`).join('')}
    <div class="sel-actions"><button class="btn sm" id="sel-follow" aria-pressed="true">Follow camera</button><button class="btn sm" data-jump="${first}">Jump to first sighting</button></div>`;
  panel.querySelectorAll('[data-jump]').forEach(b => b.addEventListener('click', () => jump(+b.dataset.jump, false)));
  const close = document.getElementById('sel-close'); if (close) close.addEventListener('click', () => select(null));
  const fb = document.getElementById('sel-follow'); if (fb) fb.addEventListener('click', () => { follow = !follow; fb.setAttribute('aria-pressed', follow); });
}
function showZone(zone) {
  selected = null; selRing.visible = false; clearPath();
  const area = (meta.areas || {})[zone.id];
  if (panel) panel.innerHTML = `<div class="sel-head"><span class="sel-dot" style="background:var(--critical)"></span><b>${esc(zone.label)}</b><button class="btn sm" id="sel-close">✕</button></div>
    <dl class="sel-facts"><dt>Type</dt><dd>${esc(zone.type.replace(/_/g, ' '))}</dd><dt>Exposure</dt><dd>${area ? area.exposure.toFixed(0) + ' worker-seconds' : 'none recorded'}</dd><dt>Workers inside</dt><dd>${area ? area.workers : 0}</dd></dl>`;
  const close = document.getElementById('sel-close'); if (close) close.addEventListener('click', () => select(null));
}
function jump(at, andSelect = true, subject = null) {
  t = Math.max(0, at - 1); if (video) video.currentTime = t;
  if (andSelect && subject) { const item = workers.find(w => w.w.id === subject); if (item) select(item); }
}
function handleClick(x, y) {
  ray.setFromCamera({ x: (x / host.clientWidth) * 2 - 1, y: -(y / host.clientHeight) * 2 + 1 }, camera);
  const hits = ray.intersectObjects(workers.filter(w => w.g.visible).map(w => w.g), true);
  if (hits.length) { let o = hits[0].object; while (o.parent && !o.userData.id) o = o.parent; const item = workers.find(w => w.g === o); if (item) { select(item); return; } }
  const zoneHit = ray.intersectObjects(zoneMeshes, false);
  if (zoneHit.length) { showZone(zoneHit[0].object.userData.zone); return; }
  select(null);
}
// incident markers above the scrubber
const marks = document.getElementById('twin-marks');
if (marks) marks.innerHTML = (meta.incidents || []).map(i =>
  `<button class="mark" style="left:${(100 * i.start / duration).toFixed(2)}%;background:${esc(i.color)}" data-at="${i.start}" data-subject="${esc(i.subject)}" title="${esc(i.rule)} · ${esc(i.title)} at ${fmt(i.start)}"></button>`).join('');
if (marks) marks.querySelectorAll('.mark').forEach(b => b.addEventListener('click', () => jump(+b.dataset.at, true, b.dataset.subject)));
// keyboard, only while the twin section is on screen
document.addEventListener('keydown', e => {
  if (!document.getElementById('page-twin').classList.contains('active') || /INPUT|TEXTAREA/.test(document.activeElement.tagName)) return;
  if (e.key === ' ') { e.preventDefault(); setPlaying(!playing); }
  else if (e.key === 'ArrowRight') { jump(t + 3, false); }
  else if (e.key === 'ArrowLeft') { jump(t - 1, false); }
  else if (e.key === '1') goto('cctv'); else if (e.key === '2') goto('iso'); else if (e.key === '3') goto('top');
  else if (e.key === 'Escape') select(null);
  else if (e.key.toLowerCase() === 'f' && selected) { follow = !follow; const fb = document.getElementById('sel-follow'); if (fb) fb.setAttribute('aria-pressed', follow); }
});
select(null);
const live = document.getElementById('twin-live'), counts = document.getElementById('twin-counts');
let liveKey = '';

function resize() {
  const w = host.clientWidth, h = host.clientHeight; renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + 'px'; renderer.domElement.style.height = h + 'px';
  camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host); resize();
views.cctv(); applyOrbit();

function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (video && !video.paused) t = video.currentTime; else if (playing) { t += dt * speed; if (t >= duration) { t = 0; } }
  if (scrub && document.activeElement !== scrub) scrub.value = t;
  if (timeEl) timeEl.textContent = `${fmt(t)} / ${fmt(duration)}`;
  if (tween) {
    const k = Math.min(1, (now - tween.start) / 650); const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    orbit.theta = tween.from.theta + (tween.to.theta - tween.from.theta) * e;
    orbit.phi = tween.from.phi + (tween.to.phi - tween.from.phi) * e;
    orbit.radius = tween.from.radius + (tween.to.radius - tween.from.radius) * e;
    orbit.target.lerpVectors(tween.from.target, tween.to.target, e); if (k >= 1) tween = null;
  }
  if (selected && !sampleAt(selected.w, t)) selRing.visible = false;
  applyOrbit();
  let visible = 0, alerting = []; const pulse = 0.85 + 0.15 * Math.sin(now / 160);
  for (const item of workers) {
    const s = sampleAt(item.w, t);
    const hidden = !s || (show.focus && s.state === 0 && item !== selected);
    if (hidden) { item.g.visible = false; item.trail.visible = false; if (item.tag) item.tag.visible = false; if (s) visible++; continue; }
    visible++;
    item.g.visible = true; item.g.position.copy(P(s.x, s.y));
    const next = s.seg.xy.length > 2 * (s.i + 1) + 1 ? P(s.seg.xy[2 * (s.i + 1)], s.seg.xy[2 * (s.i + 1) + 1]) : null;
    if (next && next.distanceTo(item.g.position) > 0.02) item.g.lookAt(next.x, 0, next.z);
    item.torso.material = MAT[s.state]; item.hat.visible = s.helmet === 'h';
    item.ring.visible = s.state > 0; item.ring.material.color.setHex(s.state === 2 ? 0xe0454a : 0xfab219); item.ring.scale.setScalar(s.state === 2 ? pulse : 1);
    item.g.scale.setScalar(s.ok ? 1 : 0.92);
    const rules = ruleNames(s.rules);
    if (s.state > 0) alerting.push({ label: item.w.label, state: s.state, rules });
    const want = show.labels && s.state > 0 ? `${item.w.label} · ${rules.join(' ')}${s.state === 2 ? '' : ' pending'}` : '';
    if (want !== item.tagText) {
      if (item.tag) { scene.remove(item.tag); item.tag.material.map.dispose(); item.tag.material.dispose(); item.tag = null; }
      if (want) { item.tag = label(want, '#fff', s.state === 2 ? 'rgba(208,59,59,0.95)' : 'rgba(214,150,0,0.95)', 0.75); scene.add(item.tag); }
      item.tagText = want;
    }
    if (item.tag) { item.tag.visible = true; item.tag.position.copy(item.g.position).setY(1.35); }
    if (item === selected) {
      selRing.visible = true; selRing.position.copy(item.g.position).setY(0.013);
      if (follow && !drag) orbit.target.lerp(item.g.position.clone().setY(0), 0.08);
    }
    if (show.trails) {
      const pts = []; const seg = s.seg; const from = Math.max(0, s.i - 12);
      for (let k = from; k <= s.i; k++) pts.push(P(seg.xy[2 * k], seg.xy[2 * k + 1], 0.02));
      pts.push(P(s.x, s.y, 0.02)); item.trail.geometry.setFromPoints(pts); item.trail.visible = pts.length > 1;
    } else item.trail.visible = false;
  }
  for (const { m, mesh } of machines) {
    let k = -1; for (let i = 0; i < m.t.length; i++) { if (m.t[i] <= t) k = i; else break; }
    if (k < 0 || t - m.t[k] > 1.0) { mesh.visible = false; continue; }
    const poly = m.polygons[k]; const xs2 = poly.map(p => p[0]), ys2 = poly.map(p => p[1]);
    const w = Math.max(...xs2) - Math.min(...xs2), d = Math.max(...ys2) - Math.min(...ys2);
    const h = m.class === 'vehicle' ? 1.5 : 2.4;
    mesh.visible = true; mesh.scale.set(Math.max(w, 0.3), h, Math.max(d, 0.3));
    mesh.position.copy(P((Math.max(...xs2) + Math.min(...xs2)) / 2, (Math.max(...ys2) + Math.min(...ys2)) / 2, h / 2));
  }
  // live panel
  if (counts) counts.innerHTML = `<b>${visible}</b> in view · <b>${alerting.filter(a => a.state === 2).length}</b> alerting`;
  const key = alerting.map(a => a.label + a.state + a.rules.join()).join('|');
  if (live && key !== liveKey) {
    liveKey = key;
    live.innerHTML = alerting.length ? alerting.map(a => `<li><span class="dot" style="background:${a.state === 2 ? 'var(--critical)' : 'var(--warning)'}"></span>Worker ${esc(a.label)}<span class="muted" style="margin-left:auto">${a.rules.join(' ')} ${a.state === 2 ? 'alert' : 'pending'}</span></li>`).join('')
      : '<li class="twin-empty">No one is in a hazard condition right now.</li>';
  }
  // hover
  if (tip && pointer.x >= 0) {
    ray.setFromCamera({ x: (pointer.x / host.clientWidth) * 2 - 1, y: -(pointer.y / host.clientHeight) * 2 + 1 }, camera);
    const hits = ray.intersectObjects(workers.filter(w => w.g.visible).map(w => w.g), true);
    if (hits.length) {
      let o = hits[0].object; while (o.parent && !o.userData.id) o = o.parent;
      const item = workers.find(w => w.g === o); const s = item && sampleAt(item.w, t);
      if (s) {
        const rules = ruleNames(s.rules);
        tip.innerHTML = `<b>Worker ${esc(item.w.label)}</b>${s.state === 2 ? 'Alert: ' + rules.join(', ') : s.state === 1 ? 'Pending: ' + rules.join(', ') : 'No hazard condition'}<br>Helmet: ${s.helmet === 'h' ? 'yes' : s.helmet === 'n' ? 'no' : 'not sure'}<br><span class="muted">${s.ok ? 'position calibrated' : 'outside the calibrated area'}</span>`;
        tip.style.display = 'block'; tip.style.left = Math.min(pointer.x + 14, host.clientWidth - 190) + 'px'; tip.style.top = (pointer.y + 14) + 'px';
      }
    } else tip.style.display = 'none';
  }
  scene.updateMatrixWorld(); declutter();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
host.addEventListener('pointerleave', () => { pointer.x = -1; if (tip) tip.style.display = 'none'; });
requestAnimationFrame(frame);
})();
"""


def section(
    data: dict,
    video: str | None,
    poster: str | None = None,
    *,
    incidents: list[dict] | None = None,
    areas: dict[str, dict] | None = None,
) -> str:
    """The Digital twin page body (the caller adds the page head)."""
    from html import escape

    feed = (
        f'<video id="twin-video" muted playsinline preload="auto" src="{escape(video)}"'
        + (f' poster="{escape(poster)}"' if poster else "")
        + "></video>"
        if video
        else '<p class="twin-empty">No camera video in this bundle.</p>'
    )
    approx_pill = (
        '<span class="pill" style="color:var(--warning)" title="The camera could not be calibrated; positions are approximate">Approximate layout</span>'
        if (data.get("camera") or {}).get("approximate")
        else ""
    )
    legend = (
        '<div class="twin-legend">'
        '<span><i style="background:#6b819e"></i>Worker, no hazard</span>'
        '<span><i style="background:#fab219"></i>Hazard condition (pending)</span>'
        '<span><i style="background:#e0454a"></i>Alert</span>'
        '<span><i style="background:#f6c343"></i>Hard hat detected</span>'
        '<span><i style="background:#d03b3b;border-radius:2px"></i>Restricted zone</span>'
        '<span><i style="background:#f5c518;border-radius:2px"></i>Open edge</span>'
        '<span><i style="background:#3b82f6;border-radius:2px;opacity:.6"></i>Camera coverage</span>'
        "</div>"
    )
    return f"""
<div class="twin-layout">
  <div class="card twin-stage">
    <div id="twin-canvas" role="img" aria-label="Interactive 3D digital twin of the site">
      <div class="twin-hud"><span class="pill" id="twin-counts"></span>{approx_pill}<span class="pill">1 grid square = 1 worker height</span></div>
      <div class="twin-views"><button data-view="cctv" aria-pressed="true">CCTV view</button><button data-view="iso">Isometric</button><button data-view="top">Top-down</button></div>
      <div class="twin-tip" id="twin-tip"></div>
    </div>
    <div class="twin-bar">
      <button class="twin-play" id="twin-play" aria-label="Play or pause">▶</button>
      <span class="time" id="twin-time">0:00.0</span>
      <div class="twin-scrubwrap"><div id="twin-marks" aria-label="Incidents on the timeline"></div><input type="range" id="twin-scrub" min="0" step="0.05" value="0" aria-label="Time"></div>
      <div class="seg speed" role="group" aria-label="Speed"><button data-speed="1" aria-pressed="true">1×</button><button data-speed="2" aria-pressed="false">2×</button><button data-speed="4" aria-pressed="false">4×</button></div>
      <div class="twin-toggles"><button data-toggle="labels" aria-pressed="true">Labels</button><button data-toggle="trails" aria-pressed="true">Trails</button><button data-toggle="coverage" aria-pressed="true">Coverage</button><button data-toggle="heat" aria-pressed="false">Heat</button><button data-toggle="focus" aria-pressed="false">Hazards only</button></div>
    </div>
    {legend}
    <div class="twin-keys"><kbd>Space</kbd> play · <kbd>←</kbd><kbd>→</kbd> step · <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> views · click a worker to follow, a zone for its exposure · <kbd>F</kbd> follow · <kbd>Esc</kbd> clear</div>
  </div>
  <aside class="twin-side stack-gap">
    <div class="card"><div class="card-head"><h2>Camera feed</h2><span class="hint">in sync with the twin</span></div><div class="card-body">{feed}</div></div>
    <div class="card"><div class="card-head"><h2>Selection</h2></div><div class="card-body" id="twin-selected"></div></div>
    <div class="card"><div class="card-head"><h2>Right now</h2></div><div class="card-body"><ul class="twin-live" id="twin-live"></ul></div></div>
    <details class="card fold"><summary>How to read the twin</summary><div class="card-body small sub">
      Positions come from the camera calibrated on the workers themselves, smoothed over time,
      in worker heights (WH), not metres. Drag to orbit, right-drag or shift-drag to pan, scroll to zoom.
      Workers outside the calibrated area are drawn slightly smaller: their position is less certain.</div></details>
  </aside>
</div>
{embed(data)}
{embed_meta({"incidents": incidents or [], "areas": areas or {}})}"""
