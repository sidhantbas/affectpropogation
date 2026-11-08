"""
Lab Zero — Interactive LucidLyra × VoxLution × JEPA 3D World Holo-Deck on Port 8899.

Features:
  1. Real-Time VoxLution Coarse-to-Fine Evolution Slider (25cm blocks -> 12.5cm -> 6.25cm -> 3.12cm -> 1.56cm HD Splats)
  2. 4-Mode 3D Shader Switcher (Photorealistic RGB, LucidLyra 512-D CLIP PCA, 3D Object Instances, ESP32 Affect Survival)
  3. Open-Vocabulary 3D Object Search & Clickable Object Chips with Live Samsung S22 Camera Photo HUD
  4. Fun 3D Telekinesis & Physics Playground:
     - Zero-G Object Levitation (levitate any single 3D object instance in your room!)
     - 3D Exploded View (separate all 6 room objects in mid-air)
     - Object-to-Object Critical Affect Wave (Gamma = b * g)
     - Auto-Orbit Cinema Tour
"""

from __future__ import annotations

import http.server
import json
import socketserver
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RAW_FRAMES_DIR = DATA_DIR / "raw_frames"
PORT = 8899

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>AffectSplat Holo-Deck — LucidLyra × VoxLution × JEPA</title>
<style>
  * { box-sizing: border-box; }
  body {
    background: #070b14; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 12px; text-align: center;
  }
  .container {
    max-width: 960px; margin: 0 auto;
  }
  .card {
    background: #111827; border: 1px solid #1e293b; border-radius: 18px;
    padding: 18px; margin-bottom: 16px; box-shadow: 0 12px 36px rgba(0,0,0,0.6);
  }
  .badge {
    display: inline-block; background: linear-gradient(90deg, #065f46, #0f766e); color: #6ee7b7;
    padding: 5px 14px; border-radius: 999px; font-size: 11px; font-weight: 800; letter-spacing: 0.5px;
    margin-bottom: 8px;
  }
  h1 { color: #f8fafc; font-size: 22px; margin: 0 0 4px 0; }
  .subtitle { color: #94a3b8; font-size: 13px; margin-bottom: 12px; }
  h2 { color: #38bdf8; font-size: 16px; margin: 18px 0 8px 0; text-align: left; border-bottom: 1px solid #1e293b; padding-bottom: 6px; }

  .holo-grid {
    display: grid; grid-template-columns: 1.55fr 1fr; gap: 14px; align-items: start;
  }
  @media (max-width: 780px) {
    .holo-grid { grid-template-columns: 1fr; }
  }

  #splatCanvas {
    width: 100%; height: 420px; background: radial-gradient(circle at center, #0f172a 0%, #030712 100%);
    border-radius: 14px; border: 1.5px solid #38bdf8; touch-action: none; cursor: grab;
  }

  .control-box {
    background: #0b0f19; border: 1px solid #1e293b; border-radius: 14px; padding: 12px; text-align: left;
  }
  .ctrl-title {
    font-size: 11px; font-weight: 800; text-transform: uppercase; color: #94a3b8;
    letter-spacing: 0.6px; margin: 8px 0 6px 0;
  }
  .ctrl-title:first-child { margin-top: 0; }

  .slider-row {
    background: #111827; padding: 10px; border-radius: 10px; border: 1px solid #334155;
  }
  .slider-row input[type=range] {
    width: 100%; accent-color: #22c55e; cursor: pointer;
  }
  .stage-lbl {
    display: flex; justify-content: space-between; font-size: 11px; color: #cbd5e1; font-weight: 700; margin-top: 4px;
  }

  .btn-row {
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
  }
  .btn {
    padding: 8px 10px; border-radius: 9px; border: 1px solid #334155; background: #1e293b;
    color: #e2e8f0; font-weight: 700; font-size: 11.5px; cursor: pointer; transition: all 0.15s;
  }
  .btn:hover { border-color: #38bdf8; background: #0f172a; }
  .btn.active { background: #0284c7; border-color: #38bdf8; color: #ffffff; }
  .btn-fun { background: #311042; border-color: #a855f7; color: #f3e8ff; }
  .btn-fun.active { background: #9333ea; border-color: #d8b4fe; color: #fff; }

  .chips {
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px;
  }
  .chip {
    padding: 6px 10px; border-radius: 999px; font-size: 11px; font-weight: 700;
    cursor: pointer; border: 1.5px solid #334155; background: #111827; color: #cbd5e1;
  }
  .chip.active { border-color: #ffffff; transform: scale(1.04); color: #fff; }

  .search-bar {
    display: flex; gap: 6px; margin-top: 6px;
  }
  .search-bar input {
    flex: 1; padding: 8px 10px; border-radius: 8px; border: 1px solid #334155;
    background: #111827; color: #f8fafc; font-size: 12px;
  }

  .hud-card {
    margin-top: 10px; background: #111827; border: 1.5px solid #22c55e; border-radius: 12px;
    padding: 10px; display: grid; grid-template-columns: 110px 1fr; gap: 10px; align-items: center;
  }
  .hud-card img {
    width: 110px; height: 82px; object-fit: cover; border-radius: 8px; border: 1px solid #475569;
  }
  .hud-meta { font-size: 11.5px; line-height: 1.45; color: #cbd5e1; }
  .hud-meta strong { color: #f8fafc; font-size: 13px; }

  img.chart { width: 100%; border-radius: 12px; margin-top: 8px; border: 1px solid #1e293b; }
</style>
</head>
<body>
<div class="container">
  <div class="card">
    <div class="badge">🚀 ACT II LIVE • LUCIDLYRA × VOXLUTION × JEPA × ESP32-S3 HOLO-DECK</div>
    <h1>Interactive 3D Room Sandbox (129,441 Micro-Splats)</h1>
    <div class="subtitle">
      Drag to orbit in 3D • Slide to evolve voxels from <b>25cm → 1.56cm</b> • Click or type any object in your room to <b>Spotlight, Levitate, Explode, or Pulse</b> it!
    </div>

    <div class="holo-grid">
      <div>
        <canvas id="splatCanvas" width="600" height="420"></canvas>
        <div class="chips" id="objChips"></div>
      </div>

      <div class="control-box">
        <div class="ctrl-title">🧊 1. VoxLution Coarse-to-Fine Evolution Slider</div>
        <div class="slider-row">
          <input type="range" id="voxSlider" min="0" max="4" step="1" value="4" oninput="setVoxStage(this.value)">
          <div class="stage-lbl">
            <span id="stageName" style="color:#4ade80;">Stage 4: 1.56cm HD Leaf Splats</span>
            <span id="stageStats">84,953 Voxels (99.7% RAM Saved)</span>
          </div>
        </div>

        <div class="ctrl-title">🎨 2. LucidLyra 3D Brain Shader</div>
        <div class="btn-row">
          <button class="btn active" id="mode-rgb" onclick="setMode('rgb')">📸 True RGB Color</button>
          <button class="btn" id="mode-pca" onclick="setMode('pca')">🧠 CLIP 512-D PCA</button>
          <button class="btn" id="mode-inst" onclick="setMode('inst')">🎯 6 Object Twins</button>
          <button class="btn" id="mode-trust" onclick="setMode('trust')">📡 ESP32 Survival I(n)</button>
        </div>

        <div class="ctrl-title">🔍 3. Open-Vocabulary 3D Text Search (CLIP)</div>
        <div class="search-bar">
          <input type="text" id="searchInput" placeholder="Try: 'laptop', 'mug', 'flag', 'door', 'esp32'..." oninput="runTextSearch(this.value)">
          <button class="btn" onclick="selectObject(-1)">Reset</button>
        </div>

        <div class="ctrl-title">🕹️ 4. Fun 3D Telekinesis & Physics Modes</div>
        <div class="btn-row">
          <button class="btn btn-fun" id="btn-levitate" onclick="toggleLevitate()">🛸 Zero-G Levitate Object</button>
          <button class="btn btn-fun" id="btn-explode" onclick="toggleExplode()">💥 Explode 6 Objects</button>
          <button class="btn btn-fun" onclick="triggerWave(1.35)">⚡ Fire Object Affect Wave</button>
          <button class="btn btn-fun" id="btn-tour" onclick="toggleTour()">🎥 Auto-Cinema Tour</button>
        </div>

        <div class="hud-card" id="hudCard">
          <img id="hudImg" src="/raw_frames/frame_0008.jpg" alt="Room Object">
          <div class="hud-meta" id="hudMeta">
            <strong>🌐 Whole Room World Model</strong><br>
            • <b>6 Semantic Objects</b> + <b>ESP32-S3 Seed (x_0)</b><br>
            • Click any object chip below the 3D view or type in the search bar to isolate & levitate it!
          </div>
        </div>
      </div>
    </div>

    __CONTENT_BLOCK__
  </div>
</div>

<script>
  let lucidData = null;
  let splats = [];
  let vox25 = [], vox12 = [], vox6 = [];
  let classes = [];
  let seed = [-0.28, -0.26, 0.82];

  let voxStage = 4; // 0=25cm, 1=12.5cm, 2=6.25cm, 3=3.12cm, 4=1.56cm
  let renderMode = 'rgb'; // 'rgb', 'pca', 'inst', 'trust'
  let selectedObj = -1; // -1 = all, 0..5 = specific object instance
  let levitateOn = false;
  let explodeOn = false;
  let explodeFactor = 0.0;
  let tourOn = false;

  let rotY = -0.48, rotX = 0.22, zoom = 310;
  let dragging = false, lastX = 0, lastY = 0;
  let waveTime = -1, waveGamma = 1.35, waveOrigin = [-0.28, -0.26, 0.82];

  const canvas = document.getElementById('splatCanvas');
  const ctx = canvas.getContext('2d');

  const STAGE_INFO = [
    { name: "Stage 0: 25.0 cm Coarse Cubes", stats: "1,860 Voxels • Minecraft Layout" },
    { name: "Stage 1: 12.5 cm Octant Subdivision", stats: "10,643 Voxels • 1-Byte Octant Stream" },
    { name: "Stage 2: 6.25 cm Mid Evolution", stats: "41,109 Voxels • Structural Surfaces" },
    { name: "Stage 3: 3.125 cm Sharp Geometry", stats: "75,554 Voxels • High-Surprise Split" },
    { name: "Stage 4: 1.5625 cm HD Leaf Splats", stats: "84,953 Voxels • 99.7% RAM Saved!" }
  ];

  const OCTANT_CODES = ["0x9F (159)", "0xE7 (231)", "0x7B (123)", "0xD5 (213)", "0xFE (254)", "0xBC (188)"];

  fetch('/room_splats_lucid.json').then(r => r.json()).then(data => {
    lucidData = data;
    seed = data.esp32_seed_xyz || seed;
    classes = data.semantic_classes || [];
    vox25 = data.voxels_25cm || [];
    vox12 = data.voxels_12_5cm || [];
    vox6 = data.voxels_6_25cm || [];
    // Subsample every 2nd HD splat (~43k splats) for silky 60fps interactive rendering
    splats = data.splats.filter((_, idx) => idx % 2 === 0);
    buildChips();
    requestAnimationFrame(draw);
  });

  function buildChips() {
    const box = document.getElementById('objChips');
    let html = `<button class="chip active" id="chip--1" onclick="selectObject(-1)">🌐 All 6 Objects (${splats.length * 2} Splats)</button>`;
    classes.forEach((c, i) => {
      html += `<button class="chip" id="chip-${i}" style="border-color:${c.color_hex}" onclick="selectObject(${i})">
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${c.color_hex};margin-right:4px;"></span>
        ${c.label}
      </button>`;
    });
    box.innerHTML = html;
  }

  function setVoxStage(val) {
    voxStage = parseInt(val, 10);
    document.getElementById('stageName').textContent = STAGE_INFO[voxStage].name;
    document.getElementById('stageStats').textContent = STAGE_INFO[voxStage].stats;
  }

  function setMode(m) {
    renderMode = m;
    ['rgb', 'pca', 'inst', 'trust'].forEach(k => {
      document.getElementById('mode-' + k).classList.toggle('active', k === m);
    });
  }

  function selectObject(id) {
    selectedObj = id;
    document.querySelectorAll('.chip').forEach(el => el.classList.remove('active'));
    const activeChip = document.getElementById('chip-' + id);
    if (activeChip) activeChip.classList.add('active');

    const hudCard = document.getElementById('hudCard');
    const hudImg = document.getElementById('hudImg');
    const hudMeta = document.getElementById('hudMeta');

    if (id === -1) {
      hudCard.style.borderColor = '#22c55e';
      hudImg.src = '/raw_frames/frame_0008.jpg';
      hudMeta.innerHTML = `<strong>🌐 Whole Room World Model (1.56cm)</strong><br>
        • <b>129,441 Anisotropic Splats</b> • <b>99.7% Voxel RAM Saved</b><br>
        • Click any object chip or use <b>Zero-G Levitate</b> / <b>Explode</b>!`;
      waveOrigin = seed;
    } else {
      const c = classes[id];
      hudCard.style.borderColor = c.color_hex;
      hudImg.src = '/raw_frames/' + c.frame_file;
      hudMeta.innerHTML = `<strong style="color:${c.color_hex}">${c.label}</strong><br>
        • <b>Prompt:</b> "${c.prompt.slice(0, 48)}..."<br>
        • <b>Octant Code:</b> ${OCTANT_CODES[id]} • <b>ESP32 Survival:</b> I=${c.survival_I.toFixed(2)} (${c.rssi_dbm} dBm)`;
      waveOrigin = c.center_xyz;
    }
  }

  function runTextSearch(q) {
    const s = q.trim().toLowerCase();
    if (!s) { selectObject(-1); return; }
    for (let i = 0; i < classes.length; i++) {
      const hay = (classes[i].label + ' ' + classes[i].prompt).toLowerCase();
      if (hay.includes(s)) {
        selectObject(i);
        return;
      }
    }
  }

  function toggleLevitate() {
    levitateOn = !levitateOn;
    document.getElementById('btn-levitate').classList.toggle('active', levitateOn);
    if (levitateOn && selectedObj === -1) {
      selectObject(2); // Default to levitating the Steel Tumbler Mug if none selected!
    }
  }

  function toggleExplode() {
    explodeOn = !explodeOn;
    document.getElementById('btn-explode').classList.toggle('active', explodeOn);
  }

  function toggleTour() {
    tourOn = !tourOn;
    document.getElementById('btn-tour').classList.toggle('active', tourOn);
  }

  function triggerWave(gamma) {
    waveGamma = gamma;
    waveOrigin = selectedObj >= 0 ? classes[selectedObj].center_xyz : seed;
    waveTime = performance.now();
  }

  canvas.addEventListener('pointerdown', e => { dragging = true; tourOn = false; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener('pointerup', () => dragging = false);
  window.addEventListener('pointermove', e => {
    if (!dragging) return;
    rotY += (e.clientX - lastX) * 0.0085;
    rotX = Math.max(-1.0, Math.min(1.0, rotX + (e.clientY - lastY) * 0.0085));
    lastX = e.clientX; lastY = e.clientY;
  });

  function draw(now) {
    ctx.fillStyle = '#050811';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (tourOn) {
      rotY = -0.48 + Math.sin(now * 0.0007) * 0.55;
      rotX = 0.20 + Math.cos(now * 0.0005) * 0.14;
      const tourIdx = Math.floor((now * 0.0004) % 7) - 1;
      if (tourIdx !== selectedObj) selectObject(tourIdx);
    }

    // Smoothly animate Exploded View factor
    const targetExplode = explodeOn ? 0.55 : 0.0;
    explodeFactor += (targetExplode - explodeFactor) * 0.12;

    const cy = Math.cos(rotY), sy = Math.sin(rotY);
    const cx = Math.cos(rotX), sx = Math.sin(rotX);
    const w = canvas.width, h = canvas.height;

    let dt = waveTime > 0 ? (now - waveTime) * 0.0018 : -1;
    if (dt > 4.2) { waveTime = -1; dt = -1; }

    const projected = [];
    // Choose active dataset based on VoxLution slider stage
    let activeSet = splats;
    let isVoxelCube = false;
    let baseRadius = 1.0;

    if (voxStage === 0) { activeSet = vox25; isVoxelCube = true; baseRadius = 0.13; }
    else if (voxStage === 1) { activeSet = vox12; isVoxelCube = true; baseRadius = 0.07; }
    else if (voxStage === 2) { activeSet = vox6; isVoxelCube = true; baseRadius = 0.04; }
    else if (voxStage === 3) { activeSet = splats; baseRadius = 1.45; }

    const step = (voxStage === 3) ? 2 : 1;

    for (let i = 0; i < activeSet.length; i += step) {
      const s = activeSet[i];
      let x = s[0], y = s[1], z = s[2];
      const instId = isVoxelCube ? s[6] : s[8];
      const cls = classes[instId] || classes[0];

      // 1. Apply 3D Exploded View offset per semantic object
      if (explodeFactor > 0.005 && cls) {
        x += cls.center_xyz[0] * explodeFactor;
        y += (cls.center_xyz[1] + 0.15) * explodeFactor;
        z += (cls.center_xyz[2] - 1.35) * explodeFactor;
      }

      // 2. Apply Zero-G Levitation to selected object!
      if (levitateOn && instId === selectedObj) {
        const hover = 0.28 + 0.14 * Math.sin(now * 0.004 + x * 4.0);
        y += hover;
      }

      const zCam = z - 1.55;
      const rx = x * cy - zCam * sy;
      const rz = x * sy + zCam * cy;
      const ry = y * cx - rz * sx;
      const fz = y * sx + rz * cx + 2.75;
      if (fz < 0.25) continue;

      let r = s[3], g = s[4], b = s[5];
      let op = isVoxelCube ? 0.92 : s[7] * 0.82;

      if (renderMode === 'pca' && !isVoxelCube) {
        r = s[9]; g = s[10]; b = s[11];
      } else if (renderMode === 'inst' && cls) {
        r = cls.color_rgb[0]; g = cls.color_rgb[1]; b = cls.color_rgb[2];
      } else if (renderMode === 'trust' && cls) {
        const surv = cls.survival_I;
        r = Math.round(255 * (1 - surv));
        g = Math.round(235 * surv + 20);
        b = Math.round(120 * surv);
      }

      // Spotlight isolation when a specific object is selected
      if (selectedObj >= 0) {
        if (instId !== selectedObj) {
          op *= 0.12;
          r = Math.round(r * 0.35); g = Math.round(g * 0.35); b = Math.round(b * 0.45);
        } else {
          op = Math.min(1.0, op * 1.25);
        }
      }

      // Critical Affect Wave animation
      if (dt >= 0) {
        const dist = Math.hypot(s[0] - waveOrigin[0], s[1] - waveOrigin[1], s[2] - waveOrigin[2]);
        const waveFront = dt * 1.45;
        if (Math.abs(dist - waveFront) < 0.35) {
          r = 250; g = 204; b = 21; op = 0.99;
        }
      }

      const screenX = w * 0.5 + (rx / fz) * zoom;
      const screenY = h * 0.5 - (ry / fz) * zoom;
      const rad = isVoxelCube ? (baseRadius / fz) * zoom : (s[6] * baseRadius / fz) * zoom * 2.2;

      projected.push([fz, screenX, screenY, rad, r, g, b, op]);
    }

    projected.sort((a, b) => b[0] - a[0]);

    for (let i = 0; i < projected.length; i++) {
      const p = projected[i];
      ctx.fillStyle = `rgba(${p[4]},${p[5]},${p[6]},${p[7]})`;
      const sz = Math.max(1.6, p[3]);
      if (isVoxelCube) {
        ctx.fillRect(p[1] - sz, p[2] - sz, sz * 1.9, sz * 1.9);
      } else {
        ctx.beginPath();
        ctx.arc(p[1], p[2], sz, 0, 6.283);
        ctx.fill();
      }
    }

    // Draw 3D Object Instance Anchors & Labels
    for (let i = 0; i < classes.length; i++) {
      const c = classes[i];
      let ox = c.center_xyz[0], oy = c.center_xyz[1], oz = c.center_xyz[2];
      if (explodeFactor > 0.005) {
        ox += ox * explodeFactor;
        oy += (oy + 0.15) * explodeFactor;
        oz += (oz - 1.35) * explodeFactor;
      }
      if (levitateOn && i === selectedObj) {
        oy += 0.28 + 0.14 * Math.sin(now * 0.004 + ox * 4.0);
      }
      const ozCam = oz - 1.55;
      const orx = ox * cy - ozCam * sy;
      const orz = ox * sy + ozCam * cy;
      const ory = oy * cx - orz * sx;
      const ofz = oy * sx + orz * cx + 2.75;
      if (ofz > 0.25) {
        const px = w * 0.5 + (orx / ofz) * zoom;
        const py = h * 0.5 - (ory / ofz) * zoom;
        const isSel = (i === selectedObj);
        ctx.fillStyle = c.color_hex;
        ctx.strokeStyle = isSel ? '#ffffff' : '#0f172a';
        ctx.lineWidth = isSel ? 2.5 : 1.5;
        ctx.beginPath();
        ctx.arc(px, py, isSel ? 7.5 : 5.0, 0, 6.283);
        ctx.fill();
        ctx.stroke();
        if (selectedObj === -1 || isSel) {
          ctx.fillStyle = isSel ? '#ffffff' : c.color_hex;
          ctx.font = isSel ? 'bold 12px sans-serif' : 'bold 10px sans-serif';
          ctx.fillText(c.label, px + 9, py + 4);
        }
      }
    }

    requestAnimationFrame(draw);
  }
</script>
</body>
</html>
"""


class PortalHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        clean_path = self.path.split("?")[0]
        if clean_path == "/room_splats_lucid.json":
            splat_file = DATA_DIR / "room_splats_lucid.json"
            if splat_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(splat_file.read_bytes())
                return
            self.send_error(404)
            return

        if clean_path == "/room_splats.json":
            splat_file = DATA_DIR / "room_splats.json"
            if splat_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(splat_file.read_bytes())
                return
            self.send_error(404)
            return

        if clean_path.startswith("/figures/"):
            fig_file = FIG_DIR / Path(clean_path).name
            if fig_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(fig_file.read_bytes())
                return
            self.send_error(404)
            return

        if clean_path.startswith("/raw_frames/"):
            frame_file = RAW_FRAMES_DIR / Path(clean_path).name
            if frame_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(frame_file.read_bytes())
                return
            self.send_error(404)
            return

        ts = int(time.time())
        blocks = []
        gallery = [
            (
                "epic_object_anatomy.png",
                "🎯 Anatomy of Your 3D Room: Real Samsung S22 Photos vs. Isolated 3D Object Twins",
            ),
            (
                "voxlution_cascade.png",
                "🧊 VoxLution Coarse-to-Fine Sparse Octant Evolution (25cm → 1.56cm, 99.7% RAM Saved)",
            ),
            (
                "lucidlyra_semantics.png",
                "🧠 LucidLyra Single-Pass RGB + 512-D CLIP Semantic Field + 6-Object Segmentation",
            ),
            (
                "jepa_expA_ablation.png",
                "📡 JEPA Mini-Exp A: Dividing the Role of Samsung Camera (z_k) vs. ESP32-S3 Radar (a_k)",
            ),
            (
                "jepa_expB_hd_splats.png",
                "✨ JEPA Mini-Exp B: 12× ViT Patch-Surprise Gaussian Densification (129,441 Micro-Splats)",
            ),
            (
                "jepa_expC_blindspot.png",
                "🕶️ JEPA Mini-Exp C: 18-Step Camera Blind-Spot / Visual Blackout Prediction",
            ),
            (
                "hour5_benchmark.png",
                "🏆 Hour 5 Benchmark: 28.8× Faster CPU Time, 9.6× Lower RAM, 85.8% Coherence",
            ),
        ]
        for fname, title in gallery:
            if (FIG_DIR / fname).exists():
                blocks.append(f'<h2>{title}</h2><img class="chart" src="/figures/{fname}?t={ts}">')

        content = "\n".join(blocks)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.replace("__CONTENT_BLOCK__", content).encode("utf-8"))


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), PortalHandler) as httpd:
        print(f"AffectSplat Holo-Deck listening on http://127.0.0.1:{PORT}", flush=True)
        httpd.serve_forever()
