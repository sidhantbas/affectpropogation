# AffectSplat: Low-Spec 3D Gaussian World Models via Multiplicative Affect Propagation

<p align="center">
  <img src="figures/exp_camera_filmstrip.png" alt="Physical Hardware & Visual Chain Capture Filmstrip" width="100%">
</p>

> **A zero-GPU, commodity-hardware research benchmark proving that multiplicative chain composition ($I(n) = \prod g_k$), weak-link gatekeeping, and a bare ESP32-S3 Wi-Fi CSI Seed Anchor reduce 3D Gaussian Splatting compute by 28.8× (`80.06s → 2.78s`) and memory by 9.6× (`412 MB → 42.8 MB`) while lifting 3D trajectory coherence from `4.8%` to `85.8%`.**

---

## 📸 Physical Hardware & Experimental Capture Gallery

All experiments in this repository were executed on a real indoor environment using **zero external sensors and zero cloud GPUs**:
- **Origin Seed Node ($x_0$):** `1x Bare ESP32-S3` (`QFN56`, 8MB PSRAM, `esp32-csi-node v0.4.3.1`, MAC `10:b4:1d:c4:00:44`) on `/dev/cu.usbmodem5AE70749591` streaming live 128-byte Wi-Fi Channel State Information (CSI) + RSSI radar reflections.
- **Mobile Explorer Node ($x_1 \dots x_N$):** `1x Samsung Galaxy S22 Ultra` (`SM-S908E`, Serial `R5CT33YQERR`) tethered via USB/ADB (`localhost:8899`).
- **Host Compute:** Standard Mac CPU (`Apple Silicon / CPU-only execution`).

| **(a) Origin Seed $x_0$ (`Frame #1`)** | **(b) Rejected Weak Link (`Frame #51`)** | **(c) Golden Keyframe (`Frame #81`)** | **(d) Terminal Keyframe (`Frame #168`)** |
| :---: | :---: | :---: | :---: |
| <img src="figures/frame_seed_esp32.jpg" width="210"> | <img src="figures/frame_blur_rejected.jpg" width="210"> | <img src="figures/frame_mid_kept.jpg" width="210"> | <img src="figures/frame_end_kept.jpg" width="210"> |
| **ESP32-S3 on Desk**<br>`I(0) = 1.000` • `s_blur = 0.92`<br>*Locks 3D origin coordinate* | **Motion Blur (Hand Shake)**<br>`s_blur = 0.48 < 0.76`<br>*Blocked by Mission 3 Gatekeeper* | **Mid-Room Scan (Nepal Flags)**<br>`s_blur = 0.99` • `KEPT`<br>*High-sharpness 3D splat source* | **End of Sweep (Room Door)**<br>`I_anchored = 0.235` • `KEPT`<br>*Rescued by ESP32-S3 CSI Seed* |

---

## 🏆 Head-to-Head Low-Spec Benchmark (`Hour 5`)

<p align="center">
  <img src="figures/hour5_benchmark.png" alt="Hour 5 Head-to-Head Benchmark" width="100%">
</p>

| Pipeline | Frames Processed | Motion-Blur Floaters | 3D Coherence ($I \ge 0.10$) | Peak RAM | Mac CPU Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive All-Frame 3DGS Baseline** | `168` | `1,450` | `4.8%` (`8 / 168`) | `412.0 MB` | `80.06 s` |
| **Chain Diet Only (No ESP32-S3)** | `106` | `0` | `7.5%` (`8 / 106`) | `68.5 MB` | `3.20 s` |
| **AffectSplat (Ours: S22 + ESP32-S3)** | **`28` (`106` golden)** | **`0`** | **`85.8%` (`91 / 106`)** | **`42.8 MB` (`9.6× lighter`)** | **`2.78 s` (`28.8× faster`)** |

---

## 🔬 Staged Empirical Validation (Act 0 – Hour 4)

Every experiment in `world_model/` tests a falsifiable prediction from [`MISSION_empirical_validation.md`](MISSION_empirical_validation.md).

### Act 0: Live Silicon Handshake & ESP32-S3 CSI Radar Heartbeat
**Script:** [`world_model/exp00_hardware_pulse.py`](world_model/exp00_hardware_pulse.py) • **Data:** [`data/act0_hardware_pulse.json`](data/act0_hardware_pulse.json)

Verifies both hardware nodes on the USB bus (`ESP32-S3` + `SM-S908E`) and captures a live multi-channel (`ch 1, 6, 11`) Wi-Fi CSI radar stream (`1,000` callbacks, `-66 dBm` to `-81 dBm`) from the physical seed $x_0$.

<p align="center">
  <img src="figures/act0_seed_heartbeat.png" alt="Act 0 Live Seed Heartbeat" width="85%">
</p>

---

### Hour 1: The "Chain Diet" — Pre-Registered Geometric Decay & Weak-Link Gatekeeper
**Script:** [`world_model/exp01_chain_filter.py`](world_model/exp01_chain_filter.py) • **Math Engine:** [`core/affect_math.py`](core/affect_math.py) • **Data:** [`data/hour1_summary.json`](data/hour1_summary.json)

Tests **Mission 0 (Instrument Floor Subtraction)**, **Mission 1 (Out-of-Sample Multiplicative Decay)**, and **Mission 3 (Weak-Link Annihilation)** on the `168`-frame (`15 fps`) Samsung S22 Ultra room video:
1. **Instrument Floor (`Mission 0`):** Unrelated cross-room frames sit at cosine similarity $\text{floor} = 0.6311$, which is subtracted before computing link gains:
   $$g_k = \frac{\cos(E(x_{k-1}), E(x_k)) - \text{floor}}{1 - \text{floor}} \cdot s_{\text{blur}}(x_k)$$
2. **Weak-Link Gatekeeper (`Mission 3`):** Automatically detects and rejects **`27` motion-blur frames** (red dots where $s_{\text{blur}} < 0.76$) and skips **`35` redundant frames**, distilling `106` clean Golden Keyframes in **`2.40 seconds` on Mac CPU**.
3. **Pre-Registered Decay (`Mission 1`):** Single-parameter geometric prediction $I(k) = \hat{g}^{\,k-1}$ ($\hat{g} = 0.729$, yellow dashed curve) tracks the observed multiplicative chain $I(n) = \prod g_k$ (green curve).

<p align="center">
  <img src="figures/hour1_chain_filter.png" alt="Hour 1 Chain Diet & Multiplicative Decay" width="95%">
</p>

---

### Hour 2: Bare ESP32-S3 "Seed Anchor" — Defeating 3D Drift via Mission 2A
**Script:** [`world_model/exp02_esp32_anchor.py`](world_model/exp02_esp32_anchor.py) • **Data:** [`data/hour2_summary.json`](data/hour2_summary.json)

Tests **Mission 2A ("Seed Visible" Control)**:
- In an **unanchored chain** (orange dashed curve), multiplicative decay $I(k) = \prod g_j$ drops below the coherence threshold $\theta = 0.10$ after just **8 keyframes** (`7.5%` survival), flatlining at `0.0000`.
- When coupled to the live **ESP32-S3 Wi-Fi CSI Radar (`-86 dBm` to `-68 dBm`)** at origin $x_0$, parallel causal composition:
  $$I_{\text{anchored}}(k) = 1 - \bigl(1 - g_k \cdot I_{\text{anchored}}(k-1)\bigr)\bigl(1 - \alpha_{\text{seed}}(k)\bigr)$$
  repeatedly rescues coordinate trust against drift, keeping **`91 / 106` keyframes (`85.8%`) above $\theta = 0.10$** and closing the loop at `I = 0.235`.

<p align="center">
  <img src="figures/hour2_esp32_anchor.png" alt="Hour 2 ESP32-S3 Seed Anchor Validation" width="95%">
</p>

---

### Hour 3: "Lite-Splat" — 10,752 3D Gaussian Splats in `0.12 Seconds` on Mac CPU
**Script:** [`world_model/exp03_lite_splat.py`](world_model/exp03_lite_splat.py) • **Data:** [`data/hour3_summary.json`](data/hour3_summary.json)

Unprojects `28` sampled Golden Keyframes into **`10,752` 3D Gaussian Splats** (`x, y, z, r, g, b, scale, opacity`) in **`0.116 seconds` on CPU**, weighting each splat's opacity by its ESP32-S3 anchored trust score $I_{\text{anchored}}(k)$:
- **Left (Naive All-Frame 3DGS):** Includes the `27` motion-blurred frames at full opacity, contaminating the 3D volume with **`1,450` red "floater" artifacts**.
- **Right (AffectSplat — Ours):** Produces a clean 3D room reconstruction anchored at the **ESP32-S3 Origin Seed (`x_0 = [-0.28m, -0.26m, 0.82m]`)**.

<p align="center">
  <img src="figures/hour3_lite_splat.png" alt="Hour 3 Lite-Splat 3D Gaussian World Comparison" width="100%">
</p>

---

### Hour 4: Live 3D Affect Waves & The Critical Point ($\Gamma = b \cdot g = 1.0$)
**Script:** [`world_model/exp04_affect_wave_3d.py`](world_model/exp04_affect_wave_3d.py) • **Data:** [`data/hour4_summary.json`](data/hour4_summary.json)

Tests **Mission 4 (Critical Branching Transition)** on the 3D Gaussian room graph triggered by live ESP32-S3 CSI packet `#811900`:
- With measured room link gain $g = 0.78$, the theoretical critical branching threshold sits at:
  $$b_{\text{crit}} = \frac{1}{g} = \frac{1}{0.78} = 1.282$$
- **Sub-Critical Regime ($\Gamma = 0.78 < 1$, Bottom-Left):** A local desk perturbation activates only **`1,284 / 10,752` splats** immediately surrounding the ESP32-S3 on the desk and dies out naturally.
- **Super-Critical Regime ($\Gamma = 1.56 \ge 1$, Bottom-Right):** Crossing $b_{\text{crit}} = 1.282$ triggers a 3-order-of-magnitude phase transition (`10 → 18,865` active graph nodes), illuminating all **`10,752 / 10,752` 3D Gaussians** across the room.

<p align="center">
  <img src="figures/hour4_critical_wave.png" alt="Hour 4 Critical Branching Phase Transition" width="100%">
</p>

---

## 📋 Mission 5: Official Protocol Verdict Table ([`mission/05_verdict/VERDICT.md`](mission/05_verdict/VERDICT.md))

| Mission | Prediction if Theory Holds | Prediction if Null Holds | Observed (`SM-S908E` + `ESP32-S3`) | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Mission 0 (Instrument)** | Non-zero floor, no depth artifact | Depth trend in unrelated pairs | `floor = 0.6311` across distant room views | **✅ PASSED** |
| **Mission 1 (Decay)** | Log-linear at registered slope $\hat{g}$ | Linear or early plateau | `g_hat = 0.729` predicts $I(n) = \prod g_k$ | **✅ PASSED** |
| **Mission 2A (Seed Visible)** | Slower decay, high plateau | Identical to Mission 1 | ESP32-S3 CSI anchor lifts survival from `7.5%` to **`85.8%`** | **✅ PASSED** |
| **Mission 3 (Weak Link)** | 1 weak link annihilates naive chain | High survival near control | `27` blur frames annihilated naive chain (`1,450` floaters); `0` with gate | **✅ PASSED** |
| **Mission 4 (Critical Point)** | Sharp bend near $b = 1/g$ | Smooth growth, no threshold | Sharp 3-order-of-magnitude transition at $b_{\text{crit}} = 1.282$ | **✅ PASSED** |

---

## 🛠️ Repository Layout & Reproduction Commands

```text
affectpropogation/
├── README.md                          # Visual showcase, theory, & empirical benchmark tables
├── HACKATHON.md                       # 5-Hour Low-Spec Hackathon guide & sprint rules
├── MISSION_empirical_validation.md    # Pre-registered theoretical protocol (Missions 0–5)
├── hackathon.py                       # Interactive Guru CLI & milestone verifier
├── core/
│   └── affect_math.py                 # Floor subtraction, I(n) = Π g_k, weak-link gatekeeper
├── world_model/
│   ├── exp00_hardware_pulse.py        # Act 0: USB bus check & live ESP32-S3 CSI radar capture
│   ├── exp01_chain_filter.py          # Hour 1: Video -> Golden Keyframes (Missions 0, 1, 3)
│   ├── exp02_esp32_anchor.py          # Hour 2: ESP32-S3 CSI Seed Anchor (Mission 2A)
│   ├── exp03_lite_splat.py            # Hour 3: 0.12s CPU 3D Gaussian Splatting (.ply/.json)
│   ├── exp04_affect_wave_3d.py        # Hour 4: Critical branching wave Γ = b·g (Mission 4)
│   ├── exp05_benchmark.py             # Hour 5: RAM / CPU / Coherence benchmark suite
│   └── phone_portal.py                # Interactive 60fps 3D WebGL/Canvas server (port 8899)
├── mission/
│   ├── 00_instrument/generate_chains.py
│   └── 05_verdict/VERDICT.md          # Official Mission 5 Verdict table
├── data/                              # JSON telemetry summaries & checkpoints
└── figures/                           # Camera filmstrip, raw frames, & benchmark plots
```

### Run All Benchmarks (`<10 Seconds` on Mac CPU)

```bash
# 1. Check 5-Hour Hackathon Scoreboard
./venv/bin/python hackathon.py status

# 2. Run the 5-Stage Experimental Suite
./venv/bin/python world_model/exp00_hardware_pulse.py
./venv/bin/python world_model/exp01_chain_filter.py
./venv/bin/python world_model/exp02_esp32_anchor.py
./venv/bin/python world_model/exp03_lite_splat.py
./venv/bin/python world_model/exp04_affect_wave_3d.py
./venv/bin/python world_model/exp05_benchmark.py

# 3. Launch Interactive 60fps 3D World Viewer on Samsung Phone over USB Tunnel
./venv/platform-tools/adb reverse tcp:8899 tcp:8899
./venv/bin/python world_model/phone_portal.py
```
