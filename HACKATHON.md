# AffectSplat: The 5-Hour Low-Spec 3D World Model Hackathon

**Hardware Allowed:** `1x Samsung Android Phone` + `1x Bare ESP32` + `This Laptop`  
**External Parts / Cloud GPUs Allowed:** `NONE ($0 Budget)`  
**Time Limit:** `5 Hours (5 x 60-Minute Sprints)`

---

## Why This Exists (The Research Benchmark Goal)

Normal **3D Gaussian Splatting (3DGS)** is built for rich labs with $2,000 NVIDIA GPUs. A 15-second phone video has **450 frames**. Standard pipelines run heavy math on *all 450 frames*—even when 80% of those frames are near-duplicates and 10% are motion-blurred from hand shake.

This hackathon turns the **Affect Propagation** theory ([`MISSION_empirical_validation.md`](MISSION_empirical_validation.md)) into a **low-spec 3D World Modeling benchmark**:

1. **Phone Video is a Causal Chain (`x_0 -> x_1 -> ... -> x_N`):**
   Every frame step has a link quality `g_k` (from `0` to `1`). Total coordinate trust at step `n` multiplies:
   $$I(n) = \prod_{k=1}^{n} g_k$$
2. **Mission 3 Weak-Link Pruning (10x Less Compute):**
   Because multiplication by `0` destroys everything after it (`I(N) = g_weak * g_strong^(N-1)`), one blurry frame ruins a 3D map. By filtering out **weak links** (`g_k ≈ floor`) and **redundant frames** (`g_k > 0.96`) *before* 3D reconstruction, we shrink 450 frames down to **~25 Golden Keyframes**—cutting RAM and CPU time by ~90%.
3. **Mission 2A Bare-ESP32 Anchor (Zero-Cost Loop Closure):**
   A bare ESP32 has a built-in Wi-Fi radio, a `BOOT` button, and bare-metal **capacitive touch pins** (`touchRead`). Placed on your desk as **Seed `x_0`**, its Wi-Fi RSSI peak tells the phone when you return to the start—resetting drift `I(n) -> 1.0` with zero extra sensors.
4. **Mission 4 Live 3D Affect Waves (`Γ = b · g`):**
   Touch a bare pin on your ESP32 or walk past it, and watch a state wave ripple across your 3D Gaussian room—settling locally when `Γ < 1` or triggering a room-wide phase transition when `Γ >= 1`.

---

## The 5-Hour Sprint Map

| Hour | Sprint Name | What You Learn (Simple Version) | Checkpoint Artifact |
| :--- | :--- | :--- | :--- |
| **Hour 1** | **The Chain Diet** | How to score frame links (`g_k`) and prune 90% of video frames using multiplicative decay. | `figures/hour1_chain_filter.png` |
| **Hour 2** | **The Bare ESP32 Seed** | How to use a bare ESP32's Wi-Fi RSSI & touch pins as Origin Seed `x_0` to reset drift. | `figures/hour2_esp32_anchor.png` |
| **Hour 3** | **Lite-Splat 3D** | How 3D Gaussian Splatting works (`xyz`, `rgb`, `scale`, `opacity`) on CPU in `<60s`. | `data/room_world.splat` + `.html` viewer |
| **Hour 4** | **3D Affect Wave** | How `Γ = b · g` turns a dead 3D photo into a live, reactive World Model. | `figures/hour4_critical_wave.png` |
| **Hour 5** | **GitHub Benchmark** | How to prove our pipeline beats brute-force 3DGS in RAM, speed, and coherence. | `figures/hour5_benchmark.png` + `README.md` |

---

## Quick Commands (Your Guru CLI)

Run the interactive Hackathon Mentor CLI at any time:

```bash
# View your live 5-hour scoreboard and checkpoint status
./venv/bin/python hackathon.py status

# Get the Guru's plain-English lesson for Hour 1..5
./venv/bin/python hackathon.py learn 1

# Verify and unlock a checkpoint once built
./venv/bin/python hackathon.py check 1
```

