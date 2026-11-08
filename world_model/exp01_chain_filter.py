"""
Hour 1 Experiment: The "Chain Diet" (Video -> Multiplicative Affect Graph -> Golden Keyframes)

Implements Missions 0, 1, and 3 on real Samsung Galaxy S22 Ultra video:
  1. Extracts frames from data/samsung_room.mp4 (or pulls newest video via ADB).
  2. Computes Mission 0 instrument floor from distant frame pairs.
  3. Measures per-step link gain g_k and scene-normalized optical sharpness (catching motion blur).
  4. Compares Naive Multiplicative Chain (destroyed by hand-shake blur & redundancy, Mission 3)
     against AffectSplat Gatekeeper (skips redundant frames + rejects weak links).
  5. Saves Golden Keyframes to data/keyframes/ and plots figures/hour1_chain_filter.png.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.affect_math import (
    compute_instrument_floor,
    cosine_sim,
    floor_corrected_similarity,
    propagate_multiplicative_chain,
)

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RAW_FRAMES_DIR = DATA_DIR / "raw_frames"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
VIDEO_PATH = DATA_DIR / "samsung_room.mp4"
ADB_BIN = ROOT / "venv" / "platform-tools" / "adb"
FFMPEG_BIN = "/opt/homebrew/bin/ffmpeg"


def ensure_video_exists() -> Path:
    if ADB_BIN.exists():
        try:
            out = subprocess.check_output(
                [str(ADB_BIN), "shell", "ls -t /sdcard/DCIM/Camera/*.mp4 2>/dev/null | head -n 1"],
                text=True,
                timeout=5,
            ).strip()
            if out and out.endswith(".mp4") and not VIDEO_PATH.exists():
                print(f"[Hour 1] Pulling newest video from Samsung S22 Ultra: {out}")
                subprocess.check_call([str(ADB_BIN), "pull", out, str(VIDEO_PATH)])
        except Exception as exc:
            print(f"[Hour 1] ADB note: {exc}")

    if VIDEO_PATH.exists() and VIDEO_PATH.stat().st_size > 10_000:
        print(f"[Hour 1] Using video at {VIDEO_PATH} ({VIDEO_PATH.stat().st_size / 1e6:.2f} MB)")
        return VIDEO_PATH

    raise FileNotFoundError(f"No video found at {VIDEO_PATH} yet.")


def extract_frames(video_path: Path, fps: int = 15, max_frames: int = 240) -> list[Path]:
    if RAW_FRAMES_DIR.exists():
        shutil.rmtree(RAW_FRAMES_DIR)
    RAW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale=480:-2",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "2",
        str(RAW_FRAMES_DIR / "frame_%04d.jpg"),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return sorted(RAW_FRAMES_DIR.glob("frame_*.jpg"))


def compute_frame_features_and_sharpness(frame_paths: list[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes:
      1. Spatial-color + gradient embeddings for frame similarity.
      2. Local-window normalized sharpness so moving from a text-heavy laptop screen
         to a smooth wall still catches real motion-blur dips without false alarms!
    """
    embeddings = []
    raw_edge_energy = []

    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        small = img.resize((32, 32), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0

        gx = np.diff(arr, axis=1, prepend=arr[:, :1, :])
        gy = np.diff(arr, axis=0, prepend=arr[:1, :, :])
        feat = np.concatenate([arr.flatten(), gx.flatten() * 1.5, gy.flatten() * 1.5])
        embeddings.append(feat)

        gray = np.asarray(img.convert("L"), dtype=np.float32)
        lap = np.abs(
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        # Ratio of 98th percentile edge strength to mean intensity (invariant to room brightness)
        edge_score = float(np.percentile(lap, 98) / (np.mean(gray) + 10.0))
        raw_edge_energy.append(edge_score)

    emb_arr = np.vstack(embeddings)
    edge_arr = np.array(raw_edge_energy, dtype=np.float64)

    # Compare each frame's sharpness to its local 1.5-second neighborhood envelope
    # so sudden camera shakes/twitches show up as sharp downward spikes!
    local_envelope = uniform_filter1d(edge_arr, size=15, mode="nearest")
    local_envelope = np.maximum(local_envelope, 1e-6)
    relative_sharpness = np.clip((edge_arr / local_envelope) * 0.92, 0.20, 0.99)

    return emb_arr, relative_sharpness


def run_hour1_experiment() -> dict:
    t0 = time.monotonic()
    video_path = ensure_video_exists()
    frame_paths = extract_frames(video_path, fps=15, max_frames=240)
    n_total = len(frame_paths)

    embeddings, sharpness = compute_frame_features_and_sharpness(frame_paths)
    floor = compute_instrument_floor(embeddings, min_gap=max(10, n_total // 4))

    # Reject bottom ~15% sharpest motion-blur dips (Mission 3 Weak Links)
    sharp_thresh = float(np.percentile(sharpness, 16))

    naive_steps, _ = propagate_multiplicative_chain(
        embeddings,
        sharpness,
        floor=floor,
        filter_enabled=False,
    )

    filtered_steps, keyframe_indices = propagate_multiplicative_chain(
        embeddings,
        sharpness,
        floor=floor,
        redundancy_threshold=0.86,
        weak_link_threshold=0.45,
        sharpness_threshold=sharp_thresh,
        filter_enabled=True,
    )

    kf_gains = [s.gain_g for s in filtered_steps if s.status == "KEYFRAME"]
    cal_gains = kf_gains[: min(5, max(1, len(kf_gains) // 3))]
    g_hat = float(np.mean(cal_gains)) if cal_gains else 0.91

    if KEYFRAMES_DIR.exists():
        shutil.rmtree(KEYFRAMES_DIR)
    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for rank, f_idx in enumerate(keyframe_indices):
        shutil.copy2(frame_paths[f_idx], KEYFRAMES_DIR / f"kf_{rank:03d}_frame_{f_idx:04d}.jpg")

    elapsed = time.monotonic() - t0
    weak_rejected = [s.frame_idx for s in filtered_steps if s.status == "WEAK_LINK_REJECTED"]
    redundant_skipped = [s.frame_idx for s in filtered_steps if s.status == "REDUNDANT_SKIP"]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), dpi=140)

    frames_x = np.arange(n_total)
    axes[0].plot(frames_x, sharpness, color="#38bdf8", lw=1.8, label="Local Optical Sharpness s_blur(x_k)")
    axes[0].axhline(sharp_thresh, color="#f87171", ls="--", lw=1.3, label=f"Mission 3 Weak-Link Cutoff ({sharp_thresh:.2f})")
    if weak_rejected:
        axes[0].scatter(
            weak_rejected,
            sharpness[weak_rejected],
            color="#ef4444",
            s=50,
            zorder=5,
            label=f"Rejected Motion-Blur Weak Links ({len(weak_rejected)} frames)",
        )
    axes[0].scatter(
        keyframe_indices,
        sharpness[keyframe_indices],
        color="#22c55e",
        s=55,
        zorder=4,
        label=f"Golden Keyframes Kept ({len(keyframe_indices)} frames across full scan)",
    )
    axes[0].set_title(
        f"Hour 1 'Chain Diet': {n_total} Raw Samsung Frames → {len(keyframe_indices)} Golden Keyframes "
        f"({100.0 * (1.0 - len(keyframe_indices)/n_total):.1f}% Pruned in {elapsed:.2f}s)"
    )
    axes[0].set_xlabel("Samsung S22 Ultra Video Frame Index (15 fps)")
    axes[0].set_ylabel("Scene-Normalized Sharpness")
    axes[0].legend(loc="lower left", fontsize=8.5)
    axes[0].grid(True, alpha=0.25)

    naive_I = [s.cumulative_I for s in naive_steps]
    filt_I = [s.cumulative_I for s in filtered_steps]
    axes[1].plot(frames_x, naive_I, color="#ef4444", lw=2.0, ls=":", label="Naive All-Frame Chain (Collapses to 0 from Redundancy & Blur)")
    axes[1].plot(frames_x, filt_I, color="#22c55e", lw=2.5, label="AffectSplat Filtered Chain I(n) = Π g_k (Ours)")

    pred_I = [g_hat ** r for r in range(len(keyframe_indices))]
    axes[1].plot(
        keyframe_indices,
        pred_I,
        color="#facc15",
        lw=1.8,
        ls="--",
        marker="o",
        ms=4,
        label=f"Mission 1 Predicted Decay g_hat^(k-1) (g_hat={g_hat:.3f}, floor={floor:.2f})",
    )
    axes[1].set_title("Multiplicative Trust Propagation I(n): Naive Collapse vs. AffectSplat Gatekeeper")
    axes[1].set_xlabel("Samsung S22 Ultra Video Frame Index")
    axes[1].set_ylabel("Propagated Seed Trust I(n)")
    axes[1].legend(loc="upper right", fontsize=8.5)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig_path = FIG_DIR / "hour1_chain_filter.png"
    fig.savefig(fig_path)
    plt.close(fig)

    summary = {
        "video_path": str(video_path),
        "total_raw_frames": n_total,
        "golden_keyframes_kept": len(keyframe_indices),
        "redundant_frames_skipped": len(redundant_skipped),
        "weak_links_rejected": len(weak_rejected),
        "compression_ratio_pct": round(100.0 * (1.0 - len(keyframe_indices) / n_total), 1),
        "mission0_floor": round(floor, 4),
        "mission1_g_hat": round(g_hat, 4),
        "final_naive_I": round(float(naive_I[-1]), 6),
        "final_affectsplat_I": round(float(filt_I[-1]), 4),
        "cpu_wall_time_seconds": round(elapsed, 3),
        "keyframe_indices": keyframe_indices,
        "figure": str(fig_path),
    }
    (DATA_DIR / "hour1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_hour1_experiment()
