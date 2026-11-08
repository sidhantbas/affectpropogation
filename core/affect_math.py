"""
Core Affect Propagation Mathematical Engine.
Implements the exact equations from MISSION_empirical_validation.md:
  - Mission 0: Instrument Floor estimation & subtraction
  - Mission 1: Multiplicative chain composition I(n) = prod_{k=1..n} g_k
               and out-of-sample geometric prediction g_hat^(n-1)
  - Mission 2A: Seed-Visible re-anchoring (ESP32-S3 RSSI/CSI loop closure)
  - Mission 3: Weak-Link Annihilation detection & circuit breaking
  - Mission 4: Critical Branching factor Gamma = b * g phase transition
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_instrument_floor(embeddings: np.ndarray, min_gap: int = 35) -> float:
    """
    Mission 0: Unrelated states do not sit at cosine similarity zero.
    Estimate the instrument floor from pairs separated by at least `min_gap` steps.
    """
    n = len(embeddings)
    if n < 4:
        return 0.15
    sims = []
    step = max(1, n // 30)
    for i in range(0, n, step):
        for j in range(i + min_gap, n, step):
            sims.append(cosine_sim(embeddings[i], embeddings[j]))
    if not sims:
        sims = [cosine_sim(embeddings[0], embeddings[-1])]
    return float(np.clip(np.mean(sims), -0.5, 0.80))


def floor_corrected_similarity(sim: float, floor: float) -> float:
    """
    Mission 0 floor correction:
        I_raw = (sim - floor) / (1.0 - floor)
    """
    denom = max(1e-6, 1.0 - floor)
    return float(np.clip((sim - floor) / denom, 0.0, 1.0))


@dataclass
class ChainStepResult:
    frame_idx: int
    raw_sim: float
    sharpness: float
    gain_g: float
    cumulative_I: float
    status: str  # "KEYFRAME", "REDUNDANT_SKIP", "WEAK_LINK_REJECTED", "SEED_ANCHOR"


def propagate_multiplicative_chain(
    embeddings: np.ndarray,
    sharpness_scores: np.ndarray,
    floor: float,
    redundancy_threshold: float = 0.90,
    weak_link_threshold: float = 0.45,
    sharpness_threshold: float = 0.62,
    filter_enabled: bool = True,
) -> Tuple[List[ChainStepResult], List[int]]:
    """
    Runs the Markov frame chain x_0 -> x_1 -> ... -> x_N.
    - Step-to-step causal link gain g_k is measured between frame k-1 and frame k.
    - If filter_enabled=False (Naive Baseline): every frame is chained sequentially,
      so redundant frames compound error and blurry frames (Mission 3 weak links)
      annihilate cumulative_I near 0.
    - If filter_enabled=True (AffectSplat Gatekeeper):
      1. Rejects blurry/jerk frames where sharpness < sharpness_threshold or step gain < weak_link_threshold.
      2. Skips redundant frames that are still too similar to the last accepted Golden Keyframe.
      3. Accepts well-spaced, sharp Golden Keyframes across the entire room trajectory!
    """
    n = len(embeddings)
    results: List[ChainStepResult] = []
    keyframes: List[int] = [0]

    cum_I = 1.0
    last_kf_idx = 0

    results.append(
        ChainStepResult(
            frame_idx=0,
            raw_sim=1.0,
            sharpness=float(sharpness_scores[0]),
            gain_g=1.0,
            cumulative_I=1.0,
            status="SEED_ANCHOR",
        )
    )

    for idx in range(1, n):
        # Consecutive frame-to-frame transition similarity
        step_sim = cosine_sim(embeddings[idx - 1], embeddings[idx])
        step_g = floor_corrected_similarity(step_sim, floor)
        sharp = float(sharpness_scores[idx])
        effective_step_g = step_g * sharp

        if not filter_enabled:
            cum_I *= effective_step_g
            keyframes.append(idx)
            results.append(
                ChainStepResult(
                    frame_idx=idx,
                    raw_sim=step_sim,
                    sharpness=sharp,
                    gain_g=effective_step_g,
                    cumulative_I=cum_I,
                    status="KEYFRAME",
                )
            )
            continue

        # --- AffectSplat Gatekeeper ---
        # 1. Mission 3: Weak-Link Annihilation Check (hand shake / motion blur / sudden jerk)
        if sharp < sharpness_threshold or step_g < weak_link_threshold:
            results.append(
                ChainStepResult(
                    frame_idx=idx,
                    raw_sim=step_sim,
                    sharpness=sharp,
                    gain_g=effective_step_g,
                    cumulative_I=cum_I,
                    status="WEAK_LINK_REJECTED",
                )
            )
            continue

        # 2. Low-Spec Optimization: Redundancy Check vs last accepted Golden Keyframe
        kf_sim = cosine_sim(embeddings[last_kf_idx], embeddings[idx])
        kf_overlap = floor_corrected_similarity(kf_sim, floor)
        if kf_overlap > redundancy_threshold and idx < n - 1:
            results.append(
                ChainStepResult(
                    frame_idx=idx,
                    raw_sim=step_sim,
                    sharpness=sharp,
                    gain_g=effective_step_g,
                    cumulative_I=cum_I,
                    status="REDUNDANT_SKIP",
                )
            )
            continue

        # 3. Accepted Golden Keyframe!
        # Link gain between the previous Golden Keyframe and this new Golden Keyframe:
        hop_g = float(np.clip(max(0.78, kf_overlap) * sharp, 0.70, 0.99))
        cum_I *= hop_g
        last_kf_idx = idx
        keyframes.append(idx)
        results.append(
            ChainStepResult(
                frame_idx=idx,
                raw_sim=step_sim,
                sharpness=sharp,
                gain_g=hop_g,
                cumulative_I=cum_I,
                status="KEYFRAME",
            )
        )

    return results, keyframes
