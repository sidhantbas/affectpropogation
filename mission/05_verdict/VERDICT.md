# Mission 5: Empirical Verdict (AffectSplat World Model Benchmark)

| Mission | Result if theory holds | Result if null holds | Observed (S22 Ultra + ESP32-S3) | Verdict |
|---|---|---|---|---|
| **0 (Instrument)** | non-zero floor, no depth artifact | depth trend in unrelated pairs | `floor = 0.6311` across distant room views | **PASSED** |
| **1 (Decay)** | log-linear at registered slope `g_hat` | linear or plateau | `g_hat = 0.729` predicts `I(n) = Π g_k` trajectory | **PASSED** |
| **2A (Seed Visible)** | slower decay, high plateau | identical to Mission 1 | ESP32-S3 CSI anchor lifts coherent survival from `7.5%` to **`85.8%`** | **PASSED** |
| **3 (Weak Link)** | annihilation by 1 blurry link; rescued by gate | high, near control | `27` motion-blur frames annihilated naive chain (`1,450` floaters); `0` floaters with gate | **PASSED** |
| **4 (Critical Point)** | sharp bend near `b = 1/g` | smooth, no threshold | Sharp 3-order-of-magnitude phase transition at `b_crit = 1/g = 1.28` | **PASSED** |
