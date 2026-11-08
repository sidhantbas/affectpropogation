"""
AffectSplat Hackathon Guru CLI.
Interactive mentor, 5-hour milestone tracker, and plain-English concept teacher.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "data" / "hackathon_state.json"
FIGURES_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

console = Console()

HOURS = {
    1: {
        "title": "Hour 1: The 'Chain Diet' (Video -> Golden Keyframes)",
        "domain": "Computer Vision & Multiplicative Decay (Missions 0, 1, 3)",
        "artifact": "figures/hour1_chain_filter.png",
        "lesson": """
### Guru Lesson 1: Why Your Phone Video Needs a "Chain Diet"

Imagine filming a 15-second video of your desk with your Samsung phone.
At 30 frames per second, that is **450 photos**.
If you give all 450 photos to a 3D program on a normal laptop, it freezes. Why?
1. **Redundancy:** Frame 1 and Frame 2 are 99% identical (`g > 0.96`). You don't learn anything new.
2. **Weak Links (Motion Blur):** At Frame 87, your hand twitched. The photo is blurry (`g ≈ 0`).

**The Affect Propagation Trick:**
In our math, confidence multiplies down the chain:
`Trust(n) = g_1 × g_2 × ... × g_n`

If you multiply a chain of numbers by `0` (one blurry photo), the whole chain becomes `0`!
So in Hour 1, we build a **Gatekeeper** (`world_model/exp01_chain_filter.py`):
- It shrinks every frame to a tiny thumbnail on your CPU (takes milliseconds).
- It measures the **Floor** (how similar two random frames are, Mission 0).
- It skips duplicate frames (`g > 0.95`) and **deletes blurry Weak Links** (`g < 0.35`).
- **Result:** 450 frames become **~25 crystal-clear Golden Keyframes**, saving 90% of RAM and CPU!
""",
    },
    2: {
        "title": "Hour 2: The Bare ESP32 'Seed Anchor' (Zero Extra Sensors)",
        "domain": "Embedded Systems & Loop Closure (Mission 2A)",
        "artifact": "figures/hour2_esp32_anchor.png",
        "lesson": """
### Guru Lesson 2: Turning a Bare ESP32 into a 3D Anchor (With $0 Spent)

You might wonder: *"My ESP32 doesn't have a LiDAR or camera attached—how can it help build a 3D world?"*

A bare ESP32 chip actually has **three built-in superpowers** with zero external wires:
1. **Wi-Fi Signal Strength (RSSI):** When your Samsung phone is right next to the ESP32, the signal is strong (`-35 dBm`). When you walk across the room, it drops (`-75 dBm`).
2. **Bare-Pin Capacitive Touch:** Pins like `GPIO 4` (`T0`) detect your finger's electricity the instant you touch the bare metal pin!
3. **The `BOOT` Button:** Built right onto the board.

**How This Solves 3D Drift (Mission 2A):**
As you walk around filming with your Samsung phone, your 3D trust slowly fades (`0.90 -> 0.81 -> 0.73...`).
By placing the ESP32 on your desk as **Seed `x_0`**, whenever you walk back near the desk, the Wi-Fi RSSI spikes!
In Mission 2A ("Seed Visible"), a direct connection to the Seed **resets decay back to `1.0`**.
""",
    },
    3: {
        "title": "Hour 3: 'Lite-Splat' — Low-Spec 3D Gaussian World in <60s",
        "domain": "3D Gaussian Splatting on CPU (Mission 1 Weighting)",
        "artifact": "figures/hour3_lite_splat.png",
        "lesson": """
### Guru Lesson 3: What the Heck is a "3D Gaussian Splat"?

Forget complicated math papers. Think of **Pointillism painting** or a **can of spray paint**:
- A normal 2D photo is a flat grid of colored squares (pixels).
- If we know *roughly* how far away each pixel is (depth), we can push each pixel out into 3D space (`X, Y, Z`).
- Instead of drawing hard squares in 3D (which look blocky with gaps), we draw each point as a **soft, fuzzy 3D oval ("a Gaussian Splat")**.

Every Splat is just **4 simple things**:
1. **Position (`x, y, z`):** Where it floats in the room.
2. **Color (`r, g, b`):** Taken straight from your Samsung phone pixel.
3. **Size (`scale`):** How puff-ball big it is.
4. **Trust (`opacity`):** How solid vs. see-through it is!

**Our Low-Spec Secret:** We set each Splat's **opacity = `I(n)`** (our Affect Propagation trust score from Hour 1!). High-trust frames near your ESP32 paint solid, sharp splats; drifted frames paint soft, translucent splats so they never ruin the room!
""",
    },
    4: {
        "title": "Hour 4: Live 3D Affect Waves (Critical Branching Γ = b · g)",
        "domain": "World Models & Phase Transitions (Mission 4)",
        "artifact": "figures/hour4_critical_wave.png",
        "lesson": """
### Guru Lesson 4: From a "Dead 3D Photo" to a "Living World Model"

A normal 3D scan of a room is frozen in time—like a statue.
A **World Model** reacts when something happens in the real world.

Once our 3D Gaussian room is built, your **ESP32 stays on your desk** as node `x_0`.
Every 3D splat cluster connects to `b` neighbors (branching factor `b`), passing signals with strength `g`.
Look at the magic number from Mission 4:
`Γ = b × g`

- **When `Γ < 1` (Local Ripple):** You lightly tap the ESP32 touch pin. Each hop gets weaker. The color/energy wave lights up the splats right around your desk and naturally stops.
- **When `Γ >= 1` (Critical Runaway):** You hold the ESP32 button (or trigger an alert). Now each splat wakes up *more* than 1 neighbor. The wave crosses the tipping point and illuminates the **entire 3D room**!
""",
    },
    5: {
        "title": "Hour 5: The Low-Spec Research Benchmark & GitHub Polish",
        "domain": "Empirical Benchmarking & Reproducible Science (Mission 5)",
        "artifact": "figures/hour5_benchmark.png",
        "lesson": """
### Guru Lesson 5: Making Your GitHub Repo Stand Out to Researchers

Anyone can upload messy hackathon code. A **real research repo** proves its claim with numbers.
In Hour 5, we run a head-to-head benchmark on your Mac comparing:
- **Method A (Naive Brute-Force):** Keep all frames, ignore weak links, equal opacity.
- **Method B (AffectSplat - Ours):** Floor-corrected `I(n)` pruning + Weak-Link Gate + ESP32 Seed Anchor.

We automatically generate a publication-ready table and chart comparing:
1. **Frame Count & Memory Used (MB)**
2. **Wall-Clock Processing Time (seconds)**
3. **Downstream Geometric Coherence (`I(N)` at end of trajectory)**
""",
    },
}


def load_state() -> Dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        state = {
            "started_at": time.time(),
            "completed_hours": [],
            "benchmarks": {},
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))
        return state
    return json.loads(STATE_FILE.read_text())


def save_state(state: Dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def cmd_status() -> None:
    state = load_state()
    elapsed_min = int((time.time() - state["started_at"]) / 60)
    remaining_min = max(0, 300 - elapsed_min)

    console.print(
        Panel.fit(
            "[bold cyan]AffectSplat: 5-Hour Low-Spec 3D World Model Hackathon[/bold cyan]\n"
            "[dim]Hardware: 1x Samsung Phone + 1x Bare ESP32 + Mac | Budget: $0[/dim]\n"
            f"⏱️  Elapsed: [bold yellow]{elapsed_min}m[/bold yellow] / 300m   "
            f"Remaining: [bold green]{remaining_min}m[/bold green]",
            border_style="cyan",
        )
    )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Hour", style="bold", width=8)
    table.add_column("Sprint Milestone", width=44)
    table.add_column("Target Artifact", width=30)
    table.add_column("Status", width=14)

    completed = set(state.get("completed_hours", []))
    next_up_shown = False

    for h, info in HOURS.items():
        art_path = ROOT / info["artifact"]
        is_done = (h in completed) or art_path.exists()
        if is_done:
            status = "[bold green]✅ PASSED[/bold green]"
        elif not next_up_shown:
            status = "[bold yellow]🔥 ACTIVE[/bold yellow]"
            next_up_shown = True
        else:
            status = "[dim]🔒 LOCKED[/dim]"

        table.add_row(f"Hour {h}", info["title"], info["artifact"], status)

    console.print(table)
    console.print(
        "\n[bold]Next Mentor Commands:[/bold]\n"
        "  • Read plain-English lesson:  [cyan]./venv/bin/python hackathon.py learn 1[/cyan]\n"
        "  • Verify checkpoint artifact: [cyan]./venv/bin/python hackathon.py check 1[/cyan]"
    )


def cmd_learn(hour: int) -> None:
    if hour not in HOURS:
        console.print(f"[red]Choose an hour between 1 and 5 (got {hour}).[/red]")
        sys.exit(1)
    info = HOURS[hour]
    console.print(
        Panel(
            Markdown(info["lesson"]),
            title=f"[bold yellow]{info['title']}[/bold yellow]",
            subtitle=f"[dim]{info['domain']}[/dim]",
            border_style="yellow",
        )
    )


def cmd_check(hour: int) -> None:
    if hour not in HOURS:
        console.print(f"[red]Choose an hour between 1 and 5 (got {hour}).[/red]")
        sys.exit(1)
    state = load_state()
    info = HOURS[hour]
    art_path = ROOT / info["artifact"]
    if art_path.exists():
        if hour not in state["completed_hours"]:
            state["completed_hours"].append(hour)
            save_state(state)
        console.print(
            Panel(
                f"🎉 [bold green]CHECKPOINT {hour} PASSED![/bold green]\n"
                f"Verified artifact: [underline]{info['artifact']}[/underline]\n"
                f"Ready for Hour {min(5, hour + 1)}!",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold yellow]Checkpoint {hour} not complete yet.[/bold yellow]\n"
                f"Missing artifact: [cyan]{info['artifact']}[/cyan]\n"
                f"Let's run the Hour {hour} experiment to generate it!",
                border_style="yellow",
            )
        )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "status":
        cmd_status()
    elif sys.argv[1] == "learn":
        hour = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        cmd_learn(hour)
    elif sys.argv[1] == "check":
        hour = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        cmd_check(hour)
    else:
        cmd_status()


if __name__ == "__main__":
    main()

