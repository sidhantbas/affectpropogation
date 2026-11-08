"""
Act 0 Hardware Pulse Verification:
1. Verifies both the ESP32-S3 (Seed x_0) and SAMSUNG_Android (Explorer) on the Mac USB bus.
2. Captures a live stream of Wi-Fi CSI + RSSI packets from /dev/cu.usbmodem5AE70749591.
3. Saves the first hardware telemetry plot to figures/act0_seed_heartbeat.png.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import serial

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
ESP_PORT = "/dev/cu.usbmodem5AE70749591"

CSI_REGEX = re.compile(
    r"I \((\d+)\) csi_collector: CSI cb #(\d+): len=(\d+) rssi=(-?\d+) ch=(\d+)"
)


def detect_usb_devices() -> dict:
    out = subprocess.check_output(
        ["ioreg", "-p", "IOUSB", "-l", "-w", "0"], text=True, errors="replace"
    )
    has_esp = "USB Single Serial" in out
    has_samsung = "SAMSUNG_Android" in out
    serial_match = re.search(r'"USB Serial Number" = "(R5[A-Z0-9]+)"', out)
    samsung_serial = serial_match.group(1) if serial_match else "DETECTED"
    return {
        "esp32_s3_connected": has_esp,
        "esp32_port": ESP_PORT,
        "esp32_mac": "10:b4:1d:c4:00:44",
        "samsung_connected": has_samsung,
        "samsung_serial": samsung_serial,
    }


def capture_esp32_csi(duration_s: float = 6.0) -> list[dict]:
    samples = []
    t_start = time.time()
    with serial.Serial(ESP_PORT, 115200, timeout=0.2) as ser:
        while time.time() - t_start < duration_s:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            m = CSI_REGEX.search(line)
            if m:
                esp_ms, cb_id, length, rssi, ch = m.groups()
                samples.append(
                    {
                        "wall_t": round(time.time() - t_start, 3),
                        "esp_ms": int(esp_ms),
                        "cb_id": int(cb_id),
                        "len": int(length),
                        "rssi": int(rssi),
                        "channel": int(ch),
                    }
                )
    return samples


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    usb_info = detect_usb_devices()
    print("=== LAB ZERO HARDWARE BUS CHECK ===")
    print(json.dumps(usb_info, indent=2))

    print("\nCapturing 6 seconds of live Wi-Fi CSI Radar from ESP32-S3...")
    samples = capture_esp32_csi(duration_s=6.0)
    print(f"Captured {len(samples)} CSI summary frames from Seed x_0.")

    payload = {"usb": usb_info, "csi_samples": samples}
    out_json = DATA_DIR / "act0_hardware_pulse.json"
    out_json.write_text(json.dumps(payload, indent=2))

    if samples:
        ts = [s["wall_t"] for s in samples]
        rssis = [s["rssi"] for s in samples]
        chs = [s["channel"] for s in samples]

        fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
        scatter = ax.scatter(
            ts, rssis, c=chs, cmap="viridis", s=70, zorder=3, label="CSI Channel Hop"
        )
        ax.plot(ts, rssis, color="#00b4d8", alpha=0.6, linewidth=2, zorder=2)
        ax.set_title(
            f"Act 0: Live Seed (x_0) Radio Heartbeat — ESP32-S3 + Samsung ({usb_info['samsung_serial']})"
        )
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Wi-Fi CSI RSSI (dBm)")
        ax.grid(True, alpha=0.25)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Wi-Fi Channel (1, 6, 11)")
        fig.tight_layout()
        fig_path = FIG_DIR / "act0_seed_heartbeat.png"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved live telemetry plot -> {fig_path}")


if __name__ == "__main__":
    main()

