#!/usr/bin/env bash
# Lab Zero 1-Command Wakeup Script
# Run this whenever you plug your Samsung S22 Ultra & ESP32-S3 back in!
set -e
cd "$(dirname "$0")"

echo "=== Lab Zero: Reconnecting Hardware & 3D World Portal ==="
./venv/bin/python hackathon.py status

if ./venv/platform-tools/adb devices | grep -q "device$"; then
    echo "✅ Samsung Galaxy S22 Ultra detected on USB! Re-opening port 8899 tunnel..."
    ./venv/platform-tools/adb reverse tcp:8899 tcp:8899
    ./venv/platform-tools/adb shell am start -a android.intent.action.VIEW -d "http://localhost:8899" >/dev/null 2>&1 || true
else
    echo "ℹ️  Samsung phone not plugged in yet (you can still view http://localhost:8899 on your Mac!)."
fi

echo "🚀 Starting Lab Zero 3D World Portal on http://localhost:8899 ..."
./venv/bin/python world_model/phone_portal.py
