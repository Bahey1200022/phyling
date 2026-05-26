"""Simple real-time posture alarm demo for NanoPhyling devices.

Usage:
    python posture_alarm.py --name NanoPhyling_01 --threshold 80.0

The script connects to the BLE device, registers a data callback and plays
an audible beep when the pitch angle (computed from acc_x/y/z) drops below
the given threshold (in degrees). A short cooldown prevents spamming the alarm.

Pitch is defined as atan2(acc_x, sqrt(acc_y² + acc_z²)) in degrees.
At rest upright the sensor reads ~90°; leaning forward reduces this value.
"""
import argparse
import time
import threading

import numpy as np
import sounddevice as sd

from phyling.ble.nanophyling import NanoPhyling


def make_beep(samplerate=22050, duration=0.25, freq=880.0, amp=0.3):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    tone = amp * np.sin(2 * np.pi * freq * t)
    # simple fade out to avoid clicks
    fade = np.linspace(1.0, 0.0, tone.size)
    return (tone * fade).astype("float32"), samplerate


def run_alarm_loop(beep_buf, sr):
    sd.play(beep_buf, sr)
    sd.wait()


def compute_pitch(df):
    """Return pitch angle in degrees from accelerometer columns.
    acc_y carries gravity when upright (~90°); acc_x changes as posture shifts."""
    return np.degrees(np.arctan2(df["acc_y"], np.sqrt(df["acc_x"] ** 2 + df["acc_z"] ** 2)))


def main():
    parser = argparse.ArgumentParser(description="NanoPhyling posture alarm demo")
    parser.add_argument("--name", help="BLE device name (e.g. NanoPhyling_01)", required=False)
    parser.add_argument("--threshold", type=float, default=60.0, help="pitch threshold in degrees (alarm when pitch < threshold)")
    parser.add_argument("--cooldown", type=float, default=3.0, help="seconds between alarms")
    args = parser.parse_args()

    device = NanoPhyling(ble_name="NanoPhyling_38" if args.name is None else args.name)

    beep_buf, sr = make_beep()
    last_play = {"t": 0.0}

    def on_data(df):
        try:
            required = {"acc_x", "acc_y", "acc_z"}
            if not required.issubset(df.columns):
                return
            pitch = compute_pitch(df)
            if (pitch.abs() < args.threshold).any():
                now = time.time()
                if now - last_play["t"] < args.cooldown:
                    return
                last_play["t"] = now
                current_pitch = pitch.iloc[-1]
                print(f"Bad posture detected! pitch={current_pitch:.1f}° < {args.threshold}°")
                # play in background thread to avoid blocking BLE loop
                threading.Thread(target=run_alarm_loop, args=(beep_buf, sr), daemon=True).start()
        except Exception as e:
            print(f"Alarm callback error: {e}")

    device.on_data(on_data)

    print(f"Connecting to {args.name} and monitoring pitch, alarm when pitch < {args.threshold}°...")
    try:
        device.run(None)
    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == "__main__":
    main()
