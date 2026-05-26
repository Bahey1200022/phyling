"""Real-time posture alarm — triggers when pitch exceeds threshold.

Usage:
    python posture_alarm_high.py --name NanoPhyling_01 --threshold 50.0

Plays an audible beep when the pitch angle (computed from acc_x/y/z) rises above
the given threshold (in degrees). A short cooldown prevents spamming the alarm.

Pitch is defined as atan2(acc_y, sqrt(acc_x² + acc_z²)) in degrees.
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
    fade = np.linspace(1.0, 0.0, tone.size)
    return (tone * fade).astype("float32"), samplerate


def run_alarm_loop(beep_buf, sr):
    sd.play(beep_buf, sr)
    sd.wait()


def compute_pitch(df):
    return np.degrees(np.arctan2(df["acc_y"], np.sqrt(df["acc_x"] ** 2 + df["acc_z"] ** 2)))


def main():
    parser = argparse.ArgumentParser(description="NanoPhyling posture alarm — high pitch")
    parser.add_argument("--name", help="BLE device name (e.g. NanoPhyling_01)", required=False)
    parser.add_argument("--threshold", type=float, default=50.0, help="pitch threshold in degrees (alarm when pitch > threshold)")
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
            if (pitch.abs() > args.threshold).any():
                now = time.time()
                if now - last_play["t"] < args.cooldown:
                    return
                last_play["t"] = now
                current_pitch = pitch.iloc[-1]
                print(f"current_pitch={current_pitch:.1f}° ")
                print(f"Posture alert! pitch={current_pitch:.1f}° > {args.threshold}°")
                threading.Thread(target=run_alarm_loop, args=(beep_buf, sr), daemon=True).start()
        except Exception as e:
            print(f"Alarm callback error: {e}")

    device.on_data(on_data)

    print(f"Connecting to {args.name} and monitoring pitch, alarm when pitch > {args.threshold}°...")
    try:
        device.run(None)
    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == "__main__":
    main()
