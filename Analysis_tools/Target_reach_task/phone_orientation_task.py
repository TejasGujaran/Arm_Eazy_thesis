"""
phone_orientation_task.py
Target-reaching task driven by phone tilt (roll/pitch), with a pluggable
orientation filter: naive / complementary / madgwick / mahony.

Cursor velocity = estimated tilt angle x gain (rate control, same style as
the gamepad task). Since the neutral/level position should read as zero
tilt, any residual drift in the orientation estimate shows up directly as
unwanted cursor drift when the phone is held still - this is what makes
the naive filter's failure mode visible and comparable to the others.

Phone setup (phyphox): you need an experiment exposing BOTH accelerometer
(accX/Y/Z) and gyroscope (gyrX/Y/Z) simultaneously. If your installed
experiment only exposes one, use phyphox's experiment editor to build a
combined one, or check "Sensor Fusion" / "Inclination"-style experiments
which often expose both raw sensors alongside the fused output.

Requirements:
    pip install pygame requests numpy ahrs

Usage:
    python phone_orientation_task.py --ip 192.168.1.42 --filter naive
    python phone_orientation_task.py --ip 192.168.1.42 --filter madgwick
    python phone_orientation_task.py --ip 192.168.1.42 --filter mahony --out mahony_run1.csv
"""

import argparse
import sys
import threading
import time

import requests

from task_common import TargetTask
from imu_filters import ORIENTATION_FILTERS

DEFAULT_VARS = "gyrX,gyrY,gyrZ,gyr_time,accX,accY,accZ,acc_time"


class PhonePoller:
    def __init__(self, base_url, variables, poll_hz=60.0):
        self.base_url = base_url
        self.variables = variables
        self.poll_interval = 1.0 / poll_hz
        self._lock = threading.Lock()
        self._buffer = []
        self._stop = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop = True

    def _run(self):
        query = "&".join(self.variables)
        url = f"{self.base_url}/get?{query}"
        while not self._stop:
            try:
                resp = requests.get(url, timeout=1.0)
                resp.raise_for_status()
                data = resp.json().get("buffer", {})
                series = {v: data.get(v, {}).get("buffer", []) for v in self.variables}
                lengths = [len(s) for s in series.values() if s is not None]
                n_new = min(lengths) if lengths else 0
                if n_new > 0:
                    rows = [{v: series[v][i] for v in self.variables} for i in range(n_new)]
                    with self._lock:
                        self._buffer.extend(rows)
            except requests.exceptions.RequestException as e:
                self.error = str(e)
            time.sleep(self.poll_interval)

    def get_new_samples(self):
        with self._lock:
            samples, self._buffer = self._buffer, []
        return samples


def main():
    parser = argparse.ArgumentParser(description="Phone orientation (tilt) target-reaching task.")
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vars", type=str, default=DEFAULT_VARS,
                         help="gyrX,gyrY,gyrZ[,gyrT],accX,accY,accZ - time var optional but recommended")
    parser.add_argument("--filter", type=str, default="naive", choices=list(ORIENTATION_FILTERS.keys()))
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--gain", type=float, default=4000.0, help="tilt angle (rad) -> px/s cursor speed")
    parser.add_argument("--out", type=str, default=None, help="Output CSV (default: phone_orientation_<filter>.csv)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    out_csv = args.out or f"phone_orientation_{args.filter}.csv"

    variables = [v.strip() for v in args.vars.split(",") if v.strip()]
    time_var = next((v for v in variables if v.endswith("T")), None)
    gyr_vars = [v for v in variables if v.startswith("gyr")]
    acc_vars = [v for v in variables if v.startswith("acc")]
    if len(gyr_vars) < 3 or len(acc_vars) < 3:
        print("Need gyrX,gyrY,gyrZ AND accX,accY,accZ in --vars for orientation fusion.")
        sys.exit(1)
    gx_var, gy_var, gz_var = gyr_vars[:3]
    ax_var, ay_var, az_var = acc_vars[:3]

    base_url = f"http://{args.ip}:{args.port}"
    print(f"Connecting to phyphox at {base_url} ...")
    try:
        r = requests.get(f"{base_url}/get?{'&'.join(variables)}", timeout=2.0)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Could not reach phyphox: {e}")
        sys.exit(1)
    print("Connected.\n")

    poller = PhonePoller(base_url, variables)
    poller.start()

    orient_filter = ORIENTATION_FILTERS[args.filter]()

    extra_header = ["roll", "pitch", "filter"]
    task = TargetTask(out_csv, n_targets=args.targets, seed=args.seed, extra_header=extra_header)

    print(f"Filter: {args.filter}")
    print(f"Reach {args.targets} targets by tilting the phone. Hold cursor inside each for 0.6s. ESC to quit.\n")

    import pygame
    last_time = time.time()
    roll, pitch = 0.0, 0.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        now = time.time()
        dt_frame = now - last_time
        last_time = now

        samples = poller.get_new_samples()
 
        if samples:
            prev_t = None
            for s in samples:
                gx = float(s.get(gx_var, 0.0) or 0.0)
                gy = float(s.get(gy_var, 0.0) or 0.0)
                gz = float(s.get(gz_var, 0.0) or 0.0)
                ax = float(s.get(ax_var, 0.0) or 0.0)
                ay = float(s.get(ay_var, 0.0) or 0.0)
                az = float(s.get(az_var, 0.0) or 0.0)

                if time_var and s.get(time_var) is not None:
                    t = float(s[time_var])
                    dt = (t - prev_t) if prev_t is not None else dt_frame / len(samples)
                    prev_t = t
                else:
                    dt = dt_frame / len(samples)
                dt = max(dt, 1e-4)

                roll, pitch = orient_filter.update(gx, gy, gz, ax, ay, az, dt)

            task.nudge_cursor(pitch * args.gain * dt_frame, roll * args.gain * dt_frame)
        else:
            task.nudge_cursor(pitch * args.gain * dt_frame, roll * args.gain * dt_frame)

        still_running = task.update(extra_values=[round(roll, 4), round(pitch, 4), args.filter])

        hud = [
            f"filter: {args.filter}",
            f"roll={math_degrees(roll):+.1f}deg  pitch={math_degrees(pitch):+.1f}deg",
            f"targets completed: {len(task.results)}/{args.targets}",
        ]
        task.draw(hud_lines=hud)
        task.clock.tick(60)

        if not still_running:
            running = False

    poller.stop()
    task.print_summary()
    task.close()


def math_degrees(rad):
    return rad * 180.0 / 3.14159265358979


if __name__ == "__main__":
    main()
