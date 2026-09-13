"""
phone_position_task.py
Target-reaching task driven by phone accelerometer, with a pluggable
position (dead-reckoning) filter: naive / zupt / kalman_zupt.

Run the same task once per filter to get directly comparable completion
time / path efficiency / final error numbers for your evaluation chapter.

Phone setup (phyphox): load "Acceleration (without g)" if available
(removes gravity via the OS linear-acceleration sensor). Enable remote
access, same network as this laptop.

Requirements:
    pip install pygame requests numpy ahrs

Usage:
    python phone_position_task.py --ip 192.168.1.42 --filter naive
    python phone_position_task.py --ip 192.168.1.42 --filter zupt
    python phone_position_task.py --ip 192.168.1.42 --filter kalman_zupt --out kalman_run1.csv
"""

import argparse
import sys
import time

import requests

from task_common import TargetTask
from imu_filtersV3 import POSITION_FILTERS, apply_accel_deadband, DirectionLock, DEFAULT_VELOCITY_DAMPING

DEFAULT_VARS = "linX,linY,lin_time,gyrX,gyrY,gyr_time"


class PhonePoller:
    import threading

    def __init__(self, base_url, variables, poll_hz=60.0):
        self.base_url = base_url
        self.variables = variables
        self.poll_interval = 1.0 / poll_hz
        self._lock = self.threading.Lock()
        self._buffer = []
        self._stop = False
        self.error = None
        self.thread = self.threading.Thread(target=self._run, daemon=True)

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
    parser = argparse.ArgumentParser(description="Phone position dead-reckoning target-reaching task.")
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vars", type=str, default=DEFAULT_VARS)
    parser.add_argument("--filter", type=str, default="naive", choices=list(POSITION_FILTERS.keys()))
    parser.add_argument("--targets", type=int, default=8)
    parser.add_argument("--gain", type=float, default=3000.0, help="accel(m/s^2) -> px/s^2 scale")
    parser.add_argument("--zupt-thresh", type=float, default=None,
                         help="Override ZUPT stillness threshold (default: filter class's calibrated default)")
    parser.add_argument("--zupt-window", type=float, default=None,
                         help="Override ZUPT stillness window (default: filter class's calibrated default)")
    parser.add_argument("--damping", type=float, default=None,
                         help="Velocity damping (1/s), higher = stops faster but bleeds off sustained "
                              "motion faster too (default: filter class's default, ~%.1f)" % DEFAULT_VELOCITY_DAMPING)
    parser.add_argument("--out", type=str, default=None, help="Output CSV (default: phone_position_<filter>.csv)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    out_csv = args.out or f"phone_position_{args.filter}.csv"

    variables = [v.strip() for v in args.vars.split(",") if v.strip()]
    time_var = next((v for v in variables if v.endswith("T") or v.endswith("_time")), None)
    accel_vars = [v for v in variables if v != time_var]
    ax_var, ay_var = accel_vars[0], accel_vars[1]

    filter_kwargs = {}
    if args.filter in ("zupt", "kalman_zupt", "kalman_zupt_bias", "glrt_zupt"):
        if args.damping is not None:
            filter_kwargs["damping"] = args.damping
    if args.filter in ("zupt", "kalman_zupt", "kalman_zupt_bias"):
        if args.zupt_thresh is not None:
            filter_kwargs["threshold"] = args.zupt_thresh
        if args.zupt_window is not None:
            filter_kwargs["window"] = args.zupt_window
    pos_filter = POSITION_FILTERS[args.filter](**filter_kwargs)
    needs_gyro = getattr(pos_filter, "needs_gyro", False)
    vel_lock = DirectionLock()

    # filters like glrt_zupt need gyro alongside accel to judge stillness;
    # extend the polled variable set (kept positionally aligned with
    # PhonePoller's zip-by-index assumption) rather than requiring the user
    # to pass --vars manually for every gyro-based filter
    gyro_vars = ["gyrX", "gyrY", "gyrZ"]
    if needs_gyro:
        variables = accel_vars + gyro_vars + ([time_var] if time_var else [])

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

    extra_header = ["accel_x", "accel_y", "vel_x", "vel_y", "zupt_active", "filter"]
    task = TargetTask(out_csv, n_targets=args.targets, seed=args.seed, extra_header=extra_header)

    print(f"Filter: {args.filter}")
    print(f"Reach {args.targets} targets. Hold cursor inside each for 0.6s. ESC to quit early.\n")

    import pygame
    last_time = time.time()
    running = True

    # vx/vy/active/ax_latest/ay_latest are tracked as loop-persistent locals
    # (not re-derived from pos_filter's internals each frame) so carry-over
    # between sensor updates works identically regardless of filter type --
    # Kalman-based filters store velocity in an internal state array with no
    # public vx/vy attribute, so the previous pos_filter.__dict__.get("vx", ...)
    # pattern silently returned 0.0 for those filters on every frame with no
    # new sample, effectively freezing the cursor between updates.
    ax_latest, ay_latest = 0.0, 0.0
    gx_latest, gy_latest, gz_latest = 0.0, 0.0, 0.0
    vx, vy, active = 0.0, 0.0, False

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
                ax = float(s.get(ax_var, 0.0) or 0.0)
                ay = float(s.get(ay_var, 0.0) or 0.0)
                ax_latest, ay_latest = ax, ay  # raw values, kept for logging/HUD
                ax_f, ay_f = apply_accel_deadband(ax, ay)  # deadbanded, fed to filter

                if time_var and s.get(time_var) is not None:
                    t = float(s[time_var])
                    dt = (t - prev_t) if prev_t is not None else dt_frame / len(samples)
                    prev_t = t
                else:
                    dt = dt_frame / len(samples)
                dt = max(dt, 1e-4)

                if needs_gyro:
                    gx_latest = float(s.get("gyrX", 0.0) or 0.0)
                    gy_latest = float(s.get("gyrY", 0.0) or 0.0)
                    gz_latest = float(s.get("gyrZ", 0.0) or 0.0)
                    vx, vy, active = pos_filter.step(ax_f, ay_f, gx_latest, gy_latest, gz_latest, dt)
                else:
                    vx, vy, active = pos_filter.step(ax_f, ay_f, dt)
                vx, vy = vel_lock.apply(vx, vy, dt)

        # nudge every frame regardless of whether a new sample arrived this
        # frame -- vx/vy correctly carry the last real filter output either way
        task.nudge_cursor(vx * args.gain * dt_frame, vy * args.gain * dt_frame)

        still_running = task.update(
            extra_values=[round(ax_latest, 4), round(ay_latest, 4),
                          round(vx, 3), round(vy, 3), int(active), args.filter]
        )

        hud = [
            f"filter: {args.filter}",
            f"accel=({ax_latest:+.2f},{ay_latest:+.2f})  vel=({vx:+.2f},{vy:+.2f})",
            f"targets completed: {len(task.results)}/{args.targets}",
        ]
        task.draw(hud_lines=hud)
        task.clock.tick(60)

        if not still_running:
            running = False

    poller.stop()
    task.print_summary()
    task.close()


if __name__ == "__main__":
    main()