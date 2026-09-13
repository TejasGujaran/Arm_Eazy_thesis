#!/usr/bin/env python3
"""
ws_latency_bench.py — Autonomous round-trip latency benchmark for
bridge_server.py's WebSocket bridge, using arm_eazy's real protocol.

WHAT THIS DOES, AUTOMATICALLY (no manual target/seed lookup needed)
--------------------------------------------------------------------
1. Loads your real robot config from disk (local file — no HTTP fetch,
   so this can't be broken by bridge_server.py's static-file-serving bug).
2. Establishes a seed pose (default: all joints at 0 deg) — this is what
   gets sent as JOINT_STATE, and is the pose the solver's Jacobian starts
   iterating from for the very first solve.
3. Establishes the IK target(s) — see "SETTING THE TARGET(S)" below.
4. Connects the WebSocket and sends the same message sequence a real
   browser client sends on startup: CONFIG, then JOINT_STATE (this is
   important — session.apply_ik() seeds the solver from session.joint_state,
   which is only updated by JOINT_STATE messages, NOT from the "seed"
   field inside IK_TARGET, which session.py only logs and never uses for
   solving), then GRIPPER_POSE, then SOLVE_IK_BUTTON(pressed=True) to
   mirror the real button-hold gesture.
5. Sends IK_TARGET --n times, serially (wait for SET_JOINTS before
   sending the next). In two-point mode, ALSO sends a JOINT_STATE with
   the joints SET_JOINTS just returned, before sending the next IK_TARGET
   — see "WHY JOINT_STATE IS RE-SENT IN OSCILLATION MODE" below.
6. Sends SOLVE_IK_BUTTON(pressed=False) at the end, mirroring pointerup.

SETTING THE TARGET(S)
-----------------------
Three modes, single point (unchanged from before) or two-point oscillation
(new):

  Single point (unchanged):
    --seed        Joint pose sent as JOINT_STATE at startup — where the
                  FIRST solve's Jacobian iteration starts from.
    --target      XYZ (mm) sent as every IK_TARGET's "target" field.
    --target-seed Alternate to --target: a joint pose whose FK position
                  is used as the target instead of raw XYZ.
    If none of --target/--target-seed given, target defaults to FK(seed)
    (near-zero residual — not a real solve; a warning is printed).

  Two-point oscillation (new):
    --target-a, --target-b     XYZ (mm) each. Give BOTH to enable
                                oscillation mode — the benchmark alternates
                                IK_TARGET between point A and point B every
                                other sample: A, B, A, B, ... for --n total
                                samples. --target / --target-seed are
                                ignored if both --target-a and --target-b
                                are given.
    --target-a-seed / --target-b-seed   Same idea as --target-seed, but
                                one for each point — lets you specify each
                                point as "FK of this joint pose" instead of
                                raw XYZ. Ignored per-point if the
                                corresponding --target-a/-b is given.

WHY JOINT_STATE IS RE-SENT IN OSCILLATION MODE
-------------------------------------------------
session.apply_ik() always seeds the solver from session.joint_state, and
that is ONLY updated by an explicit JOINT_STATE message — never from a
solve's own result. A real browser client updates its local joint state
after every SET_JOINTS and reports that back via JOINT_STATE before the
next IK_TARGET (this is exactly what RobotState / slider_adapter.js do on
receiving SET_JOINTS). If this script skipped that step, every solve in
the oscillation would keep re-seeding from the ORIGINAL --seed pose
instead of "wherever the arm actually is now" — meaning your A<->B
oscillation would not be modeling continuous back-and-forth motion, it
would be modeling "solve seed->A, then throw that away and solve
seed->B, repeat" — the same trivial-ish jump every single time, with no
accumulated realism. Re-sending JOINT_STATE after each SET_JOINTS keeps
each solve honestly seeded from the previous solve's actual output, which
is what makes this a real oscillation benchmark rather than two unrelated
single-shot solves interleaved.

Single-point mode does NOT re-send JOINT_STATE, matching prior behavior:
every solve there is intentionally re-seeded from the same fixed --seed
each time, since there's only one target and no "motion" to track.

CAVEAT FOR YOUR THESIS
-----------------------
This RTT includes a real scipy least_squares solve, not pure network
transport — there is no protocol message that is solve-free and also a
true request/reply pair (HEARTBEAT is server-initiated broadcast only —
a different, separate metric). In oscillation mode, report the A<->B
displacement alongside the RTT numbers, since solve time scales with how
far the solver has to move — and note that JOINT_STATE round trips
(fire-and-forget, not awaited) are NOT included in the timed RTT, only
the IK_TARGET -> SET_JOINTS pair is.

FK ASSUMPTION (used only to compute default targets from a --seed/-a-seed/
-b-seed, and to print residual displacement; NOT used if raw --target/
--target-a/--target-b XYZ is supplied directly, other than for that
printed comparison — since it's not the same code as ik.py)
--------------------------------------------------------------------------
Standard serial-chain convention: at each joint i, rotate the current
frame about joint_axes["JointI"] by the joint angle, then translate along
the rotated frame by translations_mm["JointI"] to reach the next joint's
origin. BaseTransform and Ik_bone are applied as fixed (non-rotating)
translations at the start and end of the chain respectively. If your
actual ik.py uses a different convention (e.g. translate-then-rotate),
any FK-derived default target will be slightly off zero-residual but the
solver will simply do a bit more work — it will not fail, since any
config-valid target should still be within some reachable neighborhood.

USAGE
-----
    pip install websockets numpy   # numpy only needed for the FK step

    # Old single-point behavior (near-zero-residual):
    python ws_latency_bench.py --config-path config.json --n 200

    # Single-point, real displacement:
    python ws_latency_bench.py --config-path config.json \
        --seed 0,0,0,0,0,0 --target 550,120,480 --n 200

    # Two-point oscillation, raw XYZ for both points:
    python ws_latency_bench.py --config-path config.json \
        --seed 0,0,0,0,0,0 \
        --target-a 550,120,480 --target-b 400,-120,520 --n 200

    # Two-point oscillation, points specified as joint poses instead:
    python ws_latency_bench.py --config-path config.json \
        --seed 0,0,0,0,0,0 \
        --target-a-seed 20,15,-10,0,30,0 \
        --target-b-seed -20,10,-5,0,-20,0 --n 200

    # Save results to disk (writes results/run1.csv + results/run1_summary.json):
    python ws_latency_bench.py --config-path config.json \
        --target-a 550,120,480 --target-b 400,-120,520 --n 200 \
        --output results/run1
"""

import argparse
import asyncio
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

JOINT_KEYS = ["j1", "j2", "j3", "j4", "j5", "j6"]
WARMUP = 10


# =============================================================================
# Forward kinematics (see FK ASSUMPTION in the module docstring)
# =============================================================================

def _rotation_matrix(axis, angle_deg):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(angle_deg)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ])


def forward_kinematics(joints_deg: dict, config: dict):
    """Returns the tool-point XYZ (mm) for a given joint pose and config."""
    translations = config["translations_mm"]
    axes = config["joint_axes"]

    pos = np.array(translations["BaseTransform"], dtype=float)
    rot = np.eye(3)

    for i in range(1, 7):
        key = f"Joint{i}"
        jkey = f"j{i}"
        angle = joints_deg.get(jkey, 0.0)
        local_rot = _rotation_matrix(axes[key], angle)
        rot = rot @ local_rot
        pos = pos + rot @ np.array(translations[key], dtype=float)

    pos = pos + rot @ np.array(translations["Ik_bone"], dtype=float)
    return pos  # [x, y, z] in mm


def _resolve_point(label, xyz_arg, seed_arg, config, fallback_xyz=None):
    """Resolve one target point: explicit XYZ > FK(seed) > fallback."""
    if xyz_arg is not None:
        return {"x": xyz_arg[0], "y": xyz_arg[1], "z": xyz_arg[2]}, f"--{label} (explicit XYZ)"
    if seed_arg is not None:
        pose = dict(zip(JOINT_KEYS, seed_arg))
        xyz = forward_kinematics(pose, config)
        return ({"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])},
                f"--{label}-seed (FK of pose {pose})")
    if fallback_xyz is not None:
        return ({"x": float(fallback_xyz[0]), "y": float(fallback_xyz[1]), "z": float(fallback_xyz[2])},
                "default: FK(seed) — near-zero residual, NOT a real solve")
    raise ValueError(f"No way to resolve target '{label}': give --{label} or --{label}-seed.")


# =============================================================================
# Benchmark
# =============================================================================

async def run_benchmark(url, config_path, seed_deg,
                         target_xyz_arg, target_seed_deg,
                         target_a_xyz, target_a_seed, target_b_xyz, target_b_seed,
                         n, output):
    import websockets

    config = json.loads(Path(config_path).read_text())
    print(f"[bench] Loaded config from {config_path} "
          f"({len(config)} top-level keys)")

    seed = dict(zip(JOINT_KEYS, seed_deg))
    seed_fk_xyz = forward_kinematics(seed, config)
    print(f"[bench] Seed pose (sent once at startup, and re-sent as current "
          f"position after each solve in oscillation mode): {seed}")
    print(f"[bench] FK(seed) (mm): x={seed_fk_xyz[0]:.3f} y={seed_fk_xyz[1]:.3f} z={seed_fk_xyz[2]:.3f}")

    oscillate = target_a_xyz is not None or target_a_seed is not None \
        or target_b_xyz is not None or target_b_seed is not None
    if oscillate and not ((target_a_xyz is not None or target_a_seed is not None)
                           and (target_b_xyz is not None or target_b_seed is not None)):
        raise ValueError("Oscillation mode needs BOTH point A and point B — "
                          "give --target-a/--target-a-seed AND --target-b/--target-b-seed.")

    if oscillate:
        point_a, source_a = _resolve_point("target-a", target_a_xyz, target_a_seed, config)
        point_b, source_b = _resolve_point("target-b", target_b_xyz, target_b_seed, config)
        ab_displacement_mm = float(np.linalg.norm(
            np.array([point_a["x"], point_a["y"], point_a["z"]])
            - np.array([point_b["x"], point_b["y"], point_b["z"]])
        ))
        print(f"[bench] Point A (mm): x={point_a['x']:.3f} y={point_a['y']:.3f} z={point_a['z']:.3f}  "
              f"(source: {source_a})")
        print(f"[bench] Point B (mm): x={point_b['x']:.3f} y={point_b['y']:.3f} z={point_b['z']:.3f}  "
              f"(source: {source_b})")
        print(f"[bench] A<->B displacement: {ab_displacement_mm:.3f} mm  "
              f"(oscillating every sample: A, B, A, B, ...)")
    else:
        target, source = _resolve_point("target", target_xyz_arg, target_seed_deg,
                                         config, fallback_xyz=seed_fk_xyz)
        displacement_mm = float(np.linalg.norm(
            np.array([target["x"], target["y"], target["z"]]) - seed_fk_xyz
        ))
        print(f"[bench] IK target (mm): x={target['x']:.3f} y={target['y']:.3f} z={target['z']:.3f}  "
              f"(source: {source})")
        print(f"[bench] Displacement from FK(seed) to target: {displacement_mm:.3f} mm")
        if displacement_mm < 1e-6:
            print("[bench] WARNING: displacement is ~0 — this run measures transport/session "
                  "overhead, not a real IK solve. Pass --target or --target-seed to change that.")

    rtts_ms = []
    records = []  # one dict per timed sample, written out at the end if --output given
    current_joints = dict(seed)  # tracks what we last told the server as JOINT_STATE

    async with websockets.connect(url) as ws:
        # ---- Startup sequence, mirroring a real browser client -----------
        await ws.send(json.dumps({"type": "CONFIG", "config": config}))
        await asyncio.sleep(0.1)

        await ws.send(json.dumps({
            "type": "JOINT_STATE",
            "joints": current_joints,
            "timestamp": int(time.time() * 1000),
        }))
        await asyncio.sleep(0.05)

        startup_gripper = point_a if oscillate else target
        await ws.send(json.dumps({
            "type": "GRIPPER_POSE",
            "x": startup_gripper["x"], "y": startup_gripper["y"], "z": startup_gripper["z"],
            "roll": 0, "pitch": 0, "yaw": 0,
            "timestamp": int(time.time() * 1000),
        }))
        await asyncio.sleep(0.05)

        await ws.send(json.dumps({"type": "SOLVE_IK_BUTTON", "pressed": True}))
        await asyncio.sleep(0.05)

        # ---- Timed IK_TARGET / SET_JOINTS round trips ---------------------
        success_count = 0
        for i in range(n + WARMUP):
            this_target = (point_a if (i % 2 == 0) else point_b) if oscillate else target

            ik_msg = {
                "type": "IK_TARGET",
                "target": this_target,
                "seed": current_joints,
                "orientation": None,
                "targetRPY": {"roll": 0, "pitch": 0, "yaw": 0},
                "useOrientation": False,
            }

            t_send = time.perf_counter()
            await ws.send(json.dumps(ik_msg))

            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                if data.get("type") == "SET_JOINTS":
                    break
                # HEARTBEAT and anything else interleaved: ignored.

            t_recv = time.perf_counter()
            rtt_ms = (t_recv - t_send) * 1000

            if i >= WARMUP:
                rtts_ms.append(rtt_ms)
                if data.get("success") is True:
                    success_count += 1
                point_label = "" if not oscillate else ("A" if i % 2 == 0 else "B")
                label = "" if not oscillate else f" ->{point_label}"
                print(f"  sample={i - WARMUP:4d}{label}  rtt={rtt_ms:7.3f} ms  "
                      f"success={data.get('success')}  joints={data.get('joints')}")
                records.append({
                    "sample": i - WARMUP,
                    "point": point_label,
                    "target_x": this_target["x"], "target_y": this_target["y"], "target_z": this_target["z"],
                    "rtt_ms": rtt_ms,
                    "success": data.get("success"),
                    "err_norm": data.get("errNorm"),
                    "joints_j1": data.get("joints", {}).get("j1"),
                    "joints_j2": data.get("joints", {}).get("j2"),
                    "joints_j3": data.get("joints", {}).get("j3"),
                    "joints_j4": data.get("joints", {}).get("j4"),
                    "joints_j5": data.get("joints", {}).get("j5"),
                    "joints_j6": data.get("joints", {}).get("j6"),
                })

            # Oscillation mode: tell the server where we actually ended up,
            # so the NEXT solve seeds from real current position rather than
            # the original --seed every time. See module docstring
            # "WHY JOINT_STATE IS RE-SENT IN OSCILLATION MODE".
            if oscillate and data.get("joints"):
                current_joints = data["joints"]
                await ws.send(json.dumps({
                    "type": "JOINT_STATE",
                    "joints": current_joints,
                    "timestamp": int(time.time() * 1000),
                }))
                # Not awaited/timed — this is bookkeeping, not part of the
                # measured IK_TARGET -> SET_JOINTS round trip.

        await ws.send(json.dumps({"type": "SOLVE_IK_BUTTON", "pressed": False}))

    print(f"\n=== WebSocket RTT summary (n={len(rtts_ms)}, {WARMUP} warmup discarded) ===")
    if oscillate:
        print(f"    mode: two-point oscillation  |  A<->B displacement: {ab_displacement_mm:.3f} mm")
    else:
        print(f"    mode: single target  |  source: {source}  |  displacement: {displacement_mm:.3f} mm")
    if "success" in data:
        print(f"    solver reported success on {success_count}/{len(rtts_ms)} samples")
    print(f"  mean:   {statistics.mean(rtts_ms):7.3f} ms")
    print(f"  median: {statistics.median(rtts_ms):7.3f} ms")
    print(f"  stdev:  {statistics.stdev(rtts_ms):7.3f} ms")
    sorted_r = sorted(rtts_ms)
    p95 = sorted_r[int(0.95 * len(sorted_r)) - 1]
    print(f"  p95:    {p95:7.3f} ms")
    print(f"  min/max: {min(rtts_ms):.3f} / {max(rtts_ms):.3f} ms")

    # ---- Save results to disk -------------------------------------------
    if output:
        out_base = Path(output)
        out_base.parent.mkdir(parents=True, exist_ok=True)
        csv_path = out_base.with_suffix(".csv")
        json_path = out_base.with_name(out_base.stem + "_summary.json")

        with open(csv_path, "w", newline="") as f:
            fieldnames = ["sample", "point", "target_x", "target_y", "target_z",
                          "rtt_ms", "success", "err_norm",
                          "joints_j1", "joints_j2", "joints_j3",
                          "joints_j4", "joints_j5", "joints_j6"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\n[bench] Per-sample data written to {csv_path}")

        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "url": url,
            "config_path": str(config_path),
            "mode": "oscillation" if oscillate else "single_target",
            "n": len(rtts_ms),
            "warmup": WARMUP,
            "seed": seed,
            "mean_ms": statistics.mean(rtts_ms),
            "median_ms": statistics.median(rtts_ms),
            "stdev_ms": statistics.stdev(rtts_ms),
            "p95_ms": p95,
            "min_ms": min(rtts_ms),
            "max_ms": max(rtts_ms),
            "success_count": success_count if "success" in data else None,
        }
        if oscillate:
            summary["point_a"] = point_a
            summary["point_b"] = point_b
            summary["ab_displacement_mm"] = ab_displacement_mm
        else:
            summary["target"] = target
            summary["target_source"] = source
            summary["displacement_mm"] = displacement_mm

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[bench] Summary stats written to {json_path}")


def parse_csv_floats(s: str):
    return [float(x) for x in s.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://localhost:9090",
                         help="WebSocket URL of bridge_server.py")
    parser.add_argument("--config-path", required=True,
                         help="local path to your robot config JSON "
                              "(e.g. Kuka_KR_4_R600_config.json)")
    parser.add_argument("--seed", default="0,0,0,0,0,0",
                         help="j1,j2,j3,j4,j5,j6 in degrees — sent as JOINT_STATE at startup "
                              "(default: all zero)")

    parser.add_argument("--target", default=None,
                         help="[single-point mode] x,y,z in mm. Ignored if --target-a/-b given.")
    parser.add_argument("--target-seed", default=None,
                         help="[single-point mode] j1..j6 in degrees; target = FK of this pose. "
                              "Ignored if --target is also given, or if --target-a/-b given.")

    parser.add_argument("--target-a", default=None,
                         help="[oscillation mode] x,y,z in mm for point A. Requires --target-b too.")
    parser.add_argument("--target-a-seed", default=None,
                         help="[oscillation mode] j1..j6 in degrees; point A = FK of this pose. "
                              "Ignored if --target-a is also given.")
    parser.add_argument("--target-b", default=None,
                         help="[oscillation mode] x,y,z in mm for point B. Requires --target-a too.")
    parser.add_argument("--target-b-seed", default=None,
                         help="[oscillation mode] j1..j6 in degrees; point B = FK of this pose. "
                              "Ignored if --target-b is also given.")

    parser.add_argument("--n", type=int, default=200, help="number of timed samples (after warmup)")
    parser.add_argument("--output", default=None,
                         help="base path to save results, e.g. 'results/run1'. Writes "
                              "'<output>.csv' (one row per timed sample: rtt, success, "
                              "joints, which point A/B, etc.) and '<output>_summary.json' "
                              "(aggregate stats + run metadata). Omit to skip saving — "
                              "results still print to stdout either way.")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        url=args.url,
        config_path=args.config_path,
        seed_deg=parse_csv_floats(args.seed),
        target_xyz_arg=parse_csv_floats(args.target) if args.target is not None else None,
        target_seed_deg=parse_csv_floats(args.target_seed) if args.target_seed is not None else None,
        target_a_xyz=parse_csv_floats(args.target_a) if args.target_a is not None else None,
        target_a_seed=parse_csv_floats(args.target_a_seed) if args.target_a_seed is not None else None,
        target_b_xyz=parse_csv_floats(args.target_b) if args.target_b is not None else None,
        target_b_seed=parse_csv_floats(args.target_b_seed) if args.target_b_seed is not None else None,
        n=args.n,
        output=args.output,
    ))