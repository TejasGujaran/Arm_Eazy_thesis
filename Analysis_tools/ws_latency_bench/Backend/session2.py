"""
session.py  (ABLATION-TEST VARIANT)
====================================
Identical to the original session.py, EXCEPT: every per-message diagnostic
print() has been gated behind a module-level DEBUG_PRINTS flag, which
defaults to OFF.

WHY THIS FILE EXISTS
---------------------
Thesis hypothesis under test: the baseline WebSocket RTT tail (observed
even at zero displacement, where no meaningful IK work happens) may be
caused by the volume of synchronous print() calls in the hot path —
specifically _handle_ik_target's 7 print statements and on_message's
result print, both of which run in the same asyncio event loop that is
also trying to time/serve the next request. A blocking stdout write can
stall the whole event loop, which would explain both (a) a tail that
exists even without real IK work, and (b) the observed clustering of slow
samples (one stall delays whatever is queued right behind it).

HOW TO USE THIS FOR THE ABLATION
----------------------------------
Run 1 (prints ON  — reproduces original behavior):
    set DEBUG_PRINTS = True below, or run with:
        BENCH_DEBUG_PRINTS=1 python bridge_server.py
Run 2 (prints OFF — the test condition):
    leave DEBUG_PRINTS = False (default), or run with:
        BENCH_DEBUG_PRINTS=0 python bridge_server.py

Re-run ws_latency_bench.py identically (same --config-path, --seed,
--target/-a/-b, --n) for both, ideally with the client-side print also
disabled (see bridge_server.py / ws_latency_bench.py ablation notes).
Compare tail stats (samples >4ms, p99, max) between the two runs.

Everything else below is unchanged from the original session.py.
"""

import asyncio
import os

import message_types as mt
from ik import solve_ik, ConfigError

JOINT_KEYS = ["j1", "j2", "j3", "j4", "j5", "j6"]

# Toggle via env var so you don't have to edit code between the two runs:
#   BENCH_DEBUG_PRINTS=1  -> prints ON  (reproduces original/baseline behavior)
#   BENCH_DEBUG_PRINTS=0  -> prints OFF (default; the ablation test condition)
DEBUG_PRINTS = os.environ.get("BENCH_DEBUG_PRINTS", "0") == "1"


class Session:
    """All state for a single browser connection."""

    def __init__(self):
        self.joint_state = {k: 0.0 for k in JOINT_KEYS}

        self.gripper_pose = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        }

        self.config = None

        self.ik_target = {
            "target": {"x": 0.0, "y": 0.0, "z": 0.0},
            "seed": {k: 0.0 for k in JOINT_KEYS},
            "targetRPY": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "useOrientation": False,
        }

        self.solve_ik_pressed = False
        self._ik_solve_in_flight = False

    # -------------------------------------------------------------------
    # Inbound message handlers
    # -------------------------------------------------------------------

    def _handle_joint_state(self, data):
        self.joint_state.update(data.get("joints", {}))

    def _handle_gripper_pose(self, data):
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            if key in data:
                self.gripper_pose[key] = data[key]

    def _handle_config(self, data):
        self.config = data.get("config", {})
        if DEBUG_PRINTS:
            print(f"[session] CONFIG received: {list(self.config.keys())}")

    def _handle_ik_target(self, data):
        self.ik_target["target"]         = data.get("target", self.ik_target["target"])
        self.ik_target["seed"]           = data.get("seed", self.ik_target["seed"])
        self.ik_target["targetRPY"]      = data.get("targetRPY", self.ik_target["targetRPY"])
        self.ik_target["useOrientation"] = data.get("useOrientation", False)

        # ---- ABLATION: this entire block (7 print calls in the original)
        # ---- is the primary suspect for event-loop stalls. Gated below.
        if DEBUG_PRINTS:
            t   = self.ik_target["target"]
            rpy = self.ik_target["targetRPY"]
            print("─" * 50)
            print("  IK_TARGET received")
            print(f"  Target XYZ     : x={t['x']}  y={t['y']}  z={t['z']}")
            print(f"  Target RPY     : roll={rpy['roll']}  pitch={rpy['pitch']}  yaw={rpy['yaw']}")
            print(f"  Use orientation: {self.ik_target['useOrientation']}")
            print(f"  Seed joints    : {self.ik_target['seed']}")
            print("─" * 50)

    def _handle_solve_ik_button(self, data):
        self.solve_ik_pressed = bool(data.get("pressed", False))
        if DEBUG_PRINTS:
            print(f"[session] SOLVE_IK_BUTTON pressed={self.solve_ik_pressed}")

    def dispatch(self, msg_type, data):
        """Route one inbound message to the right handler."""
        if msg_type == mt.JOINT_STATE:
            self._handle_joint_state(data)
        elif msg_type == mt.GRIPPER_POSE:
            self._handle_gripper_pose(data)
        elif msg_type == mt.CONFIG:
            self._handle_config(data)
        elif msg_type == mt.IK_TARGET:
            self._handle_ik_target(data)
        elif msg_type == mt.SOLVE_IK_BUTTON:
            self._handle_solve_ik_button(data)
        else:
            if DEBUG_PRINTS:
                print(f"[session] Unknown message type: {msg_type}")

    # -------------------------------------------------------------------
    # IK
    # -------------------------------------------------------------------

    def apply_ik(self):
        result = solve_ik(
            joint_angles=self.joint_state,
            config=self.config,
            target_xyz=self.ik_target["target"],
            target_rpy=self.ik_target["targetRPY"] if self.ik_target["useOrientation"] else None,
        )
        result["joints"] = {k: round(v, 4) for k, v in result["joints"].items()}
        return result

    async def apply_ik_async(self):
        loop = asyncio.get_running_loop()

        joint_state_snapshot = dict(self.joint_state)
        config_snapshot = dict(self.config) if self.config is not None else None
        ik_target_snapshot = {
            "target": dict(self.ik_target["target"]),
            "targetRPY": dict(self.ik_target["targetRPY"]),
            "useOrientation": self.ik_target["useOrientation"],
        }

        def _solve():
            result = solve_ik(
                joint_angles=joint_state_snapshot,
                config=config_snapshot,
                target_xyz=ik_target_snapshot["target"],
                target_rpy=ik_target_snapshot["targetRPY"] if ik_target_snapshot["useOrientation"] else None,
            )
            result["joints"] = {k: round(v, 4) for k, v in result["joints"].items()}
            return result

        return await loop.run_in_executor(None, _solve)

    # -------------------------------------------------------------------
    # Per-message reaction — called after every inbound message
    # -------------------------------------------------------------------

    async def on_message(self, websocket, msg_type, send_joints_fn):
        if msg_type != mt.IK_TARGET:
            return

        if self._ik_solve_in_flight:
            return
        if self.config is None:
            if DEBUG_PRINTS:
                print("[session] IK_TARGET received but no CONFIG loaded yet for this session — skipping")
            return

        self._ik_solve_in_flight = True
        try:
            result = await self.apply_ik_async()
            # ---- ABLATION: this print of the full result dict is the
            # ---- other major suspect alongside _handle_ik_target's block.
            if DEBUG_PRINTS:
                print(result)
            await send_joints_fn(websocket, result["joints"])
        except ConfigError as e:
            if DEBUG_PRINTS:
                print(f"[session] IK solve skipped: {e}")
        finally:
            self._ik_solve_in_flight = False
