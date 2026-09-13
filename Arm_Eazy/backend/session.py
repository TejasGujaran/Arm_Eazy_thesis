"""
session.py
==========
Per-connection state for the WebSocket bridge.

Previously this lived as module-level globals in bridge_server.py
(joint_state, gripper_pose, robot_config, ik_target, solve_ik_pressed) —
shared by every connected client. On a public server with more than one
browser tab open, that means two people would silently overwrite each
other's config and joint state. bridge_server.py now creates one Session
per connection instead.
"""

import asyncio

import message_types as mt
from ik import solve_ik, ConfigError

JOINT_KEYS = ["j1", "j2", "j3", "j4", "j5", "j6"]


class Session:
    """All state for a single browser connection."""

    def __init__(self):
        # Current joint values received from the browser (degrees for
        # "rotate" joints, raw slider units for "translate" joints)
        self.joint_state = {k: 0.0 for k in JOINT_KEYS}

        # Current gripper pose received from the browser (mm + degrees)
        self.gripper_pose = {
            "x": 0.0, "y": 0.0, "z": 0.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        }

        # Full arm config received from the browser. None until a CONFIG
        # message arrives — solve_ik() will raise ConfigError until then.
        self.config = None

        # Last IK target request received from the browser
        self.ik_target = {
            "target": {"x": 0.0, "y": 0.0, "z": 0.0},
            "seed": {k: 0.0 for k in JOINT_KEYS},
            "targetRPY": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "useOrientation": False,
        }

        # Whether the Solve IK button is currently held down in the browser
        self.solve_ik_pressed = False

        # Guards against overlapping solves for THIS connection only — under
        # continuous gamepad input, IK_TARGET messages can arrive faster
        # than scipy can solve them; drop new requests while one is running
        # rather than queue them, since by the time a queued solve would
        # run, ik_target has usually moved on anyway.
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
        print(f"[session] CONFIG received: {list(self.config.keys())}")

    def _handle_ik_target(self, data):
        self.ik_target["target"]         = data.get("target", self.ik_target["target"])
        self.ik_target["seed"]           = data.get("seed", self.ik_target["seed"])
        self.ik_target["targetRPY"]      = data.get("targetRPY", self.ik_target["targetRPY"])
        self.ik_target["useOrientation"] = data.get("useOrientation", False)

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
            print(f"[session] Unknown message type: {msg_type}")

    # -------------------------------------------------------------------
    # IK
    # -------------------------------------------------------------------

    def apply_ik(self):
        """
        Solve IK using this session's current state. Decoupled from
        WebSocket — callable from anywhere a solve is needed. Raises
        ConfigError if no CONFIG has been received yet.
        Returns the result dict from ik.solve_ik(); use result['joints']
        as the SET_JOINTS payload.
        """
        result = solve_ik(
            joint_angles=self.joint_state,
            config=self.config,
            target_xyz=self.ik_target["target"],
            target_rpy=self.ik_target["targetRPY"] if self.ik_target["useOrientation"] else None,
        )
        result["joints"] = {k: round(v, 4) for k, v in result["joints"].items()}
        return result

    async def apply_ik_async(self):
        """
        Runs apply_ik() in a worker thread instead of the asyncio event
        loop. solve_ik() is a blocking scipy.optimize.least_squares call —
        under gamepad control, IK_TARGET messages can arrive in quick
        bursts, and a blocking solve stalls this connection's heartbeat and
        websockets' own ping/pong, which is what causes a "Closed —
        retrying" cycle in the browser. Snapshot this session's state
        before handing off, since it can be mutated by the next inbound
        message while the thread runs.
        """
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
        """
        Called after dispatch() for every inbound message from this
        connection. send_joints_fn(websocket, joints_dict) sends SET_JOINTS
        back to this connection's browser.
        """
        if msg_type != mt.IK_TARGET:
            return

        if self._ik_solve_in_flight:
            return
        if self.config is None:
            print("[session] IK_TARGET received but no CONFIG loaded yet for this session — skipping")
            return

        self._ik_solve_in_flight = True
        try:
            result = await self.apply_ik_async()
            print(result)
            await send_joints_fn(websocket, result["joints"])
        except ConfigError as e:
            print(f"[session] IK solve skipped: {e}")
        finally:
            self._ik_solve_in_flight = False
