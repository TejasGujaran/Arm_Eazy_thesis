"""
bridge_server.py  (ABLATION-TEST VARIANT)
===========================================
Identical to the original bridge_server.py, EXCEPT:

  1. send_joints()'s per-message print is gated behind the same
     BENCH_DEBUG_PRINTS flag used in session.py (must match — set both
     env vars the same way for a given run).

  2. The heartbeat loop can be disabled entirely via BENCH_HEARTBEAT=0,
     as a SEPARATE, independent ablation axis. This tests the second
     hypothesis: that the heartbeat task (sent every 1s on the same
     event loop) occasionally delays/queues behind a timed exchange,
     contributing its own share of the tail — independent of printing.

RUNNING THE FULL 2x2 ABLATION
-------------------------------
For each of the 3 displacement conditions (zero/20mm/400mm), run all 4
combinations, keeping --n and every other benchmark parameter identical:

    # 1. Baseline (reproduces original behavior: prints ON, heartbeat ON)
    BENCH_DEBUG_PRINTS=1 BENCH_HEARTBEAT=1 python bridge_server.py

    # 2. Prints OFF, heartbeat ON
    BENCH_DEBUG_PRINTS=0 BENCH_HEARTBEAT=1 python bridge_server.py

    # 3. Prints ON, heartbeat OFF
    BENCH_DEBUG_PRINTS=1 BENCH_HEARTBEAT=0 python bridge_server.py

    # 4. Prints OFF, heartbeat OFF (fully quiet)
    BENCH_DEBUG_PRINTS=0 BENCH_HEARTBEAT=0 python bridge_server.py

Restart the server fresh between each run (env vars are read once at
import time) and re-run ws_latency_bench.py identically against each.
Compare tail stats (count >4ms, p99, max, and the adjacent-sample
clustering fraction) across the 4 runs to isolate how much each factor
contributes.

NOTE: for a clean test, also silence the client's per-sample print in
ws_latency_bench.py's run_benchmark() loop (line ~313), or pass its
output through `> /dev/null` when timing — otherwise the client's own
print() calls remain a confound on the client side regardless of what
you change here on the server.
"""

import asyncio
import json
import os
from pathlib import Path

import websockets
from websockets.http11 import Response

import message_types as mt
from session2 import Session, DEBUG_PRINTS

REPO_ROOT     = Path(__file__).resolve().parent.parent
FRONTEND_DIR  = REPO_ROOT / "frontend"
CONFIG_DIR    = REPO_ROOT / "config"

HOST = "localhost"
PORT = 9090

HEARTBEAT_INTERVAL = 1.0  # seconds between heartbeats

# ---- ABLATION axis #2: heartbeat on/off, independent of printing ----------
HEARTBEAT_ENABLED = os.environ.get("BENCH_HEARTBEAT", "1") == "1"

CONTENT_TYPES = {
    ".html": "text/html",
    ".js":   "application/javascript",
    ".css":  "text/css",
    ".json": "application/json",
}

# =============================================================================
# Static file serving (unchanged from original)
# =============================================================================

def _resolve_static_path(url_path):
    if url_path == "/":
        url_path = "/index.html"

    if url_path.startswith("/config/"):
        root = CONFIG_DIR
        rel  = url_path[len("/config/"):]
    else:
        root = FRONTEND_DIR
        rel  = url_path.lstrip("/")

    file_path = (root / rel).resolve()
    try:
        file_path.relative_to(root)
    except ValueError:
        return None

    return file_path


async def process_request(connection, request):
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None

    url_path  = request.path.split("?")[0]
    file_path = _resolve_static_path(url_path)

    if file_path is None:
        return Response(403, "Forbidden", {}, b"Forbidden")
    if not file_path.is_file():
        return Response(404, "Not Found", {}, b"Not found")

    content_type = CONTENT_TYPES.get(file_path.suffix, "application/octet-stream")
    return Response(200, "OK", {"Content-Type": content_type}, file_path.read_bytes())


# =============================================================================
# Send helper
# =============================================================================

async def send_joints(websocket, joints):
    """Send SET_JOINTS to one connection's browser."""
    await websocket.send(json.dumps({"type": mt.SET_JOINTS, "joints": joints}))
    # ---- ABLATION: gated, same flag as session.py's diagnostic prints.
    if DEBUG_PRINTS:
        print(f"[bridge] → {mt.SET_JOINTS} {joints}")


async def _heartbeat_loop(websocket):
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        await websocket.send(json.dumps({"type": mt.HEARTBEAT}))


# =============================================================================
# WebSocket handler — one Session per connection
# =============================================================================

async def handler(websocket):
    print(f"[bridge] Client connected: {websocket.remote_address}  "
          f"(debug_prints={DEBUG_PRINTS}, heartbeat={HEARTBEAT_ENABLED})")

    session = Session()

    # ---- ABLATION axis #2: only start the heartbeat task if enabled.
    heartbeat = asyncio.ensure_future(_heartbeat_loop(websocket)) if HEARTBEAT_ENABLED else None

    try:
        async for raw in websocket:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[bridge] Bad JSON: {e}")
                continue

            msg_type = data.get("type")
            session.dispatch(msg_type, data)

            try:
                await session.on_message(websocket, msg_type, send_joints)
            except Exception as e:
                print(f"[bridge] on_message error (connection kept alive): {e!r}")
    finally:
        if heartbeat is not None:
            heartbeat.cancel()

    print(f"[bridge] Client disconnected: {websocket.remote_address}")


# =============================================================================
# Entry point
# =============================================================================

async def main():
    async with websockets.serve(handler, HOST, PORT, process_request=process_request):
        print(f"[bridge] Serving frontend + WebSocket on http://{HOST}:{PORT}  "
              f"(debug_prints={DEBUG_PRINTS}, heartbeat={HEARTBEAT_ENABLED})")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
