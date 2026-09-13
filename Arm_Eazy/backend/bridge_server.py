"""
bridge_server.py
================
The only thing you need to run.

Serves the frontend (frontend/) as static files AND handles the WebSocket
connection to it, on the same port — one command, one port, no separate
dev server needed.

Run:
    python bridge_server.py

Then open:
    http://localhost:9090

Each browser tab gets its own Session (see session.py) — joint state,
gripper pose, config, and IK target are no longer module-level globals,
so multiple people/tabs can use this at once without corrupting each
other's state. This matters as soon as this is deployed anywhere other
than one person's own machine.
"""

import asyncio
import json
from pathlib import Path

import websockets
from websockets.http11 import Response

import message_types as mt
from session import Session

REPO_ROOT     = Path(__file__).resolve().parent.parent
FRONTEND_DIR  = REPO_ROOT / "frontend"
CONFIG_DIR    = REPO_ROOT / "config"   # outside frontend/ — mapped to /config/ below

HOST = "localhost"
PORT = 9090

HEARTBEAT_INTERVAL = 1.0  # seconds between heartbeats

CONTENT_TYPES = {
    ".html": "text/html",
    ".js":   "application/javascript",
    ".css":  "text/css",
    ".json": "application/json",
}

# =============================================================================
# Static file serving (plain HTTP GET) — shares the port with the WebSocket
# =============================================================================

def _resolve_static_path(url_path):
    """
    Maps a request path to a file on disk. /config/... is served from
    CONFIG_DIR (outside frontend/, since it's shared with nothing else
    frontend-specific); everything else is served from FRONTEND_DIR.
    Returns None if the resolved path would escape its root (blocks ../).
    """
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
        return None  # attempted to escape root via ../

    return file_path


async def process_request(connection, request):
    # Only intercept plain HTTP GETs; let WebSocket upgrade requests through
    # untouched so websockets.serve can handle the handshake as usual.
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
    print(f"[bridge] → {mt.SET_JOINTS} {joints}")


async def _heartbeat_loop(websocket):
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        await websocket.send(json.dumps({"type": mt.HEARTBEAT}))


# =============================================================================
# WebSocket handler — one Session per connection
# =============================================================================

async def handler(websocket):
    print(f"[bridge] Client connected: {websocket.remote_address}")

    session = Session()
    heartbeat = asyncio.ensure_future(_heartbeat_loop(websocket))

    try:
        async for raw in websocket:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"[bridge] Bad JSON: {e}")
                continue

            msg_type = data.get("type")
            session.dispatch(msg_type, data)

            # Wrapped defensively: an unhandled exception here (e.g. from
            # solve_ik / scipy) would otherwise propagate out of handler()
            # and tear down the whole connection — which looks identical to
            # a network drop from the browser's side.
            try:
                await session.on_message(websocket, msg_type, send_joints)
            except Exception as e:
                print(f"[bridge] on_message error (connection kept alive): {e!r}")
    finally:
        heartbeat.cancel()

    print(f"[bridge] Client disconnected: {websocket.remote_address}")


# =============================================================================
# Entry point
# =============================================================================

async def main():
    async with websockets.serve(handler, HOST, PORT, process_request=process_request):
        print(f"[bridge] Serving frontend + WebSocket on http://{HOST}:{PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
