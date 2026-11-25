"""Integração real de câmera, HTTP e reset; pesos reais são uma avaliação separada."""
import asyncio
from dataclasses import replace
import json
import time
import unittest

from backend.config import VisionConfig, load_backend_config
from backend.websocket_server import AsyncWSClient, GripperWebSocketServer


async def request(port,path,method="GET",body=None):
    reader,writer = await asyncio.open_connection("127.0.0.1",port)
    raw = json.dumps(body).encode() if body else b""
    writer.write(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(raw)}\r\n\r\n".encode()+raw)
    await writer.drain()
    response = await asyncio.wait_for(reader.read(),10)
    writer.close()
    await writer.wait_closed()
    head,data = response.split(b"\r\n\r\n",1)
    return int(head.split()[1]),data


async def _run():
    cfg = load_backend_config(host="127.0.0.1",port=0,log_level="WARNING",
                              vision=VisionConfig(enabled=True))
    server = GripperWebSocketServer(cfg)
    # Este cenário testa câmera/transporte/ciclo de vida, sem exigir o pacote
    # opcional YOLO ou pesos locais. Os ensaios físicos usam o detector real.
    class EmptyDetector:
        identifier = "integration-empty"
        manifest = {"validated_for_pick": False}
        def __init__(self, config): pass
        def detect(self, rgb): return []
    server.vision.detector_factory = EmptyDetector
    started = time.monotonic()
    port = await server.start()
    checks = {}
    try:
        await asyncio.sleep(.15)
        code,raw = await request(port,"/api/vision/status")
        idle = json.loads(raw)
        checks["no_capture_before_command"] = idle["status"] == "idle" and idle["frame_id"] is None and idle["frames_captured"] == 0
        code,_ = await request(port,"/api/command","POST",{"command":"vision_detect","class_name":"dental_tool"})
        packet = None
        deadline = time.monotonic()+75
        while time.monotonic()<deadline:
            code,raw = await request(port,"/api/vision/status")
            packet = json.loads(raw)
            if not packet["request_active"] and packet.get("request_id"): break
            await asyncio.sleep(.05)
        count = packet["frames_captured"]
        await asyncio.sleep(.2)
        code,raw = await request(port,"/api/vision/status")
        checks["no_capture_after_request"] = json.loads(raw)["frames_captured"] == count
        checks["camera_rendered"] = packet.get("frame_id") is not None
        checks["pick_requires_opt_in"] = not packet.get("pick_enabled")
        if packet.get("frame_id"):
            code,image = await request(port,f"/api/vision/frame?frame_id={packet['frame_id']}")
            checks["jpeg"] = code == 200 and image.startswith(b"\xff\xd8")
        code,_ = await request(port,"/api/command","POST",{"command":"vision_pick","scene_revision":0,"frame_id":0,"detection_id":"0:0"})
        checks["invalid_pick_rejected"] = code == 400
        revision = server.env.scene_revision
        code,_ = await request(port,"/api/reset","POST")
        checks["reset_invalidates_revision"] = code == 200 and server.env.scene_revision == revision+1
        code,raw = await request(port,"/health")
        checks["server_alive"] = code == 200 and json.loads(raw)["status"] == "ok"
        welcomes = []
        for _ in range(2):
            async with AsyncWSClient(f"ws://127.0.0.1:{port}/ws") as client:
                found = False
                for _ in range(8):
                    message = await client.recv_json(timeout=2)
                    if message.get("type") == "vision":
                        found = True
                        break
                welcomes.append(found)
        checks["ws_reconnect_receives_vision"] = all(welcomes)
        return {"success":all(checks.values()),"duration_ms":1000*(time.monotonic()-started),
                "metrics":{**checks,"fps_with_preview":server.measured_fps,
                           "vision_status":packet["status"],"vision_error":packet.get("error")}}
    finally:
        await server.stop()


def run_scenario(): return asyncio.run(_run())


class TestVisionIntegration(unittest.TestCase):
    def test_camera_http_reset(self):
        report = run_scenario()
        self.assertTrue(report["success"],report)
