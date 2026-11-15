"""Cenário WebSocket 2: Mede sincronização em 60 FPS e latência comando -> renderização.

Verifica:
    - Cadência contínua do stream WebSocket (~60 FPS).
    - Latência de ponta a ponta entre o envio do comando no cliente, processamento físico
      no MuJoCo e atualização interpolada para renderização (< 100 ms exigido).
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import load_backend_config  # noqa: E402
from backend.websocket_server import AsyncWSClient, GripperWebSocketServer  # noqa: E402


async def _run_async() -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_backend_config(host="127.0.0.1", port=0, fps=60, log_level="WARNING")
    server = GripperWebSocketServer(config=cfg)
    port = await server.start()
    ws_url = f"ws://127.0.0.1:{port}/ws"

    try:
        async with AsyncWSClient(ws_url) as client:
            # 1. Mede taxa real de quadros recebidos pelo frontend durante 1.0 segundo
            stream_frames = 0
            window_start = time.perf_counter()
            while (time.perf_counter() - window_start) < 1.0:
                pkt = await client.recv_json(timeout=1.0)
                if pkt.get("type") == "stream":
                    stream_frames += 1
            elapsed_window = time.perf_counter() - window_start
            effective_fps = stream_frames / elapsed_window

            # 2. Mede 24 ciclos de comando -> ACK -> passo de interpolação 3D do frontend
            cmd_to_render_samples: List[float] = []
            commands = [
                "move_right",
                "move_forward",
                "move_down",
                "close",
                "move_up",
                "move_left",
                "move_backward",
                "open",
            ]
            rendered_pos = [0.0, 0.0, 0.238]

            for idx in range(24):
                cmd = commands[idx % len(commands)]
                t_cmd = time.perf_counter()
                ack, _ = await client.send_command_wait_ack(
                    cmd, speed=0.6, force=45, duration=0.06
                )
                target_pos = ack["state"]["pos"]
                # Simula passo LERP do requestAnimationFrame no browser
                alpha = 0.35
                rendered_pos = [
                    rendered_pos[i] + (target_pos[i] - rendered_pos[i]) * alpha
                    for i in range(3)
                ]
                cmd_to_render_ms = (time.perf_counter() - t_cmd) * 1000.0
                cmd_to_render_samples.append(cmd_to_render_ms)

            sorted_samples = sorted(cmd_to_render_samples)
            avg_ms = sum(sorted_samples) / len(sorted_samples)
            p95_ms = sorted_samples[int(0.95 * (len(sorted_samples) - 1))]
            max_ms = sorted_samples[-1]

            fps_ok = 45.0 <= effective_fps <= 72.0
            latency_ok = max_ms < 100.0

            return {
                "scenario": "real_time_sync",
                "success": fps_ok and latency_ok,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "metrics": {
                    "measured_stream_fps": round(effective_fps, 2),
                    "target_fps": 60,
                    "samples_count": len(sorted_samples),
                    "avg_cmd_to_render_ms": round(avg_ms, 3),
                    "p95_cmd_to_render_ms": round(p95_ms, 3),
                    "max_cmd_to_render_ms": round(max_ms, 3),
                    "latency_sla_under_100ms": latency_ok,
                    "fps_stability_ok": fps_ok,
                },
            }
    finally:
        await server.stop()


def run_scenario() -> Dict[str, Any]:
    return asyncio.run(_run_async())


class TestRealTimeSync(unittest.TestCase):
    def test_60fps_sync_and_sub_100ms_latency(self) -> None:
        report = run_scenario()
        self.assertTrue(report["success"], msg=f"Falha de sincronia: {report}")


if __name__ == "__main__":
    res = run_scenario()
    print(res)
    sys.exit(0 if res["success"] else 1)
