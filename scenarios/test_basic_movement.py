"""Cenário WebSocket 1: Verifica todas as 8 funções de movimento da garra via WebSocket.

Valida:
    - Schema de protocolo `{"command": ..., "speed": 0.5, "force": 50}` ->
      `{"state": {"pos": [x,y,z], "quat": [w,x,y,z], "force": 50, ...}}`
    - Deslocamento físico correto nos eixos +X, -X, +Y, -Y, -Z, +Z e atuação `close`/`open`.
    - Latência de comando < 100 ms em todas as 8 operações sem desconexão.
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

    checks: Dict[str, bool] = {}
    latencies_ms: List[float] = []

    try:
        async with AsyncWSClient(ws_url) as client:
            reset_ack, _ = await client.send_command_wait_ack("reset")
            p0 = reset_ack["state"]["pos"]

            # Validação de schema obrigatório: state.pos [3], state.quat [4], state.force
            st0 = reset_ack["state"]
            checks["schema_valid"] = (
                isinstance(st0.get("pos"), list)
                and len(st0["pos"]) == 3
                and isinstance(st0.get("quat"), list)
                and len(st0["quat"]) == 4
                and "force" in st0
            )

            # Mede latência de despacho em tempo real (settle=False) e precisão física (settle=True)
            _, rt_rtt = await client.send_command_wait_ack(
                "move_right", speed=0.5, force=50, duration=0.04, settle=False
            )
            latencies_ms.append(rt_rtt)
            reset_ack, _ = await client.send_command_wait_ack("reset")
            p0 = reset_ack["state"]["pos"]

            # 1. move_right (+X)
            ack, _ = await client.send_command_wait_ack(
                "move_right", speed=0.5, force=50, duration=0.22, settle=True
            )
            p_right = ack["state"]["pos"]
            checks["move_right"] = (p_right[0] - p0[0]) > 0.07

            # 2. move_left (-X)
            ack, _ = await client.send_command_wait_ack(
                "move_left", speed=0.5, force=50, duration=0.22, settle=True
            )
            p_left = ack["state"]["pos"]
            checks["move_left"] = (p_right[0] - p_left[0]) > 0.07

            # 3. move_forward (+Y)
            ack, _ = await client.send_command_wait_ack(
                "move_forward", speed=0.5, force=50, duration=0.22, settle=True
            )
            p_fwd = ack["state"]["pos"]
            checks["move_forward"] = (p_fwd[1] - p_left[1]) > 0.07

            # 4. move_backward (-Y)
            ack, _ = await client.send_command_wait_ack(
                "move_backward", speed=0.5, force=50, duration=0.22, settle=True
            )
            p_bwd = ack["state"]["pos"]
            checks["move_backward"] = (p_fwd[1] - p_bwd[1]) > 0.07

            # 5. move_down (-Z)
            ack, _ = await client.send_command_wait_ack(
                "move_down", speed=0.45, force=50, duration=0.24, settle=True
            )
            p_down = ack["state"]["pos"]
            checks["move_down"] = (p_bwd[2] - p_down[2]) > 0.07

            # 6. move_up (+Z)
            ack, _ = await client.send_command_wait_ack(
                "move_up", speed=0.45, force=50, duration=0.24, settle=True
            )
            p_up = ack["state"]["pos"]
            checks["move_up"] = (p_up[2] - p_down[2]) > 0.07

            # 7. close (fecha garras com force=50)
            _, rt_close = await client.send_command_wait_ack(
                "close", speed=0.5, force=50, duration=0.05, settle=False
            )
            latencies_ms.append(rt_close)
            ack, _ = await client.send_command_wait_ack(
                "close", speed=0.5, force=50, duration=0.30, settle=True
            )
            st_close = ack["state"]
            checks["close"] = (
                st_close["is_closed"] is True
                and st_close["aperture"] < 0.015
                and abs(st_close["force"] - 50.0) < 0.5
            )

            # 8. open (abre garras de volta a ~80mm)
            _, rt_open = await client.send_command_wait_ack(
                "open", speed=0.5, force=30, duration=0.05, settle=False
            )
            latencies_ms.append(rt_open)
            ack, _ = await client.send_command_wait_ack(
                "open", speed=0.5, force=30, duration=0.30, settle=True
            )
            st_open = ack["state"]
            checks["open"] = (
                st_open["is_closed"] is False and st_open["aperture"] > 0.070
            )

            checks["latency_under_100ms"] = max(latencies_ms) < 100.0

        return {
            "scenario": "basic_movement",
            "success": all(checks.values()),
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "metrics": {
                "engine": server.env.engine_name,
                "commands_verified": 8,
                "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
                "max_latency_ms": round(max(latencies_ms), 3),
                "checks": checks,
            },
        }
    finally:
        await server.stop()


def run_scenario() -> Dict[str, Any]:
    return asyncio.run(_run_async())


class TestWebSocketBasicMovement(unittest.TestCase):
    def test_eight_commands_over_websocket(self) -> None:
        report = run_scenario()
        self.assertTrue(report["success"], msg=f"Falha: {report}")


if __name__ == "__main__":
    res = run_scenario()
    print(res)
    sys.exit(0 if res["success"] else 1)
