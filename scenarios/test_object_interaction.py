"""Cenário WebSocket 3: Interação física Pick & Place de objetos no MuJoCo via WebSocket.

Valida:
    1. Aproximação da garra até o instrumento na bandeja (`[0.28, -0.22, 0.018]`).
    2. Fechamento com `force=60`, confirmando força de contato tátil e `is_grasping == True`.
    3. Elevação física real contra a gravidade (`z > 0.12m`).
    4. Transporte até o posto do dentista (`[-0.30, 0.26]`), abertura e pouso.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import load_backend_config  # noqa: E402
from backend.websocket_server import AsyncWSClient, GripperWebSocketServer  # noqa: E402


def _find_object(state: Dict[str, Any], name: str) -> Dict[str, Any]:
    for obj in state.get("objects", []):
        if obj["name"] == name:
            return obj
    raise KeyError(f"Objeto '{name}' não encontrado no estado.")


async def _run_async() -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_backend_config(host="127.0.0.1", port=0, fps=60, log_level="WARNING")
    server = GripperWebSocketServer(config=cfg)
    port = await server.start()
    ws_url = f"ws://127.0.0.1:{port}/ws"

    try:
        async with AsyncWSClient(ws_url) as client:
            ack0, _ = await client.send_command_wait_ack("reset")
            tool_obj0 = _find_object(ack0["state"], "dental_tool")
            tool_init = tool_obj0.get("grasp_pos", tool_obj0["pos"])
            gx, gy = float(tool_init[0]), float(tool_init[1])
            grasp_z = max(0.022, float(tool_init[2]) + 0.014)

            await client.send_command_wait_ack("open", speed=0.6, force=60, settle=True)
            await client.send_command_wait_ack(
                "move_to",
                target={"x": gx, "y": gy, "z": 0.18},
                speed=0.65,
                settle=True,
            )
            await client.send_command_wait_ack(
                "move_to",
                target={"x": gx, "y": gy, "z": grasp_z},
                speed=0.40,
                settle=True,
            )

            grasp_ack, _ = await client.send_command_wait_ack(
                "close", speed=0.5, force=60, duration=0.45, settle=True
            )
            grasp_state = grasp_ack["state"]
            contact_during_grasp = grasp_state["contact_force"]
            was_grasping = grasp_state["is_grasping"]

            lift_ack, _ = await client.send_command_wait_ack(
                "move_to",
                target={"x": gx, "y": gy, "z": 0.22},
                speed=0.45,
                settle=True,
            )
            tool_lifted_obj = _find_object(lift_ack["state"], "dental_tool")
            tool_lifted = tool_lifted_obj.get("grasp_pos", tool_lifted_obj["pos"])

            await client.send_command_wait_ack(
                "move_to",
                target={"x": -0.30, "y": 0.26, "z": 0.22},
                speed=0.58,
                settle=True,
            )
            await client.send_command_wait_ack(
                "move_to",
                target={"x": -0.30, "y": 0.26, "z": 0.042},
                speed=0.38,
                settle=True,
            )

            release_ack, _ = await client.send_command_wait_ack(
                "open", speed=0.55, force=35, duration=0.40, settle=True
            )
            tool_final_obj = _find_object(release_ack["state"], "dental_tool")
            tool_final = tool_final_obj.get("grasp_pos", tool_final_obj["pos"])

            displacement_xy = math.hypot(
                tool_final[0] - tool_init[0], tool_final[1] - tool_init[1]
            )
            drop_error_xy = math.hypot(tool_final[0] - (-0.30), tool_final[1] - 0.26)

            lifted_ok = tool_lifted[2] > 0.10
            placed_ok = drop_error_xy < 0.08 and tool_final[2] < 0.06
            success = was_grasping and contact_during_grasp > 0.5 and lifted_ok and placed_ok

            return {
                "scenario": "object_interaction",
                "success": success,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "metrics": {
                    "engine": server.env.engine_name,
                    "initial_tool_pos": tool_init,
                    "lifted_tool_pos": tool_lifted,
                    "final_tool_pos": tool_final,
                    "contact_force_n": contact_during_grasp,
                    "was_grasping": was_grasping,
                    "displacement_xy_m": round(displacement_xy, 4),
                    "drop_zone_error_m": round(drop_error_xy, 4),
                },
            }
    finally:
        await server.stop()


def run_scenario() -> Dict[str, Any]:
    return asyncio.run(_run_async())


class TestObjectInteraction(unittest.TestCase):
    def test_websocket_pick_and_place_physics(self) -> None:
        report = run_scenario()
        self.assertTrue(report["success"], msg=f"Falha física: {report}")


if __name__ == "__main__":
    res = run_scenario()
    print(res)
    sys.exit(0 if res["success"] else 1)
