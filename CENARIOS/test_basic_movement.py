"""Cenário 1: Teste individual de cada uma das 8 funções principais da garra.

Valida:
    1. Execução isolada das 8 funções (`move_right`, `move_left`, `move_forward`,
       `move_backward`, `move_down`, `move_up`, `close`, `open`).
    2. Consistência dos deslocamentos nos eixos X, Y e Z e abertura/força dos dedos.
    3. Comunicação fim a fim via barramento estilo socket (`via_socket=True`).
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CODIGO.config import ProjectConfig, load_config  # noqa: E402
from CODIGO.controller import GripperController  # noqa: E402
from FUNCOES.gripper_commands import (  # noqa: E402
    close,
    move_backward,
    move_down,
    move_forward,
    move_left,
    move_right,
    move_up,
    open as gripper_open,
)


def run_scenario(config: Optional[ProjectConfig] = None) -> Dict[str, Any]:
    """Executa a bateria de testes das 8 funções básicas e retorna métricas."""
    cfg = config or load_config()
    t0 = time.perf_counter()
    controller = GripperController(config=cfg, enable_socket_server=True)
    checks: Dict[str, bool] = {}
    latencies_ms: List[float] = []
    artifacts: List[str] = []

    try:
        g = controller.gripper
        g.reset()
        p0 = g.get_state()["position"]

        # 1. move_right (+X)
        r_right = move_right(speed=0.25, force=20.0, duration=0.40, gripper=g)
        checks["move_right"] = (
            r_right.success and (r_right.position["x"] - p0["x"]) > 0.09
        )
        latencies_ms.append(r_right.elapsed_ms)

        # 2. move_left (-X de volta à origem)
        r_left = move_left(speed=0.25, force=20.0, duration=0.40, gripper=g)
        checks["move_left"] = (
            r_left.success and abs(r_left.position["x"] - p0["x"]) < 1e-3
        )
        latencies_ms.append(r_left.elapsed_ms)

        # 3. move_forward (+Y)
        r_fwd = move_forward(speed=0.25, force=20.0, duration=0.40, gripper=g)
        checks["move_forward"] = (
            r_fwd.success and (r_fwd.position["y"] - p0["y"]) > 0.09
        )
        latencies_ms.append(r_fwd.elapsed_ms)

        # 4. move_backward (-Y de volta à origem)
        r_bwd = move_backward(speed=0.25, force=20.0, duration=0.40, gripper=g)
        checks["move_backward"] = (
            r_bwd.success and abs(r_bwd.position["y"] - p0["y"]) < 1e-3
        )
        latencies_ms.append(r_bwd.elapsed_ms)

        # 5. move_down (-Z)
        r_down = move_down(speed=0.20, force=25.0, duration=0.50, gripper=g)
        checks["move_down"] = (
            r_down.success and (p0["z"] - r_down.position["z"]) > 0.09
        )
        latencies_ms.append(r_down.elapsed_ms)

        # 6. move_up (+Z)
        r_up = move_up(speed=0.20, force=25.0, duration=0.50, gripper=g)
        checks["move_up"] = r_up.success and abs(r_up.position["z"] - p0["z"]) < 1e-3
        latencies_ms.append(r_up.elapsed_ms)

        # 7. close (fecha os dedos e aplica força)
        r_close = close(speed=0.15, force=35.0, duration=0.30, gripper=g)
        checks["close"] = (
            r_close.success
            and r_close.aperture < 0.01
            and abs(r_close.grip_force - 35.0) < 1e-2
        )
        latencies_ms.append(r_close.elapsed_ms)

        # 8. open (abre os dedos e zera a força de preensão)
        r_open = gripper_open(speed=0.15, force=15.0, duration=0.30, gripper=g)
        checks["open"] = (
            r_open.success and r_open.aperture >= 0.075 and r_open.grip_force == 0.0
        )
        latencies_ms.append(r_open.elapsed_ms)

        # 9. Verificação extra via transporte TCP socket modular
        r_sock = controller.execute_command(
            "move_right", speed=0.20, duration=0.20, via_socket=True
        )
        checks["socket_roundtrip"] = r_sock.success and r_sock.position["x"] > 0.03

        if cfg.simulation.save_artifacts:
            saved = controller.simulator.save_frame(
                cfg.artifacts_dir / "basic_movement_final.png"
            )
            if saved:
                artifacts.append(str(saved))

        overall_success = all(checks.values())
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "scenario": "basic_movement",
            "success": overall_success,
            "duration_ms": round(total_ms, 2),
            "metrics": {
                "simulator_engine": controller.simulator.engine_name,
                "functions_tested": 8,
                "functions_passed": sum(
                    1 for k, v in checks.items() if k != "socket_roundtrip" and v
                ),
                "socket_transport_verified": checks["socket_roundtrip"],
                "avg_command_latency_ms": round(
                    sum(latencies_ms) / len(latencies_ms), 3
                ),
                "individual_results": checks,
            },
            "artifacts": artifacts,
        }
    finally:
        controller.shutdown()


class TestBasicMovement(unittest.TestCase):
    """Integração compatível com `unittest` e `pytest`."""

    def test_all_eight_gripper_functions(self) -> None:
        report = run_scenario(load_config(save_artifacts=False))
        self.assertTrue(report["success"], msg=f"Falha no cenário: {report}")
        self.assertEqual(report["metrics"]["functions_passed"], 8)


if __name__ == "__main__":
    result = run_scenario()
    status_str = "PASS" if result["success"] else "FAIL"
    print(f"[test_basic_movement] {status_str} | Métricas: {result['metrics']}")
    sys.exit(0 if result["success"] else 1)
