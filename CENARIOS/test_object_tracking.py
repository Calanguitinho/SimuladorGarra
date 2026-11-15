"""Cenário 4: Rastreamento, captura e confirmação de mudança de posição de múltiplos objetos.

Verifica se o pipeline percepção (Google API) + controlador + garra detecta dois
objetos distintos na bancada (`red_cube` e `blue_cylinder`), transporta ambos para
novas coordenadas alvo e confirma fisicamente a mudança de posição 3D.
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


def run_scenario(config: Optional[ProjectConfig] = None) -> Dict[str, Any]:
    """Executa rastreamento e relocação de múltiplos objetos validando posições antes/depois."""
    cfg = config or load_config()
    t0 = time.perf_counter()
    controller = GripperController(config=cfg)
    artifacts: List[str] = []

    try:
        controller.gripper.reset()
        controller.simulator.set_object_position("red_cube", x=0.22, y=0.16)
        controller.simulator.set_object_position("blue_cylinder", x=-0.20, y=-0.16)

        if cfg.simulation.save_artifacts:
            before_path = controller.simulator.save_frame(
                cfg.artifacts_dir / "tracking_before.png"
            )
            if before_path:
                artifacts.append(str(before_path))

        # Confirma detecção inicial dos dois objetos via Google API
        initial_frame = controller.simulator.render_frame()
        scene_info = controller.api.analyze_scene(
            initial_frame,
            prompt="Identifique todos os objetos na bancada e retorne suas coordenadas 3D.",
        )
        detected_names = {
            obj["name"] for obj in scene_info.get("detected_objects", [])
        }
        both_detected = {"red_cube", "blue_cylinder"}.issubset(detected_names)

        # Transporte 1: red_cube -> (-0.24, 0.20)
        red_transfer = controller.perceive_and_pick_place(
            object_name="red_cube",
            destination={"x": -0.24, "y": 0.20, "z": 0.025},
        )

        # Transporte 2: blue_cylinder -> (0.24, -0.20)
        blue_transfer = controller.perceive_and_pick_place(
            object_name="blue_cylinder",
            destination={"x": 0.24, "y": -0.20, "z": 0.028},
        )

        if cfg.simulation.save_artifacts:
            after_path = controller.simulator.save_frame(
                cfg.artifacts_dir / "tracking_after.png"
            )
            if after_path:
                artifacts.append(str(after_path))

        overall_success = (
            both_detected and red_transfer["success"] and blue_transfer["success"]
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "scenario": "object_tracking",
            "success": overall_success,
            "duration_ms": round(duration_ms, 2),
            "metrics": {
                "both_objects_detected": both_detected,
                "red_cube": {
                    "from": red_transfer["initial_object_position"],
                    "to": red_transfer["final_object_position"],
                    "displacement_m": red_transfer["displacement_m"],
                    "goal_error_m": red_transfer["goal_error_m"],
                    "success": red_transfer["success"],
                },
                "blue_cylinder": {
                    "from": blue_transfer["initial_object_position"],
                    "to": blue_transfer["final_object_position"],
                    "displacement_m": blue_transfer["displacement_m"],
                    "goal_error_m": blue_transfer["goal_error_m"],
                    "success": blue_transfer["success"],
                },
            },
            "artifacts": artifacts,
        }
    finally:
        controller.shutdown()


class TestObjectTracking(unittest.TestCase):
    """Teste automatizado de rastreamento e relocação de objetos."""

    def test_multi_object_tracking_and_relocation(self) -> None:
        report = run_scenario(load_config(save_artifacts=False))
        self.assertTrue(report["success"], msg=f"Falha em rastreamento: {report}")


if __name__ == "__main__":
    res = run_scenario()
    print(
        f"[test_object_tracking] {'PASS' if res['success'] else 'FAIL'} | "
        f"Métricas: {res['metrics']}"
    )
    sys.exit(0 if res["success"] else 1)
