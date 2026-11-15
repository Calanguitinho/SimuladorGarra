"""Cenário 3: Superfície suja e ruído óptico severo na câmera (`dirty_surface`).

Injeta manchas de óleo/poeira na bancada e ruído impulsivo na lente da câmera,
mede a queda de PSNR (dB), verifica a detecção da Google API e valida a manipulação
precisa do cilindro azul (`blue_cylinder`) sob 70% de contaminação visual.
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
    """Executa o cenário de sujeira/ruído visual na superfície e retorna métricas."""
    cfg = config or load_config()
    t0 = time.perf_counter()
    controller = GripperController(config=cfg)
    artifacts: List[str] = []

    try:
        dirt_levels = [0.0, 0.35, 0.70, 0.85]
        noise_evaluations: List[Dict[str, Any]] = []

        for idx, level in enumerate(dirt_levels):
            controller.simulator.set_surface_dirt(level, seed=100 + idx)
            frame = controller.simulator.render_frame()
            perception = controller.api.analyze_scene(
                frame,
                prompt=f"Analise superfície contaminada (intensidade={level:.2f}) e localize objetos.",
            )
            noise_evaluations.append(
                {
                    "dirt_intensity": level,
                    "psnr_db": frame["psnr_db"],
                    "detected_noise_level": perception["noise_level"],
                    "confidence": perception["confidence"],
                }
            )
            if cfg.simulation.save_artifacts and level in (0.0, 0.70):
                label = "clean" if level == 0.0 else "dirty_70"
                saved = controller.simulator.save_frame(
                    cfg.artifacts_dir / f"surface_{label}.png"
                )
                if saved:
                    artifacts.append(str(saved))

        psnr_degraded = (
            noise_evaluations[0]["psnr_db"] > noise_evaluations[2]["psnr_db"]
        )

        # Opera a garra para capturar e mover o cilindro azul sob 70% de sujeira/ruído
        controller.gripper.reset()
        controller.simulator.set_surface_dirt(0.70, seed=777)
        pick_blue = controller.perceive_and_pick_place(
            object_name="blue_cylinder",
            destination={"x": 0.22, "y": -0.18, "z": 0.028},
        )

        if cfg.simulation.save_artifacts:
            saved_post = controller.simulator.save_frame(
                cfg.artifacts_dir / "surface_dirty_after_pick.png"
            )
            if saved_post:
                artifacts.append(str(saved_post))

        overall_success = psnr_degraded and pick_blue["success"]
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "scenario": "dirty_surface",
            "success": overall_success,
            "duration_ms": round(duration_ms, 2),
            "metrics": {
                "psnr_degradation_verified": psnr_degraded,
                "clean_psnr_db": noise_evaluations[0]["psnr_db"],
                "dirty_70_psnr_db": noise_evaluations[2]["psnr_db"],
                "pick_place_on_dirty_surface": pick_blue["success"],
                "blue_cylinder_displacement_m": pick_blue["displacement_m"],
                "blue_cylinder_goal_error_m": pick_blue["goal_error_m"],
                "noise_sweep": noise_evaluations,
            },
            "artifacts": artifacts,
        }
    finally:
        controller.shutdown()


class TestDirtySurface(unittest.TestCase):
    """Teste automatizado do cenário de ruído visual e superfície suja."""

    def test_dirty_surface_resilience(self) -> None:
        report = run_scenario(load_config(save_artifacts=False))
        self.assertTrue(report["success"], msg=f"Falha em superfície suja: {report}")


if __name__ == "__main__":
    res = run_scenario()
    print(
        f"[test_dirty_surface] {'PASS' if res['success'] else 'FAIL'} | "
        f"Métricas: {res['metrics']}"
    )
    sys.exit(0 if res["success"] else 1)
