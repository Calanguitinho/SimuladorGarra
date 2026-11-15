"""Cenário 2: Operação da garra sob visibilidade reduzida / neblina densa (`fog_environment`).

Avalia a degradação progressiva de contraste óptico (0%, 40%, 75%, 90% de neblina),
a resposta da camada Google API (compensação adaptativa e confiança) e a execução
completa de pick-and-place sob neblina densa (75%).
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
    """Executa o teste em ambiente com neblina e retorna métricas ópticas e de manipulação."""
    cfg = config or load_config()
    t0 = time.perf_counter()
    controller = GripperController(config=cfg)
    artifacts: List[str] = []

    try:
        fog_levels = [0.0, 0.40, 0.75, 0.90]
        sweep_metrics: List[Dict[str, Any]] = []

        for density in fog_levels:
            controller.simulator.set_fog(density)
            frame = controller.simulator.render_frame()
            analysis = controller.api.analyze_scene(
                frame,
                prompt=f"Avalie cena com densidade de neblina={density:.2f} e localize objetos.",
            )
            sweep_metrics.append(
                {
                    "fog_density": density,
                    "contrast_rms": frame["contrast_rms"],
                    "psnr_db": frame["psnr_db"],
                    "visibility_score": analysis["visibility_score"],
                    "api_confidence": analysis["confidence"],
                }
            )
            if cfg.simulation.save_artifacts and density in (0.0, 0.75):
                tag = int(density * 100)
                saved = controller.simulator.save_frame(
                    cfg.artifacts_dir / f"fog_density_{tag:02d}.png"
                )
                if saved:
                    artifacts.append(str(saved))

        # Confirma que a neblina realmente reduz o contraste RMS da câmera
        contrast_dropped = (
            sweep_metrics[0]["contrast_rms"] > sweep_metrics[-1]["contrast_rms"]
        )

        # Executa pick-and-place real sob 75% de neblina
        controller.gripper.reset()
        controller.simulator.set_fog(0.75)
        pick_result = controller.perceive_and_pick_place("red_cube")

        if cfg.simulation.save_artifacts:
            saved_after = controller.simulator.save_frame(
                cfg.artifacts_dir / "fog_pick_place_after.png"
            )
            if saved_after:
                artifacts.append(str(saved_after))

        overall_success = contrast_dropped and pick_result["success"]
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "scenario": "fog_environment",
            "success": overall_success,
            "duration_ms": round(duration_ms, 2),
            "metrics": {
                "contrast_degradation_verified": contrast_dropped,
                "clean_contrast_rms": sweep_metrics[0]["contrast_rms"],
                "dense_fog_75_contrast_rms": sweep_metrics[2]["contrast_rms"],
                "dense_fog_75_psnr_db": sweep_metrics[2]["psnr_db"],
                "pick_place_under_75_fog_success": pick_result["success"],
                "object_displacement_m": pick_result["displacement_m"],
                "goal_error_m": pick_result["goal_error_m"],
                "fog_sweep": sweep_metrics,
            },
            "artifacts": artifacts,
        }
    finally:
        controller.shutdown()


class TestFogEnvironment(unittest.TestCase):
    """Teste automatizado do cenário de neblina."""

    def test_fog_visibility_and_manipulation(self) -> None:
        report = run_scenario(load_config(save_artifacts=False))
        self.assertTrue(report["success"], msg=f"Falha sob neblina: {report}")


if __name__ == "__main__":
    res = run_scenario()
    print(
        f"[test_fog_environment] {'PASS' if res['success'] else 'FAIL'} | "
        f"Métricas: {res['metrics']}"
    )
    sys.exit(0 if res["success"] else 1)
