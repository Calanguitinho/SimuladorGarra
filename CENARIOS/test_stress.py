"""Cenário 5: Teste de estresse, estabilidade em alta frequência e recuperação de falhas.

Executa:
    1. Rajada de 120 comandos rápidos consecutivos alternando os 8 movimentos/ações.
    2. Saturação intencional de bordas do workspace (verificando clamping seguro).
    3. Injeção de parâmetros inválidos e timeout forçado (verificando rollback de estado).
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


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(
        len(sorted_values) - 1,
        max(0, int(round((pct / 100.0) * (len(sorted_values) - 1)))),
    )
    return sorted_values[idx]


def run_scenario(config: Optional[ProjectConfig] = None) -> Dict[str, Any]:
    """Executa rajada de estresse e testes de resiliência retornando métricas estatísticas."""
    cfg = config or load_config()
    t0 = time.perf_counter()
    controller = GripperController(config=cfg)

    sequence = [
        "move_right",
        "move_forward",
        "move_down",
        "close",
        "move_up",
        "move_left",
        "move_backward",
        "open",
    ]
    total_rapid_ops = 120
    latencies_ms: List[float] = []
    rapid_success_count = 0

    try:
        controller.gripper.reset()
        burst_start = time.perf_counter()

        for i in range(total_rapid_ops):
            action = sequence[i % len(sequence)]
            res = controller.execute_command(
                action, speed=0.85, force=30.0, duration=0.04
            )
            latencies_ms.append(res.elapsed_ms)
            if res.success:
                rapid_success_count += 1

        burst_elapsed_s = max(1e-6, time.perf_counter() - burst_start)
        throughput_ops_s = total_rapid_ops / burst_elapsed_s

        # Teste de saturação segura de limites do workspace (+X extremo)
        clamp_res = controller.execute_command(
            "move_right", speed=1.40, force=20.0, duration=2.0
        )
        max_x_bound = cfg.gripper.workspace_x[1]
        boundary_respected = (
            clamp_res.success
            and clamp_res.clamped
            and abs(clamp_res.position["x"] - max_x_bound) < 1e-4
        )

        # Teste de recuperação graciosa diante de parâmetro inválido (speed negativo)
        pos_before_invalid = controller.gripper.get_state()["position"]
        invalid_res = controller.execute_command(
            "move_left", speed=-2.0, force=20.0, duration=0.2
        )
        pos_after_invalid = controller.gripper.get_state()["position"]
        invalid_recovered = (
            not invalid_res.success
            and invalid_res.status == "failure"
            and pos_before_invalid == pos_after_invalid
        )

        # Teste de timeout e rollback automático de estado
        controller.gripper.set_simulated_delay(0.12)
        pos_before_timeout = controller.gripper.get_state()["position"]
        timeout_res = controller.execute_command(
            "move_left", speed=0.30, duration=0.20, timeout=0.03
        )
        controller.gripper.set_simulated_delay(0.0)
        pos_after_timeout = controller.gripper.get_state()["position"]
        timeout_recovered = (
            not timeout_res.success
            and timeout_res.status == "timeout"
            and pos_before_timeout == pos_after_timeout
        )

        # Confirma que a garra volta a operar normalmente após o timeout
        post_recovery_res = controller.execute_command(
            "move_left", speed=0.40, duration=0.25
        )
        operational_after_faults = post_recovery_res.success

        sorted_lat = sorted(latencies_ms)
        overall_success = (
            rapid_success_count == total_rapid_ops
            and boundary_respected
            and invalid_recovered
            and timeout_recovered
            and operational_after_faults
        )
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "scenario": "stress",
            "success": overall_success,
            "duration_ms": round(total_ms, 2),
            "metrics": {
                "rapid_operations_executed": total_rapid_ops,
                "rapid_operations_succeeded": rapid_success_count,
                "throughput_ops_per_sec": round(throughput_ops_s, 1),
                "avg_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
                "p95_latency_ms": round(_percentile(sorted_lat, 95.0), 3),
                "max_latency_ms": round(sorted_lat[-1], 3),
                "workspace_boundary_clamping_ok": boundary_respected,
                "invalid_param_rollback_ok": invalid_recovered,
                "timeout_rollback_ok": timeout_recovered,
                "post_fault_recovery_ok": operational_after_faults,
            },
            "artifacts": [],
        }
    finally:
        controller.gripper.set_simulated_delay(0.0)
        controller.shutdown()


class TestStressScenario(unittest.TestCase):
    """Teste automatizado de estresse e tolerância a falhas."""

    def test_high_frequency_and_fault_recovery(self) -> None:
        report = run_scenario(load_config(save_artifacts=False))
        self.assertTrue(report["success"], msg=f"Falha no estresse: {report}")


if __name__ == "__main__":
    res = run_scenario()
    print(
        f"[test_stress] {'PASS' if res['success'] else 'FAIL'} | "
        f"Métricas: {res['metrics']}"
    )
    sys.exit(0 if res["success"] else 1)
