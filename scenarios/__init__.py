"""Suíte de cenários de validação fim a fim via WebSocket para o SimuladorGarra 3D."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict

from scenarios import (
    test_basic_movement,
    test_object_interaction,
    test_real_time_sync,
    test_stress,
)

SCENARIO_MAP: Dict[str, Callable[[], Dict[str, Any]]] = {
    "basic_movement": test_basic_movement.run_scenario,
    "real_time_sync": test_real_time_sync.run_scenario,
    "object_interaction": test_object_interaction.run_scenario,
    "stress": test_stress.run_scenario,
}


def run_selected_scenarios(name: str = "all") -> int:
    selected = (
        list(SCENARIO_MAP.keys()) if name == "all" else [name]
    )
    all_ok = True
    for key in selected:
        if key not in SCENARIO_MAP:
            print(f"Cenário desconhecido: '{key}'. Opções: {list(SCENARIO_MAP)} ou 'all'")
            return 2
        report = SCENARIO_MAP[key]()
        status = "PASS" if report.get("success") else "FAIL"
        print(f"\n=== [{status}] scenarios/{key} ({report.get('duration_ms', 0):.1f} ms) ===")
        print(json.dumps(report.get("metrics", {}), indent=2, ensure_ascii=False))
        if not report.get("success"):
            all_ok = False
    return 0 if all_ok else 1
