"""Ponto de entrada principal (`CLI`) que orquestra o simulador e os cenários de teste.

Exemplos de uso:
    python main.py --test basic_movement
    python main.py --test fog_environment
    python main.py --test dirty_surface
    python main.py --test object_tracking
    python main.py --test stress
    python main.py --test all
    python main.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

# Garante que a raiz do projeto esteja no sys.path mesmo executando `python CODIGO/main.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CENARIOS import (  # noqa: E402
    test_basic_movement,
    test_dirty_surface,
    test_fog_environment,
    test_object_tracking,
    test_stress,
)
from CODIGO.config import load_config  # noqa: E402
from CODIGO.controller import GripperController  # noqa: E402
from FUNCOES.utils import setup_logger  # noqa: E402

SCENARIO_RUNNERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "basic_movement": test_basic_movement.run_scenario,
    "fog_environment": test_fog_environment.run_scenario,
    "dirty_surface": test_dirty_surface.run_scenario,
    "object_tracking": test_object_tracking.run_scenario,
    "stress": test_stress.run_scenario,
}


def print_scenario_report(report: Dict[str, Any]) -> None:
    """Exibe no console um resumo formatado de status e métricas de desempenho."""
    banner = "=" * 74
    status_label = "SUCESSO (PASS)" if report.get("success") else "FALHA (FAIL)"
    print(f"\n{banner}")
    print(f"CENÁRIO : {report.get('scenario', 'desconhecido')}")
    print(f"STATUS  : {status_label}")
    print(f"DURAÇÃO : {report.get('duration_ms', 0.0):.2f} ms")
    print("-" * 74)
    metrics = report.get("metrics", {})
    for key, val in metrics.items():
        if isinstance(val, (dict, list)):
            formatted = json.dumps(val, ensure_ascii=False)
            print(f"  • {key}: {formatted}")
        else:
            print(f"  • {key}: {val}")
    if report.get("artifacts"):
        print("-" * 74)
        print("ARTEFATOS GERADOS:")
        for artifact in report["artifacts"]:
            print(f"  -> {artifact}")
    print(f"{banner}\n")


def run_demo(config_kwargs: Dict[str, Any]) -> int:
    """Executa uma demonstração completa da arquitetura estilo socket + pick-and-place."""
    cfg = load_config(**config_kwargs)
    logger = setup_logger("MainDemo", level=cfg.log_level, log_dir=cfg.log_dir)
    controller = GripperController(config=cfg, enable_socket_server=True, logger=logger)

    try:
        print("\n[DEMO] Iniciando demonstração integrada SimuladorGarra...")
        socket_res = controller.execute_command(
            "move_right", speed=0.20, duration=0.25, via_socket=True
        )
        print(f"[SOCKET] Resposta TCP loopback: {socket_res.message}")

        controller.gripper.reset()
        cycle = controller.perceive_and_pick_place("red_cube")
        if cfg.simulation.save_artifacts:
            controller.simulator.save_frame(cfg.artifacts_dir / "demo_pick_place.png")

        status = "SUCESSO" if cycle["success"] else "FALHA"
        print(
            f"[PICK & PLACE] Status={status} | Objeto='{cycle['object_name']}' | "
            f"Deslocamento={cycle['displacement_m']:.3f}m | "
            f"Erro ao alvo={cycle['goal_error_m']:.3f}m | "
            f"Tempo={cycle['total_time_ms']:.2f}ms\n"
        )
        return 0 if cycle["success"] else 1
    finally:
        controller.shutdown()


def build_parser() -> argparse.ArgumentParser:
    """Constrói o parser de linha de comando (`argparse`)."""
    parser = argparse.ArgumentParser(
        prog="SimuladorGarra",
        description=(
            "Sistema modular de controle de garra robótica com MuJoCo e Google Gemini API."
        ),
    )
    parser.add_argument(
        "--test",
        "-t",
        type=str,
        choices=[*SCENARIO_RUNNERS.keys(), "all"],
        help=(
            "Executa um cenário de teste específico "
            "(basic_movement, fog_environment, dirty_surface, object_tracking, stress ou all)."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Executa uma demonstração completa do ciclo percepção + transporte + socket.",
    )
    parser.add_argument(
        "--api-mode",
        choices=["auto", "live", "mock"],
        default=None,
        help="Modo de operação da Google API (padrão: auto).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nível de verbosidade dos logs.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Desativa o salvamento de capturas PNG na pasta artifacts/.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    """Função principal de entrada via CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config_kwargs = {
        "api_mode": args.api_mode,
        "log_level": args.log_level,
        "save_artifacts": not args.no_artifacts,
    }
    cfg = load_config(**config_kwargs)

    if args.test:
        selected = (
            list(SCENARIO_RUNNERS.keys()) if args.test == "all" else [args.test]
        )
        all_passed = True
        reports: List[Dict[str, Any]] = []

        for scenario_name in selected:
            runner = SCENARIO_RUNNERS[scenario_name]
            report = runner(config=cfg)
            reports.append(report)
            print_scenario_report(report)
            if not report.get("success", False):
                all_passed = False

        if len(reports) > 1:
            passed_count = sum(1 for r in reports if r.get("success"))
            print(
                f"RESUMO GERAL: {passed_count}/{len(reports)} cenários aprovados "
                f"({'TODOS PASSARAM' if all_passed else 'HOUVERAM FALHAS'})."
            )
        return 0 if all_passed else 1

    # Caso nenhum --test seja passado, executa o modo demonstração padrão
    return run_demo(config_kwargs)


if __name__ == "__main__":
    sys.exit(main())
