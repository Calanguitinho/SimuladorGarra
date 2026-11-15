#!/usr/bin/env python3
"""Ponto de entrada unificado na raiz do projeto.

Modos suportados:
    1. Servidor Web 3D + MuJoCo WebSocket (padrão):
       `python main.py --port 8000`
    2. Execução direta dos cenários WebSocket (`scenarios/`):
       `python main.py --scenario all`
    3. Compatibilidade com a CLI anterior (`CODIGO/main.py`):
       `python main.py --test fog_environment` ou `python main.py --demo`
"""

from __future__ import annotations

import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)

    # Mantém compatibilidade integral com os flags legados (--test / --demo)
    if any(flag in raw_args for flag in ("--test", "-t", "--demo")):
        from CODIGO.main import main as legacy_main

        return legacy_main(raw_args)

    if "--scenario" in raw_args:
        idx = raw_args.index("--scenario")
        target = raw_args[idx + 1] if idx + 1 < len(raw_args) else "all"
        from scenarios import run_selected_scenarios

        return run_selected_scenarios(target)

    from backend.main import main as backend_server_main

    return backend_server_main(raw_args)


if __name__ == "__main__":
    sys.exit(main())
