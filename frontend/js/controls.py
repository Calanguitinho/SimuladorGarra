"""Mapeamento declarativo em Python dos controles de UI (botões e sliders) do frontend.

Compatível tanto com ambientes Python no navegador (Pyodide / Brython) quanto com
clientes de automação/teste em Python para geração validada de payloads JSON WebSocket:
    `{"command": "move_right", "speed": 0.5, "force": 50}`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

SUPPORTED_UI_COMMANDS: Tuple[str, ...] = (
    "move_right",
    "move_left",
    "move_forward",
    "move_backward",
    "move_down",
    "move_up",
    "close",
    "open",
)


@dataclass
class ControlPanelState:
    """Estado reativo dos sliders de velocidade (m/s) e força (N) do painel de controle."""

    speed: float = 0.50
    force: float = 50.0
    min_speed: float = 0.05
    max_speed: float = 1.50
    min_force: float = 1.0
    max_force: float = 100.0

    def set_speed(self, value: float) -> float:
        """Ajusta o slider de velocidade dentro da faixa operacional segura."""
        self.speed = round(max(self.min_speed, min(self.max_speed, float(value))), 3)
        return self.speed

    def set_force(self, value: float) -> float:
        """Ajusta o slider de força máxima das garras em Newtons."""
        self.force = round(max(self.min_force, min(self.max_force, float(value))), 2)
        return self.force

    def build_command_message(
        self, command: str, duration: float | None = None
    ) -> Dict[str, Any]:
        """Constrói o payload JSON oficial do protocolo WebSocket para o comando selecionado."""
        if command not in SUPPORTED_UI_COMMANDS and command != "reset":
            raise ValueError(
                f"Comando de UI inválido '{command}'. Válidos: {SUPPORTED_UI_COMMANDS}"
            )
        msg: Dict[str, Any] = {
            "command": command,
            "speed": self.speed,
            "force": self.force,
        }
        if duration is not None:
            msg["duration"] = round(float(duration), 3)
        return msg
