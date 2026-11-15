"""Controlador das 8 funções de movimento e preensão da garra robótica sobre o MuJoCo.

Implementa:
    - move_right(speed, force, duration)
    - move_left(speed, force, duration)
    - move_forward(speed, force, duration)
    - move_backward(speed, force, duration)
    - move_down(speed, force, duration)
    - move_up(speed, force, duration)
    - close(speed, force, duration)
    - open(speed, force, duration)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, Optional

from backend.config import BackendConfig, configure_backend_logger, load_backend_config
from backend.mujoco_env import MuJoCoGripperEnv


class InvalidCommandError(ValueError):
    """Exceção lançada quando um comando ou parâmetro WebSocket é inválido."""


class GripperController:
    """Camada de controle de movimento da garra com validação de força e velocidade."""

    DIRECTION_AXES = {
        "move_right": (1.0, 0.0, 0.0),
        "move_left": (-1.0, 0.0, 0.0),
        "move_forward": (0.0, 1.0, 0.0),
        "move_backward": (0.0, -1.0, 0.0),
        "move_up": (0.0, 0.0, 1.0),
        "move_down": (0.0, 0.0, -1.0),
    }

    def __init__(
        self,
        env: Optional[MuJoCoGripperEnv] = None,
        config: Optional[BackendConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or (env.config if env is not None else load_backend_config())
        self.env = env or MuJoCoGripperEnv(config=self.config)
        self.logger = logger or configure_backend_logger(
            "GripperController", level=self.config.server.log_level
        )
        self.command_counter: int = 0

        self._dispatch_table: Dict[str, Callable[..., Dict[str, Any]]] = {
            "move_right": self.move_right,
            "move_left": self.move_left,
            "move_forward": self.move_forward,
            "move_backward": self.move_backward,
            "move_down": self.move_down,
            "move_up": self.move_up,
            "close": self.close,
            "open": self.open,
        }

    # ------------------------------------------------------------------------
    # Validação e Execução Base
    # ------------------------------------------------------------------------

    def _validate_params(
        self,
        speed: Optional[float],
        force: Optional[float],
        duration: Optional[float],
    ) -> tuple[float, float, float]:
        gcfg = self.config.gripper
        s = gcfg.default_speed if speed is None else float(speed)
        f = gcfg.default_force if force is None else float(force)
        d = gcfg.default_step_duration if duration is None else float(duration)

        for label, val in (("speed", s), ("force", f), ("duration", d)):
            if math.isnan(val) or math.isinf(val) or val <= 0.0:
                raise InvalidCommandError(
                    f"Parâmetro '{label}' deve ser um número positivo finito (recebido: {val})."
                )

        s = max(gcfg.min_speed, min(gcfg.max_speed, s))
        f = max(gcfg.min_force, min(gcfg.max_force, f))
        d = max(0.01, min(5.0, d))
        return s, f, d

    def _apply_directional(
        self,
        command_name: str,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        step_immediately: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        s, f, d = self._validate_params(speed, force, duration)
        ux, uy, uz = self.DIRECTION_AXES[command_name]
        distance = s * d

        self.env.set_motion_target(
            dx=ux * distance,
            dy=uy * distance,
            dz=uz * distance,
            speed=s,
            force=f,
        )
        state = (
            self.env.step_until_settled(d)
            if step_immediately
            else self.env.step_physics(substeps=4)
        )

        self.command_counter += 1
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self.logger.info(
            "CMD #%d %-14s | speed=%.2fm/s force=%.1fN dur=%.2fs -> pos=%s (%.2fms)",
            self.command_counter,
            command_name,
            s,
            f,
            d,
            state["pos"],
            elapsed_ms,
        )
        return state

    def _apply_jaw(
        self,
        close_jaws: bool,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        step_immediately: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        cmd_name = "close" if close_jaws else "open"
        s, f, d = self._validate_params(speed, force, duration)

        self.env.set_motion_target(close_jaws=close_jaws, speed=s, force=f)
        state = (
            self.env.step_until_settled(max(d, 0.28))
            if step_immediately
            else self.env.step_physics(substeps=6)
        )

        self.command_counter += 1
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self.logger.info(
            "CMD #%d %-14s | speed=%.2fm/s force=%.1fN -> aperture=%.4fm contact=%.1fN (%.2fms)",
            self.command_counter,
            cmd_name,
            s,
            f,
            state["aperture"],
            state["contact_force"],
            elapsed_ms,
        )
        return state

    # ------------------------------------------------------------------------
    # 8 Funções Principais de Movimento e Preensão
    # ------------------------------------------------------------------------

    def move_right(
        self,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        *,
        step_immediately: bool = False,
    ) -> Dict[str, Any]:
        """Move a garra no sentido +X."""
        return self._apply_directional(
            "move_right", speed, force, duration, step_immediately=step_immediately
        )

    def move_left(
        self,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        *,
        step_immediately: bool = False,
    ) -> Dict[str, Any]:
        """Move a garra no sentido -X."""
        return self._apply_directional(
            "move_left", speed, force, duration, step_immediately=step_immediately
        )

    def move_forward(
        self,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        *,
        step_immediately: bool = False,
    ) -> Dict[str, Any]:
        """Move a garra no sentido +Y."""
        return self._apply_directional(
            "move_forward", speed, force, duration, step_immediately=step_immediately
        )

    def move_backward(
        self,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        *,
        step_immediately: bool = False,
    ) -> Dict[str, Any]:
        """Move a garra no sentido -Y."""
        return self._apply_directional(
            "move_backward", speed, force, duration, step_immediately=step_immediately
        )

    def move_down(
        self,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        *,
        step_immediately: bool = False,
    ) -> Dict[str, Any]:
        """Move a garra verticalmente para baixo (-Z)."""
        return self._apply_directional(
            "move_down", speed, force, duration, step_immediately=step_immediately
        )

    def move_up(
        self,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        *,
        step_immediately: bool = False,
    ) -> Dict[str, Any]:
        """Move a garra verticalmente para cima (+Z)."""
        return self._apply_directional(
            "move_up", speed, force, duration, step_immediately=step_immediately
        )

    def close(
        self,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        *,
        step_immediately: bool = False,
    ) -> Dict[str, Any]:
        """Fecha os dedos paralelos da garra com limite de força configurável."""
        return self._apply_jaw(
            True, speed, force, duration, step_immediately=step_immediately
        )

    def open(
        self,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        *,
        step_immediately: bool = False,
    ) -> Dict[str, Any]:
        """Abre os dedos paralelos da garra."""
        return self._apply_jaw(
            False, speed, force, duration, step_immediately=step_immediately
        )

    # ------------------------------------------------------------------------
    # Despacho Unificado WebSocket / REST
    # ------------------------------------------------------------------------

    def execute_payload(
        self, payload: Dict[str, Any], *, settle: bool = False
    ) -> Dict[str, Any]:
        """Processa mensagens JSON `{"command": ..., "speed": ..., "force": ...}`."""
        cmd = str(payload.get("command", "")).strip()
        if not cmd:
            raise InvalidCommandError("Campo 'command' obrigatório ausente no JSON.")

        if cmd == "reset":
            self.command_counter += 1
            self.logger.info("CMD #%d reset          | Cena MuJoCo restaurada", self.command_counter)
            return self.env.reset()

        if cmd == "get_state" or cmd == "ping":
            return self.env.get_state_snapshot()

        if cmd == "move_to":
            # Comando utilitário opcional para posicionamento cartesiano direto
            speed = payload.get("speed")
            force = payload.get("force")
            target = payload.get("target", {})
            self.env.set_absolute_target(
                x=target.get("x"),
                y=target.get("y"),
                tcp_z=target.get("z"),
                speed=speed,
                force=force,
            )
            state = (
                self.env.step_until_settled(float(payload.get("duration", 0.45)))
                if settle or payload.get("settle", False)
                else self.env.step_physics(substeps=4)
            )
            self.command_counter += 1
            self.logger.info(
                "CMD #%d move_to        | target=%s -> pos=%s",
                self.command_counter,
                target,
                state["pos"],
            )
            return state

        handler = self._dispatch_table.get(cmd)
        if handler is None:
            raise InvalidCommandError(
                f"Comando desconhecido '{cmd}'. Suportados: {sorted(self._dispatch_table)}"
            )

        return handler(
            speed=payload.get("speed"),
            force=payload.get("force"),
            duration=payload.get("duration"),
            step_immediately=bool(settle or payload.get("settle", False)),
        )
