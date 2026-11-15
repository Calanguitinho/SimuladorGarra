"""Classe base da garra robótica com inicialização, validação, estado e recuperação de falhas.

Implementa uma interface estilo socket/adaptador (`backend`) que permite acoplar
tanto o simulador MuJoCo quanto um driver de hardware real (ex.: Serial, ROS 2, TCP)
ou operar de forma autônoma em memória para testes unitários isolados.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

from CODIGO.config import GripperSpecs
from FUNCOES.utils import (
    CommandResult,
    GripperError,
    GripperTimeoutError,
    InvalidParameterError,
    SimulatorConnectionError,
    StateManager,
    clamp,
    execute_with_timeout,
    setup_logger,
)


@runtime_checkable
class GripperBackendProtocol(Protocol):
    """Protocolo estilo socket para backends de simulação (MuJoCo) ou hardware físico."""

    def apply_cartesian_delta(
        self,
        dx: float,
        dy: float,
        dz: float,
        speed: float,
        force: float,
        duration: float,
    ) -> Dict[str, Any]:
        """Aplica deslocamento cartesiano no backend e retorna telemetria atualizada."""

    def apply_jaw_action(
        self,
        close_jaws: bool,
        speed: float,
        force: float,
        duration: float,
    ) -> Dict[str, Any]:
        """Aciona abertura ou fechamento dos dedos da garra e retorna telemetria."""

    def sync_state(self) -> Dict[str, Any]:
        """Lê o estado físico atual do backend."""


class GripperBase:
    """Classe base de controle da garra robótica com gerenciamento de estado e segurança."""

    DIRECTION_VECTORS: Dict[str, Tuple[float, float, float]] = {
        "move_right": (1.0, 0.0, 0.0),
        "move_left": (-1.0, 0.0, 0.0),
        "move_forward": (0.0, 1.0, 0.0),
        "move_backward": (0.0, -1.0, 0.0),
        "move_up": (0.0, 0.0, 1.0),
        "move_down": (0.0, 0.0, -1.0),
    }

    def __init__(
        self,
        specs: Optional[GripperSpecs] = None,
        backend: Optional[GripperBackendProtocol] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.specs = specs or GripperSpecs()
        self.logger = logger or setup_logger("GripperBase")
        self._backend: Optional[GripperBackendProtocol] = None
        self.state_manager = StateManager(
            initial_position=self.specs.home_position,
            initial_orientation=self.specs.home_orientation,
            initial_aperture=self.specs.max_aperture,
        )
        self.is_initialized: bool = False
        self._simulated_latency_s: float = 0.0

        if backend is not None:
            self.attach_backend(backend)
        else:
            self.initialize()

    def initialize(self) -> bool:
        """Inicializa a garra, sincronizando com o backend se estiver conectado."""
        try:
            self.state_manager.reset(
                position=self.specs.home_position,
                orientation=self.specs.home_orientation,
                aperture=self.specs.max_aperture,
            )
            if self._backend is not None:
                telemetry = self._backend.sync_state()
                self._ingest_backend_telemetry(telemetry, command_name="initialize")
            self.is_initialized = True
            self.logger.info(
                "Garra '%s' inicializada em %s (abertura=%.3fm).",
                self.specs.name,
                self.state_manager.state.position,
                self.state_manager.state.aperture,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self.is_initialized = False
            self.logger.error("Falha ao inicializar garra: %s", exc)
            raise SimulatorConnectionError(f"Erro na inicialização da garra: {exc}") from exc

    def attach_backend(self, backend: GripperBackendProtocol) -> None:
        """Conecta um simulador MuJoCo ou controlador de hardware real como backend."""
        self._backend = backend
        self.initialize()

    def detach_backend(self) -> None:
        """Desconecta o backend atual, mantendo operação cinemática local segura."""
        self._backend = None
        self.logger.info("Backend desconectado; operando em modo cinemático autônomo.")

    def set_simulated_delay(self, delay_seconds: float) -> None:
        """Configura atraso artificial (útil para testes de timeout e resiliência)."""
        self._simulated_latency_s = max(0.0, float(delay_seconds))

    def get_state(self) -> Dict[str, Any]:
        """Retorna um snapshot completo do estado atual da garra."""
        return self.state_manager.state.snapshot()

    def reset(self) -> Dict[str, Any]:
        """Reposiciona a garra na posição home e restaura estado limpo."""
        if self._backend is not None and hasattr(self._backend, "reset"):
            self._backend.reset()  # type: ignore[attr-defined]
        self.initialize()
        return self.get_state()

    # ------------------------------------------------------------------------
    # Validação e Execução Protegida
    # ------------------------------------------------------------------------

    def validate_parameters(
        self,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Valida e normaliza velocidade (m/s), força (N) e duração (s)."""
        resolved_speed = self.specs.default_speed if speed is None else float(speed)
        resolved_force = self.specs.default_force if force is None else float(force)
        resolved_duration = (
            self.specs.default_duration if duration is None else float(duration)
        )

        for name, val in (
            ("speed", resolved_speed),
            ("force", resolved_force),
            ("duration", resolved_duration),
        ):
            if math.isnan(val) or math.isinf(val) or val <= 0.0:
                raise InvalidParameterError(
                    f"Parâmetro '{name}' inválido ({val}). Deve ser um número finito positivo."
                )

        if resolved_speed > self.specs.max_speed * 2.0:
            raise InvalidParameterError(
                f"Velocidade solicitada ({resolved_speed:.2f} m/s) excede limite crítico de segurança."
            )
        if resolved_force > self.specs.max_force * 2.0:
            raise InvalidParameterError(
                f"Força solicitada ({resolved_force:.2f} N) excede limite crítico estrutural."
            )

        resolved_speed, _ = clamp(resolved_speed, self.specs.min_speed, self.specs.max_speed)
        resolved_force, _ = clamp(resolved_force, self.specs.min_force, self.specs.max_force)
        resolved_duration, _ = clamp(
            resolved_duration, self.specs.min_duration, self.specs.max_duration
        )
        return resolved_speed, resolved_force, resolved_duration

    def execute_translation(
        self,
        command_name: str,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> CommandResult:
        """Executa um dos 6 movimentos cartesianos com proteção de timeout e rollback."""
        start_perf = time.perf_counter()
        checkpoint = self.state_manager.save_checkpoint()
        effective_timeout = (
            self.specs.command_timeout_seconds if timeout is None else float(timeout)
        )

        try:
            if not self.is_initialized:
                self.initialize()

            if command_name not in self.DIRECTION_VECTORS:
                raise GripperError(f"Comando direcional desconhecido: {command_name}")

            v_speed, v_force, v_duration = self.validate_parameters(speed, force, duration)
            ux, uy, uz = self.DIRECTION_VECTORS[command_name]
            distance = v_speed * v_duration

            current_pos = checkpoint.position
            raw_x = current_pos["x"] + ux * distance
            raw_y = current_pos["y"] + uy * distance
            raw_z = current_pos["z"] + uz * distance

            target_x, cx = clamp(raw_x, *self.specs.workspace_x)
            target_y, cy = clamp(raw_y, *self.specs.workspace_y)
            target_z, cz = clamp(raw_z, *self.specs.workspace_z)
            was_clamped = cx or cy or cz

            dx = target_x - current_pos["x"]
            dy = target_y - current_pos["y"]
            dz = target_z - current_pos["z"]

            def _step_work() -> Dict[str, Any]:
                if self._simulated_latency_s > 0:
                    time.sleep(self._simulated_latency_s)
                if self._backend is not None:
                    return self._backend.apply_cartesian_delta(
                        dx=dx,
                        dy=dy,
                        dz=dz,
                        speed=v_speed,
                        force=v_force,
                        duration=v_duration,
                    )
                return {
                    "position": {"x": target_x, "y": target_y, "z": target_z},
                    "orientation": checkpoint.orientation,
                    "grip_force": checkpoint.grip_force,
                    "aperture": checkpoint.aperture,
                    "is_closed": checkpoint.is_closed,
                    "is_grasping": checkpoint.is_grasping,
                    "grasped_object": checkpoint.grasped_object,
                }

            telemetry = execute_with_timeout(_step_work, effective_timeout)
            updated = self._ingest_backend_telemetry(telemetry, command_name=command_name)
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            status = "workspace_limit_clamped" if was_clamped else "success"
            msg = (
                f"{command_name} concluído -> pos=({updated.position['x']:.3f}, "
                f"{updated.position['y']:.3f}, {updated.position['z']:.3f})"
            )
            if was_clamped:
                msg += " [limitado às bordas seguras do workspace]"

            self.logger.info(
                "[%s] speed=%.2fm/s force=%.1fN dur=%.2fs | %s (%.2fms)",
                command_name,
                v_speed,
                v_force,
                v_duration,
                msg,
                elapsed_ms,
            )
            return CommandResult(
                success=True,
                status=status,
                command=command_name,
                position=updated.position,
                orientation=updated.orientation,
                grip_force=updated.grip_force,
                aperture=updated.aperture,
                is_grasping=updated.is_grasping,
                grasped_object=updated.grasped_object,
                clamped=was_clamped,
                elapsed_ms=elapsed_ms,
                message=msg,
                metadata={
                    "requested_speed": v_speed,
                    "requested_force": v_force,
                    "requested_duration": v_duration,
                },
            )

        except GripperTimeoutError as exc:
            recovered = self.state_manager.record_error(rollback_to=checkpoint)
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            self.logger.error("[%s] Timeout! Estado restaurado: %s", command_name, exc)
            return CommandResult(
                success=False,
                status="timeout",
                command=command_name,
                position=recovered.position,
                orientation=recovered.orientation,
                grip_force=recovered.grip_force,
                aperture=recovered.aperture,
                is_grasping=recovered.is_grasping,
                grasped_object=recovered.grasped_object,
                elapsed_ms=elapsed_ms,
                message=f"Timeout ao executar {command_name}; rollback aplicado.",
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            recovered = self.state_manager.record_error(rollback_to=checkpoint)
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            self.logger.error("[%s] Erro recuperado com rollback: %s", command_name, exc)
            return CommandResult(
                success=False,
                status="failure",
                command=command_name,
                position=recovered.position,
                orientation=recovered.orientation,
                grip_force=recovered.grip_force,
                aperture=recovered.aperture,
                is_grasping=recovered.is_grasping,
                grasped_object=recovered.grasped_object,
                elapsed_ms=elapsed_ms,
                message=f"Falha ao executar {command_name}; estado seguro mantido.",
                error=str(exc),
            )

    def execute_jaw_action(
        self,
        close_jaws: bool,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> CommandResult:
        """Executa fechamento (`close`) ou abertura (`open`) dos dedos paralelos."""
        command_name = "close" if close_jaws else "open"
        start_perf = time.perf_counter()
        checkpoint = self.state_manager.save_checkpoint()
        effective_timeout = (
            self.specs.command_timeout_seconds if timeout is None else float(timeout)
        )

        try:
            if not self.is_initialized:
                self.initialize()

            v_speed, v_force, v_duration = self.validate_parameters(speed, force, duration)
            aperture_delta = v_speed * v_duration
            if close_jaws:
                target_aperture, _ = clamp(
                    checkpoint.aperture - aperture_delta * 2.0,
                    self.specs.min_aperture,
                    self.specs.max_aperture,
                )
                # Se a duração padrão é suficiente, fecha até o fim
                if duration is None:
                    target_aperture = self.specs.min_aperture
                applied_force = v_force
            else:
                target_aperture, _ = clamp(
                    checkpoint.aperture + aperture_delta * 2.0,
                    self.specs.min_aperture,
                    self.specs.max_aperture,
                )
                if duration is None:
                    target_aperture = self.specs.max_aperture
                applied_force = 0.0

            def _jaw_work() -> Dict[str, Any]:
                if self._simulated_latency_s > 0:
                    time.sleep(self._simulated_latency_s)
                if self._backend is not None:
                    return self._backend.apply_jaw_action(
                        close_jaws=close_jaws,
                        speed=v_speed,
                        force=v_force,
                        duration=v_duration,
                    )
                return {
                    "position": checkpoint.position,
                    "orientation": checkpoint.orientation,
                    "grip_force": applied_force,
                    "aperture": target_aperture,
                    "is_closed": close_jaws,
                    "is_grasping": False,
                    "grasped_object": None,
                }

            telemetry = execute_with_timeout(_jaw_work, effective_timeout)
            updated = self._ingest_backend_telemetry(telemetry, command_name=command_name)
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0

            msg = (
                f"{command_name} concluído -> abertura={updated.aperture:.4f}m, "
                f"força={updated.grip_force:.1f}N, objeto={updated.grasped_object or 'nenhum'}"
            )
            self.logger.info("[%s] %s (%.2fms)", command_name, msg, elapsed_ms)
            return CommandResult(
                success=True,
                status="success",
                command=command_name,
                position=updated.position,
                orientation=updated.orientation,
                grip_force=updated.grip_force,
                aperture=updated.aperture,
                is_grasping=updated.is_grasping,
                grasped_object=updated.grasped_object,
                elapsed_ms=elapsed_ms,
                message=msg,
                metadata={
                    "requested_speed": v_speed,
                    "requested_force": v_force,
                    "requested_duration": v_duration,
                },
            )

        except GripperTimeoutError as exc:
            recovered = self.state_manager.record_error(rollback_to=checkpoint)
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            self.logger.error("[%s] Timeout nos dedos da garra: %s", command_name, exc)
            return CommandResult(
                success=False,
                status="timeout",
                command=command_name,
                position=recovered.position,
                orientation=recovered.orientation,
                grip_force=recovered.grip_force,
                aperture=recovered.aperture,
                is_grasping=recovered.is_grasping,
                grasped_object=recovered.grasped_object,
                elapsed_ms=elapsed_ms,
                message=f"Timeout em {command_name}; estado anterior restaurado.",
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            recovered = self.state_manager.record_error(rollback_to=checkpoint)
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            self.logger.error("[%s] Falha nos dedos da garra: %s", command_name, exc)
            return CommandResult(
                success=False,
                status="failure",
                command=command_name,
                position=recovered.position,
                orientation=recovered.orientation,
                grip_force=recovered.grip_force,
                aperture=recovered.aperture,
                is_grasping=recovered.is_grasping,
                grasped_object=recovered.grasped_object,
                elapsed_ms=elapsed_ms,
                message=f"Erro ao executar {command_name}; rollback concluído.",
                error=str(exc),
            )

    def _ingest_backend_telemetry(
        self, telemetry: Dict[str, Any], command_name: str
    ) -> Any:
        """Sincroniza o `StateManager` com os dados retornados pelo simulador/hardware."""
        grasped_obj = telemetry.get("grasped_object")
        return self.state_manager.update(
            position=telemetry.get("position"),
            orientation=telemetry.get("orientation"),
            grip_force=telemetry.get("grip_force"),
            aperture=telemetry.get("aperture"),
            is_closed=telemetry.get("is_closed"),
            is_grasping=telemetry.get("is_grasping"),
            grasped_object=grasped_obj,
            clear_grasped_object=(grasped_obj is None),
            command_name=command_name,
        )
