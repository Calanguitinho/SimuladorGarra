"""Loop principal de controle que integra garra, comandos, Google API e simulador MuJoCo.

Suporta despacho direto em memória e despacho desacoplado via `SocketCommandBridge`
(arquitetura modular estilo socket), facilitando testes automatizados e migração
direta do simulador para controladores robóticos físicos.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

from CODIGO.config import ProjectConfig, load_config
from CODIGO.simulator import MuJoCoSimulator
from FUNCOES.api_connector import GoogleAPIConnector
from FUNCOES.gripper_base import GripperBase
from FUNCOES.gripper_commands import COMMAND_REGISTRY, set_default_gripper
from FUNCOES.utils import (
    CommandPacket,
    CommandResult,
    GripperError,
    SocketCommandBridge,
    setup_logger,
)


class GripperController:
    """Orquestrador central do pipeline percepção -> planejamento -> ação da garra."""

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        simulator: Optional[MuJoCoSimulator] = None,
        api_connector: Optional[GoogleAPIConnector] = None,
        enable_socket_server: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or load_config()
        self.logger = logger or setup_logger(
            "GripperController",
            level=self.config.log_level,
            log_dir=self.config.log_dir,
        )
        self.simulator = simulator or MuJoCoSimulator(
            sim_config=self.config.simulation,
            gripper_specs=self.config.gripper,
            logger=setup_logger(
                "MuJoCoSimulator",
                level=self.config.log_level,
                log_dir=self.config.log_dir,
            ),
        )
        self.gripper = GripperBase(
            specs=self.config.gripper,
            backend=self.simulator,
            logger=setup_logger(
                "GripperBase",
                level=self.config.log_level,
                log_dir=self.config.log_dir,
            ),
        )
        # Registra a garra conectada ao simulador como garra padrão para chamadas diretas
        set_default_gripper(self.gripper)

        self.api = api_connector or GoogleAPIConnector(
            config=self.config.api,
            logger=setup_logger(
                "GoogleAPIConnector",
                level=self.config.log_level,
                log_dir=self.config.log_dir,
            ),
        )

        self.socket_bridge: Optional[SocketCommandBridge] = None
        if enable_socket_server:
            self.start_socket_interface(port=0)

    # ------------------------------------------------------------------------
    # Despacho de Comandos (Direto e via Socket)
    # ------------------------------------------------------------------------

    def start_socket_interface(self, port: int = 0) -> int:
        """Ativa o servidor TCP local para receber pacotes `CommandPacket` via socket."""
        if self.socket_bridge is None:
            self.socket_bridge = SocketCommandBridge(
                handler=self.handle_packet,
                host=self.config.simulation.socket_host,
                port=port,
                logger=self.logger,
            )
        return self.socket_bridge.start()

    def stop_socket_interface(self) -> None:
        """Encerra o servidor TCP local caso esteja em execução."""
        if self.socket_bridge is not None:
            self.socket_bridge.stop()
            self.socket_bridge = None

    def handle_packet(self, packet: CommandPacket) -> CommandResult:
        """Processa um `CommandPacket` recebido pelo barramento modular estilo socket."""
        return self.execute_command(
            packet.action,
            speed=packet.speed,
            force=packet.force,
            duration=packet.duration,
        )

    def execute_command(
        self,
        action: str,
        *,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        timeout: Optional[float] = None,
        via_socket: bool = False,
    ) -> CommandResult:
        """Executa qualquer uma das 8 funções principais pelo nome."""
        if action not in COMMAND_REGISTRY:
            raise GripperError(
                f"Comando '{action}' inválido. Opções válidas: {sorted(COMMAND_REGISTRY)}"
            )

        if via_socket:
            if self.socket_bridge is None:
                self.start_socket_interface(port=0)
            assert self.socket_bridge is not None
            pkt = CommandPacket(
                action=action, speed=speed, force=force, duration=duration
            )
            raw = self.socket_bridge.send_packet(pkt)
            return CommandResult(
                success=raw["success"],
                status=raw["status"],
                command=raw.get("command", action),
                position=raw.get("position", self.gripper.get_state()["position"]),
                orientation=raw.get(
                    "orientation", self.gripper.get_state()["orientation"]
                ),
                grip_force=raw.get("grip_force", 0.0),
                aperture=raw.get("aperture", 0.08),
                is_grasping=raw.get("is_grasping", False),
                grasped_object=raw.get("grasped_object"),
                clamped=raw.get("clamped", False),
                elapsed_ms=raw.get("elapsed_ms", 0.0),
                message=raw.get("message", ""),
                error=raw.get("error"),
                metadata=raw.get("metadata"),
            )

        cmd_fn = COMMAND_REGISTRY[action]
        return cmd_fn(
            speed=speed,
            force=force,
            duration=duration,
            gripper=self.gripper,
            timeout=timeout,
        )

    # ------------------------------------------------------------------------
    # Rotinas de Alto Nível (Aproximação Cartesiana e Pick-and-Place Guiado por Visão)
    # ------------------------------------------------------------------------

    def move_to_coordinates(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        *,
        speed: float = 0.30,
        force: float = 25.0,
    ) -> List[CommandResult]:
        """Desloca a garra até `(target_x, target_y, target_z)` usando apenas as 6 funções direcionais."""
        current_pos = self.gripper.get_state()["position"]
        plan = self.api.plan_trajectory_commands(
            gripper_position=current_pos,
            target_position={"x": target_x, "y": target_y, "z": target_z},
            step_speed=speed,
        )
        results: List[CommandResult] = []
        for step in plan:
            res = self.execute_command(
                step["command"],
                speed=step["speed"],
                force=force,
                duration=step["duration"],
            )
            results.append(res)
            if not res.success:
                break
        return results

    def perceive_and_pick_place(
        self,
        object_name: str = "red_cube",
        destination: Optional[Dict[str, float]] = None,
        *,
        speed: float = 0.32,
        grip_force: float = 32.0,
    ) -> Dict[str, Any]:
        """Executa ciclo completo: captura imagem -> consulta Google API -> pega e transporta objeto."""
        t0 = time.perf_counter()
        drop_target = destination or dict(self.simulator.drop_zone)
        initial_obj_pos = self.simulator.get_object_position(object_name)

        # 1. Percepção visual via Google Gemini Vision API (ou fallback local)
        frame = self.simulator.render_frame()
        perception = self.api.analyze_scene(
            frame,
            prompt=f"Localize o objeto '{object_name}' para operação de pick-and-place.",
        )

        detected_target: Optional[Dict[str, float]] = None
        for item in perception.get("detected_objects", []):
            if item.get("name") == object_name:
                detected_target = item["position"]
                break

        if detected_target is None:
            detected_target = dict(initial_obj_pos)

        executed_steps: List[CommandResult] = []

        # 2. Garante garra aberta em altura segura antes da aproximação
        executed_steps.append(self.execute_command("open", speed=speed, force=grip_force))
        safe_z = 0.24
        executed_steps.extend(
            self.move_to_coordinates(
                detected_target["x"], detected_target["y"], safe_z, speed=speed
            )
        )

        # 3. Desce até a altura de captura do objeto e fecha os dedos
        grasp_z = max(0.03, detected_target["z"] + 0.01)
        executed_steps.extend(
            self.move_to_coordinates(
                detected_target["x"], detected_target["y"], grasp_z, speed=speed * 0.75
            )
        )
        close_res = self.execute_command(
            "close", speed=0.18, force=grip_force, duration=0.4
        )
        executed_steps.append(close_res)

        # 4. Eleva, transporta até a zona de entrega, desce e solta
        executed_steps.extend(
            self.move_to_coordinates(
                detected_target["x"], detected_target["y"], safe_z, speed=speed
            )
        )
        executed_steps.extend(
            self.move_to_coordinates(
                drop_target["x"], drop_target["y"], safe_z, speed=speed
            )
        )
        executed_steps.extend(
            self.move_to_coordinates(
                drop_target["x"], drop_target["y"], 0.055, speed=speed * 0.75
            )
        )
        open_res = self.execute_command("open", speed=0.20, duration=0.35)
        executed_steps.append(open_res)

        # 5. Retorno de segurança em Z
        executed_steps.append(self.execute_command("move_up", speed=speed, duration=0.5))

        final_obj_pos = self.simulator.get_object_position(object_name)
        displacement = math.hypot(
            final_obj_pos["x"] - initial_obj_pos["x"],
            final_obj_pos["y"] - initial_obj_pos["y"],
        )
        goal_error = math.hypot(
            final_obj_pos["x"] - drop_target["x"],
            final_obj_pos["y"] - drop_target["y"],
        )
        all_commands_ok = all(r.success for r in executed_steps)
        success = all_commands_ok and displacement > 0.05 and goal_error < 0.065

        return {
            "success": success,
            "object_name": object_name,
            "initial_object_position": initial_obj_pos,
            "detected_object_position": detected_target,
            "final_object_position": final_obj_pos,
            "target_drop_zone": drop_target,
            "displacement_m": round(displacement, 4),
            "goal_error_m": round(goal_error, 4),
            "was_grasped": bool(close_res.is_grasping),
            "perception": {
                "provider": perception.get("provider"),
                "visibility_score": perception.get("visibility_score"),
                "confidence": perception.get("confidence"),
            },
            "commands_executed": len(executed_steps),
            "total_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    def shutdown(self) -> None:
        """Finaliza recursos ativos do controlador de forma limpa."""
        self.stop_socket_interface()
        self.logger.info("Controlador finalizado com sucesso.")
