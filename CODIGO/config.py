"""Configurações centralizadas para API Google, especificações da garra e simulação MuJoCo.

Suporta carregamento via variáveis de ambiente (ou arquivo `.env` quando `python-dotenv`
estiver instalado), mantendo valores padrão seguros para execução local e em CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv é opcional; variáveis do sistema continuam funcionando normalmente.
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class APIConfig:
    """Parâmetros de conexão e resiliência para a camada Google Gemini API."""

    api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model_name: str = field(
        default_factory=lambda: os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.5-flash")
    )
    endpoint_url: str = field(
        default_factory=lambda: os.getenv(
            "GOOGLE_API_ENDPOINT",
            "https://generativelanguage.googleapis.com/v1beta/models",
        )
    )
    mode: str = field(
        default_factory=lambda: os.getenv("GOOGLE_API_MODE", "auto")
    )  # "auto" | "live" | "mock"
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("GOOGLE_API_TIMEOUT", "10.0"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("GOOGLE_API_MAX_RETRIES", "3"))
    )
    retry_backoff_seconds: float = 0.5


@dataclass(frozen=True)
class GripperSpecs:
    """Limites físicos, cinemáticos e dinâmicos da garra robótica."""

    name: str = "GarraParalela-3DOF"
    # Limites do espaço de trabalho cartesiano (metros): (min, max)
    workspace_x: Tuple[float, float] = (-0.45, 0.45)
    workspace_y: Tuple[float, float] = (-0.45, 0.45)
    workspace_z: Tuple[float, float] = (0.02, 0.55)

    # Posição inicial segura (home)
    home_position: Tuple[float, float, float] = (0.0, 0.0, 0.30)
    home_orientation: Tuple[float, float, float] = (0.0, 180.0, 0.0)  # roll, pitch, yaw (graus)

    # Parâmetros operacionais padrão e limites
    default_speed: float = 0.20  # m/s
    min_speed: float = 0.01  # m/s
    max_speed: float = 1.50  # m/s

    default_force: float = 20.0  # N
    min_force: float = 1.0  # N
    max_force: float = 80.0  # N

    default_duration: float = 0.50  # s
    min_duration: float = 0.02  # s
    max_duration: float = 10.0  # s

    # Abertura dos dedos paralelos (metros)
    max_aperture: float = 0.08  # Totalmente aberta (8 cm)
    min_aperture: float = 0.00  # Totalmente fechada
    grasp_radius_tolerance: float = 0.065  # Raio máximo para captura de objeto (m)
    command_timeout_seconds: float = 5.0


@dataclass(frozen=True)
class SimulationConfig:
    """Parâmetros do ambiente MuJoCo, câmera virtual e perturbações visuais."""

    timestep: float = 0.002  # Passo de integração física (500 Hz)
    control_hz: float = 50.0  # Frequência de atualização do controlador
    headless: bool = field(
        default_factory=lambda: os.getenv("SIM_HEADLESS", "true").lower() in ("1", "true", "yes")
    )
    camera_width: int = 320
    camera_height: int = 240
    camera_name: str = "overhead_cam"

    # Condições visuais padrão
    default_fog_density: float = 0.0  # 0.0 (limpo) a 1.0 (neblina densa)
    default_dirt_intensity: float = 0.0  # 0.0 (limpo) a 1.0 (lente/superfície muito suja)

    # Transporte estilo socket (loopback TCP opcional para desacoplamento sim-to-real)
    socket_host: str = "127.0.0.1"
    socket_port: int = field(
        default_factory=lambda: int(os.getenv("GRIPPER_SOCKET_PORT", "50555"))
    )
    save_artifacts: bool = field(
        default_factory=lambda: os.getenv("SIM_SAVE_ARTIFACTS", "true").lower()
        in ("1", "true", "yes")
    )


@dataclass(frozen=True)
class ProjectConfig:
    """Agregador imutável de todas as configurações do sistema."""

    api: APIConfig = field(default_factory=APIConfig)
    gripper: GripperSpecs = field(default_factory=GripperSpecs)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: Path = LOGS_DIR
    artifacts_dir: Path = ARTIFACTS_DIR


def load_config(
    *,
    api_mode: str | None = None,
    headless: bool | None = None,
    log_level: str | None = None,
    save_artifacts: bool | None = None,
) -> ProjectConfig:
    """Carrega a configuração do projeto permitindo sobrescritas via CLI ou testes."""
    api_cfg = APIConfig(mode=api_mode) if api_mode is not None else APIConfig()

    sim_defaults = SimulationConfig()
    sim_cfg = SimulationConfig(
        headless=headless if headless is not None else sim_defaults.headless,
        save_artifacts=(
            save_artifacts if save_artifacts is not None else sim_defaults.save_artifacts
        ),
    )

    return ProjectConfig(
        api=api_cfg,
        gripper=GripperSpecs(),
        simulation=sim_cfg,
        log_level=(log_level or os.getenv("LOG_LEVEL", "INFO")).upper(),
    )
