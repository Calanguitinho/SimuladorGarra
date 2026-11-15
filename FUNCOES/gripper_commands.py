"""As 8 funções fundamentais de movimentação e preensão da garra robótica.

Todas as funções podem ser chamadas de forma totalmente independente (criando ou
reutilizando uma instância padrão de `GripperBase`) ou vinculadas a uma garra
específica conectada ao simulador MuJoCo / hardware via parâmetro `gripper`.

Funções exportadas:
    - move_right()
    - move_left()
    - move_forward()
    - move_backward()
    - move_down()
    - move_up()
    - close()
    - open()
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

from FUNCOES.gripper_base import GripperBase
from FUNCOES.utils import CommandResult

_DEFAULT_GRIPPER: Optional[GripperBase] = None

SpeedOrGripper = Optional[Union[float, GripperBase]]


def get_default_gripper() -> GripperBase:
    """Obtém (ou instancia sob demanda) a garra padrão para chamadas independentes."""
    global _DEFAULT_GRIPPER
    if _DEFAULT_GRIPPER is None:
        _DEFAULT_GRIPPER = GripperBase()
    return _DEFAULT_GRIPPER


def set_default_gripper(gripper: Optional[GripperBase]) -> None:
    """Define a instância ativa padrão usada quando `gripper` não é informado."""
    global _DEFAULT_GRIPPER
    _DEFAULT_GRIPPER = gripper


def _resolve_call_args(
    speed_or_gripper: SpeedOrGripper,
    gripper: Optional[GripperBase],
) -> Tuple[GripperBase, Optional[float]]:
    """Permite tanto `move_right(0.2)` quanto `move_right(gripper_obj)` ou `move_right(gripper=...)`."""
    if isinstance(speed_or_gripper, GripperBase):
        return speed_or_gripper, None
    target_gripper = gripper if gripper is not None else get_default_gripper()
    return target_gripper, speed_or_gripper


def move_right(
    speed: SpeedOrGripper = None,
    force: Optional[float] = None,
    duration: Optional[float] = None,
    *,
    gripper: Optional[GripperBase] = None,
    timeout: Optional[float] = None,
) -> CommandResult:
    """Move a garra para a direita (+X no referencial da bancada).

    Args:
        speed: Velocidade linear em m/s (opcional).
        force: Limite de força em Newtons (opcional).
        duration: Duração do movimento em segundos (opcional).
        gripper: Instância específica de `GripperBase` (opcional).
        timeout: Tempo limite máximo em segundos (opcional).

    Returns:
        `CommandResult` contendo `success`, `status`, `position` e telemetria.
    """
    active_gripper, resolved_speed = _resolve_call_args(speed, gripper)
    return active_gripper.execute_translation(
        "move_right",
        speed=resolved_speed,
        force=force,
        duration=duration,
        timeout=timeout,
    )


def move_left(
    speed: SpeedOrGripper = None,
    force: Optional[float] = None,
    duration: Optional[float] = None,
    *,
    gripper: Optional[GripperBase] = None,
    timeout: Optional[float] = None,
) -> CommandResult:
    """Move a garra para a esquerda (-X no referencial da bancada)."""
    active_gripper, resolved_speed = _resolve_call_args(speed, gripper)
    return active_gripper.execute_translation(
        "move_left",
        speed=resolved_speed,
        force=force,
        duration=duration,
        timeout=timeout,
    )


def move_forward(
    speed: SpeedOrGripper = None,
    force: Optional[float] = None,
    duration: Optional[float] = None,
    *,
    gripper: Optional[GripperBase] = None,
    timeout: Optional[float] = None,
) -> CommandResult:
    """Move a garra para frente (+Y no referencial da bancada)."""
    active_gripper, resolved_speed = _resolve_call_args(speed, gripper)
    return active_gripper.execute_translation(
        "move_forward",
        speed=resolved_speed,
        force=force,
        duration=duration,
        timeout=timeout,
    )


def move_backward(
    speed: SpeedOrGripper = None,
    force: Optional[float] = None,
    duration: Optional[float] = None,
    *,
    gripper: Optional[GripperBase] = None,
    timeout: Optional[float] = None,
) -> CommandResult:
    """Move a garra para trás (-Y no referencial da bancada)."""
    active_gripper, resolved_speed = _resolve_call_args(speed, gripper)
    return active_gripper.execute_translation(
        "move_backward",
        speed=resolved_speed,
        force=force,
        duration=duration,
        timeout=timeout,
    )


def move_down(
    speed: SpeedOrGripper = None,
    force: Optional[float] = None,
    duration: Optional[float] = None,
    *,
    gripper: Optional[GripperBase] = None,
    timeout: Optional[float] = None,
) -> CommandResult:
    """Move a garra verticalmente para baixo (-Z em direção à superfície de trabalho)."""
    active_gripper, resolved_speed = _resolve_call_args(speed, gripper)
    return active_gripper.execute_translation(
        "move_down",
        speed=resolved_speed,
        force=force,
        duration=duration,
        timeout=timeout,
    )


def move_up(
    speed: SpeedOrGripper = None,
    force: Optional[float] = None,
    duration: Optional[float] = None,
    *,
    gripper: Optional[GripperBase] = None,
    timeout: Optional[float] = None,
) -> CommandResult:
    """Move a garra verticalmente para cima (+Z afastando-se da mesa)."""
    active_gripper, resolved_speed = _resolve_call_args(speed, gripper)
    return active_gripper.execute_translation(
        "move_up",
        speed=resolved_speed,
        force=force,
        duration=duration,
        timeout=timeout,
    )


def close(
    speed: SpeedOrGripper = None,
    force: Optional[float] = None,
    duration: Optional[float] = None,
    *,
    gripper: Optional[GripperBase] = None,
    timeout: Optional[float] = None,
) -> CommandResult:
    """Fecha os dedos paralelos da garra aplicando força controlada de preensão."""
    active_gripper, resolved_speed = _resolve_call_args(speed, gripper)
    return active_gripper.execute_jaw_action(
        close_jaws=True,
        speed=resolved_speed,
        force=force,
        duration=duration,
        timeout=timeout,
    )


def open(  # noqa: A001 - Nome exigido pela especificação da interface de controle da garra
    speed: SpeedOrGripper = None,
    force: Optional[float] = None,
    duration: Optional[float] = None,
    *,
    gripper: Optional[GripperBase] = None,
    timeout: Optional[float] = None,
) -> CommandResult:
    """Abre os dedos paralelos da garra liberando qualquer objeto segurado."""
    active_gripper, resolved_speed = _resolve_call_args(speed, gripper)
    return active_gripper.execute_jaw_action(
        close_jaws=False,
        speed=resolved_speed,
        force=force,
        duration=duration,
        timeout=timeout,
    )


COMMAND_REGISTRY: Dict[str, Callable[..., CommandResult]] = {
    "move_right": move_right,
    "move_left": move_left,
    "move_forward": move_forward,
    "move_backward": move_backward,
    "move_down": move_down,
    "move_up": move_up,
    "close": close,
    "open": open,
}
