"""Utilitários de logging, tratamento de erros, gerenciamento de estado e transporte socket.

Fornece estruturas padronizadas para comunicação modular estilo socket entre
a camada de comandos, o controlador, a API Google e o simulador/hardware real.
"""

from __future__ import annotations

import builtins
import copy
import json
import logging
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ============================================================================
# Exceções Customizadas
# ============================================================================


class GripperError(Exception):
    """Exceção base para falhas operacionais no sistema de controle da garra."""


class GripperTimeoutError(GripperError):
    """Lançada quando uma operação excede o tempo máximo permitido."""


class WorkspaceBoundaryError(GripperError):
    """Lançada quando um movimento viola de forma inválida os limites operacionais."""


class InvalidParameterError(GripperError):
    """Lançada quando velocidade, força ou duração recebem valores inválidos."""


class SimulatorConnectionError(GripperError):
    """Lançada quando há falha na comunicação com o simulador MuJoCo ou hardware."""


class GoogleAPIError(GripperError):
    """Lançada quando a comunicação com a Google API falha após as tentativas."""


# ============================================================================
# Logging Centralizado
# ============================================================================


def setup_logger(
    name: str = "SimuladorGarra",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """Configura e retorna um logger padronizado com saída em console e arquivo rotativo."""
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    target_dir = log_dir or (Path(__file__).resolve().parent.parent / "logs")
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            target_dir / "gripper_system.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Em ambientes somente leitura, mantém o logger apenas no console sem falhar.
        logger.warning("Não foi possível criar diretório de logs em %s.", target_dir)

    logger.propagate = False
    return logger


# ============================================================================
# Estruturas de Estado, Comandos e Resultados
# ============================================================================


class CommandResult(dict):
    """Resultado padronizado de comandos da garra compatível com `dict` e acesso por atributo.

    Exemplo:
        res = move_right(speed=0.2)
        assert res["success"] is True
        assert res.position["x"] > 0
    """

    def __init__(
        self,
        *,
        success: bool,
        status: str,
        command: str,
        position: Dict[str, float],
        orientation: Dict[str, float],
        grip_force: float,
        aperture: float,
        is_grasping: bool = False,
        grasped_object: Optional[str] = None,
        clamped: bool = False,
        elapsed_ms: float = 0.0,
        message: str = "",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            success=success,
            status=status,
            command=command,
            position=dict(position),
            orientation=dict(orientation),
            grip_force=round(float(grip_force), 4),
            aperture=round(float(aperture), 5),
            is_grasping=bool(is_grasping),
            grasped_object=grasped_object,
            clamped=bool(clamped),
            elapsed_ms=round(float(elapsed_ms), 3),
            message=message,
            error=error,
            metadata=metadata or {},
        )

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(f"'CommandResult' não possui atributo '{item}'") from exc

    def to_json(self) -> str:
        """Serializa o resultado em JSON para transporte em arquitetura estilo socket."""
        return json.dumps(self, ensure_ascii=False)


@dataclass
class CommandPacket:
    """Pacote serializável estilo socket para envio de comandos ao controlador/atuador."""

    action: str
    speed: Optional[float] = None
    force: Optional[float] = None
    duration: Optional[float] = None
    packet_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Converte o pacote para string JSON terminada em quebra de linha."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "CommandPacket":
        """Reconstrói um `CommandPacket` a partir de payload JSON."""
        data = json.loads(raw)
        return cls(**data)


@dataclass
class GripperState:
    """Representação completa do estado instantâneo da garra robótica."""

    position: Dict[str, float] = field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.30}
    )
    orientation: Dict[str, float] = field(
        default_factory=lambda: {"roll": 0.0, "pitch": 180.0, "yaw": 0.0}
    )
    grip_force: float = 0.0
    aperture: float = 0.08
    is_closed: bool = False
    is_grasping: bool = False
    grasped_object: Optional[str] = None
    last_command: str = "init"
    command_count: int = 0
    error_count: int = 0
    updated_at: float = field(default_factory=time.time)

    def snapshot(self) -> Dict[str, Any]:
        """Retorna uma cópia profunda em formato dicionário do estado atual."""
        return copy.deepcopy(asdict(self))


class StateManager:
    """Gerenciador thread-safe de estado com histórico e rollback automático para recuperação."""

    def __init__(
        self,
        initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.30),
        initial_orientation: Tuple[float, float, float] = (0.0, 180.0, 0.0),
        initial_aperture: float = 0.08,
        max_history: int = 200,
    ) -> None:
        self._lock = threading.RLock()
        self._max_history = max_history
        self._state = GripperState(
            position={
                "x": round(initial_position[0], 5),
                "y": round(initial_position[1], 5),
                "z": round(initial_position[2], 5),
            },
            orientation={
                "roll": round(initial_orientation[0], 3),
                "pitch": round(initial_orientation[1], 3),
                "yaw": round(initial_orientation[2], 3),
            },
            aperture=round(initial_aperture, 5),
        )
        self._history: List[GripperState] = [copy.deepcopy(self._state)]

    @property
    def state(self) -> GripperState:
        """Retorna cópia segura do estado atual."""
        with self._lock:
            return copy.deepcopy(self._state)

    @property
    def history_size(self) -> int:
        """Quantidade de snapshots armazenados no histórico."""
        with self._lock:
            return len(self._history)

    def save_checkpoint(self) -> GripperState:
        """Salva um ponto de restauração antes de executar um movimento."""
        with self._lock:
            checkpoint = copy.deepcopy(self._state)
            self._history.append(checkpoint)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            return copy.deepcopy(checkpoint)

    def update(
        self,
        *,
        position: Optional[Dict[str, float]] = None,
        orientation: Optional[Dict[str, float]] = None,
        grip_force: Optional[float] = None,
        aperture: Optional[float] = None,
        is_closed: Optional[bool] = None,
        is_grasping: Optional[bool] = None,
        grasped_object: Optional[str] = None,
        clear_grasped_object: bool = False,
        command_name: Optional[str] = None,
    ) -> GripperState:
        """Atualiza atomicamente os campos do estado da garra."""
        with self._lock:
            if position is not None:
                self._state.position = {
                    axis: round(float(val), 5) for axis, val in position.items()
                }
            if orientation is not None:
                self._state.orientation = {
                    axis: round(float(val), 3) for axis, val in orientation.items()
                }
            if grip_force is not None:
                self._state.grip_force = round(float(grip_force), 4)
            if aperture is not None:
                self._state.aperture = round(float(aperture), 5)
            if is_closed is not None:
                self._state.is_closed = bool(is_closed)
            if is_grasping is not None:
                self._state.is_grasping = bool(is_grasping)
            if clear_grasped_object:
                self._state.grasped_object = None
            elif grasped_object is not None:
                self._state.grasped_object = grasped_object
            if command_name is not None:
                self._state.last_command = command_name
                self._state.command_count += 1
            self._state.updated_at = time.time()
            return copy.deepcopy(self._state)

    def record_error(self, rollback_to: Optional[GripperState] = None) -> GripperState:
        """Incrementa o contador de erros e opcionalmente restaura um estado seguro."""
        with self._lock:
            previous_errors = self._state.error_count + 1
            if rollback_to is not None:
                self._state = copy.deepcopy(rollback_to)
            self._state.error_count = previous_errors
            self._state.updated_at = time.time()
            return copy.deepcopy(self._state)

    def reset(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.30),
        orientation: Tuple[float, float, float] = (0.0, 180.0, 0.0),
        aperture: float = 0.08,
    ) -> GripperState:
        """Restaura o estado inicial padrão da garra."""
        with self._lock:
            self._state = GripperState(
                position={
                    "x": round(position[0], 5),
                    "y": round(position[1], 5),
                    "z": round(position[2], 5),
                },
                orientation={
                    "roll": round(orientation[0], 3),
                    "pitch": round(orientation[1], 3),
                    "yaw": round(orientation[2], 3),
                },
                aperture=round(aperture, 5),
            )
            self._history = [copy.deepcopy(self._state)]
            return copy.deepcopy(self._state)


# ============================================================================
# Helpers de Timeout, Validação e Socket Local
# ============================================================================


def clamp(value: float, minimum: float, maximum: float) -> Tuple[float, bool]:
    """Limita `value` ao intervalo `[minimum, maximum]` informando se houve saturação."""
    if value < minimum:
        return minimum, True
    if value > maximum:
        return maximum, True
    return value, False


def execute_with_timeout(
    func: Callable[..., Any],
    timeout_seconds: float,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Executa `func` aplicando timeout rigoroso e convertendo atrasos em `GripperTimeoutError`."""
    if timeout_seconds <= 0:
        raise GripperTimeoutError("O tempo limite de execução deve ser positivo.")

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise GripperTimeoutError(
            f"Operação excedeu o limite de {timeout_seconds:.2f}s."
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


class SocketCommandBridge:
    """Ponte TCP loopback leve (estilo socket) para comunicação desacoplada cliente-controlador.

    Permite enviar pacotes `CommandPacket` via socket TCP real em `127.0.0.1`, facilitando
    a migração direta do simulador MuJoCo para controladores industriais em rede.
    """

    def __init__(
        self,
        handler: Callable[[CommandPacket], CommandResult],
        host: str = "127.0.0.1",
        port: int = 0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._handler = handler
        self._host = host
        self._requested_port = port
        self._logger = logger or setup_logger("SocketBridge")
        self._server_socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self.bound_port: int = port

    def start(self) -> int:
        """Inicia o servidor TCP em thread dedicada e retorna a porta vinculada."""
        if self._running.is_set():
            return self.bound_port

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self._host, self._requested_port))
        srv.listen(8)
        srv.settimeout(0.25)
        self.bound_port = srv.getsockname()[1]
        self._server_socket = srv
        self._running.set()

        self._thread = threading.Thread(
            target=self._serve_loop, name="GripperSocketServer", daemon=True
        )
        self._thread.start()
        self._logger.info(
            "Servidor socket da garra ativo em %s:%d", self._host, self.bound_port
        )
        return self.bound_port

    def _serve_loop(self) -> None:
        assert self._server_socket is not None
        while self._running.is_set():
            try:
                conn, _ = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            with conn:
                conn.settimeout(5.0)
                buffer = b""
                while b"\n" not in buffer:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                if not buffer:
                    continue
                try:
                    packet = CommandPacket.from_json(buffer.decode("utf-8").strip())
                    result = self._handler(packet)
                    payload = result.to_json() + "\n"
                except Exception as exc:  # noqa: BLE001
                    error_payload = {
                        "success": False,
                        "status": "error",
                        "error": builtins.str(exc),
                    }
                    payload = json.dumps(error_payload) + "\n"
                conn.sendall(payload.encode("utf-8"))

    def send_packet(
        self, packet: CommandPacket, timeout_seconds: float = 5.0
    ) -> Dict[str, Any]:
        """Cliente socket que envia um `CommandPacket` e aguarda a resposta JSON."""
        with socket.create_connection(
            (self._host, self.bound_port), timeout=timeout_seconds
        ) as client:
            client.sendall((packet.to_json() + "\n").encode("utf-8"))
            response = b""
            while b"\n" not in response:
                chunk = client.recv(4096)
                if not chunk:
                    break
                response += chunk
        return json.loads(response.decode("utf-8").strip())

    def stop(self) -> None:
        """Encerra de forma limpa o servidor socket."""
        self._running.clear()
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._logger.info("Servidor socket da garra encerrado.")
