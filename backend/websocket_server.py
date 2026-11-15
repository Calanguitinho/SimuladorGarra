"""Servidor HTTP + WebSocket RFC 6455 em tempo real (60 FPS) para o simulador MuJoCo.

Arquitetura unificada em porta única (padrão `8000`):
    - `ws://<host>:<port>/ws` (ou `/`): streaming contínuo de estado a 60 FPS + comandos JSON
      com latência <5ms (`TCP_NODELAY` e lock de escrita atômico por conexão).
    - `GET /health`, `GET /api/state`, `POST /api/command`, `POST /api/reset`: API REST JSON.
    - `GET /`, `GET /js/*`, `GET /css/*`, `GET /standalone`: hospedagem estática do frontend 3D.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import socket
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
from urllib.parse import unquote, urlparse

from backend.config import BackendConfig, configure_backend_logger, load_backend_config
from backend.gripper_controller import GripperController, InvalidCommandError
from backend.mujoco_env import MuJoCoGripperEnv

WS_MAGIC_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

MIME_OVERRIDES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".mjcf": "application/xml; charset=utf-8",
    ".xml": "application/xml; charset=utf-8",
    ".svg": "image/svg+xml",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".stl": "model/stl",
    ".obj": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def encode_ws_frame(payload: bytes, opcode: int = 0x1, mask: bool = False) -> bytes:
    """Codifica um quadro WebSocket RFC 6455 (suporta payloads <126B, 16-bit e 64-bit)."""
    length = len(payload)
    first_byte = 0x80 | (opcode & 0x0F)
    mask_bit = 0x80 if mask else 0x00

    if length < 126:
        header = struct.pack("!BB", first_byte, mask_bit | length)
    elif length <= 0xFFFF:
        header = struct.pack("!BBH", first_byte, mask_bit | 126, length)
    else:
        header = struct.pack("!BBQ", first_byte, mask_bit | 127, length)

    if not mask:
        return header + payload

    mask_key = b"\x37\xfa\x21\x3d"
    masked_bytes = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
    return header + mask_key + masked_bytes


async def read_ws_frame(reader: asyncio.StreamReader) -> Tuple[int, bytes]:
    """Lê e decodifica um quadro completo (incluindo continuações) do stream TCP."""
    fragments = bytearray()
    message_opcode: Optional[int] = None

    while True:
        head = await reader.readexactly(2)
        b0, b1 = head[0], head[1]
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F

        if length == 126:
            ext = await reader.readexactly(2)
            length = struct.unpack("!H", ext)[0]
        elif length == 127:
            ext = await reader.readexactly(8)
            length = struct.unpack("!Q", ext)[0]

        mask_key = await reader.readexactly(4) if masked else b""
        data = await reader.readexactly(length) if length > 0 else b""

        if masked and data:
            data = bytes(b ^ mask_key[i & 3] for i, b in enumerate(data))

        # Quadros de controle (close=0x8, ping=0x9, pong=0xA) retornam imediatamente
        if opcode >= 0x8:
            return opcode, data

        if opcode != 0x0:
            message_opcode = opcode
        fragments.extend(data)

        if fin:
            return (message_opcode or 0x1), bytes(fragments)


class WSClientSession:
    """Sessão individual de um cliente WebSocket conectado ao simulador."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        peer: str,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.peer = peer
        self.write_lock = asyncio.Lock()
        self.connected_at = time.time()
        self.last_seen = self.connected_at
        self.closed = False

    async def send_json(self, data: Dict[str, Any]) -> bool:
        """Serializa e transmite JSON atomicamente sem entrelaçar quadros concorrentes."""
        if self.closed:
            return False
        raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        frame = encode_ws_frame(raw, opcode=0x1, mask=False)
        async with self.write_lock:
            if self.closed:
                return False
            try:
                self.writer.write(frame)
                await self.writer.drain()
                return True
            except (ConnectionError, OSError, RuntimeError):
                self.closed = True
                return False

    async def send_control(self, opcode: int, payload: bytes = b"") -> None:
        if self.closed:
            return
        frame = encode_ws_frame(payload, opcode=opcode, mask=False)
        async with self.write_lock:
            try:
                self.writer.write(frame)
                await self.writer.drain()
            except (ConnectionError, OSError, RuntimeError):
                self.closed = True

    async def close(self) -> None:
        if self.closed:
            return
        await self.send_control(0x8)
        self.closed = True
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except (ConnectionError, OSError, RuntimeError):
            pass


class GripperWebSocketServer:
    """Servidor assíncrono de produção para streaming MuJoCo 60 FPS + API REST + Frontend."""

    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        env: Optional[MuJoCoGripperEnv] = None,
        controller: Optional[GripperController] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or load_backend_config()
        self.logger = logger or configure_backend_logger(
            "WebSocketServer", level=self.config.server.log_level
        )
        self.env = env or MuJoCoGripperEnv(config=self.config, logger=self.logger)
        self.controller = controller or GripperController(
            env=self.env, config=self.config, logger=self.logger
        )

        self.clients: Set[WSClientSession] = set()
        self._server: Optional[asyncio.AbstractServer] = None
        self._physics_task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._started_at = 0.0
        self.measured_fps: float = float(self.config.physics.broadcast_fps)
        self.bound_port: int = self.config.server.port

    # ------------------------------------------------------------------------
    # Ciclo de Vida do Servidor
    # ------------------------------------------------------------------------

    async def start(self) -> int:
        """Inicia o listener TCP e a malha de broadcast físico em 60 FPS."""
        if self._running:
            return self.bound_port

        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self.config.server.host,
            port=self.config.server.port,
            backlog=128,
        )
        sockets = self._server.sockets or []
        if sockets:
            self.bound_port = sockets[0].getsockname()[1]

        self._running = True
        self._started_at = time.time()
        self._physics_task = asyncio.create_task(
            self._physics_broadcast_loop(), name="MuJoCo60FPSLoop"
        )
        self.logger.info(
            "Servidor ativo em http://%s:%d (WebSocket: ws://%s:%d%s | %d FPS)",
            self.config.server.host,
            self.bound_port,
            self.config.server.host,
            self.bound_port,
            self.config.server.ws_path,
            self.config.physics.broadcast_fps,
        )
        return self.bound_port

    async def serve_forever(self) -> None:
        """Inicia o servidor e bloqueia até cancelamento (SIGINT/SIGTERM)."""
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        """Desliga a malha física, fecha sessões WebSocket e libera o socket TCP."""
        self._running = False
        if self._physics_task is not None:
            self._physics_task.cancel()
            try:
                await self._physics_task
            except asyncio.CancelledError:
                pass
            self._physics_task = None

        active = list(self.clients)
        self.clients.clear()
        if active:
            await asyncio.gather(*(c.close() for c in active), return_exceptions=True)

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        self.logger.info("Servidor WebSocket/HTTP encerrado com segurança.")

    # ------------------------------------------------------------------------
    # Loop de Física e Broadcast a 60 FPS
    # ------------------------------------------------------------------------

    async def _physics_broadcast_loop(self) -> None:
        """Integra o MuJoCo e transmite estado a 60 Hz com compensação monotônica de drift."""
        target_fps = max(1, self.config.physics.broadcast_fps)
        frame_period = 1.0 / target_fps
        next_tick = time.perf_counter()
        fps_window_start = next_tick
        frames_in_window = 0

        while self._running:
            state = self.env.step_physics()
            frames_in_window += 1

            now = time.perf_counter()
            window_elapsed = now - fps_window_start
            if window_elapsed >= 0.5:
                self.measured_fps = round(frames_in_window / window_elapsed, 1)
                fps_window_start = now
                frames_in_window = 0

            if self.clients:
                packet = {
                    "type": "stream",
                    "fps": self.measured_fps,
                    "state": state,
                }
                stale: list[WSClientSession] = []
                results = await asyncio.gather(
                    *(client.send_json(packet) for client in list(self.clients)),
                    return_exceptions=True,
                )
                for client, ok in zip(list(self.clients), results):
                    if ok is not True:
                        stale.append(client)
                for dead in stale:
                    self.clients.discard(dead)

            next_tick += frame_period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                # Evita acúmulo caso o sistema operacional atrase o agendamento
                next_tick = time.perf_counter()

    # ------------------------------------------------------------------------
    # Roteador de Conexão TCP (Upgrade WebSocket vs HTTP REST / Estático)
    # ------------------------------------------------------------------------

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        sock = writer.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        peer_info = writer.get_extra_info("peername")
        peer = f"{peer_info[0]}:{peer_info[1]}" if peer_info else "unknown"

        try:
            raw_head = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=10.0
            )
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            writer.close()
            return

        header_text = raw_head.decode("latin-1", errors="replace")
        lines = header_text.split("\r\n")
        request_line = lines[0].split(" ")
        if len(request_line) < 2:
            writer.close()
            return

        method, raw_target = request_line[0].upper(), request_line[1]
        headers: Dict[str, str] = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        if headers.get("upgrade", "").lower() == "websocket":
            await self._handle_websocket(reader, writer, headers, peer)
        else:
            await self._handle_http(reader, writer, method, raw_target, headers)

    # ------------------------------------------------------------------------
    # Handshake e Sessão WebSocket
    # ------------------------------------------------------------------------

    async def _handle_websocket(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        headers: Dict[str, str],
        peer: str,
    ) -> None:
        ws_key = headers.get("sec-websocket-key", "")
        if not ws_key:
            await self._send_http_response(
                writer, 400, "text/plain", b"Missing Sec-WebSocket-Key"
            )
            return

        accept_token = base64.b64encode(
            hashlib.sha1((ws_key + WS_MAGIC_GUID).encode("ascii")).digest()
        ).decode("ascii")

        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_token}\r\n\r\n"
        )
        writer.write(handshake.encode("ascii"))
        await writer.drain()

        session = WSClientSession(reader, writer, peer)
        self.clients.add(session)
        self.logger.info(
            "Cliente WebSocket conectado: %s (ativos=%d)", peer, len(self.clients)
        )

        # Envia snapshot inicial imediatamente ao conectar
        await session.send_json(
            {
                "type": "hello",
                "fps": self.measured_fps,
                "state": self.env.get_state_snapshot(),
            }
        )

        try:
            while self._running and not session.closed:
                try:
                    opcode, raw_payload = await asyncio.wait_for(
                        read_ws_frame(reader),
                        timeout=self.config.server.client_timeout_s,
                    )
                except asyncio.TimeoutError:
                    # Envia ping RFC 6455 para verificar vivacidade da conexão ociosa
                    await session.send_control(0x9, b"keepalive")
                    continue

                session.last_seen = time.time()

                if opcode == 0x8:  # Close
                    break
                if opcode == 0x9:  # Ping -> Pong
                    await session.send_control(0xA, raw_payload)
                    continue
                if opcode == 0xA:  # Pong
                    continue
                if opcode != 0x1:  # Apenas texto JSON
                    continue

                await self._process_ws_message(session, raw_payload)

        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        finally:
            self.clients.discard(session)
            await session.close()
            self.logger.info(
                "Cliente WebSocket desconectado: %s (ativos=%d)",
                peer,
                len(self.clients),
            )

    async def _process_ws_message(
        self, session: WSClientSession, raw_payload: bytes
    ) -> None:
        recv_ts = round(time.time() * 1000.0, 3)
        try:
            msg = json.loads(raw_payload.decode("utf-8"))
            if not isinstance(msg, dict):
                raise InvalidCommandError("Mensagem JSON deve ser um objeto.")

            state = self.controller.execute_payload(msg)
            response: Dict[str, Any] = {
                "type": "ack",
                "command": msg.get("command"),
                "fps": self.measured_fps,
                "recv_ts": recv_ts,
                "state": state,
            }
            if "req_id" in msg:
                response["req_id"] = msg["req_id"]
            if "client_ts" in msg:
                response["client_ts"] = msg["client_ts"]

            await session.send_json(response)

        except (json.JSONDecodeError, InvalidCommandError, ValueError) as exc:
            self.logger.warning("Comando inválido de %s: %s", session.peer, exc)
            await session.send_json(
                {
                    "type": "error",
                    "error": str(exc),
                    "state": self.env.get_state_snapshot(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Erro interno ao processar mensagem WS: %s", exc)
            await session.send_json(
                {
                    "type": "error",
                    "error": f"Internal server error: {exc}",
                    "state": self.env.get_state_snapshot(),
                }
            )

    # ------------------------------------------------------------------------
    # Endpoints HTTP REST + Arquivos Estáticos do Frontend
    # ------------------------------------------------------------------------

    async def _handle_http(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        method: str,
        raw_target: str,
        headers: Dict[str, str],
    ) -> None:
        parsed = urlparse(raw_target)
        path = unquote(parsed.path)

        content_length = int(headers.get("content-length", "0") or "0")
        body = (
            await reader.readexactly(content_length)
            if content_length > 0
            else b""
        )

        if method == "OPTIONS":
            await self._send_http_response(writer, 204, "text/plain", b"")
            return

        if method == "GET" and path == "/health":
            payload = {
                "status": "ok",
                "engine": self.env.engine_name,
                "fps": self.measured_fps,
                "target_fps": self.config.physics.broadcast_fps,
                "connected_clients": len(self.clients),
                "uptime_s": round(time.time() - self._started_at, 2),
            }
            await self._send_json_http(writer, 200, payload)
            return

        if method == "GET" and path == "/api/state":
            await self._send_json_http(
                writer, 200, {"state": self.env.get_state_snapshot()}
            )
            return

        if method == "GET" and path == "/api/scene-manifest":
            await self._send_json_http(writer, 200, self._build_scene_manifest())
            return

        if method == "POST" and path == "/api/reset":
            state = self.env.reset()
            await self._send_json_http(writer, 200, {"state": state})
            return

        if method == "POST" and path == "/api/command":
            try:
                data = json.loads(body.decode("utf-8") or "{}")
                state = self.controller.execute_payload(data)
                await self._send_json_http(writer, 200, {"state": state})
            except Exception as exc:  # noqa: BLE001
                await self._send_json_http(
                    writer,
                    400,
                    {"error": str(exc), "state": self.env.get_state_snapshot()},
                )
            return

        if method == "GET":
            await self._serve_static_frontend(writer, path)
            return

        await self._send_json_http(writer, 405, {"error": "Method not allowed"})

    def _build_scene_manifest(self) -> Dict[str, Any]:
        """Catálogo dos modelos 3D disponíveis (odontologia + garra) para o frontend Three.js."""
        assets_root = self.config.assets_dir.resolve()
        dental_dir = assets_root / "objetos" / "odontologia"

        def _asset(rel: str, **meta: Any) -> Optional[Dict[str, Any]]:
            path = assets_root / rel
            if not path.is_file():
                return None
            return {
                "url": f"/assets/{rel.replace(chr(92), '/')}",
                "bytes": path.stat().st_size,
                **meta,
            }

        props = [
            item
            for item in (
                _asset(
                    "objetos/odontologia/dental_chair_ostem.glb",
                    id="dental_chair",
                    label="Cadeira Odontológica",
                    role="environment",
                    fit="upright",
                    flip=True,
                    targetHeightM=1.15,
                    position=[-1.35, 0.22, 0.0],
                    yawDeg=90,
                ),
                _asset(
                    "objetos/odontologia/metal_tray.glb",
                    id="metal_tray",
                    label="Bandeja Metálica",
                    role="prop",
                    fit="lieFlat",
                    flip=True,
                    targetSizeM=0.36,
                    position=[0.28, -0.22, 0.86],
                    yawDeg=-8,
                ),
                _asset(
                    "objetos/odontologia/dentist_tools.glb",
                    id="dentist_tools",
                    label="Kit de Instrumentos",
                    role="prop",
                    fit="lieFlat",
                    align="y",
                    targetSizeM=0.16,
                    position=[0.50, -0.22, 0.86],
                    yawDeg=-8,
                ),
                _asset(
                    "objetos/odontologia/dental_tool-nih3d.stl",
                    id="dental_tool_visual",
                    label="Instrumento Dental (NIH3D)",
                    role="manipulable_visual",
                    scale=0.001,
                    syncObject="dental_tool",
                ),
                _asset(
                    "objetos/odontologia/Angled_Dental_Floss_Holder.stl",
                    id="floss_holder_visual",
                    label="Suporte de Fio Dental",
                    role="manipulable_visual",
                    scale=0.001,
                    scaleVec=[0.001, 0.001, 0.005],
                    syncObject="floss_holder",
                ),
            )
            if item is not None
        ]

        robot = []
        if getattr(self.env, "_visual_parts", None):
            robot = [
                {
                    "url": p["url"],
                    "scale": p["scale"],
                    "rgba": p["rgba"],
                    "mesh": p["mesh"],
                }
                for p in self.env._visual_parts
            ]

        return {
            "theme": "dental_clinic",
            "coordinateSystem": "Z-up",
            "units": "meters",
            "robot": robot,
            "robotName": "UR5e + Robotiq 2F-85",
            "props": props,
            "dentalDirExists": dental_dir.is_dir(),
            "engine": self.env.engine_name,
        }

    async def _serve_static_frontend(
        self, writer: asyncio.StreamWriter, request_path: str
    ) -> None:
        frontend_root = self.config.frontend_dir.resolve()
        assets_root = self.config.assets_dir.resolve()

        # Serve /assets/** from project assets/ (GLB, STL, PNG Menagerie, etc.)
        if request_path.startswith("/assets/"):
            rel = request_path[len("/assets/") :]
            target_file = (assets_root / rel).resolve()
            if not str(target_file).startswith(str(assets_root)) or not target_file.is_file():
                await self._send_json_http(
                    writer, 404, {"error": f"Asset não encontrado: {request_path}"}
                )
                return
            ext = target_file.suffix.lower()
            mime = (
                MIME_OVERRIDES.get(ext)
                or mimetypes.guess_type(str(target_file))[0]
                or "application/octet-stream"
            )
            await self._send_http_response(writer, 200, mime, target_file.read_bytes())
            return

        if request_path in ("/", "/index.html"):
            target_file = frontend_root / "index.html"
        elif request_path in ("/standalone", "/standalone.html"):
            dist_file = frontend_root / "dist" / "index.html"
            target_file = dist_file if dist_file.exists() else (frontend_root / "index.html")
        else:
            rel = request_path.lstrip("/")
            target_file = (frontend_root / rel).resolve()

        if not str(target_file).startswith(str(frontend_root)) or not target_file.is_file():
            await self._send_json_http(
                writer, 404, {"error": f"Recurso não encontrado: {request_path}"}
            )
            return

        ext = target_file.suffix.lower()
        mime = (
            MIME_OVERRIDES.get(ext)
            or mimetypes.guess_type(str(target_file))[0]
            or "application/octet-stream"
        )
        content = target_file.read_bytes()
        await self._send_http_response(writer, 200, mime, content)

    async def _send_json_http(
        self, writer: asyncio.StreamWriter, status_code: int, data: Dict[str, Any]
    ) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        await self._send_http_response(
            writer, status_code, "application/json; charset=utf-8", raw
        )

    async def _send_http_response(
        self,
        writer: asyncio.StreamWriter,
        status_code: int,
        content_type: str,
        body: bytes,
    ) -> None:
        reasons = {
            200: "OK",
            204: "No Content",
            400: "Bad Request",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
        }
        reason = reasons.get(status_code, "OK")
        head = (
            f"HTTP/1.1 {status_code} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Access-Control-Allow-Origin: {self.config.server.cors_origins}\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n\r\n"
        )
        try:
            writer.write(head.encode("latin-1") + body)
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.close()


# ============================================================================
# Cliente WebSocket Assíncrono Leve para Cenários de Teste Automatizados
# ============================================================================


class AsyncWSClient:
    """Cliente WebSocket RFC 6455 compatível com `asyncio` puro (sem dependências externas)."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def __aenter__(self) -> "AsyncWSClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def connect(self) -> Dict[str, Any]:
        parsed = urlparse(self.url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/ws"

        self._reader, self._writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=self.timeout
        )
        sock = self._writer.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        raw_key = base64.b64encode(b"simulador-garra-ws").decode("ascii")
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {raw_key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._writer.write(req.encode("ascii"))
        await self._writer.drain()

        resp_head = await asyncio.wait_for(
            self._reader.readuntil(b"\r\n\r\n"), timeout=self.timeout
        )
        if b"101" not in resp_head.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"Handshake WebSocket rejeitado: {resp_head!r}")

        # Consome o quadro inicial 'hello' com o snapshot inicial
        return await self.recv_json()

    async def send_json(self, payload: Dict[str, Any]) -> None:
        assert self._writer is not None
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._writer.write(encode_ws_frame(raw, opcode=0x1, mask=True))
        await self._writer.drain()

    async def recv_json(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        assert self._reader is not None and self._writer is not None
        eff_timeout = self.timeout if timeout is None else timeout
        while True:
            opcode, data = await asyncio.wait_for(
                read_ws_frame(self._reader), timeout=eff_timeout
            )
            if opcode == 0x9:  # Ping -> Pong
                self._writer.write(encode_ws_frame(data, opcode=0xA, mask=True))
                await self._writer.drain()
                continue
            if opcode == 0x8:
                raise ConnectionError("Servidor encerrou a conexão WebSocket.")
            if opcode == 0x1:
                return json.loads(data.decode("utf-8"))

    async def send_command_wait_ack(
        self,
        command: str,
        *,
        speed: Optional[float] = None,
        force: Optional[float] = None,
        duration: Optional[float] = None,
        settle: bool = False,
        target: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], float]:
        """Envia comando e aguarda o respectivo quadro `type == 'ack'`, medindo RTT em ms."""
        req_id = f"{command}-{time.perf_counter_ns()}"
        t0 = time.perf_counter()
        payload: Dict[str, Any] = {
            "command": command,
            "req_id": req_id,
            "client_ts": round(time.time() * 1000.0, 3),
        }
        if speed is not None:
            payload["speed"] = speed
        if force is not None:
            payload["force"] = force
        if duration is not None:
            payload["duration"] = duration
        if settle:
            payload["settle"] = True
        if target is not None:
            payload["target"] = target
        if extra:
            payload.update(extra)
        if kwargs:
            payload.update(kwargs)

        await self.send_json(payload)

        while True:
            msg = await self.recv_json()
            if msg.get("type") == "error":
                raise RuntimeError(f"Erro retornado pelo backend: {msg.get('error')}")
            if msg.get("type") == "ack" and msg.get("req_id") == req_id:
                rtt_ms = (time.perf_counter() - t0) * 1000.0
                return msg, rtt_ms

    async def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.write(encode_ws_frame(b"", opcode=0x8, mask=True))
                await self._writer.drain()
                self._writer.close()
                await self._writer.wait_closed()
            except (ConnectionError, OSError, RuntimeError):
                pass
            self._writer = None
            self._reader = None
