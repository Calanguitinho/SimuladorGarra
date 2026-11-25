"""Testes HTTP e WebSocket para instruções em linguagem natural."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import websocket_server
from backend.websocket_server import GripperWebSocketServer


class CaptureWriter:
    def __init__(self):
        self.data = bytearray()
        self.closed = False

    def write(self, data):
        self.data.extend(data)

    async def drain(self):
        return None

    def close(self):
        self.closed = True


def make_server(tmp_path: Path, *, enabled=True, enable_pick=True, busy=False):
    server = GripperWebSocketServer.__new__(GripperWebSocketServer)
    server.config = SimpleNamespace(server=SimpleNamespace(cors_origins="*"))
    server.vision = SimpleNamespace(config=SimpleNamespace(
        allowed_classes=("dental_tool", "floss_holder"),
        enabled=enabled,
        enable_pick=enable_pick,
    ))
    server.vision_pick = SimpleNamespace(
        active=busy,
        status={"type": "vision_pick_status", "stage": "detecting" if busy else "idle",
                "operation_id": "already-running" if busy else None},
        task=None,
    )
    server.nl_log_path = tmp_path / "nl_log.jsonl"
    server.logger = SimpleNamespace(warning=lambda *_args: None)
    return server


def parsed_response(writer: CaptureWriter):
    head, body = bytes(writer.data).split(b"\r\n\r\n", 1)
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(body.decode("utf-8"))


async def post_instruction(server, payload):
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    reader = asyncio.StreamReader()
    reader.feed_data(raw)
    reader.feed_eof()
    writer = CaptureWriter()
    await server._handle_http(reader, writer, "POST", "/api/instruction", {
        "content-length": str(len(raw)),
    })
    return parsed_response(writer)


def test_http_aceita_tarefa_e_registra_resultado_final(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path)
        calls = []

        def finish():
            async def operation():
                server.vision_pick.status.update(stage="completed", attempt=1, max_attempts=2,
                                                 recovered=False)
            return asyncio.create_task(operation())

        server.execute_command = lambda payload: (
            calls.append(payload),
            server.vision_pick.status.update(stage="detecting", operation_id="new-operation"),
            setattr(server.vision_pick, "task", finish()),
        )
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "fetch", "class_name": "dental_tool", "confidence": 0.9,
            "needs_clarification": False, "clarification": None, "source": "keyword",
        })

        status, body = await post_instruction(server, {"text": "pegue o instrumento"})
        await asyncio.sleep(0)
        await server.vision_pick.task
        for _ in range(20):
            if server.nl_log_path.exists():
                break
            await asyncio.sleep(0.01)

        assert status == 200
        assert body["accepted"] is True
        assert body["task"]["class_name"] == "dental_tool"
        assert body["status"]["stage"] == "detecting"
        assert calls == [{"command": "vision_pick", "class_name": "dental_tool"}]
        record = json.loads(server.nl_log_path.read_text(encoding="utf-8"))
        assert record["result"]["stage"] == "completed"
        assert record["duration_ms"] >= 0

    asyncio.run(check())


@pytest.mark.parametrize("task", [
    {"intent": "unknown", "class_name": None, "confidence": 0.0,
     "needs_clarification": True, "clarification": "Qual ferramenta?", "source": "keyword"},
])
def test_http_pede_esclarecimento_sem_disparar_captura(tmp_path, monkeypatch, task):
    async def check():
        server = make_server(tmp_path)
        calls = []
        server.execute_command = lambda payload: calls.append(payload)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: task)

        status, body = await post_instruction(server, {"text": "pega aquilo"})

        assert status == 200 and body["accepted"] is False
        assert body["task"]["needs_clarification"] is True
        assert body["status"]["stage"] == "idle"
        assert calls == []
        assert json.loads(server.nl_log_path.read_text(encoding="utf-8"))["result"]["stage"] == "clarification"

    asyncio.run(check())


def test_http_recusa_quando_ocupado(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path, busy=True)
        calls = []
        server.execute_command = lambda payload: calls.append(payload)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "fetch", "class_name": "dental_tool", "confidence": 0.9,
            "needs_clarification": False, "clarification": None, "source": "gemini",
        })

        status, body = await post_instruction(server, {"text": "instrumento dental"})

        assert status == 409 and body["accepted"] is False
        assert "em andamento" in body["error"]
        assert body["status"]["stage"] == "detecting"
        assert calls == []

    asyncio.run(check())


def test_http_recusa_com_visao_desabilitada(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path, enabled=False)
        calls = []
        server.execute_command = lambda payload: calls.append(payload)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "fetch", "class_name": "dental_tool", "confidence": 0.9,
            "needs_clarification": False, "clarification": None, "source": "gemini",
        })

        status, body = await post_instruction(server, {"text": "instrumento dental"})

        assert status == 503 and body["accepted"] is False
        assert "desabilitada" in body["error"]
        assert calls == []

    asyncio.run(check())


@pytest.mark.parametrize("text", ["", "x" * 301])
def test_http_rejeita_texto_vazio_ou_maior_que_limite(tmp_path, monkeypatch, text):
    async def check():
        server = make_server(tmp_path)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "unknown", "class_name": None, "confidence": 0.0,
            "needs_clarification": True, "clarification": "Esclareça.", "source": "keyword",
        })
        status, body = await post_instruction(server, {"text": text})
        assert status == 400 and body["accepted"] is False
        assert body["error"]

    asyncio.run(check())


def test_websocket_aceita_mensagem_instruction(tmp_path, monkeypatch):
    async def check():
        server = make_server(tmp_path)
        calls = []
        server.execute_command = lambda payload: calls.append(payload)
        monkeypatch.setattr(websocket_server, "interpret", lambda *_args: {
            "intent": "unknown", "class_name": None, "confidence": 0.0,
            "needs_clarification": True, "clarification": "Qual ferramenta?", "source": "keyword",
        })
        sent = []
        session = SimpleNamespace(peer="test", send_json=lambda data: _append(sent, data))

        await server._process_ws_message(session, b'{"type":"instruction","text":"pega aquilo"}')

        assert sent[0]["type"] == "instruction_result"
        assert sent[0]["accepted"] is False
        assert sent[0]["task"]["needs_clarification"] is True
        assert calls == []

    asyncio.run(check())


async def _append(items, value):
    items.append(value)
