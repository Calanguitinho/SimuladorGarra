"""Cenário WebSocket 4: Teste de estresse com rajada de comandos e estabilidade de 60 FPS.

Executa dois clientes WebSocket simultâneos:
    - `renderer_subscriber`: consome continuamente todos os pacotes de stream a 60 FPS,
      simulando o navegador renderizando a malha 3D sem quedas de conexão.
    - `command_blaster`: dispara 160 comandos rápidos em sequência (cobrindo todas as
      8 funções + tratamento gracioso de comando inválido sem derrubar a sessão).
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import load_backend_config  # noqa: E402
from backend.websocket_server import AsyncWSClient, GripperWebSocketServer  # noqa: E402


async def _run_async() -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_backend_config(host="127.0.0.1", port=0, fps=60, log_level="WARNING")
    server = GripperWebSocketServer(config=cfg)
    port = await server.start()
    ws_url = f"ws://127.0.0.1:{port}/ws"

    commands = [
        "move_right",
        "move_forward",
        "move_down",
        "close",
        "move_up",
        "move_left",
        "move_backward",
        "open",
    ]
    total_commands = 160
    rtts_ms: List[float] = []
    stream_packets_received = 0
    stop_stream_reader = asyncio.Event()

    async def _stream_subscriber() -> None:
        nonlocal stream_packets_received
        async with AsyncWSClient(ws_url) as sub:
            while not stop_stream_reader.is_set():
                try:
                    pkt = await sub.recv_json(timeout=0.25)
                    if pkt.get("type") == "stream":
                        stream_packets_received += 1
                except asyncio.TimeoutError:
                    continue

    try:
        subscriber_task = asyncio.create_task(_stream_subscriber())
        await asyncio.sleep(0.1)

        burst_t0 = time.perf_counter()
        async with AsyncWSClient(ws_url) as blaster:
            for idx in range(total_commands):
                cmd = commands[idx % len(commands)]
                _, rtt = await blaster.send_command_wait_ack(
                    cmd, speed=0.95, force=55, duration=0.03
                )
                rtts_ms.append(rtt)

            # Testa resiliência a comando malformado/desconhecido sem desconexão
            await blaster.send_json({"command": "invalid_cmd_xyz"})
            err_msg = None
            while err_msg is None:
                pkt = await blaster.recv_json()
                if pkt.get("type") == "error":
                    err_msg = pkt.get("error")

            # Confirma que a conexão permanece íntegra após erro recuperável
            post_ack, post_rtt = await blaster.send_command_wait_ack(
                "move_right", speed=0.5, force=40, duration=0.05
            )
            rtts_ms.append(post_rtt)
            survived_after_error = "pos" in post_ack.get("state", {})

        # Mantém leitura do stream até completar ao menos 0.8s para aferir FPS sob carga
        elapsed_so_far = time.perf_counter() - burst_t0
        if elapsed_so_far < 0.85:
            await asyncio.sleep(0.85 - elapsed_so_far)

        total_window_s = time.perf_counter() - burst_t0
        stop_stream_reader.set()
        await subscriber_task

        stream_fps_under_stress = stream_packets_received / max(0.01, total_window_s)
        sorted_rtts = sorted(rtts_ms)
        avg_rtt = sum(sorted_rtts) / len(sorted_rtts)
        p95_rtt = sorted_rtts[int(0.95 * (len(sorted_rtts) - 1))]
        max_rtt = sorted_rtts[-1]

        fps_stable = stream_fps_under_stress >= 42.0
        latency_ok = max_rtt < 100.0
        success = fps_stable and latency_ok and survived_after_error

        return {
            "scenario": "stress",
            "success": success,
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "metrics": {
                "rapid_commands_sent": total_commands,
                "disconnects": 0,
                "avg_rtt_ms": round(avg_rtt, 3),
                "p95_rtt_ms": round(p95_rtt, 3),
                "max_rtt_ms": round(max_rtt, 3),
                "stream_fps_under_load": round(stream_fps_under_stress, 2),
                "server_measured_fps": server.measured_fps,
                "graceful_error_recovery": survived_after_error,
            },
        }
    finally:
        stop_stream_reader.set()
        await server.stop()


def run_scenario() -> Dict[str, Any]:
    return asyncio.run(_run_async())


class TestWebSocketStress(unittest.TestCase):
    def test_rapid_commands_and_frame_rate_stability(self) -> None:
        report = run_scenario()
        self.assertTrue(report["success"], msg=f"Falha sob estresse: {report}")


if __name__ == "__main__":
    res = run_scenario()
    print(res)
    sys.exit(0 if res["success"] else 1)
