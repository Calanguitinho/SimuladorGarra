"""Compara cadência física sem/com YOLO no mesmo host, após aquecimento."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


async def measure(args,enabled):
    from backend.config import VisionConfig,load_backend_config
    from backend.websocket_server import GripperWebSocketServer
    server = GripperWebSocketServer(load_backend_config(host="127.0.0.1",port=0,log_level="WARNING",
        vision=VisionConfig(enabled=enabled,model_path=args.model)))
    await server.start()
    try:
        # Ativar o serviço não deve capturar; medir solicitações explícitas.
        if enabled:
            await server.vision.detect("dental_tool")  # aquecimento sob demanda
        await asyncio.sleep(1)
        fps,latencies,ages,frames = [],[],[],set()
        initial_frames = server.vision.snapshot()["frames_captured"]
        measurement_started = time.monotonic()
        deadline = time.monotonic()+args.seconds
        requests = 0
        while time.monotonic()<deadline:
            if enabled:
                await server.vision.detect("dental_tool")
                requests += 1
            fps.append(server.measured_fps)
            packet = server.vision.snapshot()
            if packet["frame_id"] not in frames and packet["frame_id"] is not None:
                frames.add(packet["frame_id"])
                latencies.append(packet["inference_ms"])
                ages.append(packet["age_ms"])
            await asyncio.sleep(.05)
        elapsed = time.monotonic()-measurement_started
        captured = server.vision.snapshot()["frames_captured"] - initial_frames
        return {"detection_requests":requests,"mode":"on_demand", "fps_mean":float(np.mean(fps)),"fps_min":min(fps),
                "result_frames_observed":len(frames), "captured_frames":captured,
                "capture_fps_while_requested":captured/elapsed,
                "inference_ms_p95":float(np.percentile(latencies,95)) if latencies else None,
                "age_ms_p95":float(np.percentile(ages,95)) if ages else None}
    finally: await server.stop()


async def benchmark(args):
    baseline = await measure(args,False)
    vision = await measure(args,True)
    report = {"baseline":baseline,"vision":vision,
              "fps_drop_fraction":1-vision["fps_mean"]/baseline["fps_mean"],
              "note":"Solicitações explícitas repetidas para benchmark; em repouso não há capturas."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=ROOT/"models/vision/odontologia.pt")
    parser.add_argument("--seconds",type=float,default=5)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/vision_evaluation/benchmark.json")
    args = parser.parse_args()
    if args.seconds <= 0: parser.error("--seconds deve ser positivo")
    print(json.dumps(asyncio.run(benchmark(args)),indent=2))
