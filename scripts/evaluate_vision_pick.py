"""Ensaio físico no simulador usando cópias temporárias dos manifestos."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


async def trial(args,temporary):
    from backend.config import VisionConfig, load_backend_config
    from backend.websocket_server import GripperWebSocketServer

    # Somente ensaio simulado; não altera as validações dos artefatos originais.
    weights = temporary / "trial.pt"
    shutil.copy2(args.model,weights)
    manifest = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    manifest["validated_for_pick"] = True
    weights.with_suffix(".json").write_text(json.dumps(manifest),encoding="utf-8")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration["validated_for_pick"] = True
    calibration_path = temporary / "calibration.json"
    calibration_path.write_text(json.dumps(calibration),encoding="utf-8")
    config = load_backend_config(host="127.0.0.1",port=0,log_level="WARNING",
        vision=VisionConfig(enabled=True,enable_pick=True,model_path=weights,
                            calibration_path=calibration_path,device=args.device,
                            max_age_ms=args.max_age_ms))
    server = GripperWebSocketServer(config)
    results = []
    rng = np.random.default_rng(args.seed)
    await server.start()
    try:
        for index in range(args.trials):
            server.execute_command({"command":"reset"})
            if args.position_jitter:
                import mujoco
                with server.env._lock:
                    model, data = server.env._model, server.env._data
                    for name in ("dental_tool","floss_holder"):
                        adr = model.jnt_qposadr[model.joint(name + "_joint").id]
                        data.qpos[adr:adr+2] += rng.uniform(-args.position_jitter,args.position_jitter,2)
                    mujoco.mj_forward(model,data)
            server.execute_command({"command":"vision_pick","class_name":args.class_name})
            await asyncio.wait_for(asyncio.shield(server.vision_pick.task),90)
            snapshot = server.vision.snapshot()
            results.append({"trial":index,**server.vision_pick.status,"last_observation":snapshot})
            image = server.vision.frame(snapshot.get("frame_id"))
            if image:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.with_suffix(".jpg").write_bytes(image)
        report = {"class_name":args.class_name,"trials":results,
                  "position_jitter_m":args.position_jitter,"seed":args.seed,
                  "success_rate":sum(r["stage"]=="completed" for r in results)/len(results),
                  "note":"Ensaio simulado com orientação inicial; não aprova automaticamente os artefatos."}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        return report
    finally:
        await server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=ROOT/"models/vision/odontologia.pt")
    parser.add_argument("--calibration",type=Path,default=ROOT/"models/vision/calibration.json")
    parser.add_argument("--class-name",choices=["dental_tool","floss_holder"],default="dental_tool")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--trials",type=int,default=1)
    parser.add_argument("--max-age-ms",type=float,default=300)
    parser.add_argument("--position-jitter",type=float,default=0.0)
    parser.add_argument("--seed",type=int,default=20261123)
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/vision_evaluation/pick.json")
    args = parser.parse_args()
    if args.trials < 1: parser.error("--trials deve ser positivo")
    if not 0 <= args.position_jitter <= .025: parser.error("Jitter deve estar entre 0 e 0.025 m")
    (ROOT/"artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vision_trial_",dir=ROOT/"artifacts") as directory:
        temporary = Path(directory).resolve()
        if temporary.parent != (ROOT/"artifacts").resolve():
            raise ValueError("Diretório temporário fora da área de artefatos.")
        print(json.dumps(asyncio.run(trial(args,temporary)),indent=2))
