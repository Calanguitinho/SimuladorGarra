"""Treina detector local e exporta pesos com manifesto; não valida preensão automaticamente."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def train(args):
    from backend.vision import configure_ultralytics
    configure_ultralytics()
    import ultralytics
    import torch
    import yaml
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)

    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise ValueError("Destino já existe; use --output com um novo nome de versão.")
    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    config["path"] = str(dataset_root.resolve())
    resolved = ROOT / "runs/vision/resolved-data.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(config), encoding="utf-8")
    # YAML cria uma arquitetura do zero, sem download de pesos. Também aceita .pt local.
    if args.model.endswith(".pt") and not Path(args.model).is_file():
        raise ValueError("Pesos iniciais devem ser um arquivo local. Use yolo11n.yaml para treino do zero.")
    model = YOLO(args.model, task="detect")
    initializer_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest() if Path(args.model).is_file() else None
    model.add_callback("on_train_start", lambda trainer: torch.set_num_threads(args.threads))
    model.train(data=str(resolved), epochs=args.epochs, imgsz=args.imgsz,
                device=args.device, batch=args.batch, workers=0, seed=args.seed,
                deterministic=True, project=str(ROOT / "runs/vision"), name="train",
                pretrained=False, plots=False, amp=False, mosaic=0.0, fliplr=0.0,
                flipud=0.0, degrees=0.0, translate=0.0, scale=0.0,
                nbs=args.batch, optimizer="AdamW", lr0=0.002)
    # Inferência usa letterbox quadrado. Selecionar entre best/last na validação
    # com esse mesmo pré-processamento, nunca usando o teste para escolher pesos.
    choices = []
    for checkpoint in dict.fromkeys((Path(model.trainer.best), Path(model.trainer.last))):
        candidate = YOLO(str(checkpoint))
        validation = candidate.val(data=str(resolved), split="val", device=args.device,
                                   imgsz=args.imgsz, workers=0, plots=False, rect=False)
        choices.append((float(validation.box.map), checkpoint))
    _, best = max(choices, key=lambda item:item[0])
    trained = YOLO(str(best))
    metrics = trained.val(data=str(resolved), split="test", device=args.device,
                          imgsz=args.imgsz, workers=0, plots=False, rect=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    manifest = {"id": args.output.stem, "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "classes": list(trained.names.values()), "validated_for_pick": False,
                "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                "epochs": args.epochs, "seed": args.seed, "data": str(args.data.resolve()),
                "epochs_completed": model.trainer.epoch + 1,
                "training": {"initializer": args.model, "imgsz": args.imgsz, "batch": args.batch,
                             "initializer_sha256": initializer_hash,
                             "initial_weights_loaded": Path(args.model).suffix == ".pt",
                             "threads": args.threads, "optimizer": "AdamW", "lr0": 0.002,
                             "nbs": args.batch, "pretrained": False},
                "selection": {"preprocessing":"square letterbox rect=False",
                              "validation_candidates":{p.stem:s for s,p in choices},
                              "selected":best.stem},
                "dataset_manifest": json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8")),
                "test_metrics": {k: float(v) for k,v in metrics.results_dict.items()}}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/odontologia-v2/data.yaml")
    parser.add_argument("--model", default=str(ROOT / "models/vision/yolo_dental.yaml"))
    parser.add_argument("--output", type=Path, default=ROOT / "models/vision/odontologia.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--threads", type=int, default=4)
    print(json.dumps(train(parser.parse_args()), indent=2))
