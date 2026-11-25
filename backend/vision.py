"""Adaptador YOLO opcional, com pesos locais e manifesto verificável."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np


def configure_ultralytics():
    from backend.config import PROJECT_ROOT
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "runs/ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


class YoloDetector:
    def __init__(self, config):
        path = Path(config.model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pesos não encontrados: {path}. Treine com scripts/train_yolo.py.")
        manifest_path = path.with_suffix(".json")
        if not manifest_path.is_file():
            raise ValueError(f"Manifesto ausente: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest["sha256"]:
            raise ValueError("Hash dos pesos difere do manifesto.")
        configure_ultralytics()
        from ultralytics import YOLO

        self.config = config
        self._configure_cpu_threads()
        self.model = YOLO(str(path), task="detect")
        # select_device() pode redefinir as threads ao criar o predictor.
        # O callback reaplica o limite antes da inferência, inclusive a primeira.
        self.model.add_callback("on_predict_start", self._configure_cpu_threads)
        if tuple(self.model.names.values()) != tuple(self.manifest["classes"]):
            raise ValueError("Classes do modelo diferem do manifesto.")
        if set(self.manifest["classes"]) != set(config.allowed_classes):
            raise ValueError("Modelo não contém exatamente as classes odontológicas configuradas.")
        self.identifier = self.manifest.get("id", path.stem)
        self.detect(np.zeros((config.height, config.width, 3), dtype=np.uint8))

    def _configure_cpu_threads(self, predictor=None):
        if self.config.device == "cpu":
            import torch
            if torch.get_num_threads() != self.config.cpu_threads:
                torch.set_num_threads(self.config.cpu_threads)

    def detect(self, rgb):
        # NumPy no Ultralytics usa BGR; resultados xyxy já estão na imagem original.
        result = self.model.predict(rgb[:, :, ::-1].copy(), imgsz=self.config.imgsz,
                                    conf=self.config.confidence_threshold,
                                    device=self.config.device, max_det=20, rect=False, verbose=False)[0]
        detections = []
        if result.boxes is None:
            return detections
        for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                   result.boxes.conf.cpu().tolist(),
                                   result.boxes.cls.cpu().tolist()):
            name = self.model.names[int(cls)]
            if name in self.config.allowed_classes and np.isfinite([*box, score]).all():
                detections.append({"class_id": int(cls), "class_name": name,
                                   "confidence": float(score), "bbox_xyxy_px": box})
        return detections
