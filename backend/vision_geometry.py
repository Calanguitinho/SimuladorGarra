"""Projeção de câmera MuJoCo (-Z óptico, +Y acima) para plano em metros."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def pixel_to_world(u, v, camera, plane_z):
    fx, fy, cx, cy = camera["intrinsics"]
    origin = np.asarray(camera["position"], dtype=float)
    direction = np.asarray(camera["rotation"], dtype=float) @ np.array(
        [(u - cx) / fx, -(v - cy) / fy, -1.0])
    if not np.isfinite(direction).all() or abs(direction[2]) < 1e-9:
        raise ValueError("Raio inválido ou paralelo ao plano.")
    distance = (plane_z - origin[2]) / direction[2]
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Plano fora do campo frontal da câmera.")
    return (origin + distance * direction).tolist()


def world_to_pixel(point, camera):
    local = np.asarray(camera["rotation"]).T @ (np.asarray(point) - camera["position"])
    if local[2] >= -1e-9:
        raise ValueError("Ponto atrás da câmera.")
    fx, fy, cx, cy = camera["intrinsics"]
    return [float(cx + fx * local[0] / -local[2]),
            float(cy - fy * local[1] / -local[2])]


class VisionGeometry:
    def __init__(self, path):
        self.calibration = json.loads(Path(path).read_text(encoding="utf-8"))
        self.identifier = str(self.calibration["id"])
        for rule in self.calibration["classes"].values():
            values = [*rule["bbox_point"], rule["plane_z_m"], rule["tcp_z_m"]]
            if not np.isfinite(values).all() or not all(0 <= x <= 1 for x in rule["bbox_point"]):
                raise ValueError("Regra de preensão inválida.")

    def candidate(self, detection, camera):
        invalid = {"valid": False, "method": "plane", "position_world_m": None}
        try:
            reference = self.calibration["camera"]
            for field in ("position", "rotation", "image_size", "intrinsics"):
                if not np.allclose(camera[field], reference[field], atol=1e-5):
                    raise ValueError("Calibração não corresponde à câmera.")
            if camera["camera_id"] != reference["camera_id"]:
                raise ValueError("Identificador de câmera divergente.")
            rule = self.calibration["classes"][detection["class_name"]]
            x1, y1, x2, y2 = detection["bbox_xyxy_px"]
            width, height = camera["image_size"]
            if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
                raise ValueError("Objeto cortado ou caixa fora da imagem.")
            if "bbox_size_range_px" in rule:
                minimum, maximum = rule["bbox_size_range_px"]
                size = np.array([x2-x1,y2-y1])
                if np.any(size < minimum) or np.any(size > maximum):
                    raise ValueError("Dimensões incompatíveis; objeto parcial ou fora da escala calibrada.")
            rx, ry = rule["bbox_point"]
            point = pixel_to_world(x1 + rx * (x2 - x1), y1 + ry * (y2 - y1),
                                   camera, rule["plane_z_m"])
            xmin, xmax, ymin, ymax = self.calibration["roi_world_m"]
            if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
                raise ValueError("Alvo fora da região calibrada.")
            point[2] = float(rule["tcp_z_m"])
            validated = self.calibration.get("validated_for_pick") is True
            return {"valid": validated, "method": "plane", "position_world_m": point,
                    "reason": None if validated else "Calibração de preensão ainda não validada."}
        except (KeyError, ValueError, TypeError) as exc:
            return {**invalid, "reason": str(exc)}
