"""Avalia detecções e erro de preensão contra truth reservado; não altera validações."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate(args):
    from PIL import Image
    from backend.config import VisionConfig
    from backend.vision import YoloDetector
    from backend.vision_geometry import VisionGeometry

    detector = YoloDetector(replace(VisionConfig(), model_path=args.model))
    geometry = VisionGeometry(args.calibration)
    distances, elapsed, matches, expected, false_positives = [], [], 0, 0, 0
    per_class = {name: {"expected":0,"matches":0,"false_positives":0,"xy_errors_mm":[]}
                 for name in detector.config.allowed_classes}
    rejected = 0
    for line in (args.dataset / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        if sample["split"] != "test":
            continue
        rgb = np.array(Image.open(args.dataset / sample["image"]).convert("RGB"))
        start = time.perf_counter()
        detections = detector.detect(rgb)
        elapsed.append(1000 * (time.perf_counter() - start))
        truth = list(sample["truth"])
        expected += len(truth)
        for item in truth:
            per_class[item["class_name"]]["expected"] += 1
        for detection in sorted(detections, key=lambda d: -d["confidence"]):
            candidates = [t for t in truth if t["class_name"] == detection["class_name"]]
            def iou(t):
                a,b = detection["bbox_xyxy_px"], t["bbox_xyxy_px"]
                inter = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
                return inter / max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
            target = max(candidates, key=iou) if candidates else None
            if target is None or iou(target) < .5:
                false_positives += 1
                per_class[detection["class_name"]]["false_positives"] += 1
                continue
            matches += 1
            per_class[detection["class_name"]]["matches"] += 1
            truth.remove(target)
            candidate = geometry.candidate(detection, sample["camera"])
            if candidate["position_world_m"] is not None:
                # XY: TCP Z é uma política de controle, não a superfície do instrumento.
                error_mm = 1000 * float(np.linalg.norm(np.array(candidate["position_world_m"][:2]) - target["grasp_world_m"][:2]))
                distances.append(error_mm)
                per_class[detection["class_name"]]["xy_errors_mm"].append(error_mm)
            else:
                rejected += 1
    for values in per_class.values():
        errors = values.pop("xy_errors_mm")
        values["xy_error_mm_p95"] = float(np.percentile(errors,95)) if errors else None
        values["precision_iou_50"] = values["matches"] / max(1,values["matches"]+values["false_positives"])
        values["recall_iou_50"] = values["matches"] / max(1,values["expected"])
    report = {"test_objects": expected, "matched_iou_50": matches,
              "false_positives": false_positives, "recall_iou_50": matches / max(1,expected),
              "precision_iou_50": matches / max(1,matches+false_positives),
              "xy_error_mm_p95": float(np.percentile(distances,95)) if distances else None,
              "xy_error_mm_max": max(distances) if distances else None,
              "inference_ms_p50": float(np.percentile(elapsed,50)) if elapsed else None,
              "inference_ms_p95": float(np.percentile(elapsed,95)) if elapsed else None,
              "capture_success_rate": None,
              "geometric_rejections": rejected, "localized_matches":len(distances),
              "per_class":per_class,
              "note": "Captura exige cenário físico adicional; este relatório não autoriza preensão."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/odontologia-v2")
    parser.add_argument("--model", type=Path, default=ROOT / "models/vision/odontologia.pt")
    parser.add_argument("--calibration", type=Path, default=ROOT / "models/vision/calibration.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/vision_evaluation/report.json")
    print(json.dumps(evaluate(parser.parse_args()), indent=2))
