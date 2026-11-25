"""Ajusta ponto relativo à caixa no treino; não autoriza captura automática."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def calibrate(args):
    calibration = json.loads((ROOT / "models/vision/calibration.json").read_text(encoding="utf-8"))
    if calibration.get("validated_for_pick") and args.output.resolve() == (ROOT / "models/vision/calibration.json").resolve():
        raise ValueError("Preserve a calibração validada usando outro --output.")
    rows = [json.loads(s) for s in (args.dataset / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    for name,rule in calibration["classes"].items():
        ratios, heights, sizes = [], [], []
        for row in rows:
            if row["split"] != "train": continue
            for truth in row["truth"]:
                if truth["class_name"] != name: continue
                x1,y1,x2,y2 = truth["bbox_xyxy_px"]
                u,v = truth["grasp_pixel"]
                ratio = [(u-x1)/(x2-x1),(v-y1)/(y2-y1)]
                # Um grasp fora da caixa visível indica oclusão: não ajustar
                # a política de preensão com uma região que não contém o alvo.
                if not all(0 <= value <= 1 for value in ratio): continue
                ratios.append(ratio)
                sizes.append([x2-x1,y2-y1])
                heights.append(truth["grasp_world_m"][2])
        if not ratios:
            raise ValueError(f"Sem amostras de treino para {name}.")
        rule["bbox_point"] = np.clip(np.median(ratios,axis=0),0,1).tolist()
        rule["plane_z_m"] = float(np.median(heights))
        median_size = np.median(sizes,axis=0)
        rule["bbox_size_range_px"] = [(median_size*.60).tolist(),(median_size*1.8).tolist()]
    calibration["camera"] = {k:v for k,v in rows[0]["camera"].items()
                             if k not in ("scene_revision","sim_time_s")}
    dataset = args.dataset.resolve()
    dataset_name = dataset.relative_to(ROOT).as_posix() if dataset.is_relative_to(ROOT) else str(dataset)
    identifier = hashlib.sha256(json.dumps({"camera":calibration["camera"],"classes":calibration["classes"]},sort_keys=True).encode()).hexdigest()[:12]
    calibration.update(id=f"vision-{identifier}",validated_for_pick=False,
                       fit_dataset=dataset_name,
                       note="Ajuste em treino sintético; exige avaliação reservada e ensaio físico.")
    calibration.pop("validation_scope",None)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(calibration,indent=2),encoding="utf-8")
    return calibration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=ROOT / "datasets/odontologia-v2")
    parser.add_argument("--output",type=Path,default=ROOT / "models/vision/calibration.json")
    print(json.dumps(calibrate(parser.parse_args()),indent=2))
