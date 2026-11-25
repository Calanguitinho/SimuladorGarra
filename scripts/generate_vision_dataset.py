"""Gera dataset de detecção a partir de máscaras visuais, com splits independentes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def generate(output, count=300, seed=20261123):
    import mujoco
    import numpy as np
    from PIL import Image
    from backend.config import VisionConfig
    from backend.mujoco_env import MuJoCoGripperEnv, SCENE_OBJECT_META
    from backend.vision_camera import VisionCamera
    from backend.vision_geometry import world_to_pixel

    output = Path(output).resolve()
    if count < 10:
        raise ValueError("Gerar ao menos 10 imagens para distribuir os três conjuntos.")
    if (output / "manifest.json").exists() or (output / "samples.jsonl").exists():
        raise ValueError("Destino já contém um dataset; use outro diretório para preservar os dados.")
    output.mkdir(parents=True, exist_ok=True)
    env = MuJoCoGripperEnv()
    camera = VisionCamera(env, VisionConfig())
    rng = np.random.default_rng(seed)
    model = env._model
    base_qpos = env._data.qpos.copy()
    classes = ["dental_tool", "floss_holder"]
    records = []
    try:
        for index in range(count):
            # Cada amostra usa uma configuração independente, não quadros adjacentes.
            split = "train" if index < int(count * .70) else "val" if index < int(count * .85) else "test"
            for folder in ("images", "labels"):
                (output / folder / split).mkdir(parents=True, exist_ok=True)
            env._data.qpos[:] = base_qpos
            env._data.qvel[:] = 0
            if rng.random() > .35:
                # Diversificar oclusões e fundo com poses cinemáticas plausíveis.
                env._sync_ik_from_sim()
                target = [rng.uniform(.18,.48),rng.uniform(-.32,.12),rng.uniform(.16,.42)]
                env._solve_ik_step(target,iters=80)
                env._data.qpos[env._arm_qposadr] = env._desired_arm_q
            # Varia fundo e iluminação sem mudar física ou coordenadas da câmera.
            model.geom("workbench").rgba[:3] = rng.uniform(.08, .28, 3)
            if model.nlight:
                model.light_diffuse[:] = rng.uniform(.5, .95, model.light_diffuse.shape)
            visible = []
            for name in classes:
                adr = model.jnt_qposadr[model.joint(name + "_joint").id]
                present = bool(rng.random() > .18)
                # Pequena faixa angular para o primeiro conjunto de captura.
                yaw = float(rng.uniform(-.12, .12))
                quat = np.zeros(4)
                mujoco.mju_mulQuat(quat, np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]),
                                  base_qpos[adr+3:adr+7])
                env._data.qpos[adr:adr+2] += rng.uniform(-.045, .045, 2)
                env._data.qpos[adr+3:adr+7] = quat
                if not present:
                    env._data.qpos[adr] = 3  # fora da imagem; não torna geometria transparente
                else:
                    visible.append(name)
            mujoco.mj_forward(model, env._data)
            rgb, meta, _ = camera.capture()
            camera.renderer.enable_segmentation_rendering()
            try:
                segmentation = camera.renderer.render().copy()
            finally:
                camera.renderer.disable_segmentation_rendering()
            height, width = rgb.shape[:2]
            labels, truth = [], []
            for class_id, name in enumerate(classes):
                geom_id = model.geom(name + "_visual").id
                mask = (segmentation[:, :, 0] == geom_id) & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                ys, xs = np.where(mask)
                if len(xs) < 8:
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                labels.append(f"{class_id} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
                body = model.body(name).id
                # Offsets atuais são definidos no mundo no snapshot; girar pela variação de yaw
                # não é necessário aqui: obter o mesmo ponto local a partir da pose inicial.
                base_adr = model.jnt_qposadr[model.joint(name + "_joint").id]
                base_rot = np.zeros(9)
                mujoco.mju_quat2Mat(base_rot, base_qpos[base_adr+3:base_adr+7])
                local_offset = base_rot.reshape(3,3).T @ np.array(SCENE_OBJECT_META[name]["grasp_offset"])
                grasp = env._data.xpos[body] + env._data.xmat[body].reshape(3,3) @ local_offset
                truth.append({"class_name": name, "bbox_xyxy_px": [x1,y1,x2,y2],
                              "grasp_world_m": grasp.tolist(),
                              "grasp_pixel": world_to_pixel(grasp, meta)})
            stem = f"{seed}_{index:06d}"
            image_path = output / "images" / split / (stem + ".jpg")
            Image.fromarray(rgb).save(image_path, quality=95)
            (output / "labels" / split / (stem + ".txt")).write_text("\n".join(labels), encoding="utf-8")
            records.append({"image": image_path.relative_to(output).as_posix(), "split": split,
                            "camera": meta, "truth": truth})
        (output / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        dataset_path = output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else output.as_posix()
        (output / "data.yaml").write_text(
            f"path: {dataset_path}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: dental_tool\n  1: floss_holder\n", encoding="utf-8")
        manifest = {"seed": seed, "count": count, "classes": classes, "mujoco": mujoco.__version__,
                    "scene_sha256": hashlib.sha256((ROOT / "backend/clinic_model.py").read_bytes()).hexdigest(),
                    "annotation": "visible visual-mesh segmentation; no collision proxies",
                    "limitations": "Synthetic fixed inclined camera, narrow yaw, randomized arm poses/objects/background/light; no real camera validation."}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        camera.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/odontologia-v2")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.count, args.seed), indent=2))
