"""Câmera offscreen: todas as chamadas gráficas pertencem ao worker chamador."""
from __future__ import annotations

import time
import numpy as np


class VisionCamera:
    def __init__(self, env, config):
        import mujoco

        if env._model is None:
            raise RuntimeError("Visão indisponível no fallback cinemático.")
        self.env, self.config = env, config
        self.data = mujoco.MjData(env._model)
        self.camera_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_CAMERA,
                                         config.camera_name)
        if self.camera_id < 0:
            raise ValueError(f"Câmera ausente: {config.camera_name}")
        self.renderer = mujoco.Renderer(env._model, height=config.height, width=config.width)

    def capture(self):
        import mujoco

        revision, sim_time = self.env.copy_vision_state(self.data)
        captured = time.perf_counter()
        mujoco.mj_forward(self.env._model, self.data)
        self.renderer.update_scene(self.data, camera=self.config.camera_name)
        rgb = self.renderer.render().copy()
        cid = self.camera_id
        height, width = rgb.shape[:2]
        fy = height / (2 * np.tan(np.deg2rad(self.env._model.cam_fovy[cid]) / 2))
        metadata = {
            "camera_id": self.config.camera_name, "image_size": [width, height],
            "position": self.data.cam_xpos[cid].tolist(),
            "rotation": self.data.cam_xmat[cid].reshape(3, 3).tolist(),
            "intrinsics": [float(fy), float(fy), (width - 1) / 2, (height - 1) / 2],
            "scene_revision": revision, "sim_time_s": sim_time,
        }
        return rgb, metadata, captured

    def close(self):
        self.renderer.close()
