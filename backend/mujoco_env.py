"""Inicialização da cena MuJoCo, carregamento do modelo `.mjcf` e passo de física em tempo real.

Suporta execução nativa via `mujoco` (C/C++ bindings de alta precisão com sensores de
contato, força, torque e quatérnios reais) e fallback físico determinístico que lê o
mesmo arquivo `.mjcf` caso a biblioteca `mujoco` não esteja instalada no interpretador.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import BackendConfig, configure_backend_logger, load_backend_config

try:
    import mujoco  # type: ignore[import-not-found]

    HAS_NATIVE_MUJOCO = True
except Exception:  # noqa: BLE001
    mujoco = None  # type: ignore[assignment]
    HAS_NATIVE_MUJOCO = False


SCENE_OBJECT_META: Dict[str, Dict[str, Any]] = {
    "dental_tool": {
        "type": "mesh",
        "size": [0.098, 0.0065, 0.0065],
        "color": "#c5ced6",
        "label": "Instrumento Dental",
        "initial_pos": [0.25, -0.172, 0.010],
        # Rx 90° — comprimento no Y (centro da haste em y = -0.22), espessura 12mm no X
        "initial_quat": [0.70710678, 0.70710678, 0.0, 0.0],
        "grasp_offset": [0.0, -0.048, 0.0],
        "visual_url": "/assets/objetos/odontologia/dental_tool-nih3d.stl",
    },
    "floss_holder": {
        "type": "mesh",
        "size": [0.038, 0.0095, 0.004],
        "color": "#b8c4d0",
        "label": "Suporte de Fio Dental",
        "initial_pos": [0.315, -0.238, 0.004],
        "initial_quat": [1.0, 0.0, 0.0, 0.0],
        "grasp_offset": [0.02347, 0.01819, 0.0],
        "visual_url": "/assets/objetos/odontologia/Angled_Dental_Floss_Holder.stl",
    },
}

# Orientação desejada do site `gripper/pinch`:
#   X_pinch = +Y_mundo, Y_pinch (abertura dos dedos) = +X_mundo, Z_pinch = -Z_mundo (vertical para baixo)
_PINCH_R_DES = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)

# Limites articulares seguros "elbow-up / wrist-down" (evitam inversão do cotovelo e singularidade wrist_2=0)
_UR5E_SAFE_LIMITS: Tuple[Tuple[float, float], ...] = (
    (-5.60, 2.40),   # shoulder_pan_joint
    (-2.70, -0.35),  # shoulder_lift_joint
    (0.22, 2.78),    # elbow_joint (sempre positivo -> cotovelo para cima)
    (-3.10, -0.25),  # wrist_1_joint
    (-2.45, -0.70),  # wrist_2_joint (longe da singularidade 0.0)
    (-6.10, 6.10),   # wrist_3_joint
)

_UR5E_MIN_RADIUS = 0.185  # Fora do cilindro de offset de ombro do UR5e (d4 = 0.134 m)
_UR5E_MAX_RADIUS = 0.730  # Dentro do envelope de alcance com cotovelo flexionado


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class MuJoCoGripperEnv:
    """Ambiente físico thread-safe da garra robótica alimentado por MuJoCo."""

    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or load_backend_config()
        self.logger = logger or configure_backend_logger(
            "MuJoCoEnv", level=self.config.server.log_level
        )
        self._lock = threading.RLock()

        mjcf_path = self.config.physics.mjcf_path
        if not mjcf_path.exists():
            raise FileNotFoundError(f"Arquivo de modelo MJCF não encontrado: {mjcf_path}")

        # Valida sintaxe XML do arquivo .mjcf sempre
        self._xml_tree = ET.parse(mjcf_path)

        # Alvo cartesiano do TCP (x, y, z) + abertura da garra (0 aberta .. 255 fechada)
        self._target_ctrl: List[float] = [0.0, 0.0, 0.32, 0.0]
        self._current_ctrl: List[float] = [0.0, 0.0, 0.32, 0.0]
        self._waypoints: List[List[float]] = []
        self._active_speed: float = self.config.gripper.default_speed
        self._commanded_force: float = self.config.gripper.default_force
        self._is_closed: bool = False
        self._seq: int = 0
        self._sim_time: float = 0.0
        self._substep_counter: int = 0

        # Estruturas para modo nativo MuJoCo vs fallback determinístico
        self._model: Any = None
        self._data: Any = None
        self._ik_data: Any = None
        self._jacp: Any = None
        self._jacr: Any = None
        self._desired_arm_q: Any = None
        self._body_ids: Dict[str, int] = {}
        self._object_geoms: Dict[str, int] = {}
        self._arm_dof: List[int] = []
        self._arm_qposadr: List[int] = []
        self._arm_range: List[Tuple[float, float]] = []
        self._visual_parts: List[Dict[str, Any]] = []
        self._pinch_site: int = -1
        self._gripper_act: int = -1
        self._left_pad_geom: int = -1
        self._right_pad_geom: int = -1
        self._uses_ur5e: bool = False
        self._fallback_objects: Dict[str, Dict[str, Any]] = {}
        self._fallback_grasped: Optional[str] = None

        self._load_model()

    def _load_model(self) -> None:
        """Carrega UR5e + Robotiq no motor nativo MuJoCo ou inicializa o fallback 3D."""
        if HAS_NATIVE_MUJOCO:
            from backend.clinic_model import ARM_JOINTS, compile_clinic_model, visual_part_catalog

            self._model, self._data = compile_clinic_model()
            self._ik_data = mujoco.MjData(self._model)
            self._jacp = np.zeros((3, self._model.nv), dtype=np.float64)
            self._jacr = np.zeros((3, self._model.nv), dtype=np.float64)
            self._uses_ur5e = True
            self._visual_parts = visual_part_catalog(self._model)
            self._pinch_site = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch"
            )
            self._gripper_act = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator"
            )
            self._left_pad_geom = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/left_pad1"
            )
            self._right_pad_geom = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_GEOM, "gripper/right_pad1"
            )
            self._arm_dof = []
            self._arm_qposadr = []
            self._arm_range = []
            for idx, jname in enumerate(ARM_JOINTS):
                jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                self._arm_dof.append(int(self._model.jnt_dofadr[jid]))
                self._arm_qposadr.append(int(self._model.jnt_qposadr[jid]))
                lo_xml, hi_xml = self._model.jnt_range[jid]
                lo_safe, hi_safe = _UR5E_SAFE_LIMITS[idx]
                self._arm_range.append((max(float(lo_xml), lo_safe), min(float(hi_xml), hi_safe)))

            self._sync_ik_from_sim()
            pinch = self._data.site_xpos[self._pinch_site]
            self._target_ctrl = [float(pinch[0]), float(pinch[1]), float(pinch[2]), 0.0]
            self._current_ctrl = list(self._target_ctrl)
            self._waypoints = []
            self._apply_actuator_force_limits()

            for obj_name in SCENE_OBJECT_META:
                bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
                if bid >= 0:
                    self._body_ids[obj_name] = bid
                gid = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_GEOM, f"{obj_name}_col"
                )
                if gid >= 0:
                    self._object_geoms[obj_name] = gid

            # Assenta brevemente os instrumentos sobre o fundo físico da bandeja
            for _ in range(30):
                self._data.ctrl[:6] = self._desired_arm_q
                self._data.qfrc_applied[self._arm_dof] = self._data.qfrc_bias[self._arm_dof]
                mujoco.mj_step(self._model, self._data)

            self.engine_name = f"mujoco-{mujoco.__version__}-ur5e"
        else:
            self.engine_name = "mujoco-mjcf-kinematic-fallback"
            self._reset_fallback_objects()

        self.logger.info(
            "Cena carregada de '%s' [motor=%s | dt=%.4fs | stream=%d FPS]",
            self.config.physics.mjcf_path.name,
            self.engine_name,
            self.config.physics.timestep,
            self.config.physics.broadcast_fps,
        )

    def _sync_ik_from_sim(self) -> None:
        if self._ik_data is None or self._data is None:
            return
        self._ik_data.qpos[:] = self._data.qpos
        mujoco.mj_kinematics(self._model, self._ik_data)
        mujoco.mj_comPos(self._model, self._ik_data)
        self._desired_arm_q = np.array(
            [self._data.qpos[qa] for qa in self._arm_qposadr], dtype=np.float64
        )

    def _clamp_cartesian_xyz(self, x: float, y: float, z: float) -> List[float]:
        """Garante que o alvo (x, y, z) permaneça no anel alcançável fora da coluna central do UR5e."""
        gcfg = self.config.gripper
        cx = _clamp(float(x), *gcfg.workspace_x)
        cy = _clamp(float(y), *gcfg.workspace_y)
        z_limits = gcfg.workspace_z if self._uses_ur5e else gcfg.slide_z_limits
        cz = _clamp(float(z), *z_limits)

        if self._uses_ur5e:
            r = math.hypot(cx, cy)
            if r < _UR5E_MIN_RADIUS:
                if r < 1e-6:
                    cx, cy = 0.0, _UR5E_MIN_RADIUS
                else:
                    scale = _UR5E_MIN_RADIUS / r
                    cx *= scale
                    cy *= scale
            elif r > _UR5E_MAX_RADIUS:
                scale = _UR5E_MAX_RADIUS / r
                cx *= scale
                cy *= scale

        return [cx, cy, cz]

    def _plan_waypoints(self, start_xyz: List[float], goal_xyz: List[float]) -> List[List[float]]:
        """Desvia suavemente pelo arco frontal (+Y) caso a reta cruze a singularidade da base (0,0)."""
        if not self._uses_ur5e:
            return [list(goal_xyz)]

        dx = goal_xyz[0] - start_xyz[0]
        dy = goal_xyz[1] - start_xyz[1]
        seg_len2 = dx * dx + dy * dy
        if seg_len2 > 1e-6:
            t_closest = _clamp(-(start_xyz[0] * dx + start_xyz[1] * dy) / seg_len2, 0.0, 1.0)
            near_x = start_xyz[0] + t_closest * dx
            near_y = start_xyz[1] + t_closest * dy
            if math.hypot(near_x, near_y) < (_UR5E_MIN_RADIUS + 0.012) and 0.04 < t_closest < 0.96:
                safe_z = max(start_xyz[2], goal_xyz[2], 0.18)
                arc_wp = [0.06, 0.28, safe_z]
                return [arc_wp, list(goal_xyz)]

        return [list(goal_xyz)]

    def _reset_fallback_objects(self) -> None:
        self._fallback_objects = {
            name: {
                "pos": list(meta["initial_pos"]),
                "quat": list(meta.get("initial_quat", [1.0, 0.0, 0.0, 0.0])),
            }
            for name, meta in SCENE_OBJECT_META.items()
        }
        self._fallback_grasped = None

    def reset(self) -> Dict[str, Any]:
        """Restaura toda a simulação para o estado inicial."""
        with self._lock:
            self._target_ctrl = [0.0, 0.0, 0.0, 0.0]
            self._current_ctrl = [0.0, 0.0, 0.0, 0.0]
            self._waypoints = []
            self._active_speed = self.config.gripper.default_speed
            self._commanded_force = self.config.gripper.default_force
            self._is_closed = False
            self._sim_time = 0.0
            self._substep_counter = 0

            if self._model is not None and self._data is not None:
                if self._model.nkey > 0:
                    mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
                else:
                    mujoco.mj_resetData(self._model, self._data)
                self._data.qvel[:] = 0.0
                self._data.qacc[:] = 0.0
                self._data.qfrc_applied[:] = 0.0
                mujoco.mj_forward(self._model, self._data)

                if self._uses_ur5e and self._pinch_site >= 0:
                    self._sync_ik_from_sim()
                    pinch = self._data.site_xpos[self._pinch_site]
                    self._target_ctrl = [float(pinch[0]), float(pinch[1]), float(pinch[2]), 0.0]
                    self._current_ctrl = list(self._target_ctrl)
                    self._data.ctrl[:6] = self._desired_arm_q
                    if self._gripper_act >= 0:
                        self._data.ctrl[self._gripper_act] = 0.0

                self._apply_actuator_force_limits()
                if self._uses_ur5e:
                    for _ in range(25):
                        self._data.ctrl[:6] = self._desired_arm_q
                        self._data.qfrc_applied[self._arm_dof] = self._data.qfrc_bias[self._arm_dof]
                        mujoco.mj_step(self._model, self._data)
                else:
                    mujoco.mj_forward(self._model, self._data)
            else:
                self._reset_fallback_objects()

            return self.get_state_snapshot()

    def set_motion_target(
        self,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        close_jaws: Optional[bool] = None,
        speed: Optional[float] = None,
        force: Optional[float] = None,
    ) -> None:
        """Atualiza os alvos cartesianos incrementais respeitando o envelope seguro do UR5e."""
        gcfg = self.config.gripper
        with self._lock:
            if speed is not None:
                self._active_speed = _clamp(float(speed), gcfg.min_speed, gcfg.max_speed)
            if force is not None:
                self._commanded_force = _clamp(float(force), gcfg.min_force, gcfg.max_force)

            if dx != 0.0 or dy != 0.0 or dz != 0.0:
                clamped = self._clamp_cartesian_xyz(
                    self._target_ctrl[0] + dx,
                    self._target_ctrl[1] + dy,
                    self._target_ctrl[2] + dz,
                )
                self._target_ctrl[:3] = clamped
                self._waypoints = self._plan_waypoints(self._current_ctrl[:3], clamped)

            if close_jaws is not None:
                self._is_closed = bool(close_jaws)
                if self._uses_ur5e:
                    self._target_ctrl[3] = 255.0 if self._is_closed else 0.0
                else:
                    self._target_ctrl[3] = 0.040 if self._is_closed else 0.0

            self._apply_actuator_force_limits()

    def set_absolute_target(
        self,
        *,
        x: Optional[float] = None,
        y: Optional[float] = None,
        tcp_z: Optional[float] = None,
        close_jaws: Optional[bool] = None,
        speed: Optional[float] = None,
        force: Optional[float] = None,
    ) -> None:
        """Define coordenadas cartesianas absolutas do TCP com desvio automático da base."""
        gcfg = self.config.gripper
        with self._lock:
            if speed is not None:
                self._active_speed = _clamp(float(speed), gcfg.min_speed, gcfg.max_speed)
            if force is not None:
                self._commanded_force = _clamp(float(force), gcfg.min_force, gcfg.max_force)

            nx = self._target_ctrl[0] if x is None else float(x)
            ny = self._target_ctrl[1] if y is None else float(y)
            if tcp_z is None:
                nz = self._target_ctrl[2]
            elif self._uses_ur5e:
                nz = float(tcp_z)
            else:
                nz = float(tcp_z) - (gcfg.carriage_home_z + gcfg.tcp_z_offset)

            clamped = self._clamp_cartesian_xyz(nx, ny, nz)
            self._target_ctrl[:3] = clamped
            self._waypoints = self._plan_waypoints(self._current_ctrl[:3], clamped)

            if close_jaws is not None:
                self._is_closed = bool(close_jaws)
                if self._uses_ur5e:
                    self._target_ctrl[3] = 255.0 if self._is_closed else 0.0
                else:
                    self._target_ctrl[3] = 0.040 if self._is_closed else 0.0
            self._apply_actuator_force_limits()

    def _apply_actuator_force_limits(self) -> None:
        """Propaga a força solicitada aos atuadores dos dedos no modelo MuJoCo."""
        if self._model is None:
            return
        if self._uses_ur5e and self._gripper_act >= 0:
            # F_pad ≈ 8.3 * F_tendon -> peak=3.0 em force=50N gera ~25N por dedo (~50N totais)
            peak = _clamp(self._commanded_force * 0.06, 0.6, 7.0)
            self._model.actuator_forcerange[self._gripper_act] = [-peak, peak]
        elif self._model.nu >= 5:
            per_finger = max(0.5, self._commanded_force * 0.5)
            self._model.actuator_forcerange[3] = [-per_finger, per_finger]
            self._model.actuator_forcerange[4] = [-per_finger, per_finger]

    def _solve_ik_step(self, target_xyz: List[float], iters: int = 2) -> None:
        """IK 6-DOF amortecida em buffer cinemático separado (zero teletransporte de `data.qpos`)."""
        site = self._pinch_site
        dofs = self._arm_dof
        ik_d = self._ik_data
        tgt = np.asarray(target_xyz, dtype=np.float64)

        for _ in range(iters):
            mujoco.mj_kinematics(self._model, ik_d)
            mujoco.mj_comPos(self._model, ik_d)
            mujoco.mj_jacSite(self._model, ik_d, self._jacp, self._jacr, site)

            err_p = tgt - ik_d.site_xpos[site]
            R_cur = ik_d.site_xmat[site].reshape(3, 3)
            err_r = 0.5 * (
                np.cross(R_cur[:, 0], _PINCH_R_DES[:, 0])
                + np.cross(R_cur[:, 1], _PINCH_R_DES[:, 1])
                + np.cross(R_cur[:, 2], _PINCH_R_DES[:, 2])
            )
            if float(np.dot(err_p, err_p)) < 1e-10 and float(np.dot(err_r, err_r)) < 1e-8:
                break

            err = np.concatenate((err_p, 0.75 * err_r))
            J = np.vstack((self._jacp[:, dofs], self._jacr[:, dofs]))
            JJt = J @ J.T + 1.8e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            dq = np.clip(dq, -0.16, 0.16)

            for i, qadr in enumerate(self._arm_qposadr):
                lo, hi = self._arm_range[i]
                q_next = _clamp(float(ik_d.qpos[qadr] + dq[i]), lo, hi)
                ik_d.qpos[qadr] = q_next
                self._desired_arm_q[i] = q_next

    def step_physics(self, substeps: Optional[int] = None) -> Dict[str, Any]:
        """Avança `substeps` passos de física com servoing dinâmico PD + compensação gravitacional."""
        n_steps = substeps or self.config.physics.substeps_per_frame
        dt = self.config.physics.timestep

        with self._lock:
            max_cart_step = self._active_speed * dt
            jaw_rate = 820.0 if self._uses_ur5e else max(0.14, self._active_speed * 0.45)
            max_jaw_step = jaw_rate * dt

            for _ in range(n_steps):
                active_wp = self._waypoints[0] if self._waypoints else self._target_ctrl[:3]
                dx = active_wp[0] - self._current_ctrl[0]
                dy = active_wp[1] - self._current_ctrl[1]
                dz = active_wp[2] - self._current_ctrl[2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)

                if dist <= max_cart_step:
                    self._current_ctrl[0] = active_wp[0]
                    self._current_ctrl[1] = active_wp[1]
                    self._current_ctrl[2] = active_wp[2]
                    if self._waypoints:
                        self._waypoints.pop(0)
                else:
                    inv = max_cart_step / dist
                    self._current_ctrl[0] += dx * inv
                    self._current_ctrl[1] += dy * inv
                    self._current_ctrl[2] += dz * inv

                if self._uses_ur5e:
                    clamped_cur = self._clamp_cartesian_xyz(
                        self._current_ctrl[0], self._current_ctrl[1], self._current_ctrl[2]
                    )
                    self._current_ctrl[:3] = clamped_cur

                jaw_diff = self._target_ctrl[3] - self._current_ctrl[3]
                if abs(jaw_diff) <= max_jaw_step:
                    self._current_ctrl[3] = self._target_ctrl[3]
                else:
                    self._current_ctrl[3] += math.copysign(max_jaw_step, jaw_diff)

                if self._model is not None and self._data is not None:
                    if self._uses_ur5e:
                        if (self._substep_counter & 1) == 0:
                            self._solve_ik_step(self._current_ctrl[:3], iters=2)
                        self._data.ctrl[:6] = self._desired_arm_q
                        if self._gripper_act >= 0:
                            self._data.ctrl[self._gripper_act] = self._current_ctrl[3]
                        self._data.qfrc_applied[self._arm_dof] = self._data.qfrc_bias[self._arm_dof]
                    else:
                        self._data.ctrl[0] = self._current_ctrl[0]
                        self._data.ctrl[1] = self._current_ctrl[1]
                        self._data.ctrl[2] = self._current_ctrl[2]
                        self._data.ctrl[3] = self._current_ctrl[3]
                        self._data.ctrl[4] = self._current_ctrl[3]

                    mujoco.mj_step(self._model, self._data)

                    # Salvaguarda contra divergência numérica sob perturbações extremas
                    if self._uses_ur5e and (
                        not np.isfinite(self._data.qacc).all()
                        or not np.isfinite(self._data.qpos).all()
                    ):
                        self.logger.warning("Recuperando estado físico após perturbação numérica.")
                        self._data.qpos[:6] = self._desired_arm_q
                        self._data.qvel[:] = 0.0
                        self._data.qacc[:] = 0.0
                        mujoco.mj_forward(self._model, self._data)
                else:
                    self._step_fallback_kinematics()

                self._substep_counter += 1
                self._sim_time += dt

            self._seq += 1
            return self.get_state_snapshot()

    def step_until_settled(self, duration_s: float = 0.16) -> Dict[str, Any]:
        """Integra a física pelo tempo total da trajetória cartesiana/garra mais acomodação."""
        with self._lock:
            pts = [self._current_ctrl[:3]] + (
                list(self._waypoints) if self._waypoints else [self._target_ctrl[:3]]
            )
            path_len = 0.0
            for a, b in zip(pts, pts[1:]):
                path_len += math.sqrt(
                    (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2
                )
            cart_time = path_len / max(0.05, self._active_speed)
            jaw_rate = 820.0 if self._uses_ur5e else max(0.14, self._active_speed * 0.45)
            jaw_time = abs(self._target_ctrl[3] - self._current_ctrl[3]) / jaw_rate

        total_time = max(float(duration_s), cart_time, jaw_time) + 0.22
        steps = max(1, int(round(total_time / self.config.physics.timestep)))
        return self.step_physics(substeps=steps)

    def _step_fallback_kinematics(self) -> None:
        """Atualiza interação com objetos no modo fallback sem binário MuJoCo."""
        gcfg = self.config.gripper
        tcp_x = self._current_ctrl[0]
        tcp_y = self._current_ctrl[1]
        tcp_z = gcfg.carriage_home_z + self._current_ctrl[2] + gcfg.tcp_z_offset

        if self._is_closed and self._current_ctrl[3] >= 0.015:
            if self._fallback_grasped is None:
                for name, state in self._fallback_objects.items():
                    ox, oy, oz = state["pos"]
                    off = SCENE_OBJECT_META[name].get("grasp_offset", [0.0, 0.0, 0.0])
                    gx, gy, gz = ox + off[0], oy + off[1], oz + off[2]
                    if math.hypot(tcp_x - gx, tcp_y - gy) <= 0.048 and abs(tcp_z - gz) <= 0.045:
                        self._fallback_grasped = name
                        break
            if self._fallback_grasped is not None:
                meta = SCENE_OBJECT_META[self._fallback_grasped]
                half_w = meta["size"][1]
                off = meta.get("grasp_offset", [0.0, 0.0, 0.0])
                self._current_ctrl[3] = min(self._current_ctrl[3], 0.040 - half_w * 0.85)
                self._fallback_objects[self._fallback_grasped]["pos"] = [
                    tcp_x - off[0],
                    tcp_y - off[1],
                    max(half_w, tcp_z - off[2]),
                ]
        else:
            if self._fallback_grasped is not None:
                name = self._fallback_grasped
                rest_z = SCENE_OBJECT_META[name]["initial_pos"][2]
                self._fallback_objects[name]["pos"][2] = rest_z
                self._fallback_grasped = None

    def _compute_real_aperture(self) -> float:
        """Mede a abertura física real (m) entre as pastilhas internas da Robotiq 2F-85."""
        if (
            self._data is not None
            and self._left_pad_geom >= 0
            and self._right_pad_geom >= 0
        ):
            lpos = self._data.geom_xpos[self._left_pad_geom]
            rpos = self._data.geom_xpos[self._right_pad_geom]
            center_dist = float(np.linalg.norm(rpos - lpos))
            return round(max(0.0, center_dist - 0.0084), 5)
        grip = self._current_ctrl[3]
        return round(max(0.0, 0.085 * (1.0 - grip / 255.0)), 5)

    def _detect_grasped_object_native(
        self, tcp_pos: List[float], left_touch: float, right_touch: float, aperture: float
    ) -> Tuple[bool, Optional[str]]:
        """Determina se um objeto está fisicamente preso entre os dedos no MuJoCo."""
        if not self._is_closed or (left_touch + right_touch) < 0.35 or aperture <= 0.0025:
            return False, None

        if self._left_pad_geom >= 0 and self._right_pad_geom >= 0:
            mid = 0.5 * (
                self._data.geom_xpos[self._left_pad_geom]
                + self._data.geom_xpos[self._right_pad_geom]
            )
            ref_x, ref_y, ref_z = float(mid[0]), float(mid[1]), float(mid[2])
        else:
            ref_x, ref_y, ref_z = tcp_pos[0], tcp_pos[1], tcp_pos[2]

        best_name: Optional[str] = None
        best_dist = 0.075
        for name, bid in self._body_ids.items():
            gid = self._object_geoms.get(name)
            opos = self._data.geom_xpos[gid] if gid is not None else self._data.xpos[bid]
            dist = math.sqrt(
                (ref_x - float(opos[0])) ** 2
                + (ref_y - float(opos[1])) ** 2
                + (ref_z - float(opos[2])) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return (best_name is not None), best_name

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Extrai o estado completo compatível com o protocolo WebSocket em tempo real."""
        gcfg = self.config.gripper
        with self._lock:
            if self._model is not None and self._data is not None:
                if self._uses_ur5e and self._pinch_site >= 0:
                    raw_pos = self._data.site_xpos[self._pinch_site]
                    raw_mat = self._data.site_xmat[self._pinch_site]
                    raw_quat = np.zeros(4)
                    mujoco.mju_mat2Quat(raw_quat, raw_mat)
                    try:
                        left_touch = float(self._data.sensor("left_finger_touch").data[0])
                        right_touch = float(self._data.sensor("right_finger_touch").data[0])
                    except Exception:
                        left_touch = 0.0
                        right_touch = 0.0
                    try:
                        raw_wf = self._data.sensor("wrist_force").data
                        raw_wt = self._data.sensor("wrist_torque").data
                        wrist_force = [round(float(raw_wf[i]), 3) for i in range(3)]
                        wrist_torque = [round(float(raw_wt[i]), 4) for i in range(3)]
                    except Exception:
                        wrist_force = [0.0, 0.0, 0.0]
                        wrist_torque = [0.0, 0.0, 0.0]

                    aperture = self._compute_real_aperture()
                    vel = [
                        round(float(self._data.qvel[self._arm_dof[0]]), 4),
                        round(float(self._data.qvel[self._arm_dof[1]]), 4),
                        round(float(self._data.qvel[self._arm_dof[2]]), 4),
                    ]
                else:
                    raw_pos = self._data.sensor("tcp_pos").data
                    raw_quat = self._data.sensor("tcp_quat").data
                    raw_wf = self._data.sensor("wrist_force").data
                    raw_wt = self._data.sensor("wrist_torque").data
                    left_touch = float(self._data.sensor("left_finger_touch").data[0])
                    right_touch = float(self._data.sensor("right_finger_touch").data[0])
                    fl = float(self._data.sensor("jpos_finger_l").data[0])
                    fr = float(self._data.sensor("jpos_finger_r").data[0])
                    aperture = round(max(0.0, gcfg.max_aperture - (fl + fr)), 5)
                    wrist_force = [round(float(raw_wf[i]), 3) for i in range(3)]
                    wrist_torque = [round(float(raw_wt[i]), 4) for i in range(3)]
                    vel = [
                        round(float(self._data.sensor("jvel_x").data[0]), 4),
                        round(float(self._data.sensor("jvel_y").data[0]), 4),
                        round(float(self._data.sensor("jvel_z").data[0]), 4),
                    ]

                pos = [round(float(raw_pos[i]), 5) for i in range(3)]
                quat = [round(float(raw_quat[i]), 5) for i in range(4)]
                contact_force = round(left_touch + right_touch, 3)
                is_grasping, grasped_obj = self._detect_grasped_object_native(
                    pos, left_touch, right_touch, aperture
                )

                objects_payload: List[Dict[str, Any]] = []
                for name, meta in SCENE_OBJECT_META.items():
                    bid = self._body_ids.get(name)
                    if bid is None:
                        continue
                    bpos = self._data.xpos[bid]
                    bquat = self._data.xquat[bid]
                    gid = self._object_geoms.get(name)
                    gpos = self._data.geom_xpos[gid] if gid is not None else bpos
                    objects_payload.append(
                        {
                            "name": name,
                            "label": meta["label"],
                            "type": meta["type"],
                            "size": meta["size"],
                            "color": meta["color"],
                            "pos": [round(float(bpos[i]), 5) for i in range(3)],
                            "grasp_pos": [round(float(gpos[i]), 5) for i in range(3)],
                            "quat": [round(float(bquat[i]), 5) for i in range(4)],
                        }
                    )
                if self._uses_ur5e:
                    from backend.clinic_model import geom_pose_payload

                    robot_payload = geom_pose_payload(self._model, self._data, self._visual_parts)
                else:
                    robot_payload = []
            else:
                tcp_z = gcfg.carriage_home_z + self._current_ctrl[2] + gcfg.tcp_z_offset
                pos = [
                    round(self._current_ctrl[0], 5),
                    round(self._current_ctrl[1], 5),
                    round(tcp_z, 5),
                ]
                quat = [1.0, 0.0, 0.0, 0.0]
                aperture = round(max(0.0, gcfg.max_aperture - 2.0 * self._current_ctrl[3]), 5)
                is_grasping = self._fallback_grasped is not None
                grasped_obj = self._fallback_grasped
                contact_force = round(self._commanded_force if is_grasping else 0.0, 3)
                left_touch = round(contact_force * 0.5, 3)
                right_touch = round(contact_force * 0.5, 3)
                wrist_force = [0.0, 0.0, round(-0.65 if is_grasping else 0.0, 3)]
                wrist_torque = [0.0, 0.0, 0.0]
                vel = [0.0, 0.0, 0.0]
                objects_payload = [
                    {
                        "name": name,
                        "label": meta["label"],
                        "type": meta["type"],
                        "size": meta["size"],
                        "color": meta["color"],
                        "pos": [round(v, 5) for v in self._fallback_objects[name]["pos"]],
                        "grasp_pos": [
                            round(
                                self._fallback_objects[name]["pos"][i]
                                + meta.get("grasp_offset", [0.0, 0.0, 0.0])[i],
                                5,
                            )
                            for i in range(3)
                        ],
                        "quat": list(self._fallback_objects[name]["quat"]),
                    }
                    for name, meta in SCENE_OBJECT_META.items()
                ]
                robot_payload = []

            return {
                "pos": pos,
                "quat": quat,
                "force": round(self._commanded_force, 3),
                "commanded_force": round(self._commanded_force, 2),
                "contact_force": contact_force,
                "speed": round(self._active_speed, 3),
                "aperture": aperture,
                "is_closed": self._is_closed,
                "is_grasping": is_grasping,
                "grasped_object": grasped_obj,
                "velocity": vel,
                "sensors": {
                    "left_touch": round(left_touch, 3),
                    "right_touch": round(right_touch, 3),
                    "wrist_force": wrist_force,
                    "wrist_torque": wrist_torque,
                },
                "objects": objects_payload,
                "robot": robot_payload,
                "drop_zone": {
                    "pos": [-0.30, 0.26, 0.0015],
                    "size": [0.065, 0.065, 0.0015],
                },
                "sim_time": round(self._sim_time, 4),
                "seq": self._seq,
                "server_ts": round(time.time() * 1000.0, 3),
                "engine": self.engine_name,
            }
