"""Monta a cena clínica: UR5e + Robotiq 2F-85 + instrumento dental (MuJoCo MjSpec)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np

from backend.config import PROJECT_ROOT

UR5E_XML = PROJECT_ROOT / "assets" / "Garra" / "universal_robots_ur5e" / "ur5e.xml"
ROBOTIQ_XML = PROJECT_ROOT / "assets" / "Garra" / "robotiq_2f85" / "2f85.xml"
ODONTO_DIR = PROJECT_ROOT / "assets" / "objetos" / "odontologia"
DENTAL_STL = ODONTO_DIR / "dental_tool-nih3d.stl"
FLOSS_STL = ODONTO_DIR / "Angled_Dental_Floss_Holder.stl"

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

MESH_URL = {
    "ur": "/assets/Garra/universal_robots_ur5e/assets/{name}.obj",
    "gripper": "/assets/Garra/robotiq_2f85/assets/{name}.stl",
}


def build_clinic_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(UR5E_XML))
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.option.timestep = 0.002
    spec.option.noslip_iterations = 6
    spec.option.noslip_tolerance = 1e-5

    # Câmera de percepção independente da câmera orbital do navegador.
    camera = spec.worldbody.add_camera()
    camera.name = "vision_top"
    camera.pos = [0.28, -0.75, 0.60]
    # Vista frontal inclinada evita que antebraço/punho cubram toda a bandeja.
    forward = np.array([0.28, -0.22, 0.0]) - np.asarray(camera.pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack((right, up, -forward)).ravel())
    camera.quat = quat
    camera.fovy = 45
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    # Acabamento branco tipo braço industrial (HIWIN / UR)
    for mat in spec.materials:
        if mat.name == "urblue":
            mat.rgba = [0.92, 0.93, 0.94, 1.0]
        elif mat.name == "linkgray":
            mat.rgba = [0.88, 0.88, 0.89, 1.0]

    # Sintoniza servoatuadores do UR5e para resposta crítica sem sobressinal nem atraso
    for act in spec.actuators:
        if act.name in ("shoulder_pan", "shoulder_lift", "elbow"):
            act.gainprm[0] = 2400.0
            act.biasprm[:3] = [0.0, -2400.0, -180.0]
        elif act.name in ("wrist_1", "wrist_2", "wrist_3"):
            act.gainprm[0] = 650.0
            act.biasprm[:3] = [0.0, -650.0, -48.0]
            act.forcerange[:] = [-38.0, 38.0]

    gripper = mujoco.MjSpec.from_file(str(ROBOTIQ_XML))
    gripper.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    gripper.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    gripper.option.impratio = 10.0
    gripper.option.timestep = 0.002
    spec.attach(gripper, prefix="gripper/", site=spec.site("attachment_site"))

    # Bitmask de colisão:
    #   bit 1 (1): braço/mesa/bandeja
    #   bit 2 (2): pastilhas dos dedos Robotiq (não colidem entre si quando fecham vazias)
    #   bit 3 (4): instrumentos manipuláveis (colidem com mesa/bandeja=1, dedos=2 e entre si=4 -> 7)
    floor = spec.worldbody.add_geom()
    floor.name = "workbench"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.9, 0.9, 0.05]
    floor.rgba = [0.14, 0.17, 0.22, 1.0]
    floor.friction = [1.3, 0.03, 0.005]
    floor.contype = 1
    floor.conaffinity = 5  # braço (1) + instrumentos (4)

    # Fundo físico da bandeja metálica (alinha o repouso Z=0.003m com metal_tray.glb no Three.js)
    tray_yaw = math.radians(-8.0)
    tray_floor = spec.worldbody.add_geom()
    tray_floor.name = "tray_floor"
    tray_floor.type = mujoco.mjtGeom.mjGEOM_BOX
    tray_floor.pos = [0.28, -0.22, 0.0015]
    tray_floor.size = [0.16, 0.16, 0.0015]
    tray_floor.quat = [math.cos(tray_yaw * 0.5), 0.0, 0.0, math.sin(tray_yaw * 0.5)]
    tray_floor.friction = [1.35, 0.03, 0.005]
    tray_floor.contype = 1
    tray_floor.conaffinity = 5
    tray_floor.rgba = [0.22, 0.24, 0.27, 0.0]

    drop = spec.worldbody.add_site()
    drop.name = "drop_zone"
    drop.type = mujoco.mjtGeom.mjGEOM_BOX
    drop.pos = [-0.30, 0.26, 0.0015]
    drop.size = [0.07, 0.07, 0.0015]
    drop.rgba = [0.16, 0.78, 0.44, 0.45]

    def _add_stl(
        name: str,
        path: Path,
        pos: list[float],
        quat: list[float],
        rgba: list[float],
        mass: float,
        col_pos: list[float],
        col_size: list[float],
        mesh_scale: list[float] | None = None,
    ) -> None:
        """Malha só visual; colisão em caixa ajustada à haste real do instrumento.

        O compilador recentra/rotaciona a STL, então usamos colisão primitiva alinhada
        ao referencial original do arquivo STL desenhado pelo Three.js.
        """
        mesh = spec.add_mesh()
        mesh.name = f"{name}_mesh"
        mesh.file = str(path)
        mesh.scale = mesh_scale or [0.001, 0.001, 0.001]
        body = spec.worldbody.add_body()
        body.name = name
        body.pos = pos
        body.quat = quat
        fj = body.add_freejoint()
        fj.name = f"{name}_joint"

        vis = body.add_geom()
        vis.name = f"{name}_visual"
        vis.type = mujoco.mjtGeom.mjGEOM_MESH
        vis.meshname = f"{name}_mesh"
        vis.contype = 0
        vis.conaffinity = 0
        vis.group = 0
        vis.mass = 0.0
        vis.rgba = rgba

        col = body.add_geom()
        col.name = f"{name}_col"
        col.type = mujoco.mjtGeom.mjGEOM_BOX
        col.pos = col_pos
        col.size = col_size
        col.mass = mass
        col.group = 3
        col.contype = 4
        col.conaffinity = 7
        col.condim = 4
        col.priority = 1
        col.friction = [1.85, 0.08, 0.02]
        col.solimp[:] = [0.95, 0.99, 0.001, 0.5, 2.0]
        col.solref[:] = [0.004, 1.0]
        col.margin = 0.0
        col.rgba = [0.55, 0.55, 0.55, 0.0]

    # Haste cilíndrica/octogonal NIH3D (diâmetro real 12mm). Rx 90°: comprimento no Y (centro em y=-0.22).
    _add_stl(
        "dental_tool",
        DENTAL_STL,
        [0.25, -0.172, 0.010],
        [0.70710678, 0.70710678, 0.0, 0.0],
        [0.78, 0.82, 0.86, 1.0],
        0.045,
        [0.0, 0.0005, 0.048],
        [0.0065, 0.0065, 0.098],
    )
    if FLOSS_STL.is_file():
        # Suporte de fio dental ao lado direito da bandeja (centro em x≈0.338, y≈-0.22)
        _add_stl(
            "floss_holder",
            FLOSS_STL,
            [0.315, -0.238, 0.004],
            [1.0, 0.0, 0.0, 0.0],
            [0.72, 0.78, 0.86, 1.0],
            0.022,
            [0.02347, 0.01819, 0.004],
            [0.0095, 0.038, 0.004],
            mesh_scale=[0.001, 0.001, 0.005],
        )

    # Pastilhas de silicone da Robotiq 2F-85: atrito torsional elíptico e filtro limpo contra objetos
    for pad_name in (
        "gripper/left_pad1",
        "gripper/left_pad2",
        "gripper/right_pad1",
        "gripper/right_pad2",
    ):
        pad = spec.geom(pad_name)
        pad.contype = 2
        pad.conaffinity = 4
        pad.condim = 4
        pad.friction = [1.85, 0.08, 0.02]
        pad.solimp[:] = [0.95, 0.99, 0.001, 0.5, 2.0]
        pad.solref[:] = [0.004, 1.0]

    # Volume do site tátil exatamente centrado sobre pad_box1 + pad_box2 ([0, -0.0026, 0.01875])
    for pad_body, site_name in (
        ("gripper/left_pad", "left_touch_site"),
        ("gripper/right_pad", "right_touch_site"),
    ):
        body = spec.body(pad_body)
        site = body.add_site()
        site.name = site_name
        site.type = mujoco.mjtGeom.mjGEOM_BOX
        site.pos = [0.0, -0.0026, 0.01875]
        site.size = [0.014, 0.007, 0.021]
        site.rgba = [1.0, 0.5, 0.0, 0.0]

    def _sensor(name: str, typ: int, objtype: int, objname: str) -> None:
        s = spec.add_sensor()
        s.name = name
        s.type = typ
        s.objtype = objtype
        s.objname = objname

    _sensor("tcp_pos", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    _sensor("tcp_quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT, mujoco.mjtObj.mjOBJ_SITE, "gripper/pinch")
    _sensor("left_finger_touch", mujoco.mjtSensor.mjSENS_TOUCH, mujoco.mjtObj.mjOBJ_SITE, "left_touch_site")
    _sensor("right_finger_touch", mujoco.mjtSensor.mjSENS_TOUCH, mujoco.mjtObj.mjOBJ_SITE, "right_touch_site")
    _sensor("wrist_force", mujoco.mjtSensor.mjSENS_FORCE, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    _sensor("wrist_torque", mujoco.mjtSensor.mjSENS_TORQUE, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    return spec


def compile_clinic_model() -> Tuple[Any, Any]:
    spec = build_clinic_spec()
    model = spec.compile()
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[6] = 0.0  # garra aberta
    mujoco.mj_forward(model, data)
    return model, data


def visual_part_catalog(model: Any) -> List[Dict[str, Any]]:
    """Lista malhas visuais (group 2) para o frontend Three.js."""
    parts: List[Dict[str, Any]] = []
    for gi in range(model.ngeom):
        if int(model.geom_group[gi]) != 2:
            continue
        if int(model.geom_type[gi]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[gi])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh_{mid}"
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gi])) or ""
        mat_id = int(model.geom_matid[gi])
        rgba = (
            [round(float(v), 3) for v in model.mat_rgba[mat_id]]
            if mat_id >= 0
            else [round(float(v), 3) for v in model.geom_rgba[gi]]
        )
        is_gripper = mesh_name.startswith("gripper/")
        file_stem = mesh_name.split("/", 1)[-1]
        url = (MESH_URL["gripper"] if is_gripper else MESH_URL["ur"]).format(name=file_stem)
        parts.append(
            {
                "geom": gi,
                "mesh": mesh_name,
                "body": body_name,
                "url": url,
                "scale": 0.001 if is_gripper else 1.0,
                "rgba": rgba,
            }
        )
    return parts


def geom_pose_payload(model: Any, data: Any, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pose do *body* (não do geom compilado).

    O compilador do MuJoCo recentra/rotaciona a malha e grava isso em geom_pos/quat.
    O frontend carrega o OBJ/STL original (referencial do body no MJCF), então
    aplicar geom_xpos/xquat explode o braço. A pose do body reconstitui o conjunto.
    """
    poses: List[Dict[str, Any]] = []
    for part in parts:
        gi = part["geom"]
        bid = int(model.geom_bodyid[gi])
        pos = data.xpos[bid]
        quat = data.xquat[bid]
        poses.append(
            {
                "pos": [round(float(pos[0]), 5), round(float(pos[1]), 5), round(float(pos[2]), 5)],
                "quat": [round(float(quat[i]), 5) for i in range(4)],
            }
        )
    return poses
