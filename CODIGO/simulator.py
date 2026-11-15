"""Inicialização do simulador MuJoCo, modelo MJCF da garra, física de objetos e câmera RGB.

Opera de forma transparente em dois motores:
    1. **Motor MuJoCo Nativo (`mujoco`)**: quando o pacote `mujoco` está instalado,
       compila o modelo MJCF XML completo, integra a dinâmica via `mj_step` e utiliza
       renderização nativa (com fallback seguro caso não haja contexto OpenGL headless).
    2. **Motor Físico Cinemático/Dinâmico Integrado**: fallback determinístico de alta
       fidelidade que simula espaço de trabalho 3D, gravidade na bancada, captura por
       contato dos dedos paralelos e câmera sintética RGB (`Pillow`) com modelos físicos
       de neblina atmosférica (Beer-Lambert) e sujeira/ruído óptico na lente e superfície.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from CODIGO.config import GripperSpecs, SimulationConfig
from FUNCOES.utils import clamp, setup_logger

# Configura backend headless antes de tentar importar mujoco em servidores Linux sem display
if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "osmesa"

try:
    import mujoco  # type: ignore[import-not-found]

    HAS_MUJOCO = True
except Exception:  # noqa: BLE001
    mujoco = None  # type: ignore[assignment]
    HAS_MUJOCO = False

try:
    from PIL import Image, ImageDraw, ImageFilter

    HAS_PIL = True
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
    HAS_PIL = False


MUJOCO_GRIPPER_XML = """
<mujoco model="parallel_gripper_workcell">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>

  <visual>
    <map fogstart="0.6" fogend="3.5"/>
    <rgba fog="0.82 0.85 0.88 1.0"/>
  </visual>

  <worldbody>
    <light directional="true" diffuse="0.85 0.85 0.85" pos="0 0 1.8" dir="0 0 -1"/>
    <camera name="overhead_cam" pos="0 -0.55 0.85" xyaxes="1 0 0 0 0.75 0.66"/>

    <!-- Bancada de trabalho -->
    <geom name="table_top" type="plane" pos="0 0 0" size="0.6 0.6 0.05" rgba="0.22 0.25 0.29 1"/>
    <site name="drop_zone" pos="-0.22 0.18 0.002" size="0.06 0.06 0.002" type="box" rgba="0.15 0.72 0.35 0.45"/>

    <!-- Pórtico Cartesiano 3-DOF + Garra Paralela -->
    <body name="gripper_carriage" pos="0 0 0.30">
      <joint name="slide_x" type="slide" axis="1 0 0" range="-0.45 0.45" damping="18"/>
      <joint name="slide_y" type="slide" axis="0 1 0" range="-0.45 0.45" damping="18"/>
      <joint name="slide_z" type="slide" axis="0 0 1" range="-0.28 0.25" damping="22"/>
      <geom name="palm" type="box" size="0.045 0.025 0.02" rgba="0.18 0.48 0.85 1"/>

      <body name="left_finger_body" pos="-0.04 0 -0.045">
        <joint name="finger_left" type="slide" axis="1 0 0" range="0.0 0.04" damping="6"/>
        <geom name="left_pad" type="box" size="0.008 0.018 0.032" rgba="0.88 0.58 0.12 1" friction="1.4 0.02 0.002"/>
      </body>

      <body name="right_finger_body" pos="0.04 0 -0.045">
        <joint name="finger_right" type="slide" axis="-1 0 0" range="0.0 0.04" damping="6"/>
        <geom name="right_pad" type="box" size="0.008 0.018 0.032" rgba="0.88 0.58 0.12 1" friction="1.4 0.02 0.002"/>
      </body>
    </body>

    <!-- Objetos manipuláveis -->
    <body name="red_cube" pos="0.18 0.12 0.025">
      <freejoint name="red_cube_free"/>
      <geom name="red_cube_geom" type="box" size="0.024 0.024 0.024" mass="0.08" rgba="0.86 0.20 0.22 1" friction="1.2 0.01 0.001"/>
    </body>

    <body name="blue_cylinder" pos="-0.15 -0.14 0.03">
      <freejoint name="blue_cyl_free"/>
      <geom name="blue_cyl_geom" type="cylinder" size="0.022 0.028" mass="0.09" rgba="0.20 0.52 0.90 1" friction="1.2 0.01 0.001"/>
    </body>
  </worldbody>

  <actuator>
    <position name="act_x" joint="slide_x" kp="220"/>
    <position name="act_y" joint="slide_y" kp="220"/>
    <position name="act_z" joint="slide_z" kp="260"/>
    <position name="act_finger_l" joint="finger_left" kp="90"/>
    <position name="act_finger_r" joint="finger_right" kp="90"/>
  </actuator>
</mujoco>
""".strip()


@dataclass
class SimObject:
    """Representa um objeto físico sobre a bancada do simulador."""

    name: str
    shape: str
    color_rgb: Tuple[int, int, int]
    radius: float
    height: float
    position: Dict[str, float] = field(default_factory=dict)
    initial_position: Dict[str, float] = field(default_factory=dict)


class MuJoCoSimulator:
    """Ambiente de simulação robótica MuJoCo compatível com `GripperBackendProtocol`."""

    def __init__(
        self,
        sim_config: Optional[SimulationConfig] = None,
        gripper_specs: Optional[GripperSpecs] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.sim_config = sim_config or SimulationConfig()
        self.specs = gripper_specs or GripperSpecs()
        self.logger = logger or setup_logger("MuJoCoSimulator")

        self.fog_density: float = self.sim_config.default_fog_density
        self.dirt_intensity: float = self.sim_config.default_dirt_intensity
        self.dirt_seed: int = 42
        self.step_counter: int = 0

        # Estado cinemático/dinâmico interno sincronizado
        self._pos: Dict[str, float] = {
            "x": self.specs.home_position[0],
            "y": self.specs.home_position[1],
            "z": self.specs.home_position[2],
        }
        self._orientation: Dict[str, float] = {
            "roll": self.specs.home_orientation[0],
            "pitch": self.specs.home_orientation[1],
            "yaw": self.specs.home_orientation[2],
        }
        self._aperture: float = self.specs.max_aperture
        self._grip_force: float = 0.0
        self._is_closed: bool = False
        self._grasped_object: Optional[str] = None
        self.drop_zone: Dict[str, float] = {"x": -0.22, "y": 0.18, "z": 0.025}

        self._objects: Dict[str, SimObject] = {}
        self._mj_model: Any = None
        self._mj_data: Any = None
        self.engine_name: str = "kinematic-physics-fallback"

        self._initialize_scene()

    def _initialize_scene(self) -> None:
        """Inicializa os objetos manipuláveis e compila o modelo MJCF se `mujoco` existir."""
        self._objects = {
            "red_cube": SimObject(
                name="red_cube",
                shape="box",
                color_rgb=(220, 52, 56),
                radius=0.025,
                height=0.048,
                position={"x": 0.18, "y": 0.12, "z": 0.025},
                initial_position={"x": 0.18, "y": 0.12, "z": 0.025},
            ),
            "blue_cylinder": SimObject(
                name="blue_cylinder",
                shape="cylinder",
                color_rgb=(50, 132, 230),
                radius=0.022,
                height=0.056,
                position={"x": -0.15, "y": -0.14, "z": 0.028},
                initial_position={"x": -0.15, "y": -0.14, "z": 0.028},
            ),
        }

        if HAS_MUJOCO:
            try:
                self._mj_model = mujoco.MjModel.from_xml_string(MUJOCO_GRIPPER_XML)
                self._mj_data = mujoco.MjData(self._mj_model)
                mujoco.mj_forward(self._mj_model, self._mj_data)
                self.engine_name = f"mujoco-native-{mujoco.__version__}"
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Falha ao compilar MJCF nativo (%s); usando motor físico integrado.",
                    exc,
                )
                self._mj_model = None
                self._mj_data = None
                self.engine_name = "kinematic-physics-fallback"

        self.logger.info(
            "Simulador inicializado [motor=%s | headless=%s | resolução=%dx%d].",
            self.engine_name,
            self.sim_config.headless,
            self.sim_config.camera_width,
            self.sim_config.camera_height,
        )

    def reset(self) -> Dict[str, Any]:
        """Reinicia completamente a cena, objetos, neblina, sujeira e posição da garra."""
        self._pos = {
            "x": self.specs.home_position[0],
            "y": self.specs.home_position[1],
            "z": self.specs.home_position[2],
        }
        self._aperture = self.specs.max_aperture
        self._grip_force = 0.0
        self._is_closed = False
        self._grasped_object = None
        self.fog_density = self.sim_config.default_fog_density
        self.dirt_intensity = self.sim_config.default_dirt_intensity
        self.step_counter = 0

        for obj in self._objects.values():
            obj.position = dict(obj.initial_position)

        if self._mj_model is not None and self._mj_data is not None:
            mujoco.mj_resetData(self._mj_model, self._mj_data)
            mujoco.mj_forward(self._mj_model, self._mj_data)

        return self.sync_state()

    # ------------------------------------------------------------------------
    # Implementação do GripperBackendProtocol
    # ------------------------------------------------------------------------

    def sync_state(self) -> Dict[str, Any]:
        """Retorna telemetria instantânea completa do simulador."""
        return {
            "position": dict(self._pos),
            "orientation": dict(self._orientation),
            "grip_force": self._grip_force,
            "aperture": self._aperture,
            "is_closed": self._is_closed,
            "is_grasping": self._grasped_object is not None,
            "grasped_object": self._grasped_object,
        }

    def apply_cartesian_delta(
        self,
        dx: float,
        dy: float,
        dz: float,
        speed: float,
        force: float,
        duration: float,
    ) -> Dict[str, Any]:
        """Integra o movimento cartesiano no simulador e transporta objetos capturados."""
        num_steps = max(1, min(250, int(duration / self.sim_config.timestep)))
        step_dx = dx / num_steps
        step_dy = dy / num_steps
        step_dz = dz / num_steps

        for _ in range(num_steps):
            nx, _ = clamp(self._pos["x"] + step_dx, *self.specs.workspace_x)
            ny, _ = clamp(self._pos["y"] + step_dy, *self.specs.workspace_y)
            nz, _ = clamp(self._pos["z"] + step_dz, *self.specs.workspace_z)
            self._pos = {"x": nx, "y": ny, "z": nz}

            if self._grasped_object and self._grasped_object in self._objects:
                carried = self._objects[self._grasped_object]
                carried.position = {
                    "x": round(nx, 5),
                    "y": round(ny, 5),
                    "z": round(max(carried.height * 0.5, nz - 0.02), 5),
                }
            self.step_counter += 1

        self._step_native_mujoco()
        return self.sync_state()

    def apply_jaw_action(
        self,
        close_jaws: bool,
        speed: float,
        force: float,
        duration: float,
    ) -> Dict[str, Any]:
        """Executa fechamento (com detecção de contato/captura) ou abertura (com gravidade)."""
        steps = max(1, min(120, int(duration / self.sim_config.timestep)))
        self.step_counter += steps

        if close_jaws:
            self._is_closed = True
            self._grip_force = force
            captured = self._detect_graspable_object()
            if captured is not None:
                self._grasped_object = captured.name
                self._aperture = round(captured.radius * 1.8, 5)
                self.logger.info(
                    "Objeto '%s' capturado em %s com força %.1fN!",
                    captured.name,
                    captured.position,
                    force,
                )
            else:
                self._aperture = self.specs.min_aperture
        else:
            released_name = self._grasped_object
            self._is_closed = False
            self._grip_force = 0.0
            self._aperture = self.specs.max_aperture
            if released_name and released_name in self._objects:
                obj = self._objects[released_name]
                # Aplica gravidade pousando suavemente sobre a bancada (z = height/2)
                obj.position["z"] = round(obj.height * 0.5, 5)
                self.logger.info(
                    "Objeto '%s' liberado sobre a bancada em %s.",
                    released_name,
                    obj.position,
                )
            self._grasped_object = None

        self._step_native_mujoco()
        return self.sync_state()

    def _detect_graspable_object(self) -> Optional[SimObject]:
        """Verifica se há algum objeto dentro da tolerância tridimensional dos dedos."""
        best_obj: Optional[SimObject] = None
        best_dist = float("inf")

        for obj in self._objects.values():
            planar_dist = math.hypot(
                self._pos["x"] - obj.position["x"],
                self._pos["y"] - obj.position["y"],
            )
            vertical_gap = abs(self._pos["z"] - obj.position["z"])
            if (
                planar_dist <= self.specs.grasp_radius_tolerance
                and vertical_gap <= 0.075
                and planar_dist < best_dist
            ):
                best_dist = planar_dist
                best_obj = obj
        return best_obj

    def _step_native_mujoco(self) -> None:
        """Sincroniza os atuadores do modelo MuJoCo nativo quando disponível."""
        if self._mj_model is None or self._mj_data is None:
            return
        try:
            self._mj_data.ctrl[0] = self._pos["x"]
            self._mj_data.ctrl[1] = self._pos["y"]
            self._mj_data.ctrl[2] = self._pos["z"] - self.specs.home_position[2]
            finger_travel = (self.specs.max_aperture - self._aperture) * 0.5
            self._mj_data.ctrl[3] = finger_travel
            self._mj_data.ctrl[4] = finger_travel
            for _ in range(10):
                mujoco.mj_step(self._mj_model, self._mj_data)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Aviso ao avançar passo MuJoCo nativo: %s", exc)

    # ------------------------------------------------------------------------
    # Manipulação de Objetos e Perturbações Ambientais (Neblina & Sujeira)
    # ------------------------------------------------------------------------

    def set_fog(self, density: float) -> float:
        """Ajusta a densidade de neblina (`0.0` sem neblina até `1.0` neblina severa)."""
        self.fog_density, _ = clamp(float(density), 0.0, 1.0)
        if self._mj_model is not None:
            # Aproxima o plano de neblina conforme a densidade aumenta
            self._mj_model.vis.map.fogstart = max(0.15, 1.2 * (1.0 - 0.8 * self.fog_density))
            self._mj_model.vis.map.fogend = max(0.45, 3.8 * (1.0 - 0.75 * self.fog_density))
        self.logger.info("Densidade de neblina ajustada para %.0f%%.", self.fog_density * 100)
        return self.fog_density

    def set_surface_dirt(self, intensity: float, seed: int = 42) -> float:
        """Ajusta o nível de sujeira/ruído visual na superfície e câmera (`0.0` a `1.0`)."""
        self.dirt_intensity, _ = clamp(float(intensity), 0.0, 1.0)
        self.dirt_seed = int(seed)
        self.logger.info(
            "Intensidade de sujeira/ruído visual ajustada para %.0f%% (seed=%d).",
            self.dirt_intensity * 100,
            self.dirt_seed,
        )
        return self.dirt_intensity

    def set_object_position(
        self, name: str, x: float, y: float, z: Optional[float] = None
    ) -> Dict[str, float]:
        """Reposiciona um objeto da cena para configuração de cenários de teste."""
        if name not in self._objects:
            raise KeyError(f"Objeto '{name}' inexistente no simulador.")
        obj = self._objects[name]
        obj.position = {
            "x": round(float(x), 5),
            "y": round(float(y), 5),
            "z": round(float(z if z is not None else obj.height * 0.5), 5),
        }
        return dict(obj.position)

    def get_object_position(self, name: str) -> Dict[str, float]:
        """Retorna a posição 3D atual de um objeto manipulável."""
        if name not in self._objects:
            raise KeyError(f"Objeto '{name}' inexistente no simulador.")
        return dict(self._objects[name].position)

    def list_objects(self) -> Dict[str, Dict[str, Any]]:
        """Lista todos os objetos presentes na bancada e seus metadados."""
        return {
            name: {
                "shape": obj.shape,
                "position": dict(obj.position),
                "radius": obj.radius,
            }
            for name, obj in self._objects.items()
        }

    # ------------------------------------------------------------------------
    # Renderização de Quadros RGB e Métricas de Qualidade Visual
    # ------------------------------------------------------------------------

    def render_frame(self, apply_clean_reference: bool = True) -> Dict[str, Any]:
        """Renderiza a visão aérea da bancada aplicando neblina e sujeira configuradas.

        Retorna imagem `PIL.Image` (se disponível), métricas ópticas (`contrast_rms`,
        `mean_luminance`, `psnr_db`) e metadados da cena.
        """
        width = self.sim_config.camera_width
        height = self.sim_config.camera_height
        clean_img = self._render_synthetic_scene(width, height)
        degraded_img = clean_img.copy() if clean_img is not None else None

        if degraded_img is not None:
            if self.dirt_intensity > 0.0:
                degraded_img = self._apply_dirt_and_noise(
                    degraded_img, self.dirt_intensity, self.dirt_seed
                )
            if self.fog_density > 0.0:
                degraded_img = self._apply_atmospheric_fog(degraded_img, self.fog_density)

        metrics = self._compute_frame_metrics(
            clean_img if apply_clean_reference else None, degraded_img
        )

        return {
            "width": width,
            "height": height,
            "fog_density": self.fog_density,
            "dirt_intensity": self.dirt_intensity,
            "mean_luminance": metrics["mean_luminance"],
            "contrast_rms": metrics["contrast_rms"],
            "psnr_db": metrics["psnr_db"],
            "pil_image": degraded_img,
            "gripper_position": dict(self._pos),
            "scene_objects": self.list_objects(),
        }

    def save_frame(self, output_path: Path) -> Optional[Path]:
        """Renderiza o estado atual e salva como imagem PNG em disco."""
        frame = self.render_frame()
        pil_img = frame.get("pil_image")
        if pil_img is None:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(output_path, format="PNG")
        self.logger.info("Quadro da simulação salvo em %s", output_path)
        return output_path

    def _world_to_pixel(
        self, x: float, y: float, width: int, height: int
    ) -> Tuple[int, int]:
        """Projeta coordenadas da bancada `[-0.5, 0.5]m` em pixels da câmera aérea."""
        px = int(((x + 0.50) / 1.00) * width)
        py = int(((0.50 - y) / 1.00) * height)
        return max(0, min(width - 1, px)), max(0, min(height - 1, py))

    def _render_synthetic_scene(self, width: int, height: int) -> Any:
        """Gera quadro RGB limpo da bancada, zona de entrega, objetos e garra."""
        if not HAS_PIL:
            return None

        img = Image.new("RGB", (width, height), color=(48, 54, 62))
        draw = ImageDraw.Draw(img)

        # Grade métrica da bancada
        for step in range(-4, 5):
            coord = step * 0.10
            x0, y0 = self._world_to_pixel(coord, -0.48, width, height)
            x1, y1 = self._world_to_pixel(coord, 0.48, width, height)
            draw.line([(x0, y0), (x1, y1)], fill=(64, 72, 82), width=1)
            gx0, gy0 = self._world_to_pixel(-0.48, coord, width, height)
            gx1, gy1 = self._world_to_pixel(0.48, coord, width, height)
            draw.line([(gx0, gy0), (gx1, gy1)], fill=(64, 72, 82), width=1)

        # Zona de destino (drop_zone)
        dz_x, dz_y = self._world_to_pixel(
            self.drop_zone["x"], self.drop_zone["y"], width, height
        )
        dz_half = int(0.065 * width)
        draw.rectangle(
            [dz_x - dz_half, dz_y - dz_half, dz_x + dz_half, dz_y + dz_half],
            outline=(58, 205, 105),
            fill=(38, 92, 60),
            width=2,
        )

        # Objetos manipuláveis com sombra proporcional à altura Z
        for obj in self._objects.values():
            ox, oy = self._world_to_pixel(
                obj.position["x"], obj.position["y"], width, height
            )
            r_px = max(6, int(obj.radius * width * 1.15))
            shadow_offset = max(2, int(obj.position["z"] * 35))
            draw.ellipse(
                [
                    ox - r_px + shadow_offset,
                    oy - r_px + shadow_offset,
                    ox + r_px + shadow_offset,
                    oy + r_px + shadow_offset,
                ],
                fill=(26, 30, 35),
            )
            if obj.shape == "box":
                draw.rectangle(
                    [ox - r_px, oy - r_px, ox + r_px, oy + r_px],
                    fill=obj.color_rgb,
                    outline=(245, 245, 245),
                    width=1,
                )
            else:
                draw.ellipse(
                    [ox - r_px, oy - r_px, ox + r_px, oy + r_px],
                    fill=obj.color_rgb,
                    outline=(245, 245, 245),
                    width=1,
                )

        # Garra paralela (escala visual muda levemente com a altura Z)
        gx, gy = self._world_to_pixel(self._pos["x"], self._pos["y"], width, height)
        scale = 1.0 + 0.55 * self._pos["z"]
        jaw_gap_px = max(3, int((self._aperture / self.specs.max_aperture) * 16 * scale))
        palm_w = int(18 * scale)
        palm_h = int(8 * scale)

        # Dedos esquerdo e direito
        finger_color = (240, 168, 36) if not self._is_closed else (255, 115, 30)
        draw.rectangle(
            [gx - jaw_gap_px - 5, gy - 12, gx - jaw_gap_px, gy + 12],
            fill=finger_color,
            outline=(20, 20, 20),
        )
        draw.rectangle(
            [gx + jaw_gap_px, gy - 12, gx + jaw_gap_px + 5, gy + 12],
            fill=finger_color,
            outline=(20, 20, 20),
        )
        # Palma central
        draw.rectangle(
            [gx - palm_w, gy - palm_h, gx + palm_w, gy + palm_h],
            fill=(52, 136, 235),
            outline=(230, 240, 255),
            width=1,
        )
        return img

    @staticmethod
    def _apply_atmospheric_fog(img: Any, density: float) -> Any:
        """Simula espalhamento atmosférico (Beer-Lambert) + perda de nitidez óptica."""
        fog_layer = Image.new("RGB", img.size, color=(212, 218, 224))
        transmission = math.exp(-2.1 * density)
        alpha = max(0.0, min(0.94, 1.0 - transmission))
        blended = Image.blend(img, fog_layer, alpha=alpha)
        blur_radius = round(2.8 * density, 2)
        if blur_radius > 0.15:
            blended = blended.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return blended

    @staticmethod
    def _apply_dirt_and_noise(img: Any, intensity: float, seed: int) -> Any:
        """Adiciona manchas de óleo/poeira na superfície e ruído impulsivo na lente."""
        rng = random.Random(seed)
        noisy = img.copy()
        draw = ImageDraw.Draw(noisy)
        width, height = noisy.size

        # Manchas escuras/barrentas de sujeira
        num_blotches = int(28 * intensity)
        for _ in range(num_blotches):
            bx = rng.randint(0, width - 1)
            by = rng.randint(0, height - 1)
            rx = rng.randint(4, max(5, int(22 * intensity)))
            ry = rng.randint(4, max(5, int(18 * intensity)))
            shade = rng.randint(58, 118)
            draw.ellipse(
                [bx - rx, by - ry, bx + rx, by + ry],
                fill=(shade, max(45, shade - 12), max(35, shade - 24)),
            )

        # Ruído granular (salt-and-pepper + poeira)
        num_specks = int(width * height * 0.09 * intensity)
        pixels = noisy.load()
        if pixels is not None:
            for _ in range(num_specks):
                px = rng.randint(0, width - 1)
                py = rng.randint(0, height - 1)
                if rng.random() < 0.5:
                    val = rng.randint(185, 245)
                else:
                    val = rng.randint(10, 45)
                pixels[px, py] = (val, val, val)

        return noisy

    @staticmethod
    def _compute_frame_metrics(clean_img: Any, current_img: Any) -> Dict[str, float]:
        """Calcula luminância média, contraste RMS normalizado e PSNR em decibéis."""
        if current_img is None or not HAS_PIL:
            return {"mean_luminance": 128.0, "contrast_rms": 0.25, "psnr_db": 48.0}

        gray = current_img.convert("L")
        raw_bytes = list(gray.tobytes())
        n = max(1, len(raw_bytes))
        mean_lum = sum(raw_bytes) / n
        variance = sum((p - mean_lum) ** 2 for p in raw_bytes) / n
        contrast_rms = math.sqrt(variance) / 255.0

        psnr_db = 60.0
        if clean_img is not None:
            c_bytes = clean_img.tobytes()
            d_bytes = current_img.tobytes()
            mse = sum((a - b) ** 2 for a, b in zip(c_bytes, d_bytes)) / max(1, len(c_bytes))
            if mse > 1e-9:
                psnr_db = 10.0 * math.log10((255.0**2) / mse)

        return {
            "mean_luminance": round(mean_lum, 3),
            "contrast_rms": round(contrast_rms, 4),
            "psnr_db": round(psnr_db, 2),
        }
