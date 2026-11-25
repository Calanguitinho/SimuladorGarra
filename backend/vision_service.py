"""Visão sob demanda: somente comandos explícitos produzem imagens e inferência."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import io
import asyncio
import uuid
import logging
import threading
import time
import numpy as np

from backend.vision import YoloDetector
from backend.vision_camera import VisionCamera
from backend.vision_geometry import VisionGeometry


class VisionService:
    def __init__(self, env, config, detector_factory=YoloDetector, camera_factory=VisionCamera):
        self.env, self.config = env, config
        self.logger = getattr(env, "logger", logging.getLogger(__name__))
        self.detector_factory, self.camera_factory = detector_factory, camera_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._wake = threading.Event()
        self._request = None
        self._frame_seq = 0
        self._frames = OrderedDict()
        self._packet = {"type": "vision", "schema_version": 1,
                        "status": "disabled" if not config.enabled else "idle",
                        "error": None, "detections": [], "frame_id": None,
                        "scene_revision": env.scene_revision, "pick_enabled": False}
        self._captured = None

    def start(self):
        if self.config.enabled and self._thread is None:
            with self._lock:
                self._frames.clear()
                self._captured = None
                self._packet.update(status="idle", error=None, frame_id=None,
                                    scene_revision=self.env.scene_revision,
                                    detections=[], pick_enabled=False)
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="VisionWorker", daemon=True)
            self._thread.start()

    def stop(self):
        self.cancel_detection()
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join()  # called through asyncio.to_thread by the server
            self._thread = None
        with self._lock:
            self._frames.clear()
            self._captured = None
            self._packet.update(status="disabled", frame_id=None, detections=[],pick_enabled=False)

    def snapshot(self):
        with self._lock:
            packet = deepcopy(self._packet)
            captured = self._captured
            packet["request_active"] = self._request is not None
            packet["frames_captured"] = self._frame_seq
        packet["enabled"] = self.config.enabled
        packet["can_request_pick"] = self.config.enabled and self.config.enable_pick
        packet["allowed_classes"] = list(self.config.allowed_classes)
        age = (time.perf_counter() - captured) * 1000 if captured is not None else None
        packet["age_ms"] = round(age, 2) if age is not None else None
        packet["max_age_ms"] = self.config.max_age_ms
        if packet["scene_revision"] != self.env.scene_revision:
            packet.update(status="stale", detections=[], error="Cena alterada; solicite nova detecção.")
            if not packet["request_active"]:
                packet.update(status="idle" if self.config.enabled else "disabled",
                              scene_revision=self.env.scene_revision, frame_id=None,
                              error=None, pick_enabled=False)
        elif age is not None and age > self.config.max_age_ms:
            if packet["status"] == "ready":
                packet["status"] = "stale"
                packet["error"] = (
                    f"Observação expirada: idade {age:.0f} ms excede o limite de "
                    f"{self.config.max_age_ms:.0f} ms "
                    f"(processamento {packet.get('processing_ms', 0):.0f} ms; "
                    f"inferência {packet.get('inference_ms', 0):.0f} ms). "
                    "Reduza a carga de CPU ou ajuste --vision-max-age-ms para a bancada estática."
                )
            for detection in packet["detections"]:
                detection["grasp_candidate"]["valid"] = False
                detection["grasp_candidate"]["reason"] = "Observação expirada."
        return packet

    def frame(self, frame_id):
        with self._lock:
            cached = self._frames.get(frame_id)
            if cached and cached[0] == self.env.scene_revision:
                return cached[1]
        return None

    def resolve(self, revision, frame_id, detection_id):
        if self.snapshot()["status"] != "ready":
            raise ValueError("Serviço de visão indisponível ou sem observação recente.")
        with self._lock:
            cached = self._frames.get(frame_id) if isinstance(frame_id, int) else None
            packet = deepcopy(cached[2]) if cached else None
            captured = cached[3] if cached else None
        if packet is None or revision != self.env.scene_revision or revision != packet["scene_revision"]:
            raise ValueError("Quadro expirado ou cena alterada; selecione uma observação atual.")
        if (time.perf_counter() - captured) * 1000 > self.config.max_age_ms:
            raise ValueError("Observação expirada.")
        if not packet.get("pick_enabled") or packet["status"] != "ready":
            raise ValueError("Captura indisponível: confira pesos, calibração, idade e habilitação.")
        detection = next((d for d in packet["detections"] if d["detection_id"] == detection_id), None)
        if not detection or not detection["grasp_candidate"]["valid"]:
            raise ValueError("Candidato inválido ou instável.")
        return detection

    def request_detection(self, class_name):
        if not self.config.enabled or self._thread is None or not self._thread.is_alive():
            raise ValueError("Serviço de visão não está habilitado.")
        if class_name not in self.config.allowed_classes:
            raise ValueError("class_name deve identificar uma ferramenta configurada.")
        request = {"id": uuid.uuid4().hex, "class_name": class_name,
                   "scene_revision": self.env.scene_revision}
        with self._lock:
            if self._request is not None:
                raise ValueError("Já existe uma solicitação de visão em andamento.")
            self._request = request
            self._packet.update(status="detecting", error=None, detections=[],
                                request_id=request["id"], requested_class=class_name,
                                scene_revision=request["scene_revision"])
        self._wake.set()
        return request["id"]

    def cancel_detection(self, request_id=None):
        with self._lock:
            if self._request and (request_id is None or self._request["id"] == request_id):
                self._request = None
                self._packet.update(status="idle", error="Solicitação de visão cancelada.",
                                    detections=[], pick_enabled=False)
        self._wake.set()

    async def detect(self, class_name, timeout=60):
        request_id = self.request_detection(class_name)
        try:
            deadline = time.perf_counter() + timeout
            while time.perf_counter() < deadline:
                packet = self.snapshot()
                if packet.get("request_id") == request_id and not packet["request_active"]:
                    if packet["status"] != "ready":
                        raise ValueError(packet.get("error") or "Detecção indisponível ou expirada.")
                    return packet
                await asyncio.sleep(.02)
            raise ValueError("Timeout aguardando a solicitação de visão.")
        finally:
            self.cancel_detection(request_id)

    def _current(self, request):
        with self._lock:
            return not self._stop.is_set() and self._request is request

    def _run(self):
        resources = {}
        try:
            while not self._stop.is_set():
                self._wake.wait()
                self._wake.clear()
                with self._lock:
                    request = self._request
                if request is None or self._stop.is_set():
                    continue
                try:
                    self._execute_request(request, resources)
                except Exception as exc:
                    self.logger.warning("Solicitação de visão falhou: %s", exc)
                    with self._lock:
                        if self._request is request:
                            self._request = None
                            self._packet.update(status="error", error=str(exc), detections=[],
                                                pick_enabled=False)
        finally:
            if "camera" in resources:
                resources["camera"].close()

    def _execute_request(self, request, resources):
        # Inicialização tardia: repouso não abre câmera nem executa warmup do YOLO.
        if "camera" not in resources:
            resources["camera"] = self.camera_factory(self.env, self.config)
        if "geometry" not in resources:
            resources["geometry"] = VisionGeometry(self.config.calibration_path)
        if "detector" not in resources:
            resources["detector"] = self.detector_factory(self.config)
        camera, geometry, detector = (resources[k] for k in ("camera", "geometry", "detector"))
        previous = []
        maximum_frames = max(3, self.config.stable_frames * 2)
        for attempt in range(maximum_frames):
            if not self._current(request):
                return
            if request["scene_revision"] != self.env.scene_revision:
                raise ValueError("Cena alterada durante a solicitação de visão.")
            started = time.perf_counter()
            rgb, metadata, captured = camera.capture()
            self._frame_seq += 1
            frame_id = self._frame_seq
            revision = metadata["scene_revision"]
            inferred = time.perf_counter()
            detections = detector.detect(rgb)
            inference_ms = (time.perf_counter() - inferred) * 1000
            unmatched = list(previous)
            for index, detection in enumerate(detections):
                detection["detection_id"] = f"{frame_id}:{index}"
                candidate = geometry.candidate(detection, metadata) if geometry else {
                    "valid": False, "method": "plane", "position_world_m": None,
                    "reason": "Calibração indisponível."}
                point = candidate["position_world_m"]
                matches = [p for p in unmatched if p["class_name"] == detection["class_name"]
                           and point is not None and p["grasp_candidate"]["position_world_m"] is not None
                           and np.linalg.norm(np.array(point) - p["grasp_candidate"]["position_world_m"])
                           <= self.config.stability_m]
                match = min(matches, key=lambda p: np.linalg.norm(
                    np.array(point) - p["grasp_candidate"]["position_world_m"])) if matches else None
                detection["stable_frames"] = match["stable_frames"] + 1 if match else 1
                if match:
                    unmatched.remove(match)
                if detection["stable_frames"] < self.config.stable_frames:
                    candidate.update(valid=False, reason="Aguardando estabilidade espacial.")
                detection["grasp_candidate"] = candidate
            from PIL import Image
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="JPEG", quality=85)
            pick_enabled = bool(self.config.enable_pick and detector and geometry
                                and detector.manifest.get("validated_for_pick") is True
                                and geometry.calibration.get("validated_for_pick") is True)
            packet = {"type": "vision", "schema_version": 1,
                      "status": "ready" if detector and geometry else "unavailable",
                      "error": None, "frame_id": frame_id,
                      "scene_revision": revision, "camera_id": self.config.camera_name,
                      "calibration_id": geometry.identifier if geometry else None,
                      "model_id": detector.identifier if detector else None,
                      "sim_time_s": metadata["sim_time_s"],
                      "image_size": metadata["image_size"], "camera": metadata,
                      "inference_ms": round(inference_ms, 2),
                      "processing_ms": round((time.perf_counter() - captured) * 1000, 2),
                      "pick_enabled": pick_enabled, "detections": detections}
            fresh = (time.perf_counter() - captured) * 1000 <= self.config.max_age_ms
            eligible = fresh and any(d["class_name"] == request["class_name"] and
                                     d["grasp_candidate"]["valid"] for d in detections)
            finished = eligible or attempt == maximum_frames - 1
            packet.update(status="ready" if finished else "detecting",
                          request_id=request["id"], requested_class=request["class_name"])
            with self._lock:
                if self._request is not request or self._stop.is_set():
                    return
                if revision != self.env.scene_revision:
                    raise ValueError("Cena alterada durante a solicitação de visão.")
                self._packet, self._captured = packet, captured
                self._frames[frame_id] = (revision, buffer.getvalue(), deepcopy(packet), captured)
                while len(self._frames) > 4:
                    self._frames.popitem(last=False)
                if finished:
                    self._request = None
            if finished:
                self.logger.info(
                    "Detecção concluída: classe=%s quadro=%s inferência=%.1f ms "
                    "processamento=%.1f ms limite=%.1f ms elegível=%s",
                    request["class_name"], frame_id, inference_ms,
                    packet["processing_ms"], self.config.max_age_ms, eligible)
                return
            previous = detections
            self._stop.wait(max(0, 1 / self.config.target_fps - (time.perf_counter() - started)))
