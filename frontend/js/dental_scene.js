/**
 * dental_scene.js — Sala odontológica (Three.js, Z-up)
 *
 * GLB Sketchfab são Y-up: converte para Z-up antes de assentar no chão.
 * A bancada fica a COUNTER_Z; a física MuJoCo (z≈0) é desenhada sobre ela.
 * A garra visual é a Robotiq 2F-85 de assets/Garra (poses gravadas no MuJoCo).
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const COUNTER_Z = 0.86;

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpVec3(out, a, b, t) {
  out[0] = a[0] + (b[0] - a[0]) * t;
  out[1] = a[1] + (b[1] - a[1]) * t;
  out[2] = a[2] + (b[2] - a[2]) * t;
  return out;
}

function slerpQuat(out, qa, qb, t) {
  let aw = qa[0], ax = qa[1], ay = qa[2], az = qa[3];
  let bw = qb[0], bx = qb[1], by = qb[2], bz = qb[3];
  let cosHalf = aw * bw + ax * bx + ay * by + az * bz;
  if (cosHalf < 0) {
    bw = -bw; bx = -bx; by = -by; bz = -bz;
    cosHalf = -cosHalf;
  }
  if (cosHalf > 0.9995) {
    out[0] = aw + t * (bw - aw);
    out[1] = ax + t * (bx - ax);
    out[2] = ay + t * (by - ay);
    out[3] = az + t * (bz - az);
  } else {
    const half = Math.acos(Math.min(1, cosHalf));
    const sinHalf = Math.sin(half);
    const ra = Math.sin((1 - t) * half) / sinHalf;
    const rb = Math.sin(t * half) / sinHalf;
    out[0] = aw * ra + bw * rb;
    out[1] = ax * ra + bx * rb;
    out[2] = ay * ra + by * rb;
    out[3] = az * ra + bz * rb;
  }
  const len = Math.hypot(out[0], out[1], out[2], out[3]) || 1;
  out[0] /= len; out[1] /= len; out[2] /= len; out[3] /= len;
  return out;
}

function detectPerfTier() {
  const cores = navigator.hardwareConcurrency || 4;
  const mem = navigator.deviceMemory || 4;
  const mobile = window.matchMedia("(max-width: 768px)").matches;
  if (mobile || cores <= 4 || mem <= 4) return "low";
  return "high";
}

function makeMat(color, extra = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.72, metalness: 0.04, ...extra });
}

function buildClinicRoom(scene, perfTier) {
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(6.4, 5.2),
    makeMat(0xd5dbd8, { roughness: 0.9 })
  );
  floor.receiveShadow = true;
  scene.add(floor);

  const tile = new THREE.GridHelper(6.4, 16, 0xb7c2bc, 0xc5cec8);
  tile.rotation.x = Math.PI / 2;
  tile.position.z = 0.004;
  tile.material.transparent = true;
  tile.material.opacity = 0.35;
  scene.add(tile);

  const wallMat = makeMat(0xe7eef2, { roughness: 0.92 });
  const wallH = 2.55;
  const back = new THREE.Mesh(new THREE.BoxGeometry(6.4, 0.08, wallH), wallMat);
  back.position.set(0, 2.56, wallH / 2);
  back.receiveShadow = true;
  scene.add(back);
  const left = new THREE.Mesh(new THREE.BoxGeometry(0.08, 5.2, wallH), wallMat);
  left.position.set(-3.16, 0, wallH / 2);
  scene.add(left);
  const right = new THREE.Mesh(new THREE.BoxGeometry(0.08, 5.2, wallH), wallMat);
  right.position.set(3.16, 0, wallH / 2);
  scene.add(right);

  const stripe = new THREE.Mesh(
    new THREE.BoxGeometry(6.24, 0.02, 0.18),
    makeMat(0x7ec8c3, { roughness: 0.45, emissive: 0x1a4a48, emissiveIntensity: 0.25 })
  );
  stripe.position.set(0, 2.51, 1.35);
  scene.add(stripe);

  const ceiling = new THREE.Mesh(
    new THREE.PlaneGeometry(6.4, 5.2),
    makeMat(0xf3f6f8, { roughness: 1 })
  );
  ceiling.position.z = wallH;
  scene.add(ceiling);

  const panel = new THREE.Mesh(
    new THREE.BoxGeometry(1.4, 0.45, 0.06),
    new THREE.MeshStandardMaterial({
      color: 0xf7fbff,
      emissive: 0xd6e6f2,
      emissiveIntensity: 0.85,
      roughness: 0.4,
    })
  );
  panel.position.set(0.1, 0.15, wallH - 0.05);
  scene.add(panel);

  // Bancada: o plano z=0 do MuJoCo vira o tampo
  const counter = new THREE.Mesh(
    new THREE.BoxGeometry(2.15, 1.55, 0.06),
    makeMat(0xf7f4ef, { roughness: 0.42, metalness: 0.08 })
  );
  counter.position.set(0.15, 0.05, COUNTER_Z - 0.03);
  counter.receiveShadow = true;
  counter.castShadow = perfTier === "high";
  scene.add(counter);

  const cabinetMat = makeMat(0xc5d0d6, { roughness: 0.6 });
  const cabinet = new THREE.Mesh(new THREE.BoxGeometry(2.05, 1.42, COUNTER_Z - 0.08), cabinetMat);
  cabinet.position.set(0.15, 0.05, (COUNTER_Z - 0.08) / 2);
  cabinet.castShadow = true;
  scene.add(cabinet);

  const edge = new THREE.Mesh(
    new THREE.BoxGeometry(2.15, 0.02, 0.08),
    makeMat(0x9aa7b0, { metalness: 0.55, roughness: 0.3 })
  );
  edge.position.set(0.15, -0.72, COUNTER_Z);
  scene.add(edge);

  const drop = new THREE.Mesh(
    new THREE.CircleGeometry(0.07, 40),
    new THREE.MeshStandardMaterial({
      color: 0x1f9d6a,
      transparent: true,
      opacity: 0.55,
      emissive: 0x065f46,
      emissiveIntensity: 0.35,
    })
  );
  drop.position.set(-0.30, 0.26, COUNTER_Z + 0.002);
  scene.add(drop);
}

/**
 * GLB Y-up → Z-up, escala e assenta min.z = 0 no wrapper.
 * lieFlat: o eixo mais fino fica vertical (bandeja / instrumentos deitados).
 */
function layoutProp(root, prop) {
  const orient = new THREE.Group();
  const isStl = (prop.url || "").toLowerCase().endsWith(".stl");
  if (!isStl && prop.fit) {
    orient.rotation.x = -Math.PI / 2;
  }
  orient.add(root);

  const holder = new THREE.Group();
  holder.add(orient);
  holder.updateMatrixWorld(true);

  if (prop.fit === "lieFlat") {
    const size = new THREE.Box3().setFromObject(holder).getSize(new THREE.Vector3());
    const dims = [size.x, size.y, size.z];
    const minI = dims.indexOf(Math.min(...dims));
    if (minI === 0) holder.rotation.y = Math.PI / 2;
    else if (minI === 1) holder.rotation.x = Math.PI / 2;
    holder.updateMatrixWorld(true);
  }

  if (!prop.scale) {
    const size = new THREE.Box3().setFromObject(holder).getSize(new THREE.Vector3());
    const current = prop.targetHeightM ? Math.max(size.z, 1e-6) : Math.max(size.x, size.y, size.z, 1e-6);
    const desired = prop.targetHeightM || prop.targetSizeM || current;
    orient.scale.setScalar(desired / current);
    holder.updateMatrixWorld(true);
  } else {
    root.scale.setScalar(prop.scale);
    holder.updateMatrixWorld(true);
  }

  if (prop.align === "x") {
    const size = new THREE.Box3().setFromObject(holder).getSize(new THREE.Vector3());
    if (size.y > size.x) {
      const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
      holder.quaternion.premultiply(yaw);
      holder.updateMatrixWorld(true);
    }
  } else if (prop.align === "y") {
    const size = new THREE.Box3().setFromObject(holder).getSize(new THREE.Vector3());
    if (size.x > size.y) {
      const yaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
      holder.quaternion.premultiply(yaw);
      holder.updateMatrixWorld(true);
    }
  }

  if (prop.flip) {
    // 180° no eixo X do mundo: a concavidade fica para cima
    // (rotateX local falha depois do lieFlat, que já girou o eixo)
    const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
    holder.quaternion.premultiply(turn);
    holder.updateMatrixWorld(true);
  }

  const box = new THREE.Box3().setFromObject(holder);
  const center = box.getCenter(new THREE.Vector3());
  holder.position.x -= center.x;
  holder.position.y -= center.y;
  holder.position.z -= box.min.z;

  const wrapper = new THREE.Group();
  wrapper.name = prop.id || "prop";
  wrapper.add(holder);
  if (prop.yawDeg) wrapper.rotation.z = THREE.MathUtils.degToRad(prop.yawDeg);
  if (prop.position) {
    wrapper.position.set(prop.position[0], prop.position[1], prop.position[2] || 0);
  }
  return wrapper;
}

export async function createDentalClinicRenderer(canvas, options = {}) {
  const onFpsUpdate = options.onFpsUpdate || function () {};
  const onLog = options.onLog || function () {};
  const perfTier = detectPerfTier();

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: perfTier === "high",
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0xd7e2e6, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = perfTier === "high";
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, perfTier === "low" ? 1 : 1.5));

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xd7e2e6, 7.5, 16);
  scene.background = new THREE.Color(0xd7e2e6);

  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 40);
  camera.up.set(0, 0, 1);
  camera.position.set(1.85, -2.15, 1.85);

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0.05, 0.12, COUNTER_Z + 0.12);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 0.45;
  controls.maxDistance = 6.5;
  controls.update();

  scene.add(new THREE.HemisphereLight(0xf4f8fb, 0xb7c4bc, 0.85));
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const key = new THREE.DirectionalLight(0xfff6ea, 1.05);
  key.position.set(1.6, -2.2, 4.2);
  key.castShadow = perfTier === "high";
  if (key.castShadow) {
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.4;
    key.shadow.camera.far = 12;
    key.shadow.camera.left = -3;
    key.shadow.camera.right = 3;
    key.shadow.camera.top = 3;
    key.shadow.camera.bottom = -3;
  }
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xc5e4f5, 0.45);
  fill.position.set(-2.2, 1.4, 2.4);
  scene.add(fill);

  buildClinicRoom(scene, perfTier);

  // Marcador luminoso sutil da projeção vertical do TCP na bancada (auxilia preensão manual)
  const tcpRing = new THREE.Mesh(
    new THREE.RingGeometry(0.013, 0.018, 32),
    new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide,
    })
  );
  tcpRing.position.set(0, 0, COUNTER_Z + 0.0045);
  scene.add(tcpRing);

  const robotRoot = new THREE.Group();
  scene.add(robotRoot);

  const smooth = {
    pos: [0, 0, 0.32],
    quat: [1, 0, 0, 0],
    aperture: 0.08,
    contact_force: 0,
    is_grasping: false,
    objects: new Map(),
    robot: [],
  };
  let targetState = null;
  const syncMeshes = new Map();
  let robotMeshes = [];

  const gltfLoader = new GLTFLoader();
  const stlLoader = new STLLoader();
  const objLoader = new OBJLoader();
  const geomCache = new Map();

  function colorFromRgba(rgba) {
    const [r, g, b] = rgba || [0.7, 0.7, 0.7, 1];
    return new THREE.Color(r, g, b);
  }

  async function loadRobotPart(part) {
    const url = part.url;
    const scale = part.scale || 1;
    const mat = new THREE.MeshStandardMaterial({
      color: colorFromRgba(part.rgba),
      metalness: 0.45,
      roughness: 0.38,
    });
    if (url.toLowerCase().endsWith(".stl")) {
      const geom = await stlLoader.loadAsync(url);
      geom.computeVertexNormals();
      geom.scale(scale, scale, scale);
      const mesh = new THREE.Mesh(geom, mat);
      mesh.castShadow = perfTier === "high";
      return mesh;
    }
    const obj = await objLoader.loadAsync(url);
    const wrap = new THREE.Group();
    obj.traverse((n) => {
      if (n.isMesh) {
        n.material = mat;
        n.castShadow = perfTier === "high";
        if (scale !== 1) n.geometry = n.geometry.clone().scale(scale, scale, scale);
      }
    });
    wrap.add(obj);
    return wrap;
  }

  async function loadRobot(parts) {
    robotMeshes = [];
    for (const part of parts || []) {
      try {
        const mesh = await loadRobotPart(part);
        robotRoot.add(mesh);
        robotMeshes.push(mesh);
      } catch (err) {
        onLog(`Falha na malha ${part.mesh}: ${err.message}`, "warn");
        robotMeshes.push(new THREE.Group());
      }
    }
    onLog(`Braço UR5e + Robotiq: ${robotMeshes.length} malhas.`, "ok");
  }

  function applyRobot() {
    const poses = smooth.robot.length
      ? smooth.robot
      : (targetState && targetState.robot) || [];
    if (!poses || !poses.length) return;
    for (let i = 0; i < robotMeshes.length && i < poses.length; i++) {
      const p = poses[i];
      robotMeshes[i].position.set(p.pos[0], p.pos[1], p.pos[2] + COUNTER_Z);
      robotMeshes[i].quaternion.set(p.quat[1], p.quat[2], p.quat[3], p.quat[0]);
    }
    tcpRing.position.set(smooth.pos[0], smooth.pos[1], COUNTER_Z + 0.0045);
    tcpRing.material.color.setHex(smooth.is_grasping ? 0x10b981 : 0x38bdf8);
  }

  async function loadManifest() {
    try {
      const res = await fetch("/api/scene-manifest");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      onLog(`Manifesto indisponível (${err.message}); layout local.`, "warn");
      return {
        props: [
          {
            id: "dental_chair",
            url: "/assets/objetos/odontologia/dental_chair_ostem.glb",
            label: "Cadeira Odontológica",
            fit: "upright",
            targetHeightM: 1.18,
            position: [-1.35, 0.22, 0],
            yawDeg: 90,
            flip: true,
          },
          {
            id: "metal_tray",
            url: "/assets/objetos/odontologia/metal_tray.glb",
            label: "Bandeja Metálica",
            fit: "lieFlat",
            flip: true,
            targetSizeM: 0.36,
            position: [0.28, -0.22, COUNTER_Z],
            yawDeg: -8,
          },
          {
            id: "dentist_tools",
            url: "/assets/objetos/odontologia/dentist_tools.glb",
            label: "Kit de Instrumentos",
            fit: "lieFlat",
            align: "y",
            targetSizeM: 0.16,
            position: [0.50, -0.22, COUNTER_Z],
            yawDeg: -8,
          },
          {
            id: "dental_tool_visual",
            url: "/assets/objetos/odontologia/dental_tool-nih3d.stl",
            label: "Instrumento Dental",
            role: "manipulable_visual",
            scale: 0.001,
            syncObject: "dental_tool",
          },
          {
            id: "floss_holder_visual",
            url: "/assets/objetos/odontologia/Angled_Dental_Floss_Holder.stl",
            label: "Suporte de Fio Dental",
            role: "manipulable_visual",
            scale: 0.001,
            scaleVec: [0.001, 0.001, 0.005],
            syncObject: "floss_holder",
          },
        ],
      };
    }
  }

  async function loadProp(prop) {
    const url = prop.url;
    const isStl = url.toLowerCase().endsWith(".stl");
    let root;
    if (isStl) {
      const geom = await stlLoader.loadAsync(url);
      geom.computeVertexNormals();
      root = new THREE.Mesh(
        geom,
        new THREE.MeshStandardMaterial({ color: 0xc5ced6, metalness: 0.65, roughness: 0.32 })
      );
      root.castShadow = true;
    } else {
      const gltf = await gltfLoader.loadAsync(url);
      root = gltf.scene;
      root.traverse((n) => {
        if (n.isMesh) {
          n.castShadow = perfTier === "high";
          n.receiveShadow = true;
        }
      });
    }

    const synced = prop.role === "manipulable_visual" && prop.syncObject;
    let wrapper;
    if (synced) {
      // Mesma origem e escala da malha no MuJoCo — não recentralizar
      if (Array.isArray(prop.scaleVec) && prop.scaleVec.length === 3) {
        root.scale.set(prop.scaleVec[0], prop.scaleVec[1], prop.scaleVec[2]);
      } else if (prop.scale) {
        root.scale.setScalar(prop.scale);
      }
      wrapper = new THREE.Group();
      wrapper.name = prop.id || "prop";
      wrapper.add(root);
      syncMeshes.set(prop.syncObject, wrapper);
    } else {
      wrapper = layoutProp(root, prop);
    }
    scene.add(wrapper);
    onLog(`Modelo 3D: ${prop.label || prop.id}`, "ok");
    return wrapper;
  }

  const manifest = await loadManifest();
  await loadRobot(manifest.robot || []).catch((err) => {
    onLog(`Falha ao carregar UR5e: ${err.message}`, "warn");
  });
  const results = await Promise.allSettled((manifest.props || []).map(loadProp));
  const loaded = results.filter((r) => r.status === "fulfilled").length;
  results.forEach((r, i) => {
    if (r.status === "rejected") {
      onLog(`Falha ao carregar ${(manifest.props || [])[i]?.id}: ${r.reason}`, "warn");
    }
  });
  onLog(`Sala odontológica: ${loaded} modelos + bancada.`, "ok");

  const loadingEl = document.getElementById("scene-loading");
  if (loadingEl) loadingEl.classList.add("hidden");

  let running = false;
  let frameCount = 0;
  let fpsLast = performance.now();
  let lastTs = performance.now();
  let renderFps = 60;

  function resize() {
    const w = Math.max(2, canvas.clientWidth);
    const h = Math.max(2, canvas.clientHeight);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener("resize", resize);

  function applyGripperPose() {
    applyRobot();
  }

  function applySyncedObjects() {
    for (const [name, mesh] of syncMeshes.entries()) {
      const obj = smooth.objects.get(name);
      if (!obj) continue;
      mesh.position.set(obj.pos[0], obj.pos[1], obj.pos[2] + COUNTER_Z);
      mesh.quaternion.set(obj.quat[1], obj.quat[2], obj.quat[3], obj.quat[0]);
    }
    if (!targetState || !Array.isArray(targetState.objects)) return;
    for (const o of targetState.objects) {
      if (syncMeshes.has(o.name)) continue;
      let helper = scene.getObjectByName(`helper_${o.name}`);
      if (!helper) {
        const color = new THREE.Color(o.color || "#888888");
        if (o.type === "cylinder") {
          helper = new THREE.Mesh(
            new THREE.CylinderGeometry(o.size[0], o.size[0], o.size[1] * 2, 20),
            new THREE.MeshStandardMaterial({ color, metalness: 0.25, roughness: 0.45 })
          );
          helper.rotation.x = Math.PI / 2;
        } else if (o.type === "mesh") {
          continue;
        } else {
          helper = new THREE.Mesh(
            new THREE.BoxGeometry(o.size[0] * 2, o.size[1] * 2, o.size[2] * 2),
            new THREE.MeshStandardMaterial({ color, metalness: 0.2, roughness: 0.5 })
          );
        }
        helper.name = `helper_${o.name}`;
        helper.castShadow = true;
        scene.add(helper);
      }
      helper.position.set(o.pos[0], o.pos[1], o.pos[2] + COUNTER_Z);
      helper.quaternion.set(o.quat[1], o.quat[2], o.quat[3], o.quat[0]);
    }
  }

  function ingestBackendState(state) {
    targetState = state;
    if (Array.isArray(state.objects)) {
      for (const o of state.objects) {
        if (!smooth.objects.has(o.name)) {
          smooth.objects.set(o.name, { pos: o.pos.slice(), quat: o.quat.slice() });
        }
      }
    }
    if (Array.isArray(state.robot) && state.robot.length) {
      if (smooth.robot.length !== state.robot.length) {
        smooth.robot = state.robot.map((p) => ({ pos: p.pos.slice(), quat: p.quat.slice() }));
      }
    }
  }

  function setCameraPreset(preset) {
    const look = new THREE.Vector3(0.05, 0.1, COUNTER_Z + 0.15);
    if (preset === "top") camera.position.set(0.05, 0.05, COUNTER_Z + 1.7);
    else if (preset === "front") camera.position.set(0.1, -2.05, COUNTER_Z + 0.55);
    else if (preset === "side") camera.position.set(2.15, 0.05, COUNTER_Z + 0.55);
    else camera.position.set(1.85, -2.15, 1.85);
    controls.target.copy(look);
    controls.update();
  }

  function tick(now) {
    if (!running) return;
    const dt = Math.min(0.08, Math.max(0.001, (now - lastTs) / 1000));
    lastTs = now;
    const alpha = 1 - Math.exp(-22 * dt);

    if (targetState) {
      lerpVec3(smooth.pos, smooth.pos, targetState.pos, alpha);
      slerpQuat(smooth.quat, smooth.quat, targetState.quat, alpha);
      smooth.aperture = lerp(smooth.aperture, targetState.aperture ?? 0.08, alpha);
      smooth.contact_force = targetState.contact_force || 0;
      smooth.is_grasping = Boolean(targetState.is_grasping);
      if (Array.isArray(targetState.objects)) {
        for (const o of targetState.objects) {
          const cur = smooth.objects.get(o.name);
          if (cur) {
            lerpVec3(cur.pos, cur.pos, o.pos, alpha);
            slerpQuat(cur.quat, cur.quat, o.quat, alpha);
          }
        }
      }
      if (Array.isArray(targetState.robot)) {
        for (let i = 0; i < smooth.robot.length && i < targetState.robot.length; i++) {
          lerpVec3(smooth.robot[i].pos, smooth.robot[i].pos, targetState.robot[i].pos, alpha);
          slerpQuat(smooth.robot[i].quat, smooth.robot[i].quat, targetState.robot[i].quat, alpha);
        }
      }
    }

    applyGripperPose();
    applySyncedObjects();
    controls.update();
    renderer.render(scene, camera);

    frameCount++;
    if (now - fpsLast >= 500) {
      renderFps = Math.round((frameCount * 1000) / (now - fpsLast));
      frameCount = 0;
      fpsLast = now;
      onFpsUpdate(renderFps);
    }
    requestAnimationFrame(tick);
  }

  return {
    ingestBackendState,
    setCameraPreset,
    start() {
      if (running) return;
      running = true;
      lastTs = performance.now();
      requestAnimationFrame(tick);
    },
    stop() { running = false; },
    resize,
    get renderFps() { return renderFps; },
  };
}

window.createDentalClinicRenderer = createDentalClinicRenderer;
