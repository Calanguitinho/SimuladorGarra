/**
 * renderer.js
 * Motor de renderização 3D WebGL acelerado por GPU (com fallback automático Canvas 3D)
 * projetado para visualização robótica em 60 FPS com interpolação suave (LERP + SLERP)
 * entre quadros físicos do MuJoCo.
 *
 * Sistema de coordenadas MuJoCo (Z-up):
 *   +X: Direita | +Y: Frente (profundidade) | +Z: Vertical para cima
 */
(function (global) {
  "use strict";

  // ==========================================================================
  // Matemática Matricial 4x4, Vetores e Quatérnios (SLERP / LERP)
  // ==========================================================================

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
    // Formato MuJoCo: [w, x, y, z]
    let aw = qa[0], ax = qa[1], ay = qa[2], az = qa[3];
    let bw = qb[0], bx = qb[1], by = qb[2], bz = qb[3];

    let cosHalf = aw * bw + ax * bx + ay * by + az * bz;
    if (cosHalf < 0.0) {
      bw = -bw; bx = -bx; by = -by; bz = -bz;
      cosHalf = -cosHalf;
    }
    if (cosHalf >= 0.9995) {
      out[0] = aw + t * (bw - aw);
      out[1] = ax + t * (bx - ax);
      out[2] = ay + t * (by - ay);
      out[3] = az + t * (bz - az);
    } else {
      const halfAngle = Math.acos(cosHalf);
      const sinHalf = Math.sqrt(1.0 - cosHalf * cosHalf);
      const ratioA = Math.sin((1.0 - t) * halfAngle) / sinHalf;
      const ratioB = Math.sin(t * halfAngle) / sinHalf;
      out[0] = aw * ratioA + bw * ratioB;
      out[1] = ax * ratioA + bx * ratioB;
      out[2] = ay * ratioA + by * ratioB;
      out[3] = az * ratioA + bz * ratioB;
    }
    const len = Math.hypot(out[0], out[1], out[2], out[3]) || 1.0;
    out[0] /= len; out[1] /= len; out[2] /= len; out[3] /= len;
    return out;
  }

  function mat4Perspective(fovRad, aspect, near, far) {
    const f = 1.0 / Math.tan(fovRad / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0,
    ]);
  }

  function mat4LookAt(eye, target, up) {
    let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
    let zLen = Math.hypot(zx, zy, zz) || 1;
    zx /= zLen; zy /= zLen; zz /= zLen;

    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    let xLen = Math.hypot(xx, xy, xz) || 1;
    xx /= xLen; xy /= xLen; xz /= xLen;

    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    return new Float32Array([
      xx, yx, zx, 0,
      xy, yy, zy, 0,
      xz, yz, zz, 0,
      -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
      -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
      -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
      1,
    ]);
  }

  function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        out[c * 4 + r] =
          a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] +
          a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
      }
    }
    return out;
  }

  function mat4FromPosQuatScale(pos, quat, scale) {
    const w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    const sx = scale[0], sy = scale[1], sz = scale[2];
    const x2 = x + x, y2 = y + y, z2 = z + z;
    const xx = x * x2, xy = x * y2, xz = x * z2;
    const yy = y * y2, yz = y * z2, zz = z * z2;
    const wx = w * x2, wy = w * y2, wz = w * z2;

    return new Float32Array([
      (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
      (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
      (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
      pos[0], pos[1], pos[2], 1,
    ]);
  }

  function hexToRgb(hex) {
    const clean = (hex || "#cccccc").replace("#", "");
    const num = parseInt(clean, 16);
    return [
      ((num >> 16) & 255) / 255,
      ((num >> 8) & 255) / 255,
      (num & 255) / 255,
    ];
  }

  // ==========================================================================
  // Construtores de Malhas 3D (Cubo Unitário, Cilindro Z-up e Grade)
  // ==========================================================================

  function buildUnitBoxMesh() {
    // Cubo de [-1, 1]^3 com normais por face para iluminação Blinn-Phong nítida
    const faces = [
      { n: [0, 0, 1], v: [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]] },
      { n: [0, 0,-1], v: [[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,-1]] },
      { n: [1, 0, 0], v: [[1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1]] },
      { n: [-1,0, 0], v: [[-1,-1,1],[-1,1,1],[-1,1,-1],[-1,-1,-1]] },
      { n: [0, 1, 0], v: [[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]] },
      { n: [0,-1, 0], v: [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]] },
    ];
    const positions = [];
    const normals = [];
    for (const f of faces) {
      const [a, b, c, d] = f.v;
      for (const p of [a, b, c, a, c, d]) {
        positions.push(p[0], p[1], p[2]);
        normals.push(f.n[0], f.n[1], f.n[2]);
      }
    }
    return {
      positions: new Float32Array(positions),
      normals: new Float32Array(normals),
      count: positions.length / 3,
    };
  }

  function buildUnitCylinderMesh(segments = 24) {
    // Cilindro alinhado ao eixo Z com raio 1 e altura [-1, 1]
    const positions = [];
    const normals = [];
    for (let i = 0; i < segments; i++) {
      const a0 = (i / segments) * Math.PI * 2;
      const a1 = ((i + 1) / segments) * Math.PI * 2;
      const c0 = Math.cos(a0), s0 = Math.sin(a0);
      const c1 = Math.cos(a1), s1 = Math.sin(a1);

      // Parede lateral (2 triângulos)
      const sideVerts = [
        [c0, s0, -1, c0, s0, 0],
        [c1, s1, -1, c1, s1, 0],
        [c1, s1,  1, c1, s1, 0],
        [c0, s0, -1, c0, s0, 0],
        [c1, s1,  1, c1, s1, 0],
        [c0, s0,  1, c0, s0, 0],
      ];
      for (const v of sideVerts) {
        positions.push(v[0], v[1], v[2]);
        normals.push(v[3], v[4], v[5]);
      }

      // Tampa superior (+Z)
      positions.push(0, 0, 1, c0, s0, 1, c1, s1, 1);
      normals.push(0, 0, 1, 0, 0, 1, 0, 0, 1);

      // Tampa inferior (-Z)
      positions.push(0, 0, -1, c1, s1, -1, c0, s0, -1);
      normals.push(0, 0, -1, 0, 0, -1, 0, 0, -1);
    }
    return {
      positions: new Float32Array(positions),
      normals: new Float32Array(normals),
      count: positions.length / 3,
    };
  }

  function buildFloorGridLines() {
    const verts = [];
    const colors = [];
    const half = 0.5;
    const step = 0.05;
    for (let i = -10; i <= 10; i++) {
      const coord = +(i * step).toFixed(3);
      const major = i % 2 === 0;
      const col = major ? [0.23, 0.30, 0.42] : [0.15, 0.20, 0.29];

      verts.push(coord, -half, 0.0005, coord, half, 0.0005);
      colors.push(...col, ...col);

      verts.push(-half, coord, 0.0005, half, coord, 0.0005);
      colors.push(...col, ...col);
    }
    // Eixos de referência na origem (X=vermelho, Y=verde, Z=azul)
    verts.push(0, 0, 0.002, 0.14, 0, 0.002);
    colors.push(0.95, 0.28, 0.30, 0.95, 0.28, 0.30);

    verts.push(0, 0, 0.002, 0, 0.14, 0.002);
    colors.push(0.20, 0.85, 0.45, 0.20, 0.85, 0.45);

    verts.push(0, 0, 0.002, 0, 0, 0.14);
    colors.push(0.28, 0.58, 0.98, 0.28, 0.58, 0.98);

    return {
      positions: new Float32Array(verts),
      colors: new Float32Array(colors),
      count: verts.length / 3,
    };
  }

  // ==========================================================================
  // Renderizador 3D Principal
  // ==========================================================================

  class Gripper3DRenderer {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.onFpsUpdate = options.onFpsUpdate || function () {};
      this.gl =
        canvas.getContext("webgl", { antialias: true, alpha: false }) ||
        canvas.getContext("experimental-webgl");

      // Câmera orbital (Z-up)
      this.camera = {
        yaw: -0.72,
        pitch: 0.56,
        distance: 0.92,
        target: [0.0, 0.0, 0.10],
        fov: (44 * Math.PI) / 180,
      };

      // Estado alvo (recebido via WebSocket a 60Hz) vs Estado interpolado (renderizado)
      this.targetState = null;
      this.smoothState = {
        pos: [0.0, 0.0, 0.238],
        quat: [1.0, 0.0, 0.0, 0.0],
        aperture: 0.08,
        contact_force: 0.0,
        is_grasping: false,
        objects: new Map(),
        drop_zone: { pos: [-0.30, 0.26, 0.0015], size: [0.07, 0.07, 0.0015] },
      };

      this.renderFps = 60;
      this._frameCount = 0;
      this._fpsLastTs = performance.now();
      this._lastFrameTs = performance.now();
      this._running = false;

      if (this.gl) {
        this._initWebGL();
      } else {
        this.ctx2d = canvas.getContext("2d");
      }

      this._bindCameraPointerEvents();
      this.resize();
      window.addEventListener("resize", () => this.resize());
    }

    setCameraPreset(preset) {
      if (preset === "top") {
        this.camera.yaw = 0.0;
        this.camera.pitch = 1.48;
        this.camera.distance = 0.98;
        this.camera.target = [0.0, 0.0, 0.04];
      } else if (preset === "front") {
        this.camera.yaw = 0.0;
        this.camera.pitch = 0.18;
        this.camera.distance = 0.88;
        this.camera.target = [0.0, 0.0, 0.12];
      } else if (preset === "side") {
        this.camera.yaw = -Math.PI / 2;
        this.camera.pitch = 0.22;
        this.camera.distance = 0.88;
        this.camera.target = [0.0, 0.0, 0.12];
      } else {
        this.camera.yaw = -0.72;
        this.camera.pitch = 0.56;
        this.camera.distance = 0.92;
        this.camera.target = [0.0, 0.0, 0.10];
      }
    }

    ingestBackendState(state) {
      this.targetState = state;
      if (Array.isArray(state.objects)) {
        for (const obj of state.objects) {
          if (!this.smoothState.objects.has(obj.name)) {
            this.smoothState.objects.set(obj.name, {
              name: obj.name,
              type: obj.type,
              size: obj.size.slice(),
              color: obj.color,
              pos: obj.pos.slice(),
              quat: obj.quat.slice(),
            });
          }
        }
      }
    }

    start() {
      if (this._running) return;
      this._running = true;
      const tick = (now) => {
        if (!this._running) return;
        const dt = Math.min(0.1, Math.max(0.001, (now - this._lastFrameTs) / 1000));
        this._lastFrameTs = now;

        this._interpolateFrame(dt);
        this._renderScene();

        this._frameCount++;
        if (now - this._fpsLastTs >= 500) {
          this.renderFps = Math.round((this._frameCount * 1000) / (now - this._fpsLastTs));
          this._frameCount = 0;
          this._fpsLastTs = now;
          this.onFpsUpdate(this.renderFps);
          console.debug(`[Renderer3D] Render FPS=${this.renderFps}`);
        }
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    }

    stop() {
      this._running = false;
    }

    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(300, Math.floor(this.canvas.clientWidth * dpr));
      const h = Math.max(220, Math.floor(this.canvas.clientHeight * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      if (this.gl) {
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      }
    }

    _interpolateFrame(dt) {
      if (!this.targetState) return;
      // Fator exponencial livre de taxa de quadros (~22 rad/s -> super suave sem atraso)
      const alpha = 1.0 - Math.exp(-24.0 * dt);

      lerpVec3(this.smoothState.pos, this.smoothState.pos, this.targetState.pos, alpha);
      slerpQuat(this.smoothState.quat, this.smoothState.quat, this.targetState.quat, alpha);
      this.smoothState.aperture = lerp(
        this.smoothState.aperture,
        typeof this.targetState.aperture === "number" ? this.targetState.aperture : 0.08,
        alpha
      );
      this.smoothState.contact_force = this.targetState.contact_force || 0.0;
      this.smoothState.is_grasping = Boolean(this.targetState.is_grasping);
      if (this.targetState.drop_zone) {
        this.smoothState.drop_zone = this.targetState.drop_zone;
      }

      if (Array.isArray(this.targetState.objects)) {
        for (const targetObj of this.targetState.objects) {
          const cur = this.smoothState.objects.get(targetObj.name);
          if (cur) {
            lerpVec3(cur.pos, cur.pos, targetObj.pos, alpha);
            slerpQuat(cur.quat, cur.quat, targetObj.quat, alpha);
            cur.color = targetObj.color;
            cur.size = targetObj.size;
            cur.type = targetObj.type;
          }
        }
      }
    }

    // ------------------------------------------------------------------------
    // Inicialização de Shaders e Buffers WebGL
    // ------------------------------------------------------------------------

    _initWebGL() {
      const gl = this.gl;
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      const meshVS = `
        attribute vec3 aPos;
        attribute vec3 aNormal;
        uniform mat4 uMVP;
        uniform mat4 uModel;
        varying vec3 vNormal;
        varying vec3 vWorldPos;
        void main() {
          vec4 world = uModel * vec4(aPos, 1.0);
          vWorldPos = world.xyz;
          vNormal = normalize(mat3(uModel[0].xyz, uModel[1].xyz, uModel[2].xyz) * aNormal);
          gl_Position = uMVP * vec4(aPos, 1.0);
        }
      `;

      const meshFS = `
        precision mediump float;
        varying vec3 vNormal;
        varying vec3 vWorldPos;
        uniform vec4 uColor;
        uniform vec3 uEyePos;
        void main() {
          vec3 N = normalize(vNormal);
          vec3 L1 = normalize(vec3(0.35, -0.45, 0.85));
          vec3 L2 = normalize(vec3(-0.55, 0.55, 0.50));
          vec3 V = normalize(uEyePos - vWorldPos);
          vec3 H = normalize(L1 + V);

          float diff1 = max(dot(N, L1), 0.0);
          float diff2 = max(dot(N, L2), 0.0) * 0.35;
          float spec = pow(max(dot(N, H), 0.0), 36.0) * 0.28;
          float ambient = 0.34;

          vec3 lit = uColor.rgb * (ambient + diff1 * 0.68 + diff2) + vec3(spec);
          gl_FragColor = vec4(lit, uColor.a);
        }
      `;

      const lineVS = `
        attribute vec3 aPos;
        attribute vec3 aColor;
        uniform mat4 uVP;
        varying vec3 vColor;
        void main() {
          vColor = aColor;
          gl_Position = uVP * vec4(aPos, 1.0);
        }
      `;

      const lineFS = `
        precision mediump float;
        varying vec3 vColor;
        void main() {
          gl_FragColor = vec4(vColor, 1.0);
        }
      `;

      this.meshProg = this._createProgram(meshVS, meshFS);
      this.lineProg = this._createProgram(lineVS, lineFS);

      const box = buildUnitBoxMesh();
      this.boxBuffers = {
        pos: this._createBuffer(box.positions),
        norm: this._createBuffer(box.normals),
        count: box.count,
      };

      const cyl = buildUnitCylinderMesh(28);
      this.cylBuffers = {
        pos: this._createBuffer(cyl.positions),
        norm: this._createBuffer(cyl.normals),
        count: cyl.count,
      };

      const grid = buildFloorGridLines();
      this.gridBuffers = {
        pos: this._createBuffer(grid.positions),
        col: this._createBuffer(grid.colors),
        count: grid.count,
      };
    }

    _createShader(type, src) {
      const gl = this.gl;
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      return sh;
    }

    _createProgram(vsSrc, fsSrc) {
      const gl = this.gl;
      const prog = gl.createProgram();
      gl.attachShader(prog, this._createShader(gl.VERTEX_SHADER, vsSrc));
      gl.attachShader(prog, this._createShader(gl.FRAGMENT_SHADER, fsSrc));
      gl.linkProgram(prog);
      return prog;
    }

    _createBuffer(typedArray) {
      const gl = this.gl;
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, typedArray, gl.STATIC_DRAW);
      return buf;
    }

    _computeEyePosition() {
      const { yaw, pitch, distance, target } = this.camera;
      const cp = Math.cos(pitch);
      return [
        target[0] + distance * cp * Math.sin(yaw),
        target[1] - distance * cp * Math.cos(yaw),
        target[2] + distance * Math.sin(pitch),
      ];
    }

    // ------------------------------------------------------------------------
    // Desenho Completo da Cena 3D
    // ------------------------------------------------------------------------

    _renderScene() {
      if (this.gl) {
        this._renderWebGL();
      } else if (this.ctx2d) {
        this._renderCanvasFallback();
      }
    }

    _renderWebGL() {
      const gl = this.gl;
      gl.clearColor(0.055, 0.078, 0.118, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      const aspect = this.canvas.width / Math.max(1, this.canvas.height);
      const proj = mat4Perspective(this.camera.fov, aspect, 0.02, 10.0);
      const eye = this._computeEyePosition();
      const view = mat4LookAt(eye, this.camera.target, [0, 0, 1]);
      const vp = mat4Multiply(proj, view);
      const identityQuat = [1, 0, 0, 0];

      // 1. Plataforma da bancada
      this._drawMesh(this.boxBuffers, vp, eye, [0, 0, -0.012], identityQuat, [0.50, 0.50, 0.012], [0.11, 0.14, 0.19, 1.0]);

      // 2. Grade métrica e eixos cartesianos
      gl.useProgram(this.lineProg);
      const aLinePos = gl.getAttribLocation(this.lineProg, "aPos");
      const aLineCol = gl.getAttribLocation(this.lineProg, "aColor");
      gl.uniformMatrix4fv(gl.getUniformLocation(this.lineProg, "uVP"), false, vp);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffers.pos);
      gl.enableVertexAttribArray(aLinePos);
      gl.vertexAttribPointer(aLinePos, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.gridBuffers.col);
      gl.enableVertexAttribArray(aLineCol);
      gl.vertexAttribPointer(aLineCol, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINES, 0, this.gridBuffers.count);

      // 3. Zona alvo de entrega (drop_zone)
      const dz = this.smoothState.drop_zone;
      this._drawMesh(
        this.boxBuffers,
        vp,
        eye,
        dz.pos,
        identityQuat,
        [dz.size[0], dz.size[1], 0.002],
        [0.12, 0.72, 0.42, 0.52]
      );

      // 4. Estrutura do pórtico cartesiano (trilhos superiores)
      this._drawMesh(this.boxBuffers, vp, eye, [-0.47, 0, 0.28], identityQuat, [0.012, 0.48, 0.28], [0.18, 0.22, 0.30, 0.55]);
      this._drawMesh(this.boxBuffers, vp, eye, [0.47, 0, 0.28], identityQuat, [0.012, 0.48, 0.28], [0.18, 0.22, 0.30, 0.55]);

      // 5. Objetos físicos manipuláveis + sombras projetadas na bancada
      for (const obj of this.smoothState.objects.values()) {
        const rgb = hexToRgb(obj.color);
        const isCyl = obj.type === "cylinder";
        const buf = isCyl ? this.cylBuffers : this.boxBuffers;
        const scale = isCyl
          ? [obj.size[0], obj.size[0], obj.size[1]]
          : [obj.size[0], obj.size[1], obj.size[2]];

        // Sombra planar no tampo (z = 0.0015)
        this._drawMesh(
          buf,
          vp,
          eye,
          [obj.pos[0], obj.pos[1], 0.0012],
          identityQuat,
          [scale[0] * 1.06, scale[1] * 1.06, 0.0008],
          [0.03, 0.04, 0.06, 0.65]
        );
        // Corpo 3D do objeto
        this._drawMesh(buf, vp, eye, obj.pos, obj.quat, scale, [rgb[0], rgb[1], rgb[2], 1.0]);
      }

      // 6. Garra Robótica Paralela Articulada
      const tcp = this.smoothState.pos;
      const gQuat = this.smoothState.quat;
      const palmZ = tcp[2] + 0.062;
      const halfGap = Math.max(0.003, this.smoothState.aperture * 0.5);

      // Sombra do TCP na bancada (ajuda alinhamento vertical preciso)
      this._drawMesh(
        this.cylBuffers,
        vp,
        eye,
        [tcp[0], tcp[1], 0.0018],
        identityQuat,
        [0.028, 0.028, 0.0008],
        this.smoothState.is_grasping ? [0.15, 0.85, 0.48, 0.55] : [0.24, 0.55, 0.98, 0.38]
      );

      // Haste vertical Z do atuador até o teto do pórtico
      const stemTop = 0.56;
      const stemCenterZ = (palmZ + 0.02 + stemTop) * 0.5;
      const stemHalfH = Math.max(0.02, (stemTop - (palmZ + 0.02)) * 0.5);
      this._drawMesh(
        this.cylBuffers,
        vp,
        eye,
        [tcp[0], tcp[1], stemCenterZ],
        identityQuat,
        [0.013, 0.013, stemHalfH],
        [0.34, 0.40, 0.50, 1.0]
      );

      // Flange cilíndrica do punho (sensor de força/torque)
      this._drawMesh(
        this.cylBuffers,
        vp,
        eye,
        [tcp[0], tcp[1], palmZ + 0.032],
        gQuat,
        [0.024, 0.024, 0.016],
        [0.26, 0.31, 0.39, 1.0]
      );

      // Corpo azul da palma paralela
      this._drawMesh(
        this.boxBuffers,
        vp,
        eye,
        [tcp[0], tcp[1], palmZ],
        gQuat,
        [0.056, 0.028, 0.018],
        [0.18, 0.47, 0.88, 1.0]
      );

      // Cor dos dedos muda conforme força de contato nos sensores táteis
      const fRatio = Math.min(1.0, this.smoothState.contact_force / 40.0);
      const padColor = [
        0.96,
        0.62 - 0.34 * fRatio,
        0.14 + 0.12 * (this.smoothState.is_grasping ? 1 : 0),
        1.0,
      ];

      // Suporte estrutural + pastilha de silicone esquerda e direita
      for (const sign of [-1, 1]) {
        const strutX = tcp[0] + sign * (halfGap + 0.011);
        const padX = tcp[0] + sign * (halfGap + 0.004);
        this._drawMesh(
          this.boxBuffers,
          vp,
          eye,
          [strutX, tcp[1], palmZ - 0.038],
          gQuat,
          [0.006, 0.016, 0.022],
          [0.38, 0.44, 0.53, 1.0]
        );
        this._drawMesh(
          this.boxBuffers,
          vp,
          eye,
          [padX, tcp[1], tcp[2]],
          gQuat,
          [0.0045, 0.018, 0.024],
          padColor
        );
      }
    }

    _drawMesh(buffers, vp, eye, pos, quat, scale, rgba) {
      const gl = this.gl;
      gl.useProgram(this.meshProg);

      const model = mat4FromPosQuatScale(pos, quat, scale);
      const mvp = mat4Multiply(vp, model);

      gl.uniformMatrix4fv(gl.getUniformLocation(this.meshProg, "uMVP"), false, mvp);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.meshProg, "uModel"), false, model);
      gl.uniform4fv(gl.getUniformLocation(this.meshProg, "uColor"), rgba);
      gl.uniform3fv(gl.getUniformLocation(this.meshProg, "uEyePos"), eye);

      const aPos = gl.getAttribLocation(this.meshProg, "aPos");
      const aNorm = gl.getAttribLocation(this.meshProg, "aNormal");

      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.pos);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.norm);
      gl.enableVertexAttribArray(aNorm);
      gl.vertexAttribPointer(aNorm, 3, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.TRIANGLES, 0, buffers.count);
    }

    // ------------------------------------------------------------------------
    // Fallback Projetação Isométrica 2D Caso WebGL Esteja Desabilitado
    // ------------------------------------------------------------------------

    _renderCanvasFallback() {
      const ctx = this.ctx2d;
      const w = this.canvas.width;
      const h = this.canvas.height;
      ctx.fillStyle = "#0e1522";
      ctx.fillRect(0, 0, w, h);

      const projIso = (x, y, z) => {
        const scale = Math.min(w, h) * 0.78;
        const sx = w * 0.5 + (x - y) * 0.75 * scale;
        const sy = h * 0.62 + (-x - y) * 0.38 * scale - z * 0.85 * scale;
        return [sx, sy];
      };

      ctx.strokeStyle = "#223049";
      ctx.lineWidth = 1;
      for (let i = -5; i <= 5; i++) {
        const c = i * 0.09;
        const [x0, y0] = projIso(c, -0.45, 0);
        const [x1, y1] = projIso(c, 0.45, 0);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();

        const [gx0, gy0] = projIso(-0.45, c, 0);
        const [gx1, gy1] = projIso(0.45, c, 0);
        ctx.beginPath(); ctx.moveTo(gx0, gy0); ctx.lineTo(gx1, gy1); ctx.stroke();
      }

      for (const obj of this.smoothState.objects.values()) {
        const [ox, oy] = projIso(obj.pos[0], obj.pos[1], obj.pos[2]);
        ctx.fillStyle = obj.color;
        ctx.beginPath();
        ctx.arc(ox, oy, 12, 0, Math.PI * 2);
        ctx.fill();
      }

      const [gx, gy] = projIso(
        this.smoothState.pos[0],
        this.smoothState.pos[1],
        this.smoothState.pos[2]
      );
      ctx.fillStyle = "#3b82f6";
      ctx.fillRect(gx - 22, gy - 10, 44, 16);
    }

    // ------------------------------------------------------------------------
    // Interação de Câmera Orbital (Mouse + Scroll + Touch)
    // ------------------------------------------------------------------------

    _bindCameraPointerEvents() {
      let dragging = false;
      let mode = "orbit";
      let lastX = 0;
      let lastY = 0;

      this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

      this.canvas.addEventListener("mousedown", (e) => {
        dragging = true;
        mode = e.button === 2 || e.shiftKey ? "pan" : "orbit";
        lastX = e.clientX;
        lastY = e.clientY;
      });

      window.addEventListener("mouseup", () => {
        dragging = false;
      });

      window.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const dx = e.clientX - lastX;
        const dy = e.clientY - lastY;
        lastX = e.clientX;
        lastY = e.clientY;

        if (mode === "orbit") {
          this.camera.yaw += dx * 0.0075;
          this.camera.pitch = Math.max(0.08, Math.min(1.50, this.camera.pitch + dy * 0.0075));
        } else {
          const panScale = this.camera.distance * 0.0014;
          const cy = Math.cos(this.camera.yaw);
          const sy = Math.sin(this.camera.yaw);
          this.camera.target[0] -= dx * cy * panScale;
          this.camera.target[1] -= dx * sy * panScale;
          this.camera.target[2] = Math.max(0.0, this.camera.target[2] + dy * panScale);
        }
      });

      this.canvas.addEventListener(
        "wheel",
        (e) => {
          e.preventDefault();
          const factor = e.deltaY > 0 ? 1.08 : 0.92;
          this.camera.distance = Math.max(0.32, Math.min(2.4, this.camera.distance * factor));
        },
        { passive: false }
      );
    }
  }

  global.Gripper3DRenderer = Gripper3DRenderer;
})(window);
