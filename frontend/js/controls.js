/**
 * controls.js
 * Gerencia botões das 8 funções da garra, jogging contínuo ao segurar,
 * atalhos de teclado (WASD + Q/E + Espaço/O), sliders de velocidade/força,
 * telemetria de FPS/latência e rotina automática de Pick & Place.
 */
(function (global) {
  "use strict";

  const KEY_TO_COMMAND = {
    KeyD: "move_right",
    ArrowRight: "move_right",
    KeyA: "move_left",
    ArrowLeft: "move_left",
    KeyW: "move_forward",
    ArrowUp: "move_forward",
    KeyS: "move_backward",
    ArrowDown: "move_backward",
    KeyE: "move_up",
    KeyQ: "move_down",
    Space: "close",
    KeyC: "close",
    KeyO: "open",
  };

  class SimulatorUIControls {
    constructor(wsClient, renderer) {
      this.ws = wsClient;
      this.renderer = renderer;
      this.holdInterval = null;
      this.macroRunning = false;

      this.els = {
        wsInput: document.getElementById("ws-url-input"),
        wsConnectBtn: document.getElementById("ws-connect-btn"),
        connBadge: document.getElementById("conn-badge"),
        renderFpsVal: document.getElementById("render-fps-val"),
        simFpsVal: document.getElementById("sim-fps-val"),
        latencyVal: document.getElementById("latency-val"),
        engineVal: document.getElementById("engine-val"),
        fallbackBanner: document.getElementById("fallback-banner"),

        speedSlider: document.getElementById("speed-slider"),
        speedOut: document.getElementById("speed-out"),
        forceSlider: document.getElementById("force-slider"),
        forceOut: document.getElementById("force-out"),

        hudPos: document.getElementById("hud-pos"),
        hudQuat: document.getElementById("hud-quat"),
        hudAperture: document.getElementById("hud-aperture"),
        hudGrasp: document.getElementById("hud-grasp"),

        leftTouchVal: document.getElementById("left-touch-val"),
        leftTouchBar: document.getElementById("left-touch-bar"),
        rightTouchVal: document.getElementById("right-touch-val"),
        rightTouchBar: document.getElementById("right-touch-bar"),
        totalForceVal: document.getElementById("total-force-val"),
        wristForceVal: document.getElementById("wrist-force-val"),

        resetBtn: document.getElementById("btn-reset"),
        demoPickBtn: document.getElementById("btn-demo-pick"),
        demoFlossBtn: document.getElementById("btn-demo-floss"),
        logBox: document.getElementById("telemetry-log"),
      };

      this._initBindings();
    }

    get currentSpeed() {
      return parseFloat(this.els.speedSlider.value) || 0.5;
    }

    get currentForce() {
      return parseFloat(this.els.forceSlider.value) || 50.0;
    }

    _initBindings() {
      this.els.wsInput.value = this.ws.url;

      this.els.wsConnectBtn.addEventListener("click", () => {
        this.ws.connect(this.els.wsInput.value);
      });

      this.els.wsInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          this.ws.connect(this.els.wsInput.value);
        }
      });

      this.els.speedSlider.addEventListener("input", () => {
        this.els.speedOut.textContent = `${this.currentSpeed.toFixed(2)} m/s`;
      });

      this.els.forceSlider.addEventListener("input", () => {
        this.els.forceOut.textContent = `${this.currentForce.toFixed(0)} N`;
      });

      // Botões das 8 funções principais (clique único + segurar para movimento contínuo)
      document.querySelectorAll("[data-command]").forEach((btn) => {
        const cmd = btn.getAttribute("data-command");
        const isJaw = cmd === "close" || cmd === "open";

        const triggerOnce = () => {
          this.dispatchCommand(cmd, isJaw ? 0.25 : 0.12);
        };

        const startHold = (ev) => {
          ev.preventDefault();
          btn.classList.add("active-press");
          triggerOnce();
          if (!isJaw) {
            this._stopHold();
            this.holdInterval = setInterval(() => {
              this.dispatchCommand(cmd, 0.08);
            }, 65);
          }
        };

        const stopHold = () => {
          btn.classList.remove("active-press");
          this._stopHold();
        };

        btn.addEventListener("mousedown", startHold);
        btn.addEventListener("touchstart", startHold, { passive: false });
        btn.addEventListener("mouseup", stopHold);
        btn.addEventListener("mouseleave", stopHold);
        btn.addEventListener("touchend", stopHold);
      });

      // Presets de câmera 3D
      document.querySelectorAll("[data-cam]").forEach((btn) => {
        btn.addEventListener("click", () => {
          this.renderer.setCameraPreset(btn.getAttribute("data-cam"));
        });
      });

      this.els.resetBtn.addEventListener("click", () => {
        this.macroRunning = false;
        this.ws.sendCommand("reset");
        this.appendLog("Cena MuJoCo reiniciada para posição inicial.", "ok");
      });

      this.els.demoPickBtn.addEventListener("click", () => {
        this.requestVisionPick("dental_tool");
      });

      if (this.els.demoFlossBtn) {
        this.els.demoFlossBtn.addEventListener("click", () => {
          this.requestVisionPick("floss_holder");
        });
      }

      // Atalhos de teclado (ignora key-repeat automático para não enfileirar rajadas excessivas)
      window.addEventListener("keydown", (e) => {
        if (e.target?.closest("input, select, textarea, [contenteditable=true]")) return;
        const cmd = KEY_TO_COMMAND[e.code];
        if (!cmd) return;
        e.preventDefault();
        const isJaw = cmd === "close" || cmd === "open";
        if (e.repeat && isJaw) return;
        const btn = document.querySelector(`[data-command="${cmd}"]`);
        if (btn) btn.classList.add("active-press");
        this.dispatchCommand(cmd, isJaw ? 0.25 : (e.repeat ? 0.06 : 0.10));
      });

      window.addEventListener("keyup", (e) => {
        const cmd = KEY_TO_COMMAND[e.code];
        if (!cmd) return;
        const btn = document.querySelector(`[data-command="${cmd}"]`);
        if (btn) btn.classList.remove("active-press");
      });
    }

    _stopHold() {
      if (this.holdInterval) {
        clearInterval(this.holdInterval);
        this.holdInterval = null;
      }
    }

    dispatchCommand(command, duration = 0.14) {
      if (this.visionRunning) return;
      this.ws.sendCommand(command, {
        speed: this.currentSpeed,
        force: this.currentForce,
        duration,
      });
    }

    updateConnectionStatus(info) {
      const badge = this.els.connBadge;
      badge.classList.remove("status-connected", "status-fallback", "status-disconnected");

      if (info.status === "connected") {
        badge.classList.add("status-connected");
        badge.innerHTML = `WS: <strong>ONLINE</strong>`;
        this.els.fallbackBanner.classList.add("hidden");
      } else if (info.status === "fallback") {
        badge.classList.add("status-fallback");
        badge.innerHTML = `WS: <strong>FALLBACK</strong>`;
        this.els.fallbackBanner.classList.remove("hidden");
      } else {
        badge.classList.add("status-disconnected");
        badge.innerHTML = `WS: <strong>${info.status.toUpperCase()}</strong>`;
      }
    }

    updateTelemetryUI(state) {
      if (!state) return;
      const [x, y, z] = state.pos || [0, 0, 0];
      const [qw, qx, qy, qz] = state.quat || [1, 0, 0, 0];
      const apertureMm = ((state.aperture || 0) * 1000).toFixed(1);

      this.els.hudPos.textContent = `[${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)}] m`;
      this.els.hudQuat.textContent = `[${qw.toFixed(2)}, ${qx.toFixed(2)}, ${qy.toFixed(2)}, ${qz.toFixed(2)}]`;
      this.els.hudAperture.textContent = `${apertureMm} mm (${state.is_closed ? "FECHADA" : "ABERTA"})`;
      this.els.hudGrasp.textContent = state.is_grasping
        ? `CAPTURADO (${state.grasped_object})`
        : "Livre";

      const sensors = state.sensors || {};
      const lt = sensors.left_touch || 0;
      const rt = sensors.right_touch || 0;
      const wf = sensors.wrist_force || [0, 0, 0];

      this.els.leftTouchVal.textContent = `${lt.toFixed(1)} N`;
      this.els.rightTouchVal.textContent = `${rt.toFixed(1)} N`;
      this.els.leftTouchBar.style.width = `${Math.min(100, lt * 2)}%`;
      this.els.rightTouchBar.style.width = `${Math.min(100, rt * 2)}%`;
      this.els.totalForceVal.textContent = `${(state.force || 0).toFixed(1)} N`;
      this.els.wristForceVal.textContent = `[${wf[0].toFixed(1)}, ${wf[1].toFixed(1)}, ${wf[2].toFixed(1)}]`;

      this.els.simFpsVal.textContent = `${Math.round(this.ws.backendFps || 60)}`;
      if (state.engine) {
        this.els.engineVal.textContent = state.engine;
      }
    }

    updateRenderFps(fps) {
      this.els.renderFpsVal.textContent = `${fps}`;
    }

    updateLatency(sample) {
      this.els.latencyVal.textContent = `${sample.cmdToRenderMs.toFixed(1)} ms`;
      this.appendLog(
        `CMD '${sample.command}' | RTT=${sample.rttMs.toFixed(1)}ms | Cmd->Render=${sample.cmdToRenderMs.toFixed(1)}ms`,
        "ok"
      );
    }

    appendLog(text, level = "info") {
      if (!this.els.logBox) return;
      const line = document.createElement("div");
      line.className = `log-line ${level}`;
      const ts = new Date().toLocaleTimeString("pt-BR", { hour12: false });
      line.textContent = `[${ts}] ${text}`;
      this.els.logBox.appendChild(line);
      while (this.els.logBox.children.length > 60) {
        this.els.logBox.removeChild(this.els.logBox.firstChild);
      }
      this.els.logBox.scrollTop = this.els.logBox.scrollHeight;
    }

    requestVisionPick(className) {
      if (this.macroRunning || this.visionRunning) return;
      this._stopHold();
      if (this.ws.sendCommand("vision_pick", {class_name: className})) {
        this.appendLog(`Comando enviado: pegar '${className}'. Solicitando detecção ao backend.`, "info");
      }
    }

  }

  global.SimulatorUIControls = SimulatorUIControls;

  async function waitForDentalFactory(timeoutMs = 8000) {
    if (typeof window.createDentalClinicRenderer === "function") {
      return window.createDentalClinicRenderer;
    }
    return new Promise((resolve) => {
      const started = performance.now();
      const timer = setInterval(() => {
        if (typeof window.createDentalClinicRenderer === "function") {
          clearInterval(timer);
          resolve(window.createDentalClinicRenderer);
        } else if (performance.now() - started > timeoutMs) {
          clearInterval(timer);
          resolve(null);
        }
      }, 40);
    });
  }

  window.addEventListener("DOMContentLoaded", async () => {
    const canvas = document.getElementById("sim-canvas");
    if (!canvas) return;

    let uiControls = null;
    let visionPanel = null;
    const sharedOpts = {
      onFpsUpdate: (fps) => {
        if (uiControls) uiControls.updateRenderFps(fps);
      },
      onLog: (msg, lvl) => {
        if (uiControls) uiControls.appendLog(msg, lvl);
      },
    };

    let renderer = null;
    const dentalFactory = await waitForDentalFactory();
    if (dentalFactory) {
      try {
        renderer = await dentalFactory(canvas, sharedOpts);
      } catch (err) {
        console.warn("Falha na cena dental Three.js; fallback WebGL legado.", err);
      }
    }
    if (!renderer) {
      const loadingEl = document.getElementById("scene-loading");
      if (loadingEl) loadingEl.classList.add("hidden");
      renderer = new Gripper3DRenderer(canvas, sharedOpts);
    }

    const wsClient = new GripperWebSocketClient({
      onVisionPick: (packet) => { if (visionPanel) visionPanel.setOperation(packet); },
      onState: (state) => {
        renderer.ingestBackendState(state);
        if (uiControls) uiControls.updateTelemetryUI(state);
        document.querySelector(".viewport-pane")
          ?.classList.toggle("is-grasping", Boolean(state.is_grasping));
      },
      onConnectionChange: (info) => {
        if (uiControls) uiControls.updateConnectionStatus(info);
      },
      onLatencySample: (sample) => {
        if (uiControls) uiControls.updateLatency(sample);
      },
      onLog: (msg, lvl) => {
        if (uiControls) uiControls.appendLog(msg, lvl);
      },
    });

    uiControls = new SimulatorUIControls(wsClient, renderer);
    visionPanel = new VisionPanel(wsClient, uiControls);
    renderer.ingestBackendState(wsClient.lastKnownState);
    renderer.start();
    wsClient.connect();

    global.__SIM_APP__ = { renderer, wsClient, uiControls };
  });
})(window);
