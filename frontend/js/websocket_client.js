/**
 * websocket_client.js
 * Cliente WebSocket em tempo real com detecção de timeout, reconexão automática,
 * medição precisa de latência comando->renderização e fallback gracioso para o
 * último estado válido conhecido.
 */
(function (global) {
  "use strict";

  class GripperWebSocketClient {
    constructor(options = {}) {
      this.url = options.url || GripperWebSocketClient.resolveDefaultUrl();
      this.disconnectTimeoutMs = options.disconnectTimeoutMs || 2500;
      this.reconnectBaseDelayMs = options.reconnectBaseDelayMs || 600;
      this.maxReconnectDelayMs = options.maxReconnectDelayMs || 6000;

      this.onState = options.onState || function () {};
      this.onConnectionChange = options.onConnectionChange || function () {};
      this.onLatencySample = options.onLatencySample || function () {};
      this.onLog = options.onLog || function () {};
      this.onVision = options.onVision || function () {};
      this.onVisionPick = options.onVisionPick || function () {};

      this.ws = null;
      this.isConnected = false;
      this.inFallbackMode = false;
      this.manualClose = false;
      this.reconnectAttempts = 0;
      this.reconnectTimer = null;
      this.watchdogTimer = null;

      this.lastMessageAt = 0;
      this.backendFps = 0;
      this.lastLatencyMs = 0;
      this.pendingRequests = new Map();

      // Estado padrão inicial seguro para fallback imediato antes mesmo do 1º pacote
      this.lastKnownState = {
        pos: [0.0, 0.0, 0.238],
        quat: [1.0, 0.0, 0.0, 0.0],
        force: 50.0,
        commanded_force: 50.0,
        contact_force: 0.0,
        speed: 0.5,
        aperture: 0.08,
        is_closed: false,
        is_grasping: false,
        grasped_object: null,
        sensors: {
          left_touch: 0.0,
          right_touch: 0.0,
          wrist_force: [0.0, 0.0, 0.0],
          wrist_torque: [0.0, 0.0, 0.0],
        },
        objects: [
          {
            name: "dental_tool",
            label: "Instrumento Dental",
            type: "mesh",
            size: [0.098, 0.0065, 0.0065],
            color: "#c5ced6",
            pos: [0.25, -0.172, 0.010],
            grasp_pos: [0.25, -0.22, 0.010],
            quat: [0.70710678, 0.70710678, 0.0, 0.0],
          },
          {
            name: "floss_holder",
            label: "Suporte de Fio Dental",
            type: "mesh",
            size: [0.038, 0.0095, 0.004],
            color: "#b8c4d0",
            pos: [0.315, -0.238, 0.004],
            grasp_pos: [0.3385, -0.22, 0.004],
            quat: [1.0, 0.0, 0.0, 0.0],
          },
        ],
        drop_zone: { pos: [-0.30, 0.26, 0.0015], size: [0.07, 0.07, 0.0015] },
        engine: "offline-last-state",
      };
    }

    static resolveDefaultUrl() {
      const params = new URLSearchParams(window.location.search);
      const override = params.get("ws") || params.get("backend");
      if (override) {
        return override;
      }
      if (window.location.protocol === "http:" || window.location.protocol === "https:") {
        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        const host = window.location.host || "localhost:8000";
        return `${proto}//${host}/ws`;
      }
      return "ws://localhost:8000/ws";
    }

    connect(customUrl) {
      if (customUrl) {
        this.url = customUrl.trim();
      }
      this.manualClose = false;
      this._clearTimers();

      if (this.ws) {
        try {
          this.ws.onclose = null;
          this.ws.close();
        } catch (_) {}
        this.ws = null;
      }

      this._log(`Conectando a ${this.url} ...`, "info");
      this.onConnectionChange({
        status: "connecting",
        url: this.url,
        fallback: this.inFallbackMode,
      });

      try {
        this.ws = new WebSocket(this.url);
      } catch (err) {
        this._enterFallback(`URL WebSocket inválida: ${err.message}`);
        this._scheduleReconnect();
        return;
      }

      this.ws.onopen = () => {
        this.isConnected = true;
        this.inFallbackMode = false;
        this.reconnectAttempts = 0;
        this.lastMessageAt = performance.now();
        this._startWatchdog();
        this._log(`Conectado ao backend (${this.url})`, "ok");
        this.onConnectionChange({
          status: "connected",
          url: this.url,
          fallback: false,
        });
      };

      this.ws.onmessage = (event) => {
        this.lastMessageAt = performance.now();
        let packet;
        try {
          packet = JSON.parse(event.data);
        } catch (_) {
          return;
        }

        if (typeof packet.fps === "number") {
          this.backendFps = packet.fps;
        }

        if (packet.type === "error") {
          this._log(`Erro do servidor: ${packet.error}`, "warn");
        }
        if (packet.type === "vision") this.onVision(packet);
        if (packet.type === "vision_pick_status") this.onVisionPick(packet);

        if (packet.type === "ack" && packet.req_id && this.pendingRequests.has(packet.req_id)) {
          const sentPerf = this.pendingRequests.get(packet.req_id);
          this.pendingRequests.delete(packet.req_id);
          const rttMs = performance.now() - sentPerf;
          this.lastLatencyMs = rttMs;
          // Mede até a conclusão do próximo quadro de renderização do browser
          requestAnimationFrame(() => {
            const cmdToRenderMs = performance.now() - sentPerf;
            this.onLatencySample({
              rttMs,
              cmdToRenderMs,
              command: packet.command,
            });
          });
        }

        if (packet.state) {
          this.lastKnownState = packet.state;
          this.onState(packet.state, packet);
        }
      };

      this.ws.onerror = () => {
        // O evento onclose subsequente aciona fallback e reconexão controlada
      };

      this.ws.onclose = () => {
        const wasConnected = this.isConnected;
        this.isConnected = false;
        this._clearTimers();
        if (!this.manualClose) {
          this._enterFallback(
            wasConnected
              ? "Conexão WebSocket perdida; mantendo último estado conhecido."
              : "Backend indisponível; exibindo último estado conhecido."
          );
          this._scheduleReconnect();
        } else {
          this.onConnectionChange({
            status: "disconnected",
            url: this.url,
            fallback: false,
          });
        }
      };
    }

    sendCommand(command, params = {}) {
      const reqId = `${command}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const payload = Object.assign(
        {
          command,
          req_id: reqId,
          client_ts: Date.now(),
        },
        params
      );

      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        this._log(`Sem conexão ativa para '${command}' (modo fallback ativo).`, "warn");
        this.onState(this.lastKnownState, { type: "fallback_local" });
        return false;
      }

      this.pendingRequests.set(reqId, performance.now());
      if (this.pendingRequests.size > 200) {
        const oldestKey = this.pendingRequests.keys().next().value;
        this.pendingRequests.delete(oldestKey);
      }

      this.ws.send(JSON.stringify(payload));
      return true;
    }

    disconnect() {
      this.manualClose = true;
      this._clearTimers();
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
    }

    _startWatchdog() {
      this._stopWatchdog();
      this.watchdogTimer = setInterval(() => {
        if (!this.isConnected) return;
        const silenceMs = performance.now() - this.lastMessageAt;
        if (silenceMs > this.disconnectTimeoutMs) {
          this._log(
            `Timeout de stream WebSocket (${Math.round(silenceMs)}ms sem quadros). Reconectando...`,
            "warn"
          );
          try {
            this.ws.close();
          } catch (_) {}
        }
      }, 500);
    }

    _stopWatchdog() {
      if (this.watchdogTimer) {
        clearInterval(this.watchdogTimer);
        this.watchdogTimer = null;
      }
    }

    _enterFallback(reason) {
      this.inFallbackMode = true;
      this._log(reason, "warn");
      this.onConnectionChange({
        status: "fallback",
        url: this.url,
        fallback: true,
        reason,
      });
      this.onState(this.lastKnownState, { type: "fallback" });
    }

    _scheduleReconnect() {
      if (this.manualClose) return;
      const delay = Math.min(
        this.maxReconnectDelayMs,
        this.reconnectBaseDelayMs * Math.pow(1.6, this.reconnectAttempts++)
      );
      this.reconnectTimer = setTimeout(() => {
        this.connect();
      }, delay);
    }

    _clearTimers() {
      this._stopWatchdog();
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
    }

    _log(message, level = "info") {
      console.info(`[WSClient][${level.toUpperCase()}] ${message}`);
      this.onLog(message, level);
    }
  }

  global.GripperWebSocketClient = GripperWebSocketClient;
})(window);
