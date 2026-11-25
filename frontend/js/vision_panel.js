/* Preview usa imagem e caixas do mesmo quadro; alvos são resolvidos no servidor. */
(function (global) {
  "use strict";
  class VisionPanel {
    constructor(ws, controls) {
      this.ws = ws;
      this.controls = controls;
      this.busy = false;
      this.loading = false;
      this.packet = null;
      this.receivedAt = 0;
      const root = document.createElement("section");
      root.className = "card glass-card vision-card";
      root.innerHTML = `<div class="card-title">Visão YOLO</div>
        <p class="vision-status" role="status">Visão em espera — aguardando comando</p>
        <canvas class="vision-canvas" width="640" height="480" aria-label="Imagem da câmera de visão"></canvas>
        <label>Ferramenta a pegar <select class="vision-target"><option value="dental_tool">Instrumento dental</option><option value="floss_holder">Suporte de fio dental</option></select></label>
        <div class="vision-actions"><button class="vision-pick" disabled>Pegar ferramenta</button>
        <button class="vision-cancel" disabled>Cancelar captura</button></div>
        <p class="vision-operation" role="status"></p>`;
      document.querySelector("aside")?.appendChild(root);
      this.status = root.querySelector(".vision-status");
      this.canvas = root.querySelector("canvas");
      this.select = root.querySelector("select");
      this.pick = root.querySelector(".vision-pick");
      this.cancel = root.querySelector(".vision-cancel");
      this.operation = root.querySelector(".vision-operation");
      this.select.onchange = () => this.updateButtons();
      this.pick.onclick = () => {
        if (!this.packet || this.controls.macroRunning) return;
        const sent = this.ws.sendCommand("vision_pick", {
          class_name: this.select.value,
        });
        if (sent) { this.busy = true; this.updateButtons(); }
      };
      this.cancel.onclick = () => this.ws.sendCommand("vision_cancel");
      // Polling também entrega estado inicial/reconexão, sem depender de broadcast novo.
      this.timer = setInterval(() => this.poll(), 250);
      this.poll();
    }
    apiUrl(path) {
      const url = new URL(this.ws.url, location.href);
      url.protocol = url.protocol === "wss:" ? "https:" : "http:";
      url.pathname = path; url.search = ""; url.hash = "";
      return url;
    }
    updateButtons() {
      this.pick.disabled = !this.ws.isConnected || this.busy || this.controls.macroRunning ||
        !this.packet?.can_request_pick || !this.select.value;
      this.cancel.disabled = !this.busy || !this.ws.isConnected;
    }
    setOperation(packet) {
      this.busy = !["idle", "completed", "failed", "cancelled"].includes(packet.stage);
      this.controls.visionRunning = this.busy;
      const stages = {detecting: "solicitando detecção", planning: "planejando operação", revalidating: "revalidando ferramenta", idle: "aguardando", validating: "validando alvo", approaching: "aproximando",
        aligning: "alinhando", descending: "descendo", closing: "fechando garra", lifting: "elevando", delivering: "transportando",
        lowering: "depositando", retreating: "recuando", completed: "concluída", failed: "falhou", cancelled: "cancelada"};
      this.operation.textContent = `Captura: ${stages[packet.stage] || packet.stage}${packet.error ? " — " + packet.error : ""}`;
      this.updateButtons();
    }
    async poll() {
      this.updateButtons();
      if (this.loading) return;
      this.loading = true;
      const backend = this.ws.url;
      try {
        const response = await fetch(this.apiUrl("/api/vision/status"), {cache: "no-store"});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const packet = await response.json();
        const received = performance.now();
        if (backend !== this.ws.url) return;
        const states = {idle: "Visão em espera — aguardando comando", detecting: "Detectando ferramenta solicitada", disabled: "Visão desligada", starting: "Iniciando visão", ready: "Detector pronto",
          unavailable: "Detector indisponível", stale: "Aguardando observação recente", error: "Falha na visão"};
        this.status.textContent = packet.enabled && !packet.request_active && !packet.error &&
          ["idle", "ready"].includes(packet.status)
          ? "Visão em espera — aguardando comando"
          : `${states[packet.status] || packet.status}${packet.error ? " — " + packet.error : ""}`;
        if (packet.operation) this.setOperation(packet.operation);
        if (packet.frame_id == null || (this.packet?.frame_id === packet.frame_id && this.packet?.scene_revision === packet.scene_revision)) {
          if (packet.frame_id == null) {
            this.canvas.getContext("2d").clearRect(0, 0, this.canvas.width, this.canvas.height);
          }
          this.packet = packet; this.receivedAt = received;
          this.updateButtons();
          return;
        }
        const image = new Image();
        const url = this.apiUrl("/api/vision/frame");
        url.searchParams.set("frame_id", packet.frame_id);
        image.src = url.href;
        await image.decode();
        if (backend !== this.ws.url) return;
        const [width, height] = packet.image_size;
        this.canvas.width = width; this.canvas.height = height;
        const ctx = this.canvas.getContext("2d");
        ctx.drawImage(image, 0, 0);
        for (const detection of packet.detections) {
          }
        this.packet = packet; this.receivedAt = received;
        this.status.textContent += ` · quadro ${packet.frame_id} · ${packet.inference_ms} ms`;

        this.updateButtons();
      } catch (error) {
        this.status.textContent = `Visão indisponível: ${error.message}`;
        this.packet = null; this.pick.disabled = true;
      } finally { this.loading = false; }
    }
  }
  global.VisionPanel = VisionPanel;
})(window);
