# Guia de Implantação em Produção (`DEPLOYMENT.md`)

Este guia descreve como implantar o **Backend Python + MuJoCo** em um servidor dedicado ou VM em nuvem e hospedar o **Frontend 3D Single-File** em qualquer serviço de arquivos estáticos apontando para o IP/domínio remoto.

---

## 1. Arquitetura de Implantação

```text
[ Navegador do Operador (WebGL 60 FPS) ]
   │  Carrega HTML estático (CDN / S3 / Nginx / Local)
   │  Conecta via ws://<SERVER_IP>:8000/ws ou wss://sim.seudominio.com/ws
   ▼
[ Servidor Linux (Ubuntu 22.04/24.04) ]
   ├── Nginx / Caddy (Opcional: Terminação TLS HTTPS / WSS)
   └── Serviço Systemd: python main.py --host 0.0.0.0 --port 8000
         └── MuJoCo C++ Engine (500 Hz física / 60 FPS broadcast)
```

---

## 2. Implantando o Backend MuJoCo em um Servidor Remoto

### 2.1 Instalação no Servidor

```bash
# 1. Copie o repositório para /opt/simulador-garra
sudo mkdir -p /opt/simulador-garra
sudo chown $USER:$USER /opt/simulador-garra
cd /opt/simulador-garra

# 2. Crie o ambiente virtual e instale as dependências
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Teste a inicialização na interface pública
.venv/bin/python main.py --host 0.0.0.0 --port 8000
```

### 2.2 Configuração de Firewall (UFW / Security Group)

Caso conecte diretamente na porta `8000`:

```bash
sudo ufw allow 8000/tcp
```

Para verificar a saúde do backend remotamente:

```bash
curl http://SEU_IP_DO_SERVIDOR:8000/health
```

### 2.3 Execução Contínua via `systemd`

Crie `/etc/systemd/system/gripper-simulator.service`:

```ini
[Unit]
Description=MuJoCo Gripper Simulator WebSocket Server (60 FPS)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/simulador-garra
Environment="SIM_CORS_ORIGIN=*"
ExecStart=/opt/simulador-garra/.venv/bin/python /opt/simulador-garra/main.py --host 0.0.0.0 --port 8000 --fps 60
Restart=always
RestartSec=2
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
```

Ative o serviço:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gripper-simulator
sudo systemctl status gripper-simulator
```

---

## 3. Proxy Reverso Nginx com TLS (`wss://`) — Recomendado para HTTPS

Se o seu frontend estático estiver hospedado em `https://` (ex.: Netlify, Vercel, GitHub Pages, Cloudflare Pages), os navegadores exigem conexão segura **`wss://`**. Configure o Nginx no servidor backend:

```nginx
server {
    listen 443 ssl http2;
    server_name sim.seudominio.com;

    ssl_certificate     /etc/letsencrypt/live/sim.seudominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/sim.seudominio.com/privkey.pem;

    location /ws {
        proxy_pass http://127.0.0.1:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        tcp_nodelay on;
    }

    location / {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
    }
}
```

---

## 4. Gerando e Apontando o Frontend Single-File para um IP Remoto

### 4.1 Gerar o Artefato Único (`frontend/dist/index.html`)

```bash
python3 scripts/bundle_frontend.py
# Gera: frontend/dist/index.html (100% autônomo: HTML + CSS + WebGL 3D + WebSocket)
```

Você pode enviar **apenas o arquivo `frontend/dist/index.html`** para qualquer bucket S3, Cloudflare Pages, GitHub Pages, servidor Apache/Nginx ou abri-lo localmente no navegador.

### 4.2 Como Apontar o Frontend para o IP Remoto do Backend

Existem **duas formas imediatas** sem precisar recompilar o código:

1. **Via Parâmetro de URL (`?ws=`)**:
   - Sem TLS (rede interna / VPN / IP direto):
     `http://seu-host-estatico.local/index.html?ws=ws://203.0.113.42:8000/ws`
   - Com TLS em produção:
     `https://seu-app.pages.dev/?ws=wss://sim.seudominio.com/ws`

2. **Diretamente na Barra Superior da Interface 3D**:
   - Digite `ws://203.0.113.42:8000/ws` no campo de endereço WebSocket no topo da tela e clique em **Conectar**.
