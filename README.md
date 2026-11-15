# SimuladorGarra 3D — Simulador de Garra Robótica em Produção (Python MuJoCo + WebGL WebSocket 60 FPS)

Simulador 3D de garra robótica paralela construído em arquitetura **Cliente-Servidor em Tempo Real**:

- **Backend (`backend/`)**: Python + **MuJoCo** executando a dinâmica de corpos rígidos, limites de juntas, cones elípticos de atrito e sensores táteis/força a **500 Hz**, transmitindo o estado sincronizado a **60 FPS** via WebSocket RFC 6455 (`TCP_NODELAY`).
- **Frontend (`frontend/`)**: Visualizador **3D WebGL** responsivo no navegador com iluminação Blinn-Phong, câmera orbital interativa, interpolação suave entre quadros físicos (**LERP** posicional + **SLERP** em quatérnios), detecção de timeout com fallback gracioso para o último estado conhecido e build **single-file** (`frontend/dist/index.html`) pronto para qualquer host estático.

> **Decisão Arquitetural**: Não compilamos o MuJoCo para WebAssembly (WASM). Manter a física em Python + C++ MuJoCo no servidor evita binários pesados no cliente, permite escalar para GPUs/clusters de aprendizado por reforço ou acoplar hardware real (**Sim-to-Real**) mantendo o frontend leve (~38 KB) e universal.

---

## 1. Estrutura do Projeto

```text
SimuladorGarra/
├── backend/                        # Simulação física Python + MuJoCo & Servidor 60 FPS
│   ├── models/
│   │   └── gripper.mjcf            # Modelo MJCF completo (limites, atrito, atuadores e sensores)
│   ├── mujoco_env.py               # Carregamento do .mjcf, integração física e leitura de sensores
│   ├── gripper_controller.py       # 8 funções de movimento (move_right/left/..., close, open)
│   ├── websocket_server.py         # Servidor HTTP + WebSocket 60 FPS em porta única
│   ├── config.py                   # Caminhos de modelo, especificações da garra e rede
│   └── main.py                     # Ponto de entrada do servidor
│
├── frontend/                       # Visualização 3D no navegador (zero dependências externas)
│   ├── index.html                  # Interface web com viewport 3D, HUD e painel de controle
│   ├── css/style.css               # Layout responsivo industrial
│   ├── js/renderer.js              # Cena 3D WebGL, malha articulada da garra e interpolação LERP/SLERP
│   ├── js/websocket_client.js      # Cliente WebSocket, métricas RTT/FPS, timeout e fallback
│   ├── js/controls.js              # Botões das 8 funções, atalhos WASD/QE, sliders speed/force
│   ├── js/controls.py              # Bindings declarativos equivalentes em Python (UI/automação)
│   └── dist/index.html             # Build single-file autônomo para deploy estático
│
├── scenarios/                      # Cenários automatizados de teste via WebSocket
│   ├── test_basic_movement.py      # Valida as 8 funções via WebSocket e schema de estado
│   ├── test_real_time_sync.py      # Aferição de stream 60 FPS e latência comando -> render (<100ms)
│   ├── test_object_interaction.py  # Pick & Place físico real no MuJoCo via WebSocket
│   └── test_stress.py              # Alta frequência (160+ cmds), 0 quedas e estabilidade de FPS
│
├── scripts/
│   └── bundle_frontend.py          # Gera o arquivo único frontend/dist/index.html
├── main.py                         # CLI principal (`python main.py --port 8000`)
├── requirements.txt                # Dependências Python do backend
├── package.json                    # Scripts de build e execução do frontend
├── PROTOCOL.md                     # Documentação completa do protocolo JSON WebSocket & REST
└── DEPLOYMENT.md                   # Guia de deploy em servidor remoto, Nginx WSS e frontend estático
```

---

## 2. Instalação e Posicionamento do Modelo MuJoCo (`.mjcf`)

### 2.1 Instalar Dependências Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Caso utilize `uv`: `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`)*

### 2.2 Posicionamento do Modelo `.mjcf`

O modelo físico padrão reside em `backend/models/gripper.mjcf` e inclui:
- Juntas prismáticas limitadas (`joint_x`, `joint_y`, `joint_z`, `joint_finger_left`, `joint_finger_right`) com acoplamento simétrico `<equality>`.
- Pastilhas de contato com atrito elíptico elevado (`condim="4"`, `impratio="10"`).
- Sensores `<framepos>`, `<framequat>`, `<force>`, `<torque>` e sensores táteis `<touch>` em cada dedo.

Para carregar um modelo `.mjcf` customizado, passe `--model /caminho/seu_modelo.mjcf` ao iniciar o backend.

---

## 3. Como Executar o Backend e o Frontend

### 3.1 Iniciar o Servidor (Backend + Frontend Integrado na Porta 8000)

```bash
.venv/bin/python main.py --port 8000
```

Abra no navegador:
- **Interface 3D Interativa**: [http://localhost:8000](http://localhost:8000)
- **Versão Single-File Estática**: [http://localhost:8000/standalone](http://localhost:8000/standalone)
- **Healthcheck REST**: [http://localhost:8000/health](http://localhost:8000/health)

### 3.2 Gerar o Frontend Single-File para Hospedagem Estática Independente

```bash
python3 scripts/bundle_frontend.py
```

O arquivo gerado em `frontend/dist/index.html` contém todo o HTML, CSS e JavaScript WebGL embutidos e pode ser hospedado em qualquer servidor estático apontando para o backend via `?ws=ws://IP_DO_SERVIDOR:8000/ws`.

---

## 4. Controles na Interface 3D

- **Mouse Esquerdo + Arrastar**: Órbita 3D ao redor da bancada.
- **Mouse Direito (ou Shift) + Arrastar**: Panorâmica (Pan).
- **Scroll do Mouse**: Zoom.
- **8 Comandos de Movimento (Botões ou Teclado)**:
  - `W` / `S`: `move_forward` (+Y) / `move_backward` (-Y)
  - `A` / `D`: `move_left` (-X) / `move_right` (+X)
  - `E` / `Q`: `move_up` (+Z) / `move_down` (-Z)
  - `Espaço` ou `C`: `close` (fechar dedos com força controlada)
  - `O`: `open` (abrir dedos)
- **Sliders**: Ajuste em tempo real de `speed` (`0.05` – `1.50 m/s`) e `force` (`1` – `100 N`).
- **Auto Pick & Place**: Demonstração automática capturando o **instrumento dental** (STL NIH3D) e depositando-o na zona verde.
- **Clínica 3D**: modelos em `assets/objetos/odontologia/` (cadeira, bandeja, kit, instrumento) via Three.js + `/api/scene-manifest`.

---

## 5. Executando os Cenários de Teste (`scenarios/`)

```bash
# Executar todos os 4 cenários WebSocket com relatório detalhado de métricas
.venv/bin/python main.py --scenario all

# Ou executar via pytest / unittest diretamente
.venv/bin/pytest scenarios/ -v
```

Consulte [`PROTOCOL.md`](./PROTOCOL.md) para detalhes das mensagens JSON e [`DEPLOYMENT.md`](./DEPLOYMENT.md) para instruções de produção.
