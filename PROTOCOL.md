# Especificação do Protocolo WebSocket e REST (`PROTOCOL.md`)

Este documento define o contrato de comunicação em tempo real entre o **Backend Python + MuJoCo** (`backend/websocket_server.py`) e o **Frontend 3D no Navegador** (`frontend/js/websocket_client.js`).

---

## 1. Transporte e Conexão

- **Endpoint WebSocket Padrão**: `ws://<HOST>:<PORT>/ws` (em TLS/produção: `wss://<DOMAIN>/ws`)
- **Versão do Protocolo**: RFC 6455 (WebSocket Version 13), quadros UTF-8 JSON (`opcode = 0x1`)
- **Otimização de Latência**: `TCP_NODELAY` habilitado no servidor e escrita protegida por lock assíncrono por cliente
- **Taxa de Transmissão Contínua**: **60 FPS** (`16.67 ms` por quadro físico, 8 sub-passos MuJoCo de `0.002 s` por quadro)
- **Keepalive / Detecção de Desconexão**:
  - Servidor envia quadros `PING (0x9)` após `30s` de inatividade de comandos.
  - Cliente monitora watchdog de `2500ms`: se o fluxo 60 FPS cessar, ativa **Modo Fallback Gracioso** mantendo o último estado válido renderizado na tela e agenda reconexão com *exponential backoff*.

---

## 2. Mensagens Cliente → Servidor (Comandos)

### 2.1 Comandos das 8 Funções Principais da Garra

Formato canônico:

```json
{
  "command": "move_right",
  "speed": 0.5,
  "force": 50
}
```

Campos suportados:

| Campo | Tipo | Obrigatório | Faixa / Padrão | Descrição |
| :--- | :--- | :---: | :--- | :--- |
| `command` | `string` | Sim | — | Uma das 8 ações (`move_right`, `move_left`, `move_forward`, `move_backward`, `move_down`, `move_up`, `close`, `open`), além de `reset`, `move_to` ou `get_state`. |
| `speed` | `number` | Não | `0.05` – `1.50` (`0.50` m/s) | Velocidade linear alvo do movimento cartesiano ou dos dedos. |
| `force` | `number` | Não | `1.0` – `100.0` (`50.0` N) | Força máxima de atuação/preensão configurada nos servos dos dedos MuJoCo. |
| `duration` | `number` | Não | `0.01` – `5.0` (`0.16` s) | Duração do pulso incremental de deslocamento ($\Delta d = \text{speed} \times \text{duration}$). |
| `req_id` | `string` | Não | — | Identificador único ecoado no `ack` para correlação de latência RTT. |
| `client_ts` | `number` | Não | Epoch ms | Timestamp do cliente ecoado para telemetria comando → renderização. |

---

## 3. Mensagens Servidor → Cliente (Estado em Tempo Real)

Tanto os quadros contínuos de 60 FPS (`"type": "stream"`) quanto as confirmações imediatas de comando (`"type": "ack"`) e o pacote inicial de boas-vindas (`"type": "hello"`) transportam o objeto canônico `"state"`:

### 3.1 Exemplo Mínimo / Canônico de Resposta

```json
{
  "type": "ack",
  "command": "move_right",
  "req_id": "move_right-1763228400",
  "fps": 60.0,
  "state": {
    "pos": [0.08, 0.0, 0.238],
    "quat": [1.0, 0.0, 0.0, 0.0],
    "force": 50.0,
    "commanded_force": 50.0,
    "contact_force": 0.0,
    "speed": 0.5,
    "aperture": 0.08,
    "is_closed": false,
    "is_grasping": false,
    "grasped_object": null,
    "velocity": [0.12, 0.0, 0.0],
    "sensors": {
      "left_touch": 0.0,
      "right_touch": 0.0,
      "wrist_force": [0.0, 0.0, 0.0],
      "wrist_torque": [0.0, 0.0, 0.0]
    },
    "objects": [
      {
        "name": "red_cube",
        "label": "Cubo Vermelho",
        "type": "box",
        "size": [0.024, 0.024, 0.024],
        "color": "#e63d42",
        "pos": [0.16, 0.1, 0.024],
        "quat": [1.0, 0.0, 0.0, 0.0]
      }
    ],
    "drop_zone": {
      "pos": [-0.2, 0.16, 0.0015],
      "size": [0.065, 0.065, 0.0015]
    },
    "sim_time": 4.256,
    "seq": 255,
    "server_ts": 1763228400125.4,
    "engine": "mujoco-4.0.1"
  }
}
```

### 3.2 Definição do Schema `state`

- `pos`: `[x, y, z]` em metros no referencial mundial MuJoCo (`Z-up`), extraído do sensor `<framepos name="tcp_pos"/>`.
- `quat`: `[w, x, y, z]` quatérnio unitário de orientação 3D do TCP, extraído do sensor `<framequat name="tcp_quat"/>`.
- `force`: força efetiva ativa em Newtons (`N`).
- `contact_force`: soma real das leituras dos sensores táteis `<touch name="left_finger_touch"/>` e `<touch name="right_finger_touch"/>` em Newtons.
- `aperture`: distância livre atual entre as faces internas dos dedos paralelos em metros (`0.000` a `0.080`).
- `is_grasping` / `grasped_object`: indica se um objeto manipulável está fisicamente sustentado pelos dedos.
- `objects`: lista de corpos rígidos dinâmicos na cena com pose `[x, y, z]` e quatérnio `[w, x, y, z]` para sincronização de malhas 3D.

### 3.3 Quadro de Erro (`type == "error"`)

Caso um comando inválido seja enviado, a conexão **não é encerrada**; o servidor responde com diagnóstico e o estado íntegro atual:

```json
{
  "type": "error",
  "error": "Comando desconhecido 'fly_away'. Suportados: ['close', 'move_backward', ...]",
  "state": { "pos": [0.0, 0.0, 0.238], "quat": [1.0, 0.0, 0.0, 0.0], "force": 50.0 }
}
```

---

## 4. Endpoints HTTP REST Auxiliares

Na mesma porta TCP (`8000` por padrão), o servidor expõe endpoints REST compatíveis com CORS:

- `GET /health` — Status operacional, motor físico ativo, FPS medido, clientes conectados e uptime.
- `GET /api/state` — Snapshot imediato `{"state": ...}` via HTTP.
- `POST /api/command` — Executa comando JSON (mesmo payload do WebSocket) e retorna `{"state": ...}`.
- `POST /api/reset` — Reinicia a simulação MuJoCo para o estado inicial.
