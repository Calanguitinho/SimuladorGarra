# Arquitetura do Sistema e Guia de Migração Sim-to-Real (`ARCHITECTURE.md`)

Este documento descreve a arquitetura interna do **SimuladorGarra**, os contratos de dados estilo socket, o gerenciamento de estado transacional e o roteiro técnico para conectar o código diretamente a um braço/garra robótica física.

---

## 1. Visão Arquitetural em Camadas (Modular Socket-Style)

O sistema adota o padrão **Ports & Adapters (Hexagonal)** combinado com um barramento TCP loopback opcional (`SocketCommandBridge`):

```text
+-------------------------------------------------------------------------+
|                        CENARIOS/ & CLI (main.py)                        |
|  test_basic_movement | fog | dirty_surface | object_tracking | stress   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    CODIGO/controller.py (Orquestrador)                  |
|  • Despacho direto ou via Socket TCP JSON (CommandPacket/CommandResult) |
|  • Pipeline Percepção -> Planejamento -> Pick-and-Place                 |
+-----------------+-----------------------------------+-------------------+
                  |                                   |
                  v                                   v
+-----------------------------------+   +---------------------------------+
|   FUNCOES/gripper_commands.py     |   |   FUNCOES/api_connector.py      |
|  8 primitivas independentes:      |   |  Google Gemini Vision /         |
|  move_right/left/forward/backward |   |  Planejamento + Retry/Backoff   |
|  move_up/down, close, open        |   |  + Fallback Determinístico      |
+-----------------+-----------------+   +---------------------------------+
                  |
                  v
+-----------------------------------+
|     FUNCOES/gripper_base.py       |
|  • Validação física de parâmetros |
|  • Timeout isolado por thread     |
|  • StateManager (checkpoint/undo) |
+-----------------+-----------------+
                  |  (GripperBackendProtocol)
        +---------+---------+
        |                   |
        v                   v
+---------------+   +-----------------------------------+
| MuJoCoSim     |   | Driver de Hardware Real (Futuro)  |
| (simulator.py)|   | UR / Franka / Robotiq / ROS 2     |
+---------------+   +-----------------------------------+
```

---

## 2. Contratos e Estruturas Principais

### 2.1 Protocolo de Backend (`GripperBackendProtocol`)
Definido em `FUNCOES/gripper_base.py`, exige apenas três métodos para acoplar qualquer simulador ou robô físico:

1. `apply_cartesian_delta(dx, dy, dz, speed, force, duration) -> Dict[str, Any]`
2. `apply_jaw_action(close_jaws, speed, force, duration) -> Dict[str, Any]`
3. `sync_state() -> Dict[str, Any]`

### 2.2 Gerenciamento Transacional de Estado (`StateManager`)
Localizado em `FUNCOES/utils.py`, mantém em memória thread-safe (`RLock`):
- `position`: `{"x": float, "y": float, "z": float}` em metros.
- `orientation`: `{"roll": float, "pitch": float, "yaw": float}` em graus.
- `grip_force`: força aplicada pelos dedos em Newtons (`N`).
- `aperture`: abertura linear entre os dedos em metros (`0.00m` a `0.08m`).
- `is_grasping` e `grasped_object`: estado de preensão.
- **Checkpointing & Rollback**: antes de cada comando, `save_checkpoint()` grava o estado. Caso ocorra `GripperTimeoutError`, `InvalidParameterError` ou falha de comunicação no backend, `record_error(rollback_to=checkpoint)` restaura atomicamente o estado seguro anterior.

### 2.3 Comunicação via Socket TCP (`SocketCommandBridge`)
Para simular fielmente arquiteturas industriais distribuídas (onde o planejador roda em uma estação com GPU/IA e o controlador de tempo real roda em um CLP ou PC embarcado):
- **Formato de Envio (`CommandPacket`)**: linha JSON terminada por `\n`:
  ```json
  {"action": "move_right", "speed": 0.2, "force": 20.0, "duration": 0.5, "packet_id": "a1b2c3d4e5", "timestamp": 1763225000.0, "extra": {}}
  ```
- **Formato de Resposta (`CommandResult`)**: linha JSON contendo `success`, `status`, `position`, `grip_force`, `aperture`, `elapsed_ms` e `error`.

---

## 3. Pipeline de Percepção e Compensação Ambiental

1. **Neblina Atmosférica (`set_fog`)**:
   - No modelo MuJoCo nativo, ajusta dinamicamente `vis.map.fogstart` e `vis.map.fogend`.
   - No pipeline óptico RGB, aplica a lei exponencial de atenuação de Beer-Lambert $T(d) = e^{-\beta d}$ combinada com espalhamento Gaussiano, reduzindo o contraste RMS da imagem.
2. **Sujeira e Ruído Visual (`set_surface_dirt`)**:
   - Injeta manchas elípticas opacas simulando óleo/barro na bancada e ruído impulsivo *salt-and-pepper* simulando partículas na lente, monitorado via métrica **PSNR (dB)**.
3. **Integração Google Gemini API (`GoogleAPIConnector`)**:
   - Codifica o quadro PNG em Base64 e solicita resposta estruturada (`responseMimeType: "application/json"`) contendo coordenadas estimadas dos objetos, nível de visibilidade e próxima ação recomendada.

---

## 4. Guia Passo a Passo de Migração Sim-to-Real (Simulação -> Hardware)

Para migrar este projeto do MuJoCo para uma garra física (por exemplo, **Robotiq 2F-85** montada em um **Universal Robots UR5e** ou barramento **ROS 2 / Serial**):

### Etapa 1: Implementar um `HardwareGripperBackend`
Crie um arquivo `CODIGO/hardware_backend.py` implementando `GripperBackendProtocol`:

```python
import socket
import json
from typing import Any, Dict


class URRobotiqBackend:
    """Adaptador de hardware real conectando via TCP/IP ao controlador do robô."""

    def __init__(self, robot_ip: str = "192.168.1.100", port: int = 30002) -> None:
        self.robot_ip = robot_ip
        self.port = port

    def _exchange(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with socket.create_connection((self.robot_ip, self.port), timeout=3.0) as sock:
            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            raw = sock.recv(4096).decode("utf-8")
            return json.loads(raw)

    def sync_state(self) -> Dict[str, Any]:
        return self._exchange({"cmd": "read_state"})

    def apply_cartesian_delta(
        self, dx: float, dy: float, dz: float, speed: float, force: float, duration: float
    ) -> Dict[str, Any]:
        return self._exchange({
            "cmd": "movel_rel",
            "delta": [dx, dy, dz],
            "speed": speed,
            "force_limit": force,
            "duration": duration,
        })

    def apply_jaw_action(
        self, close_jaws: bool, speed: float, force: float, duration: float
    ) -> Dict[str, Any]:
        return self._exchange({
            "cmd": "gripper_jaw",
            "close": close_jaws,
            "speed": speed,
            "force": force,
        })
```

### Etapa 2: Calibrar Especificações e Câmera Real
1. Ajuste `GripperSpecs` em `CODIGO/config.py` com os limites reais de segurança da célula robótica (`workspace_x`, `workspace_y`, `workspace_z`, `max_force`).
2. Substitua `simulator.render_frame()` por um frame capturado de uma câmera industrial (ex.: Intel RealSense / USB OpenCV) convertendo a matriz BGR/RGB para `PIL.Image`, mantendo a calibração extrínseca câmera-bancada (homografia olho-na-base ou olho-na-mão).

### Etapa 3: Injetar o Backend no `GripperBase`
```python
from CODIGO.config import GripperSpecs
from FUNCOES.gripper_base import GripperBase
from FUNCOES.gripper_commands import move_down, close, move_up

real_backend = URRobotiqBackend(robot_ip="192.168.1.100")
gripper = GripperBase(specs=GripperSpecs(max_speed=0.25), backend=real_backend)

# As mesmas 8 funções agora controlam o robô físico com proteção de timeout e logging!
move_down(speed=0.08, duration=1.0, gripper=gripper)
close(force=25.0, gripper=gripper)
move_up(speed=0.08, duration=1.0, gripper=gripper)
```

### Etapa 4: Checklist de Segurança Pré-Operação Física
- [ ] Reduzir `max_speed` inicial para `<= 0.10 m/s` durante o comissionamento.
- [ ] Validar botão físico de parada de emergência (E-Stop) antes do primeiro comando.
- [ ] Executar `python main.py --test basic_movement` apontando para o robô em espaço livre sem obstáculos.
- [ ] Confirmar orientação dos eixos (`+X` direita, `+Y` frente, `+Z` cima) entre o referencial da garra e o controlador físico.
