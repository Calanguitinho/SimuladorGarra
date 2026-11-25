# Plano de integração do YOLO ao SimuladorGarra

Data: 23/11/2026. Este documento registra o plano original e os critérios de aceitação. A seção seguinte acompanha sua implementação; as seções de proposta descrevem o desenho e também evoluções ainda pendentes.

## Fluxo atual: visão acionada pelo braço

A visão contínua do desenho inicial foi substituída por solicitações pontuais. O fluxo implementado é **comando para pegar uma ferramenta → solicitação de detecção por classe → retorno validado → cálculo do plano → execução**. A interface permite escolher a ferramenta antes de existir qualquer imagem ou detecção. Os botões principais também usam esse fluxo e não leem posições reais dos objetos.

Em repouso, o worker espera um evento: não há captura, inferência nem atualização automática da imagem. Cada solicitação coleta até seis quadros com a configuração padrão, encerrando ao obter três observações estáveis. Uma revalidação após a aproximação é outra solicitação finita. Status/preview são consultas de resultados, não gatilhos de captura. O plano calculado é publicado no status da operação antes do primeiro comando físico.

`vision_pick` recebe `class_name`; referências a quadro/detecção permanecem internas ao servidor. `vision_detect` permite diagnóstico pontual e `vision_cancel`/`reset` descartam solicitações e resultados tardios. O levantamento e os exemplos abaixo registram o plano inicial; o contrato atualizado está em `PROTOCOL.md`.

## Estado da implementação

Implementados: `VisionConfig` e flags CLI; câmera fixa na cena nativa; cópia de estado e renderização no worker; detector opcional com pesos locais/hash/classes; geometria planar e ajuste de ponto por classe; estabilidade e expiração; cache limitado de JPEGs/resultados; endpoints e mensagens de visão; orquestração de captura e cancelamento; painel com overlay; bundle; geração sintética, treino, calibração e avaliação; testes unitários e cenário real de câmera/HTTP/reset.

A geração inicial produziu 120 imagens. Após verificar oclusão do braço na câmera vertical, foi criado `datasets/odontologia-v2` com 200 imagens (140 treino, 30 validação e 30 teste), câmera fixa inclinada e poses variadas do braço. As anotações vêm das malhas visuais. Os dados grandes são locais e ignorados pelo Git. A calibração foi ajustada apenas no treino. Modelo v2 e calibração estão validados para um perfil experimental limitado no simulador; métricas, capturas e recusas estão registradas em `VISION_VALIDATION.md`.

Evoluções fora da primeira versão: fonte webcam, profundidade, rastreador persistente, câmeras móveis, preensão com orientação livre, planejador genérico de colisões e validação em hardware real. O arquivo `.mjcf` legado não recebeu câmera porque a visão atual depende da cena nativa construída por `clinic_model.py`.

## 1. Objetivo e escopo

Reconhecer os instrumentos na bancada por imagem, estimar um ponto de preensão em coordenadas do simulador e executar a captura com o controlador existente. A primeira versão deve funcionar com uma câmera fixa do MuJoCo, os objetos `dental_tool` e `floss_holder` e inferência local em Python.

Entregas em ordem: imagem da câmera, detecções na interface, localização 3D validada e captura orientada por visão. Webcam, câmera no punho, profundidade de sensor real e controle de hardware ficam para uma evolução posterior, com calibração própria.

O modo manual e a demonstração atual continuam disponíveis. O modo de visão deve ser opcional e identificado na interface. Sem dependências de visão, o servidor deve continuar iniciando no modo atual.

## 2. Levantamento inicial (histórico)

| Componente | Comportamento observado | Consequência para a integração |
| --- | --- | --- |
| `main.py` | Encaminha a execução normal ao backend; mantém caminhos legados de CLI | Integrar primeiro em `backend/`, sem duplicar a implementação no legado |
| `backend/mujoco_env.py` | No modo nativo, `_load_model()` chama `compile_clinic_model()` | O modelo efetivo é o construído por `clinic_model.py` |
| `backend/clinic_model.py` | Constrói a cena UR5e/Robotiq usando `MjSpec` | Adicionar a câmera neste construtor |
| `backend/models/gripper.mjcf` | É validado como XML na inicialização; não é a origem da cena UR5e nativa atual | Alterar somente esse arquivo não adiciona a câmera ao UR5e |
| `backend/mujoco_env.py` | Tem fallback cinemático sem renderização nativa | Declarar visão indisponível nesse fallback, salvo implementação futura de outra fonte |
| `backend/websocket_server.py` | Integra física e transmite estado no mesmo loop assíncrono | Renderização, codificação de imagem e inferência não podem bloquear esse loop |
| `backend/gripper_controller.py` | Já aceita `move_to`, `open`, `close` e `reset` | Reutilizar essas operações, acrescentando validação e orquestração da captura |
| `frontend/js/controls.js` | `runDemoPickAndPlace()` faz reset, lê `objects[].grasp_pos` e usa esperas fixas | Criar um fluxo visual separado, sem usar a posição real como estimativa |
| `frontend/js/websocket_client.js` | Processa pacotes com `state`; não tem callback específico de visão | Acrescentar tratamento explícito das novas mensagens |
| `frontend/js/dental_scene.js` | Renderiza a apresentação Three.js, incluindo elementos visuais da clínica | Essa imagem difere da câmera MuJoCo; não assumir equivalência de pixels |
| `scripts/bundle_frontend.py` | Empacota scripts clássicos; mantém a cena Three.js externa | Incluir os novos recursos de visão e verificar o modo standalone |

Há também código de percepção e cenários legados em `CODIGO/`, `FUNCOES/` e `CENARIOS/`. A primeira integração deve usar a arquitetura ativa `backend/` + `frontend/` + `scenarios/`. Uma adaptação do legado pode reutilizar os contratos de visão depois, sem tornar essa migração pré-requisito.

## 3. Arquitetura proposta

```text
MuJoCo: física e comandos
          | cópia consistente de estado + identificação do quadro
          v
Worker de visão: renderização RGB -> YOLO -> geometria -> último resultado
          |                                      |
          | imagem + detecções                    | alvo validado
          v                                      v
HTTP/WebSocket -> painel de visão          orquestrador de captura
                                                 |
                                                 v
                                     controlador existente -> MuJoCo
```

O worker deve ser o único proprietário do contexto gráfico e da instância do detector. Criar, usar e liberar o contexto na mesma thread ou processo. Uma tarefa `asyncio` executando inferência síncrona continua bloqueando o servidor; é necessário um executor dedicado ou processo separado.

Para o primeiro protótipo, usar um worker dedicado com cópia de `MjData`, sincronizada sob o lock do ambiente. Liberar o lock antes de renderizar ou inferir; recalcular os dados derivados necessários na cópia. O worker não deve acessar o `MjData` vivo durante `mj_step`. Se as medições mostrarem interferência relevante no loop físico, migrar o worker para processo separado.

Usar uma fila de capacidade 1, substituindo o quadro pendente pelo mais recente. Somente o controlador altera os atuadores. O worker produz observações e candidatos, sem comandar o robô diretamente.

No reset ou reconstrução do modelo, incrementar `scene_revision`, descartar filas e resultados antigos e reconstruir os recursos do worker quando necessário. No encerramento, parar a captura, aguardar o worker e liberar câmera, renderer e memória do modelo.

## 4. Inventário de alterações

Os nomes novos são uma organização sugerida; podem ser ajustados durante a implementação sem mudar as responsabilidades.

| Arquivo | Tipo | Trabalho necessário |
| --- | --- | --- |
| `requirements-vision.txt` | Novo | Declarar dependências opcionais de visão; incluir o requirements base e selecionar versões verificadas |
| `requirements.txt` | Revisar | Preservar instalação sem YOLO; registrar a combinação de versões base validada |
| `backend/config.py` | Alterar | Adicionar `VisionConfig`, leitura de configuração, validação e valores padrão |
| `backend/main.py` | Alterar | Expor opções de visão e passá-las à configuração; mensagens claras de disponibilidade |
| `backend/clinic_model.py` | Alterar | Criar câmera fixa nomeada, iluminação e configuração de renderização; conferir visibilidade dos instrumentos |
| `backend/models/gripper.mjcf` | Condicional | Adicionar câmera apenas se esse modelo também passar a ter suporte real de renderização; documentar a diferença |
| `backend/mujoco_env.py` | Alterar | Expor cópia consistente para renderização, revisão da cena, pose da câmera e metadados; liberar recursos |
| `backend/vision_camera.py` | Novo | Encapsular renderização e fornecer RGB, profundidade opcional e metadados do mesmo quadro |
| `backend/vision.py` | Novo | Adaptador YOLO: carregar pesos uma vez, aquecer modelo e normalizar detecções |
| `backend/vision_geometry.py` | Novo | Transformação câmera/mundo, validação da região de interesse e estimativa do ponto de preensão |
| `backend/vision_service.py` | Novo | Gerenciar worker, fila, cache, erros, métricas e ciclo de vida |
| `backend/vision_pick.py` | Novo | Máquina de estados de captura, cancelamento e exclusão de comandos concorrentes |
| `backend/gripper_controller.py` | Alterar | Reforçar validação de alvos, limites e integração com o orquestrador, preservando comandos existentes |
| `backend/websocket_server.py` | Alterar | Iniciar/parar serviço, publicar resultados, disponibilizar imagem e encaminhar captura/cancelamento |
| `frontend/js/websocket_client.js` | Alterar | Eventos de visão e captura, erros, reconexão e validade das observações |
| `frontend/js/vision_panel.js` | Novo | Imagem, overlay, seleção do alvo e apresentação de estado/métricas |
| `frontend/index.html` e `frontend/css/style.css` | Alterar | Painel de câmera, caixas, indicadores e botões de detecção/captura/cancelamento |
| `frontend/js/controls.js` | Alterar | Separar demonstração por estado e captura por visão; evitar macros simultâneas |
| `frontend/js/dental_scene.js` e `frontend/js/renderer.js` | Condicional | Marcador 3D do alvo estimado, se desejado; câmera de inspeção independente da câmera de visão |
| `scripts/bundle_frontend.py` | Alterar | Empacotar painel e estilos; resolver URLs da API quando backend e página estiverem em hosts diferentes |
| `frontend/dist/index.html` | Gerado | Regenerar após mudanças no frontend, sem editar manualmente |
| `models/vision/` | Novo | Manifesto de pesos: classes, versão, hash, origem e métricas; política de armazenamento dos binários |
| `datasets/odontologia/` | Novo | Configuração e manifesto do dataset; imagens e anotações conforme política de armazenamento |
| `scripts/generate_vision_dataset.py` | Novo | Gerar imagens sintéticas e anotações, com sementes e metadados reproduzíveis |
| `scripts/train_yolo.py` | Novo | Treinamento reproduzível e exportação do manifesto do modelo |
| `scripts/evaluate_vision.py` | Novo | Medir detecção, localização, latência e desempenho de captura |
| `tests/test_vision_*.py` | Novos | Testes unitários de geometria, contratos, seleção, validade e ciclo de vida |
| `scenarios/test_vision_*.py` | Novos | Integração com renderização real, inferência e captura no simulador |
| `.gitignore` | Alterar | Excluir caches, resultados de treino e dados grandes; preservar configurações e manifestos |
| `.env.example` | Novo, se necessário | Exemplificar variáveis efetivamente suportadas, sem credenciais |
| `README.md`, `PROTOCOL.md`, `ARCHITECTURE.md`, `DEPLOYMENT.md` | Alterar | Instalação, arquitetura ativa, contratos, execução, operação e limitações |

## 5. Dependências, pesos e configuração

Adotar inicialmente a API Python do Ultralytics. A predição aceita imagens e retorna resultados estruturados como caixas, confiança e classe. Normalizar os resultados para listas e números Python antes do JSON, evitando transmitir tensores. Referência: [predição Ultralytics](https://docs.ultralytics.com/modes/predict/).

Adicionar `ultralytics` ao conjunto opcional. Usar OpenCV se houver calibração, homografia ou captura por webcam; evitar instalar simultaneamente variantes incompatíveis de OpenCV. Validar PyTorch em CPU primeiro e documentar a instalação específica para GPU/CUDA quando essa opção for testada. Não fixar versões por suposição: registrar versões exatas após validação com o Python 3.11 e o MuJoCo do ambiente.

O carregamento de dependências deve ser tardio quando a visão estiver habilitada. Pesos ausentes ou inválidos devem produzir erro de visão explícito. Não baixar pesos automaticamente durante um comando de captura.

Configuração proposta:

| Campo | Valor inicial proposto | Observação |
| --- | --- | --- |
| `enabled` | `false` | Preserva execução atual |
| `source` | `mujoco` | Webcam é uma extensão futura |
| `model_path` | `models/vision/odontologia.pt` | Caminho a ser provisionado, ainda inexistente |
| `camera_name` | `vision_top` | Nome a criar na cena nativa |
| `width`, `height` | `640`, `480` | Confirmar tamanho dos instrumentos em pixels |
| `imgsz` | `320` | Resolução inicial para CPU; combinar com a usada no treino |
| `device` | `cpu` | GPU opcional após teste |
| `target_fps` | `10` | Meta de captura/inferência, independente dos 60 Hz de estado |
| `confidence_threshold` | `0.60` | Ponto de partida; calibrar com validação |
| `max_age_ms` | `300` | Limite inicial de idade total da observação |
| `stable_frames` | `3` | Exigir consistência espacial e de classe |
| `allowed_classes` | Duas classes odontológicas | IDs devem coincidir com os pesos |
| `geometry_mode` | `plane` | Profundidade como segunda opção |
| `calibration_path` | A definir | Intrínsecos, extrínsecos, plano e versão |

Esses valores são hipóteses de partida, não resultados medidos. Validar combinações de resolução, FPS e idade máxima; se não forem atendidas, manter apenas detecção e bloquear captura automática.

Na documentação futura, os comandos de instalação devem usar `python -m pip` do ambiente virtual. Após criar o arquivo opcional, a instalação Windows será:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
```

Flags como `--vision` e `--vision-model` só poderão ser anunciadas como funcionais depois de implementadas em `backend/main.py`.

## 6. Câmera e transformação para o mundo

### 6.1 Imagem de entrada

Criar a câmera no `build_clinic_spec()` e garantir que ela veja a bandeja, os instrumentos e a região alcançável. A implementação usa vista fixa inclinada a partir da frente da bancada, mantendo o identificador `vision_top`; a vista vertical inicial foi substituída após comprovar oclusão do antebraço. A pose é versionada e a órbita do usuário no Three.js não a altera.

Verificar as malhas, grupos visuais e transparências na renderização nativa. A cena usa, por exemplo, um fundo físico de bandeja transparente; os elementos GLB da clínica exibidos no Three.js não são automaticamente parte da imagem MuJoCo. Ajustar a cena da câmera para que o dataset corresponda ao que o detector realmente recebe.

MuJoCo oferece renderização fora da tela; selecionar a API disponível na versão instalada e testar o contexto gráfico no Windows e no ambiente de deploy. Referência: [renderização MuJoCo](https://mujoco.readthedocs.io/en/latest/programming/rendering.html).

Definir explicitamente formato RGB/BGR, origem dos pixels, orientação vertical, dimensões e conversões. Para entrada NumPy no Ultralytics, conferir o contrato de canais da API e fazer uma única conversão no adaptador. Preservar a correspondência entre coordenadas das caixas e a imagem original, incluindo redimensionamento e letterbox.

### 6.2 Conversão geométrica

YOLO detecta regiões da imagem; a escolha de um ponto de preensão e sua conversão 3D são responsabilidades adicionais.

No MVP, usar raios da câmera intersectando um plano calibrado da bandeja, ou homografia para esse mesmo plano. Registrar intrínsecos, pose da câmera, convenção dos eixos, unidade em metros e plano de referência. A altura do objeto deve ser compensada; uma homografia da bancada não localiza corretamente qualquer ponto elevado.

Na evolução com profundidade, usar RGB e profundidade do mesmo quadro. Confirmar se a profundidade representa distância axial, distância ao longo do raio ou valor de buffer antes da retroprojeção. Rejeitar profundidade inválida e pontos fora da bancada.

Para instrumentos alongados, o centro da caixa não garante uma região agarrável. Definir uma política de ponto de preensão por classe e validar sua tolerância; se insuficiente, usar segmentação, caixas orientadas ou pontos-chave com anotações correspondentes. O controlador atual mantém uma orientação desejada do TCP; detectar o ângulo do instrumento não implica que o braço já possa executar qualquer orientação. No MVP, restringir a orientação dos objetos à faixa que a garra consegue capturar; ampliar orientação exige mudanças em IK, controle e validação de trajetória.

As posições reais `objects[].pos` e `grasp_pos` podem servir para geração de dados e avaliação. O fluxo de execução por visão não deve consultá-las para escolher o alvo nem como substituição silenciosa de uma detecção inválida.

## 7. Dataset e treinamento

Verificar primeiro se já existem pesos apropriados. Um detector genérico deve ser avaliado quanto às classes reais dos instrumentos antes de ser considerado suficiente.

Se necessário, produzir dataset com duas classes iniciais e imagens negativas. Variar posições, orientações, iluminação, fundo, oclusão pelo braço e presença simultânea dos objetos. Aumentar gradualmente a variedade sem incluir orientações que o controle ainda não suporta no conjunto de ensaios de captura.

O formato YOLO de detecção usa um YAML com os caminhos e nomes de classes e arquivos de anotação com classe e caixa normalizada. Seguir o [formato oficial de datasets](https://docs.ultralytics.com/datasets/detect/). Para anotações sintéticas, projetar malhas ou usar segmentação por objeto e validar visualmente as caixas; não assumir que os proxies de colisão correspondem à silhueta visual.

Separar treino, validação e teste por configurações de cena/sementes ou sessões, evitando quadros quase idênticos em conjuntos diferentes. Registrar sementes, versão da cena, câmera, anotação e pesos. Inspecionar imagens com as anotações sobrepostas antes de treinar.

Selecionar um modelo pequeno como baseline, ajustar com os dados odontológicos e medir precisão/recall por classe e mAP. Escolher o limiar de confiança com o conjunto de validação e fazer a avaliação final no teste reservado. Para webcam futura, incluir imagens reais e medir a diferença de domínio; treino sintético sozinho não comprova desempenho em câmera real.

## 8. Contratos HTTP e WebSocket propostos

Acrescentar mensagens independentes de visão, sem mudar o significado de `state.objects`. Exemplo ilustrativo de resultado:

```json
{
  "type": "vision",
  "schema_version": 1,
  "status": "ready",
  "scene_revision": 2,
  "frame_id": 125,
  "camera_id": "vision_top",
  "calibration_id": "vision-top-v1",
  "model_id": "odontologia-v1",
  "sim_time_s": 12.5,
  "age_ms": 85,
  "inference_ms": 42,
  "image_size": [640, 480],
  "detections": [
    {
      "detection_id": "125:0",
      "class_id": 0,
      "class_name": "dental_tool",
      "confidence": 0.91,
      "bbox_xyxy_px": [220, 180, 330, 220],
      "grasp_candidate": {
        "position_world_m": [0.25, -0.22, 0.024],
        "method": "plane",
        "valid": true,
        "reason": null
      }
    }
  ]
}
```

Os números acima são exemplos, não uma calibração válida. `detection_id` identifica uma observação de um quadro, não um rastreamento persistente. Para selecionar o mesmo objeto ao longo do tempo, implementar associação espacial/por classe ou adicionar um rastreador e seu identificador.

Usar relógio monotônico do servidor para expiração; não comparar diretamente `performance.now()` do navegador com o relógio Python. Calcular `age_ms` na publicação e revalidar na execução. Acrescentar timestamp UTC para logs se necessário, sem usá-lo como única base de validade.

| Interface proposta | Responsabilidade |
| --- | --- |
| `GET /api/vision/status` | Estado, versão do modelo/câmera, erro e métricas; funcionar também com visão desabilitada |
| `GET /api/vision/detections` | Último resultado, inclusive lista vazia |
| `GET /api/vision/frame?frame_id=125` | Imagem correspondente às caixas; responder explicitamente se o quadro já expirou |
| Mensagem WS `vision` | Resultado na cadência da visão, sem repetir imagens no estado a 60 Hz |
| Comando `vision_pick` | Receber revisão da cena, quadro e identificação do candidato; servidor revalida e resolve as coordenadas |
| Comando `vision_cancel` | Cancelar operação e informar a ação de parada ao cliente |
| Mensagem WS `vision_pick_status` | ID da operação, etapa, resultado e motivo de falha |

Manter cache curto e limitado de quadros/resultados para associar imagem e overlay. Não desenhar caixas de um quadro antigo sobre uma imagem nova. Definir limite de tamanho de resposta e frequência de consulta. Preview e API em outro host devem usar a URL do backend configurado, sem depender implicitamente da origem da página estática.

Atualizar `PROTOCOL.md` quando o contrato for implementado, incluindo exemplos de sucesso, ausência de objeto, alvo expirado, visão indisponível, cancelamento e operação ocupada. Clientes antigos devem continuar recebendo o estado atual sem precisar compreender visão.

## 9. Captura orientada por visão

Centralizar a operação no backend. A interface solicita a captura de uma observação; o servidor escolhe o alvo validado a partir do seu próprio resultado, sem confiar em coordenadas arbitrárias enviadas pelo navegador.

Máquina de estados proposta:

```text
ocioso -> buscando -> validando -> aproximando -> descendo
       -> fechando -> verificando -> elevando -> entregando -> concluído
```

Qualquer etapa pode terminar em falha ou cancelamento. Exigir confiança mínima, observação recente, estabilidade espacial, calibração válida, classe permitida e alvo alcançável. Bloquear comandos conflitantes durante a operação; reset e cancelamento devem interrompê-la de forma definida.

Substituir esperas fixas por condições observadas: distância do TCP ao alvo dentro de tolerância, velocidade reduzida e timeout. Revalidar a posição antes da descida; se o objeto tiver mudado, cancelar ou replanejar antes do contato. A implementação aproxima lateralmente, revalida e trava o ponto durante alinhamento/descida por no máximo seis segundos, quando a garra pode ocluir a câmera. Essa política exige bancada estática durante o intervalo e não persegue resultados atrasados.

Confirmar captura por contato/força e elevação bem-sucedida. O estado `is_grasping` atual pode apoiar essa validação no simulador, mas não fornece a estimativa visual do alvo. Limitar tentativas e registrar falhas. Cancelamento deve parar a progressão da macro e manter uma ação física definida para a etapa; não soltar automaticamente um objeto suspenso.

Reutilizar limites de velocidade, força e workspace. Validar valores finitos e alcançabilidade antes de `move_to`; não considerar um alvo aceito ou limitado por clamp como prova de captura possível. Começar com aproximação vertical sobre a bandeja e verificar colisões com braço, bordas e cena. Um planejador genérico de trajetórias é uma extensão adicional.

## 10. Etapas de implementação e critérios de conclusão

As etapas abaixo definem a ordem de trabalho e seus critérios; consulte o estado da implementação e `VISION_VALIDATION.md` para as evidências efetivamente obtidas. Um problema de geometria ou captura não deve ser mascarado pelo uso das posições reais.

| Etapa | Ações | Dependência | Critério de conclusão |
| --- | --- | --- | --- |
| 1. Baseline e decisões | Registrar versões/hardware, medir FPS e latência atuais, confirmar classes, câmera fixa e restrição de orientação | Nenhuma | Relatório reproduzível do estado atual e escopo fechado |
| 2. Configuração opcional | Criar requirements de visão, `VisionConfig`, flags e estados de disponibilidade | 1 | Servidor inicia sem visão; configuração inválida gera erro compreensível |
| 3. Câmera nativa | Alterar `clinic_model.py`, criar adaptador de câmera e salvar uma imagem real do renderer | 2 | Imagem correta dos dois instrumentos, escala/orientação conferidas e sem bloqueio relevante da física |
| 4. Worker e ciclo de vida | Copiar estado, limitar fila, implementar cache, reset e shutdown | 3 | Não há backlog, uso concorrente de `MjData` nem recursos gráficos abandonados |
| 5. Dataset | Gerar/coletar imagens, anotar, revisar e separar conjuntos | 3 | Anotações inspecionadas, classes consistentes e teste reservado |
| 6. Modelo | Treinar/ajustar, comparar baseline, registrar métricas e pesos | 5 | Modelo reconhece as duas classes no teste e tem manifesto reproduzível |
| 7. Inferência e preview | Integrar detector, API, WS e painel de imagem com overlay | 4 e 6 | Caixas correspondem ao frame exibido; falhas e lista vazia são tratadas |
| 8. Localização | Calibrar plano/câmera, estimar preensão, avaliar erro em metros | 7 | Erro dentro da tolerância mecânica definida, inclusive nas bordas da área útil |
| 9. Captura | Implementar máquina de estados, exclusão de comandos, limites e cancelamento | 8 | Captura usa apenas a estimativa visual e respeita os critérios de validade |
| 10. Robustez | Testar oclusão, expiração, múltiplos alvos, falhas do worker e desempenho | 9 | Cenários passam; sem degradação inaceitável do controle manual |
| 11. Entrega | Atualizar documentos, bundle e instruções de instalação/deploy | 10 | Instalação reproduzida em ambiente limpo e relatório final disponível |

As etapas 3 e 7 entregam demonstrações úteis antes de habilitar movimento. Se a precisão do ponto de preensão não for suficiente na etapa 8, revisar câmera, dataset ou tipo de modelo antes de seguir.

## 11. Plano de testes e métricas

| Grupo | Casos essenciais | Evidência |
| --- | --- | --- |
| Geometria | Pontos conhecidos, cantos da imagem, plano, altura, inversão vertical, RGB/depth alinhados, pontos fora da região | Erro médio, P95 e máximo de localização; erro no ponto de preensão |
| Detector | Duas classes, ausência de objeto, objeto parcial, oclusão e múltiplos objetos | Precisão/recall por classe e mAP no teste reservado |
| Contratos | Tensores convertidos, JSON finito, resultado vazio, frame expirado, revisão antiga e cliente antigo | Testes automatizados de serialização e validade |
| Concorrência | Reset durante inferência, fechamento/reabertura, worker lento, falha de renderer e comandos simultâneos | Sem deadlock; filas/memória limitadas; indisponibilidade explícita |
| Controle | Alvo distante, NaN/infinito, timeout, cancelamento em cada etapa, objeto não agarrado | Comando inválido rejeitado e resultado de operação consistente |
| Integração | Câmera real do simulador + pesos reais + captura em diferentes poses/sementes | Taxa de sucesso e motivos de falha por classe |
| Interface | Overlay redimensionado, imagem correta, reconexão, modo manual e standalone remoto | Verificação visual e registro dos fluxos |
| Desempenho | Baseline versus visão, CPU/GPU, inferência lenta e preview ativo | FPS físico, latência de comando, inferência P50/P95, idade total e memória |

Definir antes da etapa 9 as tolerâncias de posição e altura a partir da abertura, geometria dos dedos e região agarrável dos instrumentos. Como metas iniciais de desempenho, investigar perda de FPS acima de 10% do baseline e exigir idade dentro do `max_age_ms` configurado. Esses limites são propostas de engenharia, não garantias do ambiente atual.

Separar testes rápidos com detector simulado de testes de integração com pesos e contexto gráfico reais. O detector simulado testa contratos e controle de falhas; não comprova reconhecimento. Integrar os cenários novos ao runner em `scenarios/__init__.py` e ao modo `--scenario` caso devam ser executados por essa CLI. Evitar depender de download de pesos durante os testes.

## 12. Entrega, operação e conclusão

Antes da entrega, atualizar a arquitetura para distinguir o fluxo ativo do legado e documentar a diferença entre câmera MuJoCo e cena Three.js. Em deploy, registrar renderizador/contexto gráfico suportado, CPU/GPU, instalação dos pesos e caminhos de cache; expor separadamente saúde do servidor e disponibilidade da visão.

Guardar logs de versão de modelo, revisão da cena, quadro usado, idade da observação, alvo estimado e resultado de cada operação. Definir política de retenção para imagens e datasets; manter binários grandes em armazenamento apropriado e registrar seus hashes no manifesto.

Checklist de conclusão:

- [x] Câmera fixa renderiza a cena nativa e tem calibração versionada.
- [x] Dependências e pesos são provisionados de forma reproduzível.
- [x] Detector identifica as classes da cena e tem avaliação independente.
- [x] Ponto de preensão é estimado em metros com erro medido.
- [x] Inferência e preview não acumulam backlog nem bloqueiam o controle no benchmark local.
- [x] Imagem, caixas e alvo pertencem ao mesmo quadro e revisão da cena.
- [x] Visão indisponível ou expirada impede captura automática.
- [x] Captura usa observações visuais e confirma o resultado pelos sensores.
- [x] Cancelamento, reset, reconexão e comandos concorrentes foram verificados.
- [x] Modo manual, cenários existentes e bundle continuam funcionando.
- [x] README, protocolo, arquitetura e deploy refletem o comportamento implementado.

O checklist se refere ao perfil experimental do simulador. Ele não elimina as recusas de detecção observadas nem amplia a validação para câmeras ou robôs reais; os limites estão em `VISION_VALIDATION.md`.

## 13. Referências técnicas

Consultadas em 23/11/2026. São referências de API e formatos; as escolhas de arquitetura e metas deste plano são propostas específicas para o repositório.

- [Ultralytics: predição e resultados](https://docs.ultralytics.com/modes/predict/).
- [Ultralytics: datasets de detecção](https://docs.ultralytics.com/datasets/detect/).
- [MuJoCo: renderização](https://mujoco.readthedocs.io/en/latest/programming/rendering.html).
