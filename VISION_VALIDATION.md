# Validação da integração YOLO

Data: 23/11/2026. Perfil: experimental, exclusivamente no simulador MuJoCo nativo. Este relatório registra observações locais; não certifica câmera ou robô físico.

## Correção de expiração em CPU — 24/11/2026

No backend aberto, foi observado processamento de 284,21 ms (inferência de 212,07 ms) com validade máxima de 300 ms. A seleção de dispositivo do Ultralytics redefine o número de threads; o adaptador agora aplica o limite antes de carregar os pesos e em `on_predict_start`, antes de cada inferência, incluindo o aquecimento. O loop físico também cede execução ao asyncio quando excede seu orçamento. A expiração informa idade, limite e tempos de processamento, e o painel preserva a indicação de resultado expirado.

Verificação após a correção: **32 testes passaram** (`tests/test_vision.py` e `scenarios/`, 10,39 s), além de compilação Python, sintaxe JavaScript, regeneração do bundle e `git diff --check`. Uma execução concorrente anterior falhou em dois critérios de latência; a repetição sem o ensaio YOLO concorrente passou. O primeiro ensaio de captura também excedeu o prazo de inicialização antes de produzir quadros; o diagnóstico local mostrou carregamento de dependências de torchvision/PyTorch no aquecimento.

Uma nova captura completa do instrumento dental concluiu com o limite original de 300 ms; a última inferência levou 22,51 ms e o processamento 66,23 ms. Registro: `artifacts/vision_evaluation/pick_expiry_fix_dental.json`. Um ensaio separado com validade de 1000 ms também concluiu, mas não foi necessário ampliar o limite para obter a captura com o código corrigido. Esses resultados se referem à pose inicial em instância isolada; o processo já aberto precisa ser reiniciado para receber as alterações.

A captura completa do suporte de fio dental também concluiu com o limite original de 300 ms: última inferência de 26,82 ms e processamento de 52,64 ms. Registro: `artifacts/vision_evaluation/pick_expiry_fix_floss.json`.

## Atualização: visão sob demanda

A regressão do fluxo sob demanda passou em **29 testes**. O cenário nativo de transporte/ciclo de vida usa um detector substituto para continuar funcionando sem os pesos opcionais; os ensaios físicos citados nesta seção usam o YOLO real. A sintaxe JavaScript e o bundle também foram verificados.

O fluxo atual exige um comando por classe, por exemplo `{"command":"vision_pick","class_name":"dental_tool"}`. Habilitar o serviço e consultar seus endpoints não captura imagens. A coleta é finita e o braço calcula o plano antes de executar. A revalidação usa uma segunda solicitação pontual. Os testes atuais verificam repouso antes/depois da solicitação, ausência de movimento sem detecção e descarte de resultados cancelados. Os resultados de treinamento e qualidade óptica abaixo continuam válidos; as medidas do antigo streaming foram identificadas como históricas.

Foram repetidas operações reais no simulador, sem observação prévia: instrumento dental e suporte de fio dental concluíram detecção, planejamento, aproximação, revalidação, preensão e entrega. O instrumento usou nove quadros no total, distribuídos entre as duas solicitações finitas; o suporte usou seis. Ao fim de cada operação, a última observação já tinha cerca de dez segundos e o contador de quadros permanecia parado, confirmando ausência de captura durante transporte/entrega.

Registros locais: `artifacts/vision_evaluation/pick_on_demand_dental.json` e `pick_on_demand_floss.json`. A primeira solicitação também carrega o modelo, com prazo de até 60 segundos. Quadros cujo processamento excede a idade máxima não encerram antecipadamente a coleta: tenta-se outro quadro dentro do limite finito. Ao esgotar o limite, a operação é recusada. Ensaios concorrentes com carga produziram recusas por expiração; a confirmação física final foi feita isoladamente.

## Entrega

Implementados câmera fixa inclinada, inferência opcional em worker dedicado, geometria planar, estabilidade/expiração, API e WebSocket, preview com caixas, seleção de alvo e captura cancelável. Há ferramentas de geração, calibração, treinamento, avaliação e benchmark. O modo manual continua disponível sem instalar YOLO.

A câmera mantém o nome `vision_top`, mas usa uma vista frontal inclinada. O ensaio inicial com câmera vertical mostrou oclusão dos instrumentos pelo antebraço; essa configuração foi substituída e o modelo foi ajustado com imagens de poses variadas do braço. A aproximação passa por um ponto atrás do instrumento antes da revalidação, preservando a visibilidade.

Os pesos padrão locais são `models/vision/odontologia.pt`, equivalentes a `odontologia-v2.pt`. Manifestos, configuração e versões estão em `models/vision/`. Os binários e datasets são ignorados pelo Git: devem acompanhar o projeto como artefatos ao implantar em outra máquina.

## Ambiente e treinamento

Windows, Intel Core i5-1235U, 12 processadores lógicos, Python 3.11.4, MuJoCo 4.0.1, Ultralytics 8.4.216 e PyTorch 2.15.0 CPU. A lista completa está em [runtime-versions.json](models/vision/runtime-versions.json).

Modelo compacto YOLO com aproximadamente 151 mil parâmetros, cabeça de detecção de stride 8 e entrada 320. A câmera fornece RGB 640×480; o adaptador converte para BGR e usa letterbox quadrado (`rect=False`). As caixas retornadas ficam no sistema de pixels da imagem original.

Dataset atual: `datasets/odontologia-v2`, 200 imagens independentes, sendo 140 de treino, 30 de validação e 30 de teste. Classes: `dental_tool` e `floss_holder`. As caixas usam máscaras das malhas visuais, sem proxies de colisão. Há negativos, variação de posição, pequena variação angular, fundo e poses cinemáticas do braço. Os splits são por configuração de cena, não por quadros adjacentes.

O modelo v2 foi ajustado por 25 épocas a partir dos pesos v1, com batch 8, AdamW, learning rate inicial 0,002, `nbs=8` e duas threads. Os pesos v1 resultaram de 50 épocas no conjunto inicial de 120 imagens e foram preservados localmente para reprodução. Os manifestos registram hashes, sementes, inicializador e versão da cena. A seleção entre `best` e `last` usou a validação com o pré-processamento da produção, sem usar o teste para selecionar pesos.

Houve suspensão/interrupção do ambiente durante o ajuste. Os tempos acumulados no CSV de treinamento incluem esse intervalo e não representam tempo efetivo de computação.

## Detecção e localização no teste reservado

| Medida | Resultado |
| --- | --- |
| Imagens / instrumentos visíveis no teste | 30 / 44 |
| mAP@50 | 0,9941 |
| mAP@50:95 | 0,7037 |
| Precisão com confiança ≥ 0,60 e IoU ≥ 0,50 | 100%: 39 correspondências, zero falsos positivos |
| Recall nesse mesmo limiar | 88,6%: 39 de 44 |
| Correspondências com posição estimável após filtros geométricos | 38; uma rejeitada |
| Erro XY P95 / máximo nessas 38 correspondências | 5,60 mm / 8,82 mm |

O mAP resume a curva de precisão/recall e não equivale ao recall no limiar operacional de 0,60. O erro XY usa o ponto real apenas como referência de avaliação. O planejador da captura recebe exclusivamente a estimativa visual. A altura do TCP é uma política por classe, não uma estimativa de profundidade do YOLO.

Os resultados numéricos completos estão nos artefatos locais `artifacts/vision_evaluation/report_v2.json` e no manifesto dos pesos. O conjunto é pequeno e sintético; esses números não demonstram generalização para imagens reais.

## Captura física no simulador

Ensaios com orientação inicial e translações aleatórias de até ±15 mm em X/Y, semente 20261123:

| Caso | Solicitações | Concluídas | Sem alvo elegível |
| --- | --- | --- | --- |
| Instrumento dental, pose inicial | 1 | 1 | 0 |
| Instrumento dental, três variações | 3 | 2 | 1 |
| Suporte de fio dental, três variações | 3 | 3 | 0 |
| Total principal | 7 | 6 | 1 |

As seis capturas iniciadas terminaram com fechamento confirmado por contato/identidade, elevação, transporte, abertura e recuo. Uma posição do instrumento não foi detectada acima do limiar operacional; a captura não foi iniciada. A repetição isolada confirmou esse caso, portanto ele não é atribuído apenas à carga concorrente. Não foi reduzido o limiar para forçar sucesso.

Registros locais: `pick_dental_tool_v2.json`, `pick_dental_tool_isolated.json` e `pick_floss_holder_v2.json`, em `artifacts/vision_evaluation/`. Imagens de diagnóstico acompanham os relatórios. O total acima não conta a repetição isolada como novas posições independentes.

Os manifestos do modelo v2 e da calibração têm `validated_for_pick: true` para esse perfil experimental limitado. Isso permite usar `--vision-enable-pick`; não declara cobertura de toda a bancada, orientações livres, objetos novos ou hardware. Novos pesos/calibrações saem dos scripts com essa propriedade em `false` e precisam de nova avaliação.

## Desempenho do modo contínuo anterior (histórico)

O benchmark compara o mesmo servidor sem visão e com câmera/YOLO após aquecimento, em uma amostra local de cinco segundos por modo, sem clientes WebSocket e sem os ensaios físicos concorrentes. A medição final fica em `artifacts/vision_evaluation/benchmark_v2.json`.

| Medida | Resultado |
| --- | --- |
| Física média sem visão / com visão | 60,00 / 60,01 Hz |
| Física mínima com visão | 59,8 Hz |
| Cadência de visão observada | 9,4 quadros/s |
| Inferência P95 | 29,31 ms |
| Idade da observação P95 na consulta | 98,57 ms |

A diferença de FPS está dentro da variação dessa amostra curta; não comprova ganho de desempenho nem substitui um teste prolongado sob carga.

O serviço não acumula capturas pendentes e mantém no máximo quatro imagens/resultados. Captura e inferência ocorrem fora do loop físico; a cópia de estado usa lock breve. Durações e validade usam relógio monotônico de alta resolução.

## Verificações e reprodução anteriores à mudança para solicitações

Resultado da execução final: **24 testes passaram em 18,16 s**. Compilação dos módulos Python, sintaxe dos três scripts JavaScript alterados, consistência das dependências (`pip check`) e `git diff --check` também passaram. O bundle foi regenerado.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_vision.py scenarios/ -q
.\.venv\Scripts\python.exe scripts/evaluate_vision.py
.\.venv\Scripts\python.exe scripts/benchmark_vision.py
.\.venv\Scripts\python.exe scripts/evaluate_vision_pick.py --class-name dental_tool --trials 3 --position-jitter 0.015
.\.venv\Scripts\python.exe scripts/evaluate_vision_pick.py --class-name floss_holder --trials 3 --position-jitter 0.015
.\.venv\Scripts\python.exe scripts/bundle_frontend.py
```

Os testes de visão verificam projeção e convenções de eixos, calibração/escala, pesos ausentes/hash inválido, estabilidade, idade, revisão da cena, cache limitado, propriedade do contexto gráfico, reinício, cancelamento e revalidação. O cenário nativo exercita câmera, JPEG, HTTP, reset, reconexão WebSocket e rejeição de comando inválido. Os cenários existentes cobrem movimentos, interação física, sincronização e estresse.

O painel foi inspecionado no navegador com câmera e bloqueio de captura indisponível. A imagem nativa da nova aproximação também foi inspecionada para confirmar visibilidade dos instrumentos.

## Limites e uso

- Perfil restrito à cena, câmera e classes atuais, com bancada estática durante alinhamento/descida e orientação próxima da inicial.
- A captura pode ser recusada por confiança, estabilidade, escala, oclusão, expiração, revisão ou revalidação; o modo manual permanece disponível.
- O ponto travado durante oclusão é usado por no máximo seis segundos para alinhamento/descida. Reset ou cancelamento interrompem a progressão sem abrir automaticamente os dedos.
- Webcam, profundidade, rastreamento persistente, orientação livre do TCP e validação em robô real permanecem evoluções fora deste perfil.

Iniciar somente a percepção:

```powershell
.\.venv\Scripts\python.exe main.py --vision --port 8000
```

Habilitar também a captura experimental pelo painel:

```powershell
.\.venv\Scripts\python.exe main.py --vision --vision-enable-pick --port 8000
```
