# Pesos e calibração

O servidor carrega pesos locais `odontologia.pt` e o manifesto homônimo `odontologia.json`. O manifesto contém `id`, `sha256`, `classes`, versões e métricas. O script de treinamento produz ambos. Arquivos `.pt` não entram no Git; copie os pesos junto com o manifesto ao implantar em outra máquina.

`calibration.json` define a câmera e o ponto de preensão relativo à caixa para cada classe. A projeção usa o plano do ponto de referência do objeto e a altura do TCP separadamente. A câmera deve coincidir com posição, rotação, intrínsecos e resolução do arquivo.

Os scripts geram `validated_for_pick: false` nos dois manifestos. Só marque como `true` após avaliar o modelo no conjunto reservado e verificar erro e captura física em poses representativas. Registre o relatório usado na validação. Os artefatos v2 entregues já têm esse marcador para o perfil experimental de `VISION_VALIDATION.md`, com limites em `validation_scope`. A flag CLI `--vision-enable-pick` é uma terceira condição necessária. Detecção e preview funcionam sem liberar movimento.

Os parâmetros iniciais ou ajustados por dados sintéticos não comprovam a captura nem o uso em uma câmera real. Se alterar pesos, câmera, cena ou política de preensão, gere uma nova versão e repita a avaliação.
