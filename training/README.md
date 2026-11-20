# training/ — Fast-SCNN 96px (v5) — treino e exportação

Notebook de treino, arquitetura, modelos exportados e resultados do modelo
de segmentação **gato / cachorro / fundo** (3 classes, entrada 96×96×3).

> Mantido conforme produzido pelo autor (ago/2026): notebook **com** os
> outputs da execução original, modelos e plots originais — nada foi
> limpo ou regenerado.

## Conteúdo

| Item | Descrição |
|---|---|
| `gatos_cachorro_unet_v5_fast_scnn_96px.ipynb` | treino completo + exportação (ONNX via tf2onnx, ONNX int8 dinâmico, TFLite fp32, TFLite **int8 full**) |
| `tf_2_0_fast_scnn.py` | arquitetura Fast-SCNN (`build_fast_scnn`) — deve ficar **ao lado do notebook** (o notebook faz `from tf_2_0_fast_scnn import build_fast_scnn`) |
| `models/` | artefatos exportados + `MD5SUMS` |
| `results/` | histórico de treino e visualizações de val/teste (Keras, TFLite fp32 e int8) |

## Métricas (execução original, conjunto de teste)

| Runtime | IoU multiclasse |
|---|---|
| Keras (fp32, treino) | 0,8267 |
| ONNX int8 dinâmico | 0,8339 |
| TFLite fp32 | 0,8347 |
| **TFLite int8 full (o modelo embarcado)** | **0,8207** |

Parâmetros: 1,204 M · pyramid bins `(1, 3)` · normalização ImageNet
`(px/255 − mean)/std`, mean=(0,485, 0,456, 0,406), std=(0,229, 0,224, 0,225).

## ⚠️ Atenção: dois modelos int8 diferentes

| Arquivo | MD5 | O que é |
|---|---|---|
| `models/model_gatos_cachorro_unet_v5_fast_scnn_96px_infer_int8_full.tflite` | `179689ebac29524d449141e81a9c97d1` | **original de treino** (shapes 141/142 inválidas para TFLM) |
| `../firmware/data/gatos_v5_fast_scnn.tflite` | `449f7358985e6f3bd59aedc57272cb5e` | **o modelo embarcado** — int8 com o patch de shapes `SPACE_TO_BATCH_ND`/`BATCH_TO_SPACE_ND` (batch 1→16), necessário para runtimes com shapes estáticas (ver [diagnóstico das shapes](../docs/TFLM_DEBUG_PLAYBOOK.md#known-regressions)) |

Não sobrescreva um pelo outro. O LiteRT no PC produz a mesma saída para os
dois; o TFLite Micro no ESP32 exige a versão patcheada.

## Dataset de treino

O notebook original referencia um dataset privado. Para usar seu próprio
dataset, ajuste o caminho para o `manifest.csv` no notebook. O arquivo deve
ter as colunas `img_path` e `msk_path`, com uma imagem e sua máscara por
linha. Os valores das máscaras são `0` para fundo, `1` para gato e `2`
para cachorro.
