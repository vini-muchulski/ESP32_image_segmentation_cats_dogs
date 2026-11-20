# Porte ESP-IDF + esp-nn + LittleFS — Gatos/Cachorros UNet v5 Fast-SCNN 96px

## Objetivo

Embarcar a inferência do modelo **Fast-SCNN** (INT8, `96×96×3` → `96×96×3`
logits, argmax de 3 classes: 0=fundo, 1=gato, 2=cachorro) no **ESP32-S3** com
ESP-IDF via PlatformIO. [Guia de instalação](../docs/QUICKSTART.md)

1. **TFLite Micro + esp-nn** — `components/` traz os dois como submodules
   upstream (`esp-tflite-micro` v1.3.8 + `esp-nn` v1.3.0) com patches locais
   mínimos; `-DESP_NN` está **ativo**. O build de referência (sem esp-nn) foi
   validado bit a bit contra o LiteRT; a revalidação da árvore otimizada está
   pendente.
2. **Modelo em LittleFS** carregado para PSRAM em runtime (sem header C).
3. **HTTP `esp_http_server`** com `POST /predict_bin` (entrada RGB raw de
   27.648 B), retorno de máscara ou logits e telemetria `X-*` (tempos,
   memória, arena e hashes FNV-1a).

### Motivação vs UNet v3 (DeepLab MobileNetV2)

| Métrica | v3 DeepLab | **v5 Fast-SCNN** |
|---|---|---|
| Tamanho INT8 | 2.327.272 B | **1.527.800 B** (−34%) |
| Parâmetros | ~2,3 M | **1,204 M** |
| Operadores | 10 ops | **9 ops** (61 operadores) |
| Convs dilatadas | sim (dilate 2/4/8) | **nenhuma** |
| Ops com kernel esp-nn | Add, Conv2D, DepthwiseConv2D, Mul | + **AveragePool2D** |
| IoU teste (INT8, notebook) | — | 0.8207 |

## Decisões de projeto

| Tema | Decisão | Motivo |
|---|---|---|
| Framework | `framework = espidf` (PlatformIO) | Integração com TFLM, LittleFS e kernels do esp-nn |
| Modelo | `data/gatos_v5_fast_scnn.tflite` (25 chars) | `mklittlefs` limita nomes a 32 chars; original preservado no diretório de treino |
| TFLite Micro | submodule `espressif/esp-tflite-micro` @ v1.3.8 via `EXTRA_COMPONENT_DIRS` (`components/tflite-lib`) | Componente IDF com kernels esp-nn; 4 patches em `components/patches/tflite-lib/` |
| Versão do TFLM | `esp-tflite-micro` **v1.3.8** | Mantém a API usada pelo `main.cpp` (`tflite_bridge/micro_error_reporter.h`) e o esp-nn v1.3.0; a v1.4.3 exigiria port da API + esp-nn ≥ 1.4.2 |
| ESP-NN | submodule `espressif/esp-nn` @ v1.3.0, `-DESP_NN` **ativo** | v1.3.0 corrige o Conv2D S3; revalidação bit a bit pendente |
| Op resolver | `MicroMutableOpResolver<9>` | Add, AveragePool2D, BatchToSpaceNd, Concatenation, Conv2D, DepthwiseConv2D, Mul, ResizeBilinear, SpaceToBatchNd |
| Normalização de entrada | ImageNet por canal: `(px/255 - mean[c])/std[c]`, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) | **Diferente do v3** (`px/127.5 - 1`); modelo treinado com essa normalização |
| Quantização input | `q = round(v/scale) + zp`, scale=0.018658448, zp=−14 | Lida do tensor; input int8 |
| Pós-processamento | Argmax 3 classes sobre logits int8 | Quantização per-tensor ⇒ argmax int8 ≡ argmax float |
| Arena TFLM | 4 MiB em PSRAM | Uso observado de aproximadamente 325 KiB no build de referência |
| Partições | app 6 MB + littlefs 9,88 MB + coredump (16 MB) | Mesma tabela dos outros projetos |
| Classes de saída | 0=fundo, 1=gato, 2=cachorro | Validado no sanity check do host |

## Estrutura

```
ESP32_image_segmentation_cats_dogs/
├── training/                                      # treino/exportação (v5)
│   ├── gatos_cachorro_unet_v5_fast_scnn_96px.ipynb
│   ├── tf_2_0_fast_scnn.py                        # arquitetura
│   ├── models/                                    # inclui o int8_full original
│   │   └── model_..._infer_int8_full.tflite       # original 1.527.800 B (MD5 179689eb...)
│   └── results/
├── firmware/                                      # firmware para ESP32-S3
│   ├── README.md                                  # documentação do firmware
│   ├── platformio.ini                             # espidf, 16 MB QIO, littlefs
│   ├── partitions.csv                             # app 6 MB + littlefs 9,88 MB
│   ├── sdkconfig.defaults                         # 240 MHz, QIO 80M, PSRAM OPI, WDT 15 s
│   ├── CMakeLists.txt                             # EXTRA_COMPONENT_DIRS -> ../components
│   ├── src/
│   │   ├── main.cpp                               # app_main() — porta do main.cpp v3
│   │   ├── CMakeLists.txt
│   │   └── idf_component.yml                      # joltwallet/littlefs
│   ├── data/
│   │   └── gatos_v5_fast_scnn.tflite              # modelo INT8 patcheado (1.527.800 B, MD5 449f7358...)
│   └── python/
│       ├── teste_inferencia.py                    # cliente HTTP (PIL/numpy, sem TF)
│       ├── get_input_info.py                      # gera input_info.json (ai-edge-litert ou TF)
│       ├── input_info.json                        # scale/zp do input
│       ├── sanity_check_host.py                   # validação offline do pipeline
│       ├── inferencia_pc.py                       # máscara/logits de referência LiteRT
│       ├── trace_operadores_pc.py                 # comparação dos 61 operadores
│       └── imgs/                                  # imagens de teste (mesmo dataset SAM3)
└── components/                                    # dependências do firmware (submodules + patches)
    ├── tflite-lib/                                # SUBMODULE espressif/esp-tflite-micro @ v1.3.8
    ├── esp-nn/                                    # SUBMODULE espressif/esp-nn @ v1.3.0
    ├── patches/                                   # 4 patches do TFLM + library.json do esp-nn
    └── PROVENANCE.md
```

`.venv/`, `managed_components/` e as saídas em `python/results*/` são
diretórios locais ignorados pelo Git.

## Cobertura do esp-nn (v1.3.0)

Cobertura das 9 ops do modelo:

| Op | Nº | Kernel |
|---|---|---|
| CONV_2D | 29 | **esp-nn** (`esp_nn_conv_s8_esp32s3`) |
| DEPTHWISE_CONV_2D | 14 | **esp-nn** (`esp_nn_depthwise_conv_s8`) |
| ADD | 8 | **esp-nn** |
| MUL | 1 | **esp-nn** |
| AVERAGE_POOL_2D | 2 | **esp-nn** (pooling) |
| BATCH_TO_SPACE_ND | 1 | TFLM genérico (barata) |
| SPACE_TO_BATCH_ND | 1 | TFLM genérico (barata) |
| CONCATENATION | 1 | TFLM genérico (barata) |
| RESIZE_BILINEAR | 4 | TFLM genérico (barata) |

Com o esp-nn v1.3.0 ativo, **54/61 operadores (88%) têm caminho SIMD
potencial**; os demais continuam no TFLM genérico. Após o build, confirmar com
`nm`/map e repetir a comparação funcional completa.

## Status

| Item | Estado |
|---|---|
| Build e gravação | concluídos (árvore antiga) |
| Modelo no LittleFS | corrigido, MD5 `449f7358985e6f3bd59aedc57272cb5e` |
| Kernel `SPACE_TO_BATCH_ND` | corrigido por patch (backport do fix upstream v1.4.3) |
| Baseline reference | **61/61 operadores e saída final bit-idênticos ao PC** (árvore antiga) |
| Golden | `[6623, 0, 2593]` no ESP32 e PC |
| `OPTRACE` | não incluído na árvore nova (era instrumentação local) |
| Componentes | submodules esp-tflite-micro v1.3.8 + esp-nn v1.3.0, patches aplicados |
| Otimização | `-DESP_NN` **ativo**; revalidação bit a bit **pendente** |

## Diagnóstico resolvido e cuidados

O [registro técnico do SPACE_TO_BATCH_ND](README_UPSTREAM_SPACE_TO_BATCH_ND.md)
reúne a evidência, o patch e a proposta de teste de regressão.

**Resumo:** O erro estava no operador `SPACE_TO_BATCH_ND` do TFLite
Micro, cujo parâmetro `output_offset`, usado como valor de padding em tensores
quantizados, era alocado mas não inicializado; no ESP32, essa memória continha
`0xAA` (`-86` em INT8), fazendo com que 98.304 dos 153.600 bytes do tensor
fossem preenchidos incorretamente e contaminassem os logits e a máscara. A
solução foi inicializar explicitamente `output_offset` no `Prepare()` com
`output->params.zero_point`, garantindo que o padding represente o zero real do
tensor. Após a correção, os 61 operadores, todos os logits e a máscara ficaram
bit a bit idênticos entre o ESP32 e o PC.

- O modelo exportado tinha batch incorreto nos tensores 141/142; o LiteRT
  corrigia dinamicamente, mas o TFLM não.
- O kernel Micro `SPACE_TO_BATCH_ND` deixava `output_offset` sem inicializar.
  No ESP32, os 98.304 bytes de padding do tensor 141 recebiam `0xAA`; a
  correção usa `output->params.zero_point` (`0` neste modelo).
- O teste final confirmou hashes iguais em todos os 61 operadores,
  `max_abs_diff = 0` nos logits e máscaras idênticas.
- O tempo de ~15 s observado no diagnóstico é principalmente consequência
  dos kernels de referência; a instrumentação `OPTRACE` também adiciona I/O.
- O esp-nn está na v1.3.0 (submodule) com `-DESP_NN` ativo; limpar
  `.pio/build` e revalidar a árvore nova antes de confiar no resultado.
- `pio run -t uploadfs` só é necessário quando o conteúdo de `data/` mudar.
- A PSRAM permaneceu estável; não houve evidência de corrupção ou vazamento.


## Método reutilizável de debug device × PC

O diagnóstico deste projeto seguiu uma bisseção por fronteiras. Esse método
pode ser reaproveitado em outros portes TFLite Micro:

1. **Fixar a referência:** mesmo modelo, mesmos bytes RGB, mesma normalização e
   quantização no PC e no firmware. Salvar logits/máscaras como arrays, não
   extrair dados de plots.
2. **Hash da entrada:** calcular FNV-1a do tensor quantizado nos dois lados. Se
   for diferente, parar e corrigir o pré-processamento.
3. **Logits completos:** retornar/salvar o tensor anterior ao argmax. Comparar
   igualdade, erro máximo/médio, faixa e canais; argmax sozinho esconde erros.
4. **A/B de otimização:** desligar esp-nn, limpar `.pio/build`, confirmar o ELF
   com `nm` e repetir. Persistência do erro elimina os kernels SIMD como causa
   principal.
5. **Shapes:** comparar flatbuffer com as shapes efetivas do LiteRT. O LiteRT
   pode corrigir shapes dinamicamente que permanecem erradas no TFLM.
6. **Trace durante `Invoke()`:** registrar cada output logo após o operador,
   antes do reuso da arena. Usar `(op, tensor, bytes, hash, min, max, sum)`.
7. **Primeira divergência:** descartar todos os operadores anteriores e
   reproduzir o output errado a partir do último tensor correto.
8. **Prova causal:** variar sistematicamente o parâmetro suspeito. Neste caso,
   testar os 256 bytes de padding revelou que `0xAA` reproduzia exatamente o
   hash errado.
9. **Regressão:** após a correção, exigir igualdade de todos os operadores,
   logits completos e máscara antes de remover o trace ou medir desempenho.

### Comandos do trace

```bash
# Captura serial
pio device monitor | tee python/results/golden_optrace_esp32.log

# Inferência no dispositivo
INPUT_IMAGE_PATH="$PWD/python/imgs/golden_recortado.jpg" \
python python/teste_inferencia.py

# Comparação PC × ESP32
INPUT_IMAGE_PATH="$PWD/python/imgs/golden_recortado.jpg" \
python python/trace_operadores_pc.py \
python/results/golden_optrace_esp32.log
```

Interpretação:

- `input ... igual=False`: erro antes de `Invoke()`;
- entrada igual e logits diferentes: erro no grafo/runtime;
- `primeira diferença de hash`: primeiro operador a investigar;
- estatísticas diferentes no mesmo ponto: erro já material, não apenas
  diferença inofensiva de ordenação;
- 61/61 hashes iguais e `max_abs_diff = 0`: execução bit-idêntica.

A explicação detalhada, incluindo as limitações do reuso da arena e a prova do
padding `0xAA`, está nas [notas de engenharia](../docs/HISTORY_2026_09.md).

## Ambiente `pio`

O firmware usa PlatformIO Core 6.1.19 ou superior. Siga o [guia de instalação](../docs/QUICKSTART.md)
para preparar os componentes e configurar o Wi-Fi. Execute os comandos em
`firmware/` e substitua o endereço de exemplo pelo IP do ESP32.

## Comandos

```bash
source .venv/bin/activate     # opcional (venv local; não versionado)
pio run                       # build
pio run -t buildfs            # gera imagem LittleFS (data/)
pio run -t uploadfs           # grava LittleFS (1ª vez obrigatório)
pio run -t upload -t monitor  # grava app + monitor serial
ESP32_IP=192.0.2.10 .venv/bin/python python/teste_inferencia.py
```

Dependências dos scripts Python: `pip install ai-edge-litert numpy pillow
requests matplotlib` (o `teste_inferencia.py` não usa TensorFlow).
