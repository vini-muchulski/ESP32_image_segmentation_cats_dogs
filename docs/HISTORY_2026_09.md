# Experimento Fast-SCNN 96px (UNet v5) — ESP32-S3

Estas notas preservam o README anterior. Para os passos de instalação e validação da árvore atual, consulte o [guia de instalação](QUICKSTART.md) e o [guia de depuração](TFLM_DEBUG_PLAYBOOK.md).

Projeto de segmentação semântica gato/cachorro/fundo com **Fast-SCNN 96px
INT8** para **ESP32-S3** (framework ESP-IDF): treinamento (`training/`),
firmware de inferência TinyML (`firmware/`) e componentes de runtime
(`components/`) mantidos como **submodules upstream + patches mínimos**
(nov/2026).

As seções 6 e 7 registram o debug bit a bit e o playbook de bisseção que
levaram o firmware a reproduzir o LiteRT exatamente (61/61 operadores).

---

## 1. Estrutura

```
ESP32_image_segmentation_cats_dogs/
├── training/                                        # treino/exportação (onnx2tf) + modelos + plots
│   ├── gatos_cachorro_unet_v5_fast_scnn_96px.ipynb
│   ├── tf_2_0_fast_scnn.py                         # arquitetura (importada pelo notebook)
│   ├── models/                                      # artefatos fp32/int8 (ver models/MD5SUMS)
│   └── results/                                     # histórico de treino + visualizações
├── firmware/                                       # porte ESP32-S3 (ESP-IDF/PlatformIO)
│   ├── platformio.ini        # framework = espidf, LittleFS, 16MB
│   ├── CMakeLists.txt        # EXTRA_COMPONENT_DIRS → ../components
│   ├── data/                 # modelo .tflite gravado no LittleFS
│   ├── src/main.cpp          # servidor HTTP + inferência TFLM
│   └── python/               # teste_inferencia.py, sanity_check_host.py
├── components/
│   ├── tflite-lib/           # SUBMODULE espressif/esp-tflite-micro @ v1.3.8
│   ├── esp-nn/               # SUBMODULE espressif/esp-nn @ v1.3.0
│   ├── patches/              # patches locais aplicados sobre os submodules
│   └── PROVENANCE.md         # linhagem, patches e como reaplicá-los
└── README.md                 # este arquivo
```

---

## 2. Componentes (`components/` — submodules upstream + patches)

Os dois componentes são **submodules upstream** pinados por tag; as
modificações locais ficam em `components/patches/` e são aplicadas ao working
tree dos submodules. Linhagem, justificativa de cada patch e comandos de
restauração: `components/PROVENANCE.md`.

### `tflite-lib`
Submodule de [`espressif/esp-tflite-micro`](https://github.com/espressif/esp-tflite-micro)
@ **v1.3.8** (`d6465fd`) — o fork oficial do TFLite Micro como componente
ESP-IDF, com os kernels que delegam ao esp-nn
(`tensorflow/lite/micro/kernels/esp_nn/`: `conv.cc`, `depthwise_conv.cc`,
`add.cc`, `mul.cc`, etc.) e `-DESP_NN` ativo no `CMakeLists.txt` do
componente. Ops sem versão esp-nn caem automaticamente no `reference`.

**Patches aplicados (`patches/tflite-lib/`):**
- `space_to_batch_nd.patch` — inicializa `SpaceToBatchParams::output_offset`
  com o zero-point da saída: é o mesmo fix que entrou no upstream na
  **v1.4.3** (backport). Sem ele, o ESP32 preenche o padding INT8 com o
  padrão de memória `0xAA` (diagnóstico na seção 6).
- `concatenation.patch` — `kMaxInputNum` 10 → 128 (concats com >10 entradas
  no grafo convertido).
- `cmake-esp-nn-requirement.patch` — declara `esp-nn` no `CMakeLists.txt`
  (o upstream espera o IDF Component Manager; aqui o esp-nn é submodule local).
- `disable-component-manager.patch` — remove o `idf_component.yml` para evitar
  um esp-nn duplicado em `managed_components/`.

> **Divergência conhecida:** convolução dilatada (`dilation > 1`) usa o kernel
> `reference` do upstream — correto, porém sem SIMD. A otimização de sub-grids
> da árvore antiga está preservada em `patches/tflite-lib/reference/` e ainda
> não foi portada.

**Por que v1.3.8 (e não a v1.4.3)?** A v1.4.3 já traz o fix do
`space_to_batch` e a opção de Kconfig do esp-nn, mas **removeu** o
`tflite_bridge/micro_error_reporter.h` — API usada pelo `firmware/src/main.cpp`
— e exige **esp-nn ≥ 1.4.2**. A v1.3.8 mantém o firmware como está e o esp-nn
v1.3.0; migrar para a v1.4.3 é uma atualização futura (portar a API, subir o
esp-nn e regenerar/revalidar os patches).

### `esp-nn`
Submodule de [`espressif/esp-nn`](https://github.com/espressif/esp-nn) @
**v1.3.0** (`d8866fa`): corrige o kernel Conv2D S3 para padding SAME
assimétrico, grouped conv e sizing do scratch (a v1.1.2 usada durante o
desenvolvimento tinha bug conhecido de Conv2D S3). É o requisito mínimo do
projeto para `-DESP_NN`.

> O override `library.json` (build Arduino/PlatformIO via `lib_extra_dirs`)
> vive em `patches/esp-nn/` — o upstream não o publica.

### Restaurar o estado de build após um clone limpo

```bash
git clone --recurse-submodules https://github.com/vini-muchulski/ESP32_image_segmentation_cats_dogs.git
cd ESP32_image_segmentation_cats_dogs

# patches do TFLM (aplicar dentro do submodule):
for p in components/patches/tflite-lib/*.patch; do
  git -C components/tflite-lib apply "$PWD/$p"
done

# override do esp-nn (apenas builds Arduino/PlatformIO):
cp components/patches/esp-nn/library.json components/esp-nn/
```

---

## 3. Receita: habilitar esp-nn em outro projeto ESP32-S3

Esta seção é **agnóstica do projeto** — o que é preciso para que
qualquer rede TinyML rode com kernels otimizados.

### 3.1 Expor os componentes ao novo projeto

- **ESP-IDF** — no `CMakeLists.txt` raiz do projeto:
  ```cmake
  list(APPEND EXTRA_COMPONENT_DIRS
      "<caminho>/components")
  ```
  (o caminho pode ser relativo — `../../components` — ou absoluto).

- **Arduino (PlatformIO)** — no `platformio.ini`:
  ```ini
  lib_extra_dirs = <caminho>/components
  ```
  O `tflite-lib` precisa ter **`"name": "tflite-lib"`** no
  `library.json` (sem o campo, o PlatformIO não resolve a dependência).
  Cada componente em `lib_extra_dirs` é adicionado automaticamente se
  houver `library.json`.

### 3.2 Ligar o caminho otimizado

1. O componente `tflite-lib` do projeto deve declarar dependência do
   `esp-nn` (ESP-IDF: `PRIV_REQUIRES esp-nn` no `CMakeLists.txt` do
   componente) **e** compilar com o define:
   ```cmake
   target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)
   ```
2. No código do aplicativo, registrar as ops da rede no resolver
   (`MicroMutableOpResolver<N>`) — os headers `micro_op_resolver.h` do
   tflite-lib já apontam para os wrappers `esp_nn` quando o define está
   ativo. Não há mudança de API no app: o ganho vem do build.

Ops com kernel otimizado no esp-nn S3: **Conv2D, DepthwiseConv2D,
Add, Mul** (+ FullyConnected, Pooling, Softmax conforme wrappers incluídos).
Ops sem versão otimizada usam o reference do TFLite Micro — normal.

### 3.3 Verificar que o build usa os kernels otimizados

Após compilar (`pio run`), com o toolchain do ESP32-S3:

```bash
NM=~/.platformio/packages/toolchain-xtensa-esp32s3/bin/xtensa-esp32s3-elf-nm

# 1. símbolos S3 definidos no ELF (esperado: T esp_nn_conv_s8_esp32s3, etc.)
$NM <build_dir>/firmware.elf | grep " T esp_nn.*_esp32s3"

# 2. wrappers realmente CHAMAM os kernels (U = referência, não só presente)
$NM <build_dir>/**/libtflite-lib.a | grep -A1 "\.cc\.o:" | grep "U esp_nn"
#    esperado: conv.cc.o → U esp_nn_conv_s8_esp32s3, add.cc.o → U ..._add_...
```

Regras de bolso:
- ELF **com** `esp_nn_*_esp32s3` e flash maior (~+100–200 KB) → caminho
  otimizado ativo.
- ELF sem nenhum símbolo `esp_nn` → define `-DESP_NN` não chegou ao build.

### 3.4 Armadilhas conhecidas

| Armadilha | Sintoma | Solução |
|---|---|---|
| **Cache do CMake** | mudança de flags/componentes não muda o firmware; tamanho do ELF idêntico | `rm -rf .pio/build` (ou `idf.py fullclean`) após alterar flags ou componentes |
| **esp-nn antigo** | máscara/saída ruidosa ou errada **só** com `-DESP_NN` | usar ≥ v1.3.0; snapshot antigo tinha bug no Conv2D S3 (padding assimétrico) |
| **`library.json` sem `"name"`** | PlatformIO não encontra `tflite-lib` (Arduino) | adicionar `"name": "tflite-lib"` |
| **Definir flag no lugar errado** | `-DESP_NN` só no app, não na lib | o define precisa valer para o componente `tflite-lib` inteiro |
| **Shapes erradas no .tflite (S2B/B2S)** | máscara quebrada no device, **correta no host LiteRT** (e igual com esp-nn ON ou OFF) | LiteRT redimensiona saídas em runtime; TFLM usa shapes estáticas do flatbuffer — corrigir as shapes no modelo (ver seção 6) |
| **Resize float vs int no device** | divergência device vs host (~10% dos pixels) com resize float | manter o caminho **inteiro** `ResizeBilinearInteger` (Q10) — é o que o LiteRT host usa; grafo 100% inteiro = bit-exato em qualquer arquitetura |

### 3.5 Validar funcionalmente (metodologia de bisseção)

Uma rede otimizada pode acertar por sorteio — valide **sempre**:

1. Construa duas versões: com `-DESP_NN` e sem (reference puro).
2. Rode as mesmas entradas nas duas e compare a saída completa
   (máscara/imagens, não só a métrica final).
3. Se divergirem, isole kernel por kernel: no wrapper
   (`tflite-lib/.../kernels/esp_nn/<op>.cc`), troque temporariamente
   `#if ESP_NN` por `#if 0` no `Eval` da op suspeita, recompile
   (**limpando o build**) e re-teste até achar o kernel culpado.
4. Corrija na causa (ex.: atualizar esp-nn) em vez de "desligar" o kernel
   — desligar perde a otimização permanentemente.

### 3.6 Comparar desempenho

O firmware expõe o tempo de inferência no monitor serial (`Tempos |
... inferência: ...`) e via HTTP (`X-Inference-Us` no
`POST /predict_bin`, ver `python/teste_inferencia.py`). Compare:
reference vs esp-nn (ganho do SIMD) e Arduino vs ESP-IDF (overhead de
framework).

---

## 4. Estado atual

| Item | Estado |
|---|---|
| `tflite-lib` | **submodule `espressif/esp-tflite-micro` @ v1.3.8** (`d6465fd`) + 4 patches em `components/patches/tflite-lib/` (nov/2026) |
| esp-nn | **submodule `espressif/esp-nn` @ v1.3.0** (`d8866fa`); `library.json` em `patches/esp-nn/` |
| Modelo `.tflite` | patch de shapes aplicado em `data/gatos_v5_fast_scnn.tflite`, MD5 `449f7358985e6f3bd59aedc57272cb5e`; original de treino MD5 `179689ebac29524d449141e81a9c97d1` |
| `SPACE_TO_BATCH_ND` | corrigido por patch (backport do fix upstream v1.4.3): `output_offset` = zero-point |
| Validação device × PC | **61/61 operadores, 27.648 logits e 9.216 classes bit-idênticos** no build de referência (árvore antiga, set/2026) |
| Golden | classes `[6623, 0, 2593]` tanto no ESP32 quanto no PC; hash final dos logits `e00d853e` |
| Instrumentação | `OPTRACE` não incluído na árvore nova (era instrumentação local — ver `components/PROVENANCE.md`) |
| Otimização | `-DESP_NN` **ativo** com esp-nn v1.3.0; **revalidação pendente na árvore nova (TFLM v1.3.8)** — procedimento em `firmware/comandos.txt` |

## 5. Referências por projeto

- Detalhes do firmware, protocolo HTTP (`POST /predict_bin`) e telemetria:
  `firmware/src/main.cpp`
- Cliente de teste (`firmware/python/teste_inferencia.py` — salva máscara e
  logits brutos em `firmware/python/results/`), referência LiteRT
  (`firmware/python/inferencia_pc.py`) e comparação por operador
  (`firmware/python/trace_operadores_pc.py`)

---

## 6. Diagnóstico da máscara divergente no ESP32 (set/2026)

**Resumo:** O erro estava no operador `SPACE_TO_BATCH_ND` do TFLite
Micro, cujo parâmetro `output_offset`, usado como valor de padding em tensores
quantizados, era alocado mas não inicializado; no ESP32, essa memória continha
`0xAA` (`-86` em INT8), fazendo com que 98.304 dos 153.600 bytes do tensor
fossem preenchidos incorretamente e contaminassem os logits e a máscara. A
solução foi inicializar explicitamente `output_offset` no `Prepare()` com
`output->params.zero_point`, garantindo que o padding represente o zero real do
tensor. Após a correção, os 61 operadores, todos os logits e a máscara ficaram
bit a bit idênticos entre o ESP32 e o PC.

### 6.1 Sintoma inicial

As máscaras produzidas pelo ESP32 divergiam fortemente do LiteRT no PC,
embora imagem, normalização, quantização e modelo fossem nominalmente iguais.
Desabilitar o esp-nn não eliminou o problema, mostrando que a causa principal
estava no caminho genérico do TFLite Micro.

### 6.2 Primeiro defeito: shapes inválidas no modelo exportado

O `.tflite` gerado por `onnx2tf` declarava shapes impossíveis na decomposição
da convolução dilatada do decoder:

```text
SPACE_TO_BATCH_ND [4,4] → DEPTHWISE_CONV_2D → BATCH_TO_SPACE_ND
```

- tensor 141: `[1,5,5,384]`; correto: `[16,5,5,384]`;
- tensor 142: `[1,3,3,384]`; correto: `[16,3,3,384]`.

O LiteRT redimensionava esses tensores durante `Prepare()`. O TFLM usa as
shapes estáticas do flatbuffer e processava apenas 1 dos 16 batches.

A correção alterou somente o primeiro byte dos dois `int32` de dimensão
(`1 → 16`, oito bytes pertencentes aos dois valores):

```python
data[pos141:pos141+4] = np.int32(16).tobytes()
data[pos142:pos142+4] = np.int32(16).tobytes()
```

O LiteRT produz a mesma saída com o modelo original e o corrigido. O patch é
necessário para runtimes estáticos como o TFLM.

### 6.3 Segundo defeito: padding INT8 não inicializado no TFLM

Após corrigir as shapes, a saída do ESP32 ainda era determinística, mas errada.
A instrumentação confirmou:

- hash do tensor INT8 de entrada no PC e ESP32: `ed0f9095`;
- transporte dos logits por HTTP sem alteração;
- operadores 0–45 bit-idênticos;
- primeira divergência no operador 46, `SPACE_TO_BATCH_ND`, tensor 141;
- hash PC `e5f1a78b`; hash ESP32 antes da correção `d0a7c58b`.

O `Init()` de `space_to_batch_nd.cc` alocava `SpaceToBatchParams`, mas o
`Prepare()` não inicializava `output_offset`. O kernel de referência usa esse
campo diretamente:

```cpp
const int32_t pad_value = params.output_offset;
memset(out, pad_value, depth * sizeof(T));
```

Neste modelo, o zero-point dos tensores 140/141 é `0`. Dos 153.600 bytes do
tensor 141, 98.304 são padding. Substituir esses bytes, na referência, de
`0x00` por `0xAA` reproduziu **exatamente** o hash errado do ESP32
`d0a7c58b`. `0xAA` (`-86` em INT8) era o padrão determinístico da memória
persistente não inicializada. No teste x86, a memória por acaso estava zerada,
o que havia escondido o defeito.

Correção aplicada em
`tensorflow/lite/micro/kernels/space_to_batch_nd.cc`:

```cpp
SpaceToBatchParams* params =
    static_cast<SpaceToBatchParams*>(node->user_data);
params->output_offset = output->params.zero_point;
```

### 6.4 Validação final

Com o modelo e o kernel corrigidos, `golden_recortado.jpg` produziu:

| Verificação | Resultado |
|---|---|
| Entrada quantizada PC × ESP32 | hash `ed0f9095`, igual |
| Operadores | **61 de 61 hashes e estatísticas iguais** |
| Logits finais | arrays INT8 idênticos, `max_abs_diff = 0` |
| Hash dos logits | `e00d853e` nos dois runtimes |
| Máscara | arrays idênticos |
| Classes `[fundo,gato,cachorro]` | `[6623, 0, 2593]` nos dois runtimes |

A causa não era PSRAM, transporte HTTP, pré-processamento, resize, esp-nn ou
arredondamento entre arquiteturas.

### 6.5 Ferramentas de diagnóstico adicionadas

- `python/teste_inferencia.py`: solicita logits brutos, salva
  `results/*_logits_raw.npy`, calcula hashes FNV-1a e compara com o PC;
- `python/inferencia_pc.py`: salva `results_pc/*_logits_pc.npy`;
- `python/trace_operadores_pc.py`: produz a referência dos 61 operadores e
  compara com um log `OPTRACE` do ESP32;
- `micro_graph.cc`: imprime temporariamente hash, mínimo, máximo e soma logo
  após cada operador, antes do reuso da arena.

Dumps de tensores intermediários feitos somente após `Invoke()` não são
confiáveis porque a arena é reutilizada; o `OPTRACE` no limite de cada
operador evita esse problema.

### 6.6 Próximos passos

1. ~~Remover a instrumentação `OPTRACE`~~ — **feito (nov/2026)**: a árvore
   nova usa o upstream `esp-tflite-micro` v1.3.8 e não inclui a
   instrumentação (era local; ver `components/PROVENANCE.md`).
2. ~~Atualizar `esp-nn` para v1.3.0~~ — **feito (nov/2026)**: submodule
   pinado em v1.3.0 (`d8866fa`).
3. Revalidar bit a bit **na árvore nova (TFLM v1.3.8 + patches)**: limpar
   completamente o build e repetir a comparação de máscara/logits antes de
   medir desempenho.
4. Corrigir a exportação no notebook para produzir as shapes 141/142 certas,
   evitando depender do patch binário.
5. Considerar enviar a correção de `output_offset` ao fork/origem do TFLM.


---

## 7. Playbook reutilizável de debug TFLM

Este procedimento serve para investigar redes que funcionam no LiteRT/TFLite
do PC, mas produzem resultados diferentes no TFLite Micro. A regra central é
**comparar o dado bruto na fronteira mais próxima possível do operador**, sem
confiar apenas na imagem renderizada ou na métrica final.

### 7.1 Criar uma referência controlada no PC

1. Use exatamente o mesmo `.tflite` enviado ao dispositivo; registre tamanho e
   hash do arquivo.
2. Use a mesma imagem já redimensionada, ordem de canais, normalização,
   quantização, scale, zero-point e regra de arredondamento do firmware.
3. Prefira `OpResolverType.BUILTIN_REF` no LiteRT para reduzir diferenças entre
   kernels durante a investigação.
4. Salve arrays brutos (`.npy`) de logits e máscara. PNG/overlay é útil para
   visualização, mas não é fonte confiável para comparação numérica.

Neste projeto, `python/inferencia_pc.py` produz
`results_pc/*_logits_pc.npy` e `*_mask_pc.npy`.

### 7.2 Validar as fronteiras antes de culpar a inferência

Calcule um hash simples dos bytes em pontos bem definidos:

```text
imagem RGB enviada → tensor INT8 quantizado → logits brutos → máscara argmax
```

O cliente e o firmware usam FNV-1a de 32 bits. Ele é rápido e suficiente para
diagnóstico de integridade, embora não seja um hash criptográfico.

- Hash da entrada diferente: investigar resize, RGB/BGR, normalização,
  arredondamento, clipping, scale/zero-point ou transporte HTTP.
- Hash da entrada igual e logits diferentes: o erro está dentro do runtime.
- Hash dos logits enviado e recebido igual: transporte da resposta descartado.
- Máscara diferente com logits iguais: investigar apenas o pós-processamento.

Neste caso, a entrada deu `ed0f9095` no PC e no ESP32; os logits recebidos
também tinham o mesmo hash informado pelo firmware. Isso eliminou todo o
pipeline externo ao `Invoke()`.

### 7.3 Comparar logits, não somente argmax

O argmax esconde a direção e a magnitude do erro. Compare os tensores completos:

- percentual de valores iguais;
- `max_abs_diff` e `mean_abs_diff`;
- mínimo, máximo, média e saturações por canal;
- correlação por canal;
- contagem das classes apenas como resumo final.

Neste projeto, canais do ESP32 concentrados perto de `65`, o zero-point da
saída, indicaram que uma ramificação do grafo estava chegando próxima de zero.

### 7.4 Fazer A/B dos kernels

Construa uma baseline sem aceleração:

1. desative o define que seleciona kernels otimizados;
2. apague completamente o diretório de build para invalidar o cache do CMake;
3. compile e confirme com `nm` que símbolos da biblioteca otimizada não estão
   no ELF;
4. repita exatamente as mesmas entradas.

Se a divergência continuar, a otimização não é a causa principal. Neste
experimento, desligar `-DESP_NN` mudou pouco a máscara e o erro permaneceu.

### 7.5 Conferir shapes estáticas e shapes efetivas

Compare as shapes armazenadas no flatbuffer com as shapes após
`allocate_tensors()` no LiteRT. O runtime completo pode redimensionar tensores
dinamicamente e esconder um modelo inválido; TFLM normalmente depende das
shapes estáticas.

Foi assim que se encontrou o primeiro defeito deste modelo: os tensores 141 e
142 declaravam batch 1 onde o bloco `[4,4]` exigia batch 16.

### 7.6 Rastrear operador por operador

Quando entrada e saída estão bem definidas, localize a primeira divergência:

1. No PC, crie o interpretador com
   `experimental_preserve_all_tensors=True` e registre os outputs na ordem de
   `_get_ops_details()`.
2. No TFLM, registre o output **imediatamente após `registration->invoke()`** e
   antes de `ResetTempAllocations()`/próximo operador.
3. Para cada output, grave: índice do operador, nome, tensor, número de bytes,
   hash, mínimo, máximo e soma.
4. Compare os registros pela chave `(op, tensor)`.

Não leia todos os intermediários somente depois de `Invoke()`: o planejador do
TFLM reutiliza a arena, então muitos buffers já terão sido sobrescritos. O
`OPTRACE` temporário em `micro_graph.cc` evita essa armadilha.

Fluxo usado neste projeto:

```bash
# Terminal 1: capturar o serial
pio device monitor | tee python/results/golden_optrace_esp32.log

# Terminal 2: disparar a mesma entrada
INPUT_IMAGE_PATH="$PWD/python/imgs/golden_recortado.jpg" \
python python/teste_inferencia.py

# Depois de encerrar o monitor, comparar os 61 operadores
INPUT_IMAGE_PATH="$PWD/python/imgs/golden_recortado.jpg" \
python python/trace_operadores_pc.py \
python/results/golden_optrace_esp32.log
```

O primeiro resultado diferente é o ponto de investigação. Operadores anteriores
iguais ficam descartados. Aqui, os operadores 0–45 foram bit-idênticos e o
primeiro erro apareceu no operador 46, `SPACE_TO_BATCH_ND`.

### 7.7 Transformar a hipótese em prova reproduzível

Depois de localizar o operador, leia o kernel e tente reproduzir o output
incorreto a partir do último tensor correto. Não basta concluir que “o operador
está errado”. Teste parâmetros, padding, offsets, strides, shapes, aliasing e
valores não inicializados.

Para este bug:

1. tensor 140 era idêntico nos dois runtimes;
2. tensor 141 diferia apenas depois do `SPACE_TO_BATCH_ND`;
3. 98.304 dos 153.600 bytes eram padding;
4. o kernel usava `params.output_offset`, nunca inicializado;
5. testar os 256 possíveis bytes de padding mostrou que `0xAA` reproduzia
   exatamente o hash do ESP32 `d0a7c58b`;
6. inicializar o campo com o zero-point `0` produziu o hash correto `e5f1a78b`.

Essa reprodução fechou a relação causal entre memória não inicializada e a
máscara incorreta.

### 7.8 Corrigir minimamente e executar regressão completa

Aplique a correção na fonte do parâmetro, não um `memset` genérico ou ajuste no
pós-processamento. Em seguida:

1. faça clean build;
2. valide novamente todos os operadores;
3. compare os logits completos byte a byte;
4. compare a máscara;
5. repita mais de uma entrada;
6. só depois remova a instrumentação e meça desempenho;
7. reative otimizações uma por vez, repetindo a validação.

A regressão final deste projeto obteve 61/61 operadores iguais, logits com
`max_abs_diff = 0` e máscara golden `[6623, 0, 2593]` nos dois runtimes.

### 7.9 Checklist resumido

```text
[ ] mesmo modelo/hash nos dois lados
[ ] mesmos bytes RGB enviados
[ ] tensor quantizado de entrada com mesmo hash
[ ] logits brutos salvos e comparados
[ ] build reference confirmado por nm
[ ] shapes flatbuffer × runtime verificadas
[ ] primeiro operador divergente localizado durante Invoke
[ ] output errado reproduzido a partir do último tensor correto
[ ] correção mínima aplicada na causa
[ ] regressão de todos os operadores, logits e máscara
[ ] instrumentação removida antes do benchmark
```
