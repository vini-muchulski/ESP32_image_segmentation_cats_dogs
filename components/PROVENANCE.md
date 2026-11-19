# Proveniência dos componentes — `components/`

Este diretório contém as dependências do firmware como **submodules upstream**
e os patches locais mínimos aplicados sobre elas. Nenhum código de terceiros é
copiado para dentro deste repositório.

**Estado atual (nov/2026):** submodules pinados e patches **já aplicados** no
working tree; o firmware compila a partir desta pasta (o build não foi
executado nesta preparação).

---

## 1. `tflite-lib/` — TensorFlow Lite Micro (submodule)

- **Upstream:** https://github.com/espressif/esp-tflite-micro
- **Versão pinada:** `v1.3.8` — commit `d6465fd54a029641e58f2342eeec85f3492ba084`
- **Por que o fork da Espressif:** é o fork oficial do TFLM empacotado como
  componente ESP-IDF, com a integração esp-nn pronta: os wrappers
  `tensorflow/lite/micro/kernels/esp_nn/` são compilados com `-DESP_NN` pelo
  próprio `CMakeLists.txt` (linha 134) e a dependência `esp-nn` já está
  declarada. Não é mais necessário manter uma árvore local de ~390 arquivos.

### Histórico

Até nov/2026 o projeto usava uma árvore **vendored**, baseada em um snapshot
de ~mai/2023 do `tensorflow/tflite-micro` com wrappers `esp_nn` adaptados à
API da época. A migração para o upstream mantém apenas as correções que a
tag escolhida ainda não tinha.

### Patches aplicados (`patches/tflite-lib/`)

| Patch | O que faz | Por que é necessário |
|---|---|---|
| `space_to_batch_nd.patch` | `Prepare()` inicializa `SpaceToBatchParams::output_offset` com o zero-point da saída | É o mesmo fix que entrou no upstream na **v1.4.3** (backport). Sem ele, o padding INT8 usa memória não inicializada (`0xAA`) e contamina logits/máscara — diagnóstico completo no README raiz, §6 |
| `concatenation.patch` | `kMaxInputNum` 10 → 128 | O grafo Fast-SCNN convertido tem concatenations com >10 entradas; o upstream (ainda) trava em 10 |
| `cmake-esp-nn-requirement.patch` | adiciona `esp-nn` a `priv_req` no `CMakeLists.txt` | usando o esp-nn como componente local (submodule), o upstream não declara a dependência no CMake (espera o IDF Component Manager); sem isso os wrappers não encontram `esp_nn.h` |
| `disable-component-manager.patch` | remove o `idf_component.yml` do submodule | evita que o IDF Component Manager baixe um esp-nn duplicado para `managed_components/`; o projeto usa o submodule local (versão pinada) |

### Divergências conhecidas (não aplicadas)

- **Convolução dilatada com esp-nn** — o upstream cai no kernel `reference`
  quando `dilation > 1` (correto, porém sem SIMD). A árvore antiga tinha uma
  implementação própria via sub-grids de fase (`EvalQuantizedPerChannelDilated`);
  o porte para a v1.3.8 está **pendente** e é uma otimização (não correção).
  Referência: `patches/tflite-lib/reference/dilated-conv-subgrids-OLD-TREE.diff`
  (contra `conv.cc.bak`, ambos do tree antigo).
- **OPTRACE** — a instrumentação de depuração por operador da árvore antiga
  não foi trazida (era gated por `-DENABLE_OPTRACE`). O playbook do README
  raiz, §7, descreve o procedimento.

### Observação: `idf_component.yml` (resolvido por patch)

O submodule declara dependência de `espressif/esp-nn` (>= 1.1.1) via
`idf_component.yml`, pensado para o *IDF Component Manager*. Como o projeto
usa o esp-nn como submodule local (versão pinada), o arquivo é removido pelo
patch `disable-component-manager.patch` e a dependência vira explícita no
`CMakeLists.txt` (`cmake-esp-nn-requirement.patch`). Para voltar ao modelo de
componentes gerenciados, reverta os dois patches.

---

## 2. `esp-nn/` — kernels SIMD (submodule)

- **Upstream:** https://github.com/espressif/esp-nn
- **Versão pinada:** `v1.3.0` — commit `d8866fa3762ee9caf56712b4019d004d86e0f3f8`
- **Por que v1.3.0:** corrige o kernel Conv2D do S3 para padding SAME
  assimétrico, grouped conv e sizing do scratch (a v1.1.2 usada durante o
  desenvolvimento tinha bug conhecido de Conv2D S3). É o requisito mínimo do
  projeto para ativar `-DESP_NN`.
- **Override local:** `library.json` (build Arduino/PlatformIO via
  `lib_extra_dirs`) — o upstream não o publica. Cópia canônica em
  `patches/esp-nn/library.json`.

---

## 3. Restaurar o estado de build após um clone limpo

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

> Os patches deixam o working tree do submodule modificado — é esperado.
> Não rode `git submodule update --init` depois de aplicar sem antes reverter
> (`git -C components/tflite-lib checkout -- .`), ou os patches serão
> sobrescritos.

Os patches foram gerados contra exatamente a tag `v1.3.8`; ao trocar de tag,
regenerar.

---

## 4. Estado de validação

| Build | Validação |
|---|---|
| Reference (`-DESP_NN` off), árvore antiga | **bit-exato vs LiteRT**: 61/61 operadores, 27.648 logits, máscara e golden `[6623, 0, 2593]` idênticos (set/2026) |
| esp-nn v1.3.0 (`-DESP_NN` on), árvore antiga | revalidação em hardware **pendente** (D5; `firmware/comandos.txt`) |
| **v1.3.8 + patches (esta árvore)** | **revalidação pendente** — repetir o procedimento D5 antes de confiar no resultado |

Qualquer bump de versão (TFLM ou esp-nn) exige repetir a validação por
operador (playbook no README raiz, §7).
