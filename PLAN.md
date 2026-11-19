# PLAN — Organização do repositório para publicação no GitHub

> **Objetivo:** transformar esta pasta (hoje um workspace interno de 1,1 GB)
> em um repositório GitHub **público, limpo e reproduzível**, onde qualquer
> pessoa sem contato prévio com o projeto consiga:
>
> 1. rodar a inferência do modelo Fast-SCNN INT8 (gato/cachorro/fundo) no
>    ESP32-S3 com TFLite Micro;
> 2. usar os componentes reutilizáveis (`tflite-lib` + `esp-nn`) no próprio
>    projeto;
> 3. (opcionalmente) retreinar/exportar o modelo com o próprio dataset.
>
> Este arquivo é o plano de trabalho. Marque `[x]` conforme concluir.

> **Divisão de trabalho (acordo nov/2026):** o agente edita arquivos, git e
> documentação — **nunca compila, faz flash ou executa** código/firmware.
> Builds (`pio run`), testes no ESP32 e validações em hardware são
> executados **exclusivamente pelo autor**.

---

## 0. Estado atual (diagnóstico, out/2026)

| Item | Estado | Problema para publicação |
|---|---|---|
| Tamanho total | **1,1 GB** | `.pio/` (660 MB) + `.venv/` (337 MB) = ~1 GB de lixo que não deve ir para o git |
| `.git` aninhado | `fast_cnn_cats_components/esp-nn/.git` (clone v1.1.2) | Git trataria como submodule quebrado; decide-se estratégia na Fase 3 |
| Credenciais | SSID/senha WiFi **hard-coded** em `src/main.cpp` (`Starlink`/`diversao`) | Vazamento de dado privado; bloqueador de publicação |
| Notebook | 9,2 MB com outputs e caminho de dataset **privado** (`/home/jovyan/privado/...`) | Não reproduzível; expõe caminho interno |
| README raiz | Escrito para insiders (diagnóstico set/2026) | Não serve como porta de entrada pública |
| Instrumentação | `OPTRACE` ativo em `micro_graph.cc` (temporário) | Ruído de log no firmware; deveria ser opcional |
| Inconsistência | `-DESP_NN` **habilitado** no `tflite-lib/CMakeLists.txt:103` com esp-nn **v1.1.2** (versão com bug conhecido de Conv2D S3) | Release invalidado; README interno diz para só reativar com esp-nn ≥ v1.3.0 |
| Arquivos pessoais | `comandos.txt`, `sdkconfig.esp32s3`, `.vscode/`, `old_results/`, `imgs/` (12 MB) | Ruído; decidir manter/remover |
| Modelos | fp32 4,7 MB + int8 1,5 MB + ONNX 6,1 MB; modelo embarcado é o **patcheado** (MD5 `449f7358...`) | OK em tamanho; falta explicar o patch de shapes publicamente |

Tamanho estimado **após** a limpeza: ~30–50 MB (cabem no GitHub normal, sem LFS).

---

## 1. Estrutura-alvo proposta

```text
github_repo_fast_scnn_cats_dog/
├── README.md                  # porta de entrada pública (novo — ver Fase 6)
├── LICENSE                    # ver Fase 7
├── PLAN.md                    # este arquivo (removível após conclusão)
├── .gitignore                 # Fase 1
├── firmware/                  # ← vem de gatos_cachorro_unet_v5_fast_scnn_96px/esp32_s3_code/
│   ├── platformio.ini         # + CMakeLists.txt + partitions.csv + sdkconfig.defaults
│   ├── src/main.cpp           # sem credenciais hard-coded
│   ├── data/model.tflite      # modelo INT8 (patcheado) para LittleFS
│   └── python/                # clientes de teste host (teste_inferencia.py etc.)
├── training/                  # ← vem de gatos_cachorro_unet_v5_fast_scnn_96px/gatos_cachorro_unet_v5_fast_scnn_96px/
│   ├── fast_scnn_96px.ipynb   # notebook limpo (sem outputs, caminhos parametrizados)
│   ├── tf_2_0_fast_scnn.py    # arquitetura
│   ├── requirements.txt       # deps de treino (TF 2.18, tf2onnx, onnx2tf...)
│   └── models/                # artefatos finais (fp32/int8 .tflite, .onnx) + MD5
├── components/                # submodules upstream + patches (nov/2026)
│   ├── tflite-lib/            # SUBMODULE espressif/esp-tflite-micro @ v1.3.8
│   ├── esp-nn/                # SUBMODULE espressif/esp-nn @ v1.3.0
│   ├── patches/               # 4 patches do TFLM + library.json do esp-nn
│   └── PROVENANCE.md          # linhagem e como reaplicar os patches
├── tools/                     # scripts utilitários
│   └── patch_model_shapes.py  # patch de shapes S2B/B2S (1→16) como código
├── docs/
│   ├── QUICKSTART.md          # flash + uploadfs + inferência em <30 min
│   ├── HTTP_API.md           # protocolo POST /predict_bin + headers X-*
│   ├── MODEL_CARD.md         # métricas, quantização, MD5s, limitações
│   ├── PORTING_GUIDE.md       # receita: levar esp-nn a outro projeto (seção 3 do README interno)
│   └── TFLM_DEBUG_PLAYBOOK.md # playbook de bisseção (seção 7 do README interno)
└── assets/                    # 2–3 imagens de resultado p/ README (de python/imgs)
```

> **Decisão pendente D1:** nomes/idioma das pastas (`firmware/` vs manter
> `gatos_cachorro_unet_v5_fast_scnn_96px/esp32_s3_code/`; nomes em inglês ou
> PT-BR). A proposta acima usa inglês — padrão para repositórios públicos.
> Os nomes internos atuais carregam história de versões (v3, v5) que o
> público não conhece.
>
> **Execução (nov/2026):** `training/` ✅, `firmware/` ✅ e
> `components/` ✅ **criados/renomeados** (notebook + arquitetura na
> raiz, `models/` com MD5SUMS, `results/` com os plots originais —
> conteúdo preservado integralmente, sem limpar outputs; a antiga
> `fast_cnn_cats_components/` virou `components/`). O `CMakeLists.txt`
> do firmware aponta para `../components`. Pendentes: criar `tools/`,
> `docs/`, `assets/`.

---

## Fase 1 — Higiene e limpeza (P0 — bloqueador) — **EM ANDAMENTO**

- [x] 1.1 Remover do tree (não são fonte, regeneráveis):
  - `gatos_cachorro_unet_v5_fast_scnn_96px/esp32_s3_code/.pio/` (660 MB)
  - `gatos_cachorro_unet_v5_fast_scnn_96px/esp32_s3_code/.venv/` (337 MB)
  - `.../esp32_s3_code/managed_components/` (regenera via `idf_component.yml`)
  - `.../python/old_results/`, `results/`, `results_pc/` (saídas de debug)
  - caches `__pycache__`, `.DS_Store`, `*.pyc`
  - `.vscode/` (auto-gerado pelo PlatformIO) e `sdkconfig.esp32s3` (gerado)
- [x] 1.2 Criar `.gitignore` **na raiz** cobrindo PlatformIO, ESP-IDF,
      Python, results de debug e editores. O `.gitignore` local de
      `esp32_s3_code/` foi **removido**: tinha o bug `sdkconfig*`, que
      ignorava o `sdkconfig.defaults` (que deve ser versionado).
- [x] 1.3 `.vscode/` e `sdkconfig.esp32s3` removidos (regeneráveis).
      **`comandos.txt` MANTIDO de propósito**: contém o procedimento do
      teste A/B atual (ESP-NN ativo + OPTRACE) com o resultado esperado —
      absorver em `docs/` na Fase 6 (ou 3) e então remover.
      **Observação para a Fase 4:** o `sdkconfig` gerado (deletado) tinha
      `FLASHMODE="dio"` embora `sdkconfig.defaults` peça `QIO` — conferir
      qual modo o build limpo realmente usa (afeta desempenho do flash).
- [ ] 1.4 Reduzir `python/imgs/` (12 MB) para 3–5 imagens de demonstração
      (mover 2–3 para `assets/`). Decisão pendente de quais imagens usar.
- [x] **Resultado:** 1,1 GB → **42 MB** (restante: notebook+modelos 23 MB,
      imgs de teste 12 MB, componentes 5,7 MB). Critério ≤ 60 MB atendido;
      itens 1.4 e o notebook (Fase 5) ainda podem reduzir mais.

## Fase 2 — Segurança e privacidade (P0 — bloqueador)

> **Adiada (nov/2026):** o repositório está **privado** e o autor optou por
> não alterar o histórico (ver D2 e 8.1). Reativar esta fase **antes** de
> qualquer publicação pública.

- [ ] 2.1 Remover SSID/senha de `src/main.cpp`; suportar configuração por
      um dos meios (decisão pendente **D2**):
      - Kconfig (`Kconfig.projbuild` → aparece no `menuconfig`) — recomendado;
      - ou variáveis de build no `platformio.ini` com placeholders;
      - ou NVS/provisionamento.
      Documentar no QUICKSTART como o usuário define a própria rede.
- [ ] 2.2 Notebook: remover caminho absoluto do dataset privado
      (`/home/jovyan/privado/wolosky_notebooks/...`) → parametrizar
      (`DATASET_DIR` configurável) + seção "como preparar seu dataset"
      (formato do `manifest.csv`).
- [ ] 2.3 Varrer o tree por outros dados pessoais: IPs internos (ok se
      documentado como exemplo), senhas, nomes de máquina, caminhos `/home/...`.
- [ ] 2.4 Confirmar **licença/origem do dataset SAM3 gatos/cachorro**.
      Se o dataset não for redistribuível, o repo publica só o **modelo
      treinado** + instruções para dataset próprio (decisão pendente **D3**).
- [ ] **Critério de aceitação:** `grep -r "Starlink\|diversao\|/home/"` limpo;
      nenhuma credencial no repo.

## Fase 3 — Estratégia dos componentes (P0) — **CONCLUÍDA (nov/2026)**

- [x] 3.1 `esp-nn` convertido em **submodule** (@`v1.3.0`, commit
      `d8866fa`): `.git` aninhado removido; `library.json` local
      preservado (cópia canônica em `patches/esp-nn/library.json` +
      restaurado não-trackeado dentro do submodule).
- [x] 3.2 Inconsistência ESP-NN resolvida na origem: esp-nn **atualizado de
      v1.1.2 → v1.3.0** (versão mínima documentada como pré-requisito para
      `-DESP_NN`). Flag `-DESP_NN` mantida **ativa**.
      **PENDÊNCIA (D5): revalidação bit a bit em hardware** com a v1.3.0
      (procedimento em `firmware/comandos.txt`; playbook no README §7).
- [x] 3.3 `tflite-lib`: **submodule upstream** `espressif/esp-tflite-micro`
      @ **v1.3.8** (`d6465fd`, nov/2026) + patches locais em
      `components/patches/tflite-lib/` (ver `components/PROVENANCE.md`).
      Histórico: a árvore antiga era um snapshot vendored de ~mai/2023 do
      `tensorflow/tflite-micro` com wrappers adaptados; o fingerprinting
      exaustivo mostrou que nenhum commit público a reproduzia. A saída foi
      migrar para o upstream e manter apenas os diffs necessários:
      - `space_to_batch_nd.patch` — fix `output_offset` (backport do fix que
        entrou no upstream na v1.4.3);
      - `concatenation.patch` — `kMaxInputNum` 10 → 128 (concats >10 entradas);
      - `cmake-esp-nn-requirement.patch` — declara `esp-nn` no CMakeLists
        (o upstream espera o IDF Component Manager);
      - `disable-component-manager.patch` — remove `idf_component.yml` para
        não duplicar o esp-nn em `managed_components/`.
      Fora do escopo, preservado em `patches/tflite-lib/reference/`:
      conv dilatada via sub-grids (o upstream cai no `reference`; porte
      pendente) e OPTRACE (instrumentação local).
- [x] 3.4 Versões documentadas em `PROVENANCE.md`: esp-tflite-micro v1.3.8
      (`d6465fd`) e esp-nn v1.3.0 (`d8866fa`).
- [~] **Critério de aceitação:** estrutura pronta — submodules pinados e
      patches gerados contra a tag e verificados em checkout pristino.
      **Falta o build e a revalidação bit a bit** na árvore nova (D5).

## Fase 4 — Firmware pronto para público (P1)

- [ ] 4.1 Renomear endpoint/telemetria? Não — manter `POST /predict_bin`,
      mas documentar o protocolo completo em `docs/HTTP_API.md` (headers,
      formato raw RGB 27.648 B, códigos de erro JSON).
- [ ] 4.2 `main.cpp`: revisar mensagens de log (PT-BR ok? ver D6), erros de
      WiFi com instruções claras, timeout/reconexão.
- [ ] 4.3 Considerar `idf_component.yml` com versões pinadas das deps
      (`joltwallet/littlefs`).
- [ ] 4.4 **Teste de mesa limpa:** `pio run -t clean` → build ok →
      `pio run -t uploadfs` → flash → inferência via cliente Python com
      imagem de exemplo → máscara coerente.
- [ ] 4.5 Cliente Python (`python/teste_inferencia.py`): parametrizar IP
      (já usa `ESP32_IP` env), `requirements.txt` próprio (requests, PIL,
      numpy, matplotlib), path de imagem via CLI em vez de linhas
      comentadas no fonte.
- [ ] **Critério de aceitação:** pessoa externa segue QUICKSTART e obtém
      uma máscara do próprio ESP32 sem perguntar nada.

## Fase 5 — Treino reproduzível (P1)

- [ ] 5.1 Limpar outputs do notebook (9,2 MB → ~100 KB) e ocultar/omitir
      células com dados privados; rodar "Restart & Clear All" antes de commit.
- [ ] 5.2 Parametrizar caminhos (dataset, saída) no topo do notebook.
- [ ] 5.3 `training/requirements.txt` com versões exatas (TF 2.18.1,
      tf2onnx, onnx, onnx2tf, ai-edge-litert...).
- [ ] 5.4 Publicar o **patch de shapes** como script reutilizável
      (`tools/patch_model_shapes.py`) com MD5 antes/depois e explicação
      do bug S2B/B2S (docs) — hoje é patch binário não versionado.
- [ ] 5.5 `models/`: listar artefatos + `MD5SUMS` (fp32/int8/ONNX) e deixar
      claro qual é o modelo **para o dispositivo** (patcheado, `449f...`).
- [ ] 6.6 (opcional) célula no notebook que aplica o patch automaticamente
      após a conversão, eliminando o passo manual do binário.
- [ ] **Critério de aceitação:** notebook roda de ponta a ponta com um
      dataset no formato documentado; artefatos batem com os MD5s publicados.

## Fase 6 — Documentação pública (P0 — é o que o público vê)

- [ ] 6.1 **Novo `README.md` raiz** com:
      - o que é (1 parágrafo) + imagem/GIF de resultado (`assets/`);
      - badges (license, platformio CI opcional);
      - tabela de métricas (IoU 0,8207 int8 / 0,8347 fp32; modelo 1,53 MB;
        61 operadores; tempo de inferência no device — medir na Fase 3/4);
      - hardware suportado (ESP32-S3 com PSRAM; testado em N16R8);
      - quickstart resumido com link para `docs/QUICKSTART.md`;
      - estrutura do repo + "para que serve cada pasta";
      - créditos e licença.
- [ ] 6.2 `docs/QUICKSTART.md`: passo a ponta sem conhecimento prévio
      (ferramentas, build, flash, uploadfs, cliente Python, solução de
      problemas comuns — colapsar as "armadilhas" do README interno).
- [ ] 6.3 `docs/MODEL_CARD.md`: classes, normalização ImageNet,
      quantização (scale/zp), métricas, limitações (96×96, 3 classes),
      MD5s.
- [ ] 6.4 `docs/PORTING_GUIDE.md`: mover seção 3 do README interno
      (receita esp-nn agnóstica + verificação com `nm`).
- [ ] 6.5 `docs/TFLM_DEBUG_PLAYBOOK.md`: mover seções 6–7 do README interno
      (diagnóstico do SPACE_TO_BATCH_ND + playbook de bisseção) — conteúdo
      de alto valor, é a "assinatura" técnica do projeto.
- [ ] 6.6 README antigo: arquivar como `docs/HISTORY_2026_09.md` ou descartar
      (o conteúdo útil é redistribuído em 6.4/6.5).
- [ ] **Decisão pendente D6:** idioma dos docs públicos — inglês (alcance
      maior) com PT-BR opcional, ou PT-BR. Recomendo inglês + espelho
      `README.pt-BR.md`.
- [ ] **Critério de aceitação:** alguém que nunca viu o projeto entende em
      <5 min o que faz, o que precisa e como rodar.

## Fase 7 — Licenças e créditos (P0)

- [ ] 7.1 Definir licença raiz compatível com as dependências (**D7**):
      TFLM e esp-nn são Apache-2.0 → Apache-2.0 (ou MIT para o código
      próprio, mantendo notices das partes vendored).
- [ ] 7.2 `THIRD_PARTY_NOTICES.md`: espressif/esp-tflite-micro, espressif/esp-nn
      (v + tag), joltwallet/littlefs, onnx2tf/tf2onnx, Fast-SCNN (paper).
- [ ] 7.3 Citar o paper Fast-SCNN e a origem da arquitetura
      (`tf_2_0_fast_scnn.py` é derivado de implementação de referência).
- [ ] 7.4 Verificar se `esp-nn/library.json` (arquivo local para Arduino)
      deve ficar — se sim, documentar o motivo no PORTING_GUIDE.
- [ ] **Critério de aceitação:** `LICENSE` presente; créditos rastreáveis.

## Fase 8 — Git, verificação final e publicação (P0)

- [x] 8.1 Repositório criado no GitHub
      (`vini-muchulski/ESP32_image_segmentation_cats_dogs`), **privado**,
      com 4 commits enviados (nov/2026): `7eb48dc` training, `38e05cb`
      docs, `97b5e02` firmware e `51a0b55` submodules.
      Decisões registradas: (a) `components/` **incluído** como submodules
      upstream + patches (ver D4); (b) **sem correções de segurança** no
      histórico — credenciais WiFi (`97b5e02`) e caminho privado no notebook
      (`7eb48dc`) continuam nos commits (ver D2). **Não tornar o
      repositório público** sem antes reescrever o histórico e trocar a
      senha do WiFi.
- [x] 8.1a Identidade de autor: os commits usam
      `viniciusmuchulski2@gmail.com`; o noreply
      (`105316221+vini-muchulski@users.noreply.github.com`) ficou
      configurado apenas no workspace. Reavaliar se o histórico for
      reescrito.
- [ ] 8.2 Confirmar tamanho do repo: `git count-objects -vH` ≲ 50 MB;
      notebook ≤ 1 MB (outputs limpos); modelos ~12 MB ok (sem LFS).
- [ ] 8.3 Smoke test de "estranho": `git clone --recurse-submodules` →
      aplicar os patches (`components/patches/`) → `pio run` → upload →
      inferência → máscara correta.
- [ ] 8.4 (Opcional, P2) GitHub Actions: build do firmware em push/PR
      (`pio run` com cache) — sinal de qualidade para visitantes.
- [ ] 8.5 (Opcional, P2) `CONTRIBUTING.md` + templates de issue.
- [ ] 8.6 Após tudo validado: remover `PLAN.md` do repo (ou mover para
      `docs/` como registro) e publicar.
- [ ] **Critério de aceitação:** clone limpo compila e roda; primeira
      impressão do README profissional.

---

## Decisões pendentes (responder antes de executar as fases)

| # | Questão | Decisão |
|---|---|---|
| D1 | Renomear pastas para inglês (`firmware/`, `training/`, `components/`)? | **RESOLVIDA: `training/`, `firmware/` e `components/` executados (nov/2026)**; `tools/`, `docs/` e `assets/` seguem opcionais |
| D2 | Como configurar WiFi sem hard-code | **Adiada (nov/2026):** credenciais seguem hard-coded enquanto o repo for privado; aplicar Kconfig (menuconfig) **antes** de qualquer publicação |
| D3 | Dataset é redistribuível? | Pendente |
| D4 | esp-nn: vendor vs submodule | **RESOLVIDA (nov/2026):** ambos **submodules upstream** — esp-nn @ v1.3.0 e esp-tflite-micro @ v1.3.8; patches locais em `components/patches/` (ver `PROVENANCE.md`) |
| D5 | Release com esp-nn v1.3.0 otimizado ou reference puro | **Parcialmente: esp-nn v1.3.0 + `-DESP_NN` ativo; falta revalidação bit a bit na árvore nova (TFLM v1.3.8 + patches) — procedimento em `firmware/comandos.txt`** |
| D6 | Idioma dos docs públicos | Pendente (recomendo inglês + `README.pt-BR.md`) |
| D7 | Licença | Pendente (recomendo Apache-2.0) |
| D8 | Versão do TFLM upstream | **v1.3.8** (não v1.4.3): mantém a API do `main.cpp` (`tflite_bridge/micro_error_reporter.h`) e o esp-nn v1.3.0; a v1.4.3 exigiria port da API + esp-nn ≥ 1.4.2 |

## Ordem de execução sugerida

```
Fase 1 (higiene)      → Fase 2 (segurança)   → Fase 3 (componentes)
→ Fase 4 (firmware)   → Fase 5 (treino)      → Fase 6 (docs)
→ Fase 7 (licença)    → Fase 8 (git+publicar)
```

Fases 1–3 são **bloqueadoras**: sem elas o repositório nasce com lixo de 1 GB,
segredo de WiFi e submodule quebrado. Fases 4–7 podem ser intercaladas.
