# TFLite Micro debugging playbook

## Compare the host and device

1. Use the deployed model from `firmware/data/` on both runtimes. Resize the same source image to the same 96 × 96 RGB bytes.
2. Compare the quantized input hash with `X-Input-FNV1A`. Investigate an input mismatch before comparing operators.
3. Request raw logits with `X-Return-Logits: 1`. Compare all INT8 values, report the maximum absolute difference and compare the final masks.
4. Repeat with reference kernels in a clean build to isolate differences introduced by the optimized path. [Porting guide](PORTING_GUIDE.md#verify-the-optimized-build)

From `firmware/`, after following the [quickstart](QUICKSTART.md), run host inference before calling the device so the device client can compare against the saved host logits. Set `ESP32_IP` to the address from the serial monitor.

```bash
python -m pip install ai-edge-litert
export INPUT_IMAGE_PATH="$PWD/python/imgs/golden_recortado.jpg"
export MODEL_PATH="$PWD/data/gatos_v5_fast_scnn.tflite"
python python/inferencia_pc.py
python python/teste_inferencia.py
```

## Find the first divergent operator

The pinned TFLite Micro tree does not include the historical `OPTRACE` instrumentation. To trace operators, instrument each output immediately after its operator executes, while its data still occupies the arena. Log the operator and tensor indices, byte count, FNV-1a hash, minimum, maximum and sum. Capture serial output during the same request, then compare it with the host trace.

```bash
INPUT_IMAGE_PATH="$PWD/python/imgs/golden_recortado.jpg" \
  python python/trace_operadores_pc.py path/to/esp32-optrace.log
```

Use the last matching tensor to reproduce the first discrepancy. Remove the instrumentation before measuring latency. [Trace comparison script](../firmware/python/trace_operadores_pc.py)

## Known regressions

- **Static tensor shapes.** The original export declared tensors 141 and 142 with leading dimension 1. The `SPACE_TO_BATCH_ND → DEPTHWISE_CONV_2D → BATCH_TO_SPACE_ND` sequence needs dimension 16 for a 4 × 4 block. LiteRT resized these tensors at runtime. The deployed artifact carries the corrected static shapes.
- **INT8 padding.** In the older `SPACE_TO_BATCH_ND` kernel, the padding offset was uninitialized. The local patch sets it to the output tensor's zero point. The first divergent operator was 46, and the captured discrepancy was reproduced by using `0xAA` for padding. [Kernel patch](../components/patches/tflite-lib/space_to_batch_nd.patch)

The historical reference build matched the host on all 61 operators for `golden_recortado.jpg`, including all 27,648 logits and 9,216 mask pixels. The current optimized submodule build still needs its own parity run. [Original diagnosis in Portuguese](HISTORY_2026_09.md)
