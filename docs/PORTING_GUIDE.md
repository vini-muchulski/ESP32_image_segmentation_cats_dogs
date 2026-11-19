# Porting the ESP-NN components

The project pins Espressif TFLite Micro 1.3.8 and ESP-NN 1.3.0. Start with both submodules and apply the local TFLite Micro patches as described in the [quickstart](QUICKSTART.md#prepare-the-checkout). [Versions and patches](../components/PROVENANCE.md)

## Add the components to an ESP-IDF project

Keep the two component directories under the same parent. Add that parent to `EXTRA_COMPONENT_DIRS` before including the ESP-IDF project CMake file. Replace the example path with the location of the components.

```cmake
list(APPEND EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../components")
```

Declare `tflite-lib` in the application's `REQUIRES` or `PRIV_REQUIRES`. In this checkout, the patched TFLite Micro component declares `esp-nn` as a local dependency and compiles with `-DESP_NN`. Register the model's operators in `MicroMutableOpResolver`. For this model, the firmware registers nine operator types. [Component CMake](../firmware/src/CMakeLists.txt) · [Operator registration](../firmware/src/main.cpp)

## Verify the optimized build

Use the `nm` executable from the ESP32-S3 toolchain. Check both the linked ELF and the TFLite Micro archive for the environment you built. PlatformIO stores the build under `firmware/.pio/build/esp32s3/`.

```bash
"$NM" firmware.elf | rg ' T esp_nn.*_esp32s3'
"$NM" -A libtflite-lib.a | rg ' U esp_nn'
```

Run these commands with `NM` set to the toolchain executable and the paths set to the actual build artifacts. The ELF shows linked kernels. The archive shows which wrappers reference them.

Validate the same model and inputs with ESP-NN enabled and with reference kernels. For the pinned TFLite Micro version, remove `-DESP_NN` from the TFLite Micro component for the reference build and clean after changing it. Compare quantized input, raw logits and masks before comparing `X-Inference-Us`. [Debugging playbook](TFLM_DEBUG_PLAYBOOK.md) · [HTTP telemetry](HTTP_API.md#decode-the-response)
