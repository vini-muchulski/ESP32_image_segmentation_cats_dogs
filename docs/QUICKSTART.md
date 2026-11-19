# Quickstart

## Requirements

- ESP32-S3 DevKitC-1 N16R8 with 16 MB flash and 8 MB PSRAM, and a USB data cable.
- Git, PlatformIO Core, Python 3 with Tk support for the client plot, and a Wi-Fi network shared by the PC and board.

## Prepare the checkout

From the repository root, initialize the pinned submodules and apply the four local TFLite Micro patches. On an existing checkout with patches already applied, the loop skips them.

```bash
(
  set -e
  git submodule update --init --recursive
  for patch in "$PWD"/components/patches/tflite-lib/*.patch; do
    if ! git -C components/tflite-lib apply --reverse --check "$patch" 2>/dev/null; then
      git -C components/tflite-lib apply "$patch"
    fi
  done
)
```

Set the Wi-Fi SSID and password for your network in [the firmware source](../firmware/src/main.cpp). The firmware currently reads both values from that file.

## Flash the firmware and model

```bash
cd firmware
pio run -e esp32s3
pio run -e esp32s3 -t upload
pio run -e esp32s3 -t uploadfs
pio device monitor -b 115200
```

The filesystem upload writes the model from [the data directory](../firmware/data/). After reboot, use the IP address shown in the serial monitor. If PlatformIO selects a different serial port, pass `--upload-port` to the upload commands and `--port` to the monitor.

## Run inference

From `firmware/`, replace the example address with the ESP32 address shown in the monitor.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install requests pillow numpy matplotlib

export ESP32_IP=192.0.2.10
curl "http://${ESP32_IP}/status"
INPUT_IMAGE_PATH="$PWD/python/imgs/golden_recortado.jpg" python python/teste_inferencia.py
```

Check that `/status` returns `"model_initialized": true`. The client requests the model's logits, computes the mask and saves the plot as `python/results/golden_recortado_plot.png`.

If LittleFS fails to mount or the model cannot be opened, check the serial output and repeat `uploadfs` using the same PlatformIO environment. If the board is unreachable, confirm its serial-reported address and verify that the network allows clients to communicate. For output mismatches, compare the input hash before investigating the model. [Debugging playbook](TFLM_DEBUG_PLAYBOOK.md)
