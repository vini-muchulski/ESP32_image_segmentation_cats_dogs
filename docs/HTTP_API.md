# HTTP API

The firmware serves HTTP on port 80. The client and ESP32 must be able to reach each other over Wi-Fi.

| Method and path | Response |
| --- | --- |
| `GET /` | HTML endpoint summary |
| `GET /status` | JSON model readiness and available heap |
| `POST /predict_bin` | Binary mask or logits on success, JSON on application errors |

## Send an image

Send exactly **27,648 bytes** of unsigned RGB pixels for one 96 × 96 image, in row-major order with three interleaved channels per pixel. Use `Content-Type: application/octet-stream`. The firmware normalizes and quantizes the pixels before inference. [Input format and quantization](MODEL_CARD.md)

```bash
curl -sS -D response.headers \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @image.rgb \
  "http://${ESP32_IP}/predict_bin" -o mask.bin
```

The example assumes `image.rgb` already contains the resized RGB bytes. Omit `X-Return-Logits` for a mask. Send `X-Return-Logits: 1` for logits. The firmware checks for a nonempty header value.

## Decode the response

| `X-Response-Kind` | Binary body for the bundled model |
| --- | --- |
| `mask` | 9,216 unsigned bytes, one class ID per pixel in row-major order |
| `logits` | 27,648 signed INT8 bytes, shaped as height × width × 3 |

Class IDs are `0` background, `1` cat, and `2` dog. `X-Width` and `X-Height` describe both outputs. `X-Channels` reports the number of **logit** channels even when the response body is a mask. `X-Logits-Dtype` indicates the logit type.

| Header | Meaning |
| --- | --- |
| `X-Input-FNV1A`, `X-Logits-FNV1A` | FNV-1a 32-bit hashes in hexadecimal for the quantized model input and raw logits |
| `X-Input-Copy-Us`, `X-Inference-Us`, `X-Postprocess-Us`, `X-Total-Processing-Us` | Timings in microseconds. Inference measures `Invoke()`. |
| `X-Internal-Before`, `X-Internal-After`, `X-Psram-Before`, `X-Psram-After`, `X-Min-Free-Internal`, `X-Min-Free-Psram` | Heap measurements in bytes |
| `X-Arena-Used`, `X-Arena-Capacity`, `X-Model-Bytes`, `X-App-Partition-Bytes`, `X-Flash-Chip-Bytes` | Arena, model, partition and flash sizes in bytes |
| `X-Firmware-Bytes` | Placeholder. The current firmware sends `0`. |

## Handle errors

Both status and application error JSON include `success`, `model_initialized`, `mask_width`, `mask_height`, `mask_bytes`, `heap_free` and `error_message`. Prediction errors include invalid content length, allocation or receive failure, and inference failure. These responses use HTTP 200 with `Content-Type: application/json` and `"success": false`. Check the content type before interpreting the body as binary. [HTTP handlers](../firmware/src/main.cpp)
