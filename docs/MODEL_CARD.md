# Fast-SCNN model card

The model performs three-class semantic segmentation of 96 × 96 RGB images. It has approximately 1.204 million parameters and uses pyramid pooling bins of 1 and 3. Class IDs are `0` background, `1` cat, and `2` dog.

## Input and output

| Property | Value |
| --- | --- |
| Model input | `[1, 96, 96, 3]` signed INT8 |
| Model output | `[1, 96, 96, 3]` signed INT8 logits |
| Input scale | `0.01865844801068306` |
| Input zero point | `-14` |
| Mask | Per-pixel argmax over the three channels |

The host client resizes images with bilinear interpolation and sends unsigned RGB. The ESP32 normalizes each channel as `(pixel / 255 - mean) / std`, with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. It rounds after dividing by the input scale, adds the zero point and clamps to the INT8 range. [Input metadata](../firmware/python/input_info.json) · [HTTP API](HTTP_API.md)

## Evaluation

| Runtime | Multiclass IoU on the original test set |
| --- | ---: |
| Keras FP32 | 0.8267 |
| ONNX dynamic INT8 | 0.8339 |
| TFLite FP32 | 0.8347 |
| TFLite full INT8 | 0.8207 |

The scores come from the original host evaluation. Fine contours and small objects can lose detail at 96 × 96. The model assigns semantic classes to pixels and merges instances that share the same class ID. The original notebook references a private dataset. Its expected manifest columns are `img_path` and `msk_path`, with mask values from 0 to 2. [Training notes](../training/README.md) · [INT8 test examples](../training/results/viz_test_TFLITE_INT8_gatos_cachorro_unet_v5_fast_scnn_96px.png)

## Artifacts

| Artifact | Purpose | MD5 |
| --- | --- | --- |
| [Firmware INT8 model](../firmware/data/gatos_v5_fast_scnn.tflite) | LittleFS deployment, 1,527,800 bytes | `449f7358985e6f3bd59aedc57272cb5e` |
| [Original INT8 export](../training/models/model_gatos_cachorro_unet_v5_fast_scnn_96px_infer_int8_full.tflite) | Original training export | `179689ebac29524d449141e81a9c97d1` |

The firmware artifact corrects the leading dimension of two intermediate tensors from 1 to 16 for the lowered dilated convolution. Use the firmware artifact for ESP32 deployment. [Export checksums](../training/models/MD5SUMS) · [Shape diagnosis](TFLM_DEBUG_PLAYBOOK.md#known-regressions)
