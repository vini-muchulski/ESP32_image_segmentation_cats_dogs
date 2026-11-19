import os
import sys
import json

import numpy as np
import requests
from PIL import Image

import matplotlib

MPL_BACKEND = "TkAgg"
try:
    matplotlib.use(MPL_BACKEND, force=True)
except Exception as exc:
    matplotlib.use("Agg")
    print(f"Aviso: falha ao usar {MPL_BACKEND} ({exc}). Usando Agg (sem GUI).")

import matplotlib.pyplot as plt

ESP32_IP = os.environ.get("ESP32_IP", "192.168.3.22")
#ESP32_IP = os.environ.get("ESP32_IP", "192.168.3.59")

PREDICT_URL = f"http://{ESP32_IP}/predict_bin"
REQUEST_TIMEOUT = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_INFO_PATH = os.environ.get("INPUT_INFO_PATH", os.path.join(BASE_DIR, "input_info.json"))

#path_img = "imgs/golden_recortado.jpg"
#path_img = "imgs/000000574315.jpg"
#path_img = "imgs/000000475732.jpg"
#path_img = "imgs/000000052891.jpg"
#path_img = "imgs/cat_twt/cat2.jpeg"
#path_img = "imgs/cat_twt/cat8.jpeg"

#path_img = "imgs/gatinho_laranja.jpg"

#path_img = "imgs/twt/gato5.jpeg"
path_img = "imgs/twt/dog1.jpeg"

INPUT_IMAGE_PATH = os.environ.get(
    "INPUT_IMAGE_PATH", os.path.join(BASE_DIR, path_img)
)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PLOT_PATH = os.environ.get(
    "OUTPUT_PLOT_PATH",
    os.path.join(RESULTS_DIR, os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0] + "_plot.png"),
)
IMG_SIZE = (96, 96)


def load_input_info(path):
    with open(path, "r") as f:
        info = json.load(f)
    info["size"] = (int(info["size"][0]), int(info["size"][1]))
    return info


def preprocess_image(image_path, size):
    img = Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def quantize_input(img_u8, info):
    v = img_u8.astype(np.float32) / np.float32(255.0)
    v = (v - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.clip(
        np.rint(v / np.float32(info["scale"])) + int(info["zero_point"]),
        -128,
        127,
    ).astype(np.int8)


def fnv1a32(data):
    value = 2166136261
    for byte in memoryview(np.ascontiguousarray(data)).cast("B"):
        value = ((value ^ int(byte)) * 16777619) & 0xFFFFFFFF
    return value


def send_bin(qimg):
    data = qimg.tobytes()
    r = requests.post(
        PREDICT_URL,
        data=data,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Return-Logits": "1",
        },
        timeout=(5, REQUEST_TIMEOUT),
    )
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "application/octet-stream" in ctype:
        w = int(r.headers.get("X-Width", "0"))
        h = int(r.headers.get("X-Height", "0"))
        channels = int(r.headers.get("X-Channels", "0"))
        return {
            "ok": True,
            "width": w,
            "height": h,
            "channels": channels,
            "response_kind": r.headers.get("X-Response-Kind", "mask"),
            "payload": r.content,
            "headers": r.headers,
        }
    try:
        return {"ok": False, "json": r.json()}
    except ValueError:
        return {"ok": False, "error": f"Resposta inesperada: {ctype}"}


def print_telemetry(headers):
    keys = [
        "X-Input-Copy-Us",
        "X-Inference-Us",
        "X-Postprocess-Us",
        "X-Total-Processing-Us",
        "X-Arena-Used",
        "X-Arena-Capacity",
        "X-Model-Bytes",
        "X-Psram-After",
    ]
    print("\n--- Telemetria ESP32 ---")
    for k in keys:
        v = headers.get(k)
        if v is not None:
            print(f"{k}: {v}")
    inference_us = headers.get("X-Inference-Us")
    if inference_us is not None:
        print(f"inference_seconds: {int(inference_us) / 1_000_000.0:.6f}")


def class_mask_overlay(image, mask, alpha=0.45):
    image = np.asarray(image, dtype=np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    if image.shape[:2] != mask.shape:
        image = np.asarray(
            Image.fromarray((image * 255).astype(np.uint8)).resize(
                (mask.shape[1], mask.shape[0]), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0

    color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.float32) / 255.0
    overlay = image.copy()
    foreground = mask != 0
    overlay[foreground] = (1.0 - alpha) * overlay[foreground] + alpha * color_map[mask[foreground]]
    return np.clip(overlay, 0.0, 1.0)


def plot_result(image, mask, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis('off')

    axes[1].imshow(class_mask_overlay(image, mask))
    axes[1].set_title("Overlay mask vs fundo")
    axes[1].axis('off')

    color_map = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    axes[2].imshow(color_map[mask])
    axes[2].set_title("Mask (ESP32) classes")
    axes[2].axis('off')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"plot_path: {save_path}")
    plt.show()


def main():
    if os.path.exists(INPUT_INFO_PATH):
        input_info = load_input_info(INPUT_INFO_PATH)
        model_size = input_info["size"]
    else:
        input_info = None
        model_size = None
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"ERROR: Imagem não encontrada: {INPUT_IMAGE_PATH}")
        return

    if model_size is not None and IMG_SIZE is not None and tuple(IMG_SIZE) != tuple(model_size):
        print(f"Aviso: IMG_SIZE {IMG_SIZE} difere do input_info.json {model_size}")
    size = IMG_SIZE if IMG_SIZE is not None else model_size
    img_u8 = preprocess_image(INPUT_IMAGE_PATH, size)
    expected_input_hash = None
    if input_info is not None:
        expected_input_hash = fnv1a32(quantize_input(img_u8, input_info))
    res = send_bin(img_u8)
    if not res.get("ok"):
        payload = res.get("json") or res.get("error")
        print(f"Falha: {payload}")
        return

    w = res.get("width", 0)
    h = res.get("height", 0)
    channels = res.get("channels", 0)
    payload = res.get("payload", b"")
    if w <= 0 or h <= 0 or channels <= 0 or len(payload) != w * h * channels:
        print(
            f"Falha: logits inválidos (w={w}, h={h}, c={channels}, "
            f"bytes={len(payload)})"
        )
        return

    print_telemetry(res.get("headers", {}))

    logits_dtype = res["headers"].get("X-Logits-Dtype", "int8")
    dtype = np.int8 if logits_dtype == "int8" else np.uint8
    logits = np.frombuffer(payload, dtype=dtype).reshape((h, w, channels)).copy()
    mask = np.argmax(logits.astype(np.int32), axis=-1).astype(np.uint8)
    stem = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
    device_input_hash = res["headers"].get("X-Input-FNV1A")
    device_logits_hash = res["headers"].get("X-Logits-FNV1A")
    local_logits_hash = f"{fnv1a32(logits):08x}"
    print("\n--- Diagnóstico de hashes ---")
    if expected_input_hash is not None:
        expected_hex = f"{expected_input_hash:08x}"
        print(
            f"input PC={expected_hex} ESP32={device_input_hash} "
            f"igual={expected_hex == device_input_hash}"
        )
    print(
        f"logits recebidos={local_logits_hash} ESP32={device_logits_hash} "
        f"igual={local_logits_hash == device_logits_hash}"
    )
    pc_logits_path = os.path.join(RESULTS_DIR + "_pc", stem + "_logits_pc.npy")
    if os.path.exists(pc_logits_path):
        pc_logits = np.load(pc_logits_path)
        diff = logits.astype(np.int16) - pc_logits.astype(np.int16)
        print(
            "logits ESP32 vs PC: "
            f"iguais={np.mean(diff == 0):.6f} "
            f"max_abs_diff={np.max(np.abs(diff))} "
            f"mean_abs_diff={np.mean(np.abs(diff)):.6f}"
        )
    print("classes:", {int(c): int((mask == c).sum()) for c in np.unique(mask)})
    print("\n--- Resultado ---")
    print(f"mask_width: {w}")
    print(f"mask_height: {h}")
    print(f"logits_bytes: {len(payload)}")
    plot_result(img_u8, mask, save_path=OUTPUT_PLOT_PATH)


if __name__ == "__main__":
    main()
