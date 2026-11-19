import os

import numpy as np
from PIL import Image

import matplotlib

MPL_BACKEND = "TkAgg"
try:
    matplotlib.use(MPL_BACKEND, force=True)
except Exception as exc:
    matplotlib.use("Agg")
    print(f"Aviso: falha ao usar {MPL_BACKEND} ({exc}). Usando Agg (sem GUI).")

import matplotlib.pyplot as plt
from ai_edge_litert.interpreter import Interpreter, OpResolverType

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(BASE_DIR, "..", "data", "gatos_v5_fast_scnn.tflite")
)

#path_img = "imgs/golden_recortado.jpg"
#path_img = "imgs/000000574315.jpg"
path_img = "imgs/000000475732.jpg"

INPUT_IMAGE_PATH = os.environ.get(
    "INPUT_IMAGE_PATH", os.path.join(BASE_DIR, path_img)
)
RESULTS_DIR = os.path.join(BASE_DIR, "results_pc")
os.makedirs(RESULTS_DIR, exist_ok=True)
IMG_SIZE = (96, 96)


def preprocess_image(image_path, size):
    img = Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def run_inference(model_path, img_u8):
    interpreter = Interpreter(
        model_path=model_path, experimental_op_resolver_type=OpResolverType.BUILTIN_REF
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    v = img_u8.astype(np.float32) / 255.0
    v = (v - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    h, w = img_u8.shape[:2]
    q = np.clip(np.rint(v / in_scale) + in_zp, -128, 127).astype(np.int8)
    interpreter.set_tensor(inp["index"], q.reshape(1, h, w, 3))
    interpreter.invoke()

    logits = interpreter.get_tensor(out["index"])
    mask = np.argmax(logits[0].astype(np.int32), axis=-1).astype(np.uint8)
    print(f"quantizacao: in_scale={in_scale} in_zp={in_zp} | out_scale={out_scale} out_zp={out_zp}")
    return mask, logits[0]


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
    axes[2].set_title("Mask (PC) classes")
    axes[2].axis('off')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"plot_path: {save_path}")
    plt.show()


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Modelo não encontrado: {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"ERROR: Imagem não encontrada: {INPUT_IMAGE_PATH}")
        return

    img_u8 = preprocess_image(INPUT_IMAGE_PATH, IMG_SIZE)
    mask, logits = run_inference(MODEL_PATH, img_u8)

    stem = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
    np.save(os.path.join(RESULTS_DIR, stem + "_mask_pc.npy"), mask)
    np.save(os.path.join(RESULTS_DIR, stem + "_logits_pc.npy"), logits)
    print("classes:", {int(c): int((mask == c).sum()) for c in np.unique(mask)})
    print("\n--- Resultado ---")
    print(f"mask_shape: {mask.shape}")
    print(f"logits_min: {logits.min()} logits_max: {logits.max()}")
    plot_result(img_u8, mask, save_path=os.path.join(RESULTS_DIR, stem + "_plot_pc.png"))


if __name__ == "__main__":
    main()
