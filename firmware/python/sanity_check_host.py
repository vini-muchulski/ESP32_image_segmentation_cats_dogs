import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("ABSL_LOGLEVEL", "3")

stderr_fd = sys.stderr.fileno()
saved_stderr_fd = os.dup(stderr_fd)
try:
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), stderr_fd)
    from ai_edge_litert.interpreter import Interpreter
finally:
    os.dup2(saved_stderr_fd, stderr_fd)
    os.close(saved_stderr_fd)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "data" / "gatos_v5_fast_scnn.tflite"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
COLOR = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]], dtype=np.uint8)

DEFAULT_IMAGES = [
    "golden_recortado.jpg",
    "cat0.jpg",
    "img2.jpg",
    "000000574315.jpg",
]


def main():
    images = sys.argv[1:] or DEFAULT_IMAGES
    it = Interpreter(model_path=str(MODEL_PATH))
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    scale, zp = inp["quantization"]
    h, w = int(inp["shape"][1]), int(inp["shape"][2])

    for name in images:
        path = Path(name)
        if not path.is_absolute():
            path = BASE_DIR / "imgs" / name
        img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
        u8 = np.asarray(img, dtype=np.uint8)
        v = (u8.astype(np.float32) / 255.0 - MEAN) / STD
        q = np.clip(np.rint(v / scale) + zp, -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], q.reshape(1, h, w, 3))
        it.invoke()
        logits = it.get_tensor(out["index"])[0]
        mask = np.argmax(logits, axis=-1).astype(np.uint8)
        counts = np.bincount(mask.flatten(), minlength=3)
        out_path = RESULTS_DIR / f"mask_host_{path.stem}.png"
        Image.fromarray(COLOR[mask]).save(out_path)
        print(f"{path.name}: classes(fundo,gato,cachorro)={counts.tolist()} -> {out_path}")


if __name__ == "__main__":
    main()
