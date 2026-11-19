import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter, OpResolverType

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "data" / "gatos_v5_fast_scnn.tflite"
INPUT_IMAGE_PATH = Path(
    os.environ.get("INPUT_IMAGE_PATH", BASE_DIR / "imgs" / "golden_recortado.jpg")
)
INPUT_INFO_PATH = BASE_DIR / "input_info.json"
RESULTS_PC_DIR = BASE_DIR / "results_pc"
TRACE_RE = re.compile(
    r"OPTRACE op=(\d+) name=([^ ]+) tensor=(\d+) bytes=(\d+) "
    r"hash=(?:0x)?([0-9a-fA-F]+) min=(-?\d+) max=(-?\d+) sum=(-?\d+)"
)


def fnv1a32(data):
    value = 2166136261
    for byte in memoryview(np.ascontiguousarray(data)).cast("B"):
        value = ((value ^ int(byte)) * 16777619) & 0xFFFFFFFF
    return value


def tensor_stats(op_index, op_name, tensor_index, tensor):
    raw = np.ascontiguousarray(tensor)
    flat = raw.reshape(-1)
    return {
        "op": int(op_index),
        "name": op_name,
        "tensor": int(tensor_index),
        "bytes": int(raw.nbytes),
        "hash": f"{fnv1a32(raw):08x}",
        "min": int(flat.min()) if flat.size else 0,
        "max": int(flat.max()) if flat.size else 0,
        "sum": int(flat.astype(np.int64).sum()),
    }


def run_pc_trace():
    info = json.loads(INPUT_INFO_PATH.read_text())
    image = Image.open(INPUT_IMAGE_PATH).convert("RGB").resize((96, 96), Image.BILINEAR)
    image_u8 = np.asarray(image, dtype=np.uint8)
    value = image_u8.astype(np.float32) / np.float32(255.0)
    value = (value - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    quantized = np.clip(
        np.rint(value / np.float32(info["scale"])) + int(info["zero_point"]),
        -128,
        127,
    ).astype(np.int8)

    interpreter = Interpreter(
        model_path=str(MODEL_PATH),
        experimental_preserve_all_tensors=True,
        experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], quantized[None])
    interpreter.invoke()

    records = []
    for op in interpreter._get_ops_details():
        for tensor_index in op["outputs"]:
            tensor_index = int(tensor_index)
            if tensor_index < 0:
                continue
            tensor = interpreter.get_tensor(tensor_index)
            records.append(
                tensor_stats(op["index"], op["op_name"], tensor_index, tensor)
            )
    return records


def parse_device_trace(path):
    records = []
    for match in TRACE_RE.finditer(Path(path).read_text(errors="replace")):
        op, name, tensor, size, hash_value, min_value, max_value, sum_value = match.groups()
        records.append(
            {
                "op": int(op),
                "name": name,
                "tensor": int(tensor),
                "bytes": int(size),
                "hash": f"{int(hash_value, 16):08x}",
                "min": int(min_value),
                "max": int(max_value),
                "sum": int(sum_value),
            }
        )
    return records


def main():
    pc_records = run_pc_trace()
    stem = INPUT_IMAGE_PATH.stem
    output_path = RESULTS_PC_DIR / f"{stem}_optrace_pc.json"
    output_path.write_text(json.dumps(pc_records, indent=2) + "\n")
    print(f"trace_pc_path: {output_path}")
    for record in pc_records:
        print(
            "OPTRACE_PC "
            + " ".join(f"{key}={record[key]}" for key in (
                "op", "name", "tensor", "bytes", "hash", "min", "max", "sum"
            ))
        )

    if len(sys.argv) < 2:
        print("\nInforme o log ESP32 como argumento para comparar automaticamente.")
        return

    device_records = parse_device_trace(sys.argv[1])
    device_by_key = {(r["op"], r["tensor"]): r for r in device_records}
    print(f"\nregistros_pc={len(pc_records)} registros_esp32={len(device_records)}")
    first_hash_difference = None
    first_stats_difference = None
    for pc in pc_records:
        device = device_by_key.get((pc["op"], pc["tensor"]))
        if device is None:
            continue
        if pc["hash"] != device["hash"] and first_hash_difference is None:
            first_hash_difference = (pc, device)
        stats_equal = all(pc[key] == device[key] for key in ("bytes", "min", "max", "sum"))
        if not stats_equal and first_stats_difference is None:
            first_stats_difference = (pc, device)

    for label, difference in (
        ("primeira diferença de hash", first_hash_difference),
        ("primeira diferença estatística", first_stats_difference),
    ):
        print(f"\n{label}:")
        if difference is None:
            print("nenhuma")
        else:
            pc, device = difference
            print("PC:   ", pc)
            print("ESP32:", device)


if __name__ == "__main__":
    main()
