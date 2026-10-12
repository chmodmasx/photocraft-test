#!/usr/bin/env python3
"""Standalone CPU-only MI-GAN inpainting probe. NOT yet a PhotoCraft command.

External mask: white=remove, black=preserve. Model mask: black=remove, white=keep.
All pixels outside the requested mask are preserved byte-for-byte.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import numpy as np
from PIL import Image
from download_model import verify

def prepare(image: Image.Image, mask: Image.Image):
    if image.size != mask.size:
        raise ValueError("Image and mask sizes differ")
    w, h = image.size
    if not (0 < w <= 2048 and 0 < h <= 2048 and w * h <= 4194304):
        raise ValueError("Evaluation limit: 2048 pixels per side, 4MP total")
    source = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    selection = np.asarray(mask.convert("L"), dtype=np.uint8).copy()
    if not np.any(selection) or np.all(selection):
        raise ValueError("Selection must contain both changed and unchanged pixels")
    rgb = np.ascontiguousarray(source.transpose(2, 0, 1)[None])
    known = np.where(selection > 0, 0, 255).astype(np.uint8)[None, None]
    return rgb, known, source, selection

def composite(source: np.ndarray, generated: np.ndarray, selection: np.ndarray):
    if generated.shape != source.shape or generated.dtype != np.uint8 or selection.shape != source.shape[:2]:
        raise ValueError("Invalid model result shape or dtype")
    a = selection.astype(np.float32)[:, :, None] / 255.0
    out = np.rint(source.astype(np.float32) * (1-a) + generated.astype(np.float32) * a)
    out = out.clip(0, 255).astype(np.uint8)
    out[selection == 0] = source[selection == 0]
    return out

def run(model: Path, image: Path, mask: Path, output: Path, threads: int):
    verify(model)
    if output.resolve() in {p.resolve() for p in (model, image, mask)}:
        raise ValueError("Output cannot overwrite an input or the model")
    with Image.open(image) as im, Image.open(mask) as m:
        rgb, known, source, selected = prepare(im, m)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    start = perf_counter()
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    loaded = perf_counter()
    types = {i.name: i.type for i in session.get_inputs()}
    if types != {"image": "tensor(uint8)", "mask": "tensor(uint8)"}:
        raise ValueError(f"Unexpected model interface: {types}")
    predictions = session.run(None, {"image": rgb, "mask": known})
    ended = perf_counter()
    if len(predictions) != 1 or predictions[0].shape != rgb.shape or predictions[0].dtype != np.uint8:
        raise ValueError("Unexpected output from MI-GAN")
    generated = np.ascontiguousarray(predictions[0][0].transpose(1,2,0))
    result = composite(source, generated, selected)
    unchanged = selected == 0
    changed = selected > 0
    outside = int(np.count_nonzero(np.any(result[unchanged] != source[unchanged], axis=1)))
    inside = int(np.count_nonzero(np.any(result[changed] != source[changed], axis=1)))
    if outside != 0:
        raise ValueError("Outside-mask pixel integrity violation")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(dir=output.parent, prefix=".migan-", suffix=".png")
    os.close(fd)
    tmp = Path(tmpname)
    try:
        Image.fromarray(result, "RGB").save(tmp, "PNG")
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return dict(width=source.shape[1], height=source.shape[0],
                load_seconds=round(loaded-start,3), inference_seconds=round(ended-loaded,3),
                outside_mask_changed_pixels=outside, inside_mask_changed_pixels=inside,
                selected_pixels=int(np.count_nonzero(changed)), cpu_threads=threads)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "image", "mask", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--metrics", type=Path)
    p.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    args = p.parse_args()
    if not 1 <= args.threads <= 64:
        p.error("threads must be in [1,64]")
    metrics = run(args.model, args.image, args.mask, args.output, args.threads)
    print(json.dumps(metrics, indent=2))
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
