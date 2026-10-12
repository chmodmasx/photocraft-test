#!/usr/bin/env python3
"""Produce synthetic, redistribution-safe images for CPU smoke testing."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def make(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    w = h = 384
    yy, xx = np.mgrid[:h, :w]
    pixels = np.empty((h, w, 3), dtype=np.uint8)
    pixels[:,:,0] = np.clip(150 + xx//9, 0, 255)
    pixels[:,:,1] = np.clip(180 + yy//9, 0, 255)
    pixels[:,:,2] = 205
    image = Image.fromarray(pixels, "RGB")
    draw = ImageDraw.Draw(image)
    draw.line((20, 174, 364, 174), fill=(80,85,90), width=6)
    draw.rectangle((142,124,245,253), fill=(185,45,37))
    mask = Image.new("L", (w,h), 0)
    ImageDraw.Draw(mask).rectangle((137,119,250,258), fill=255)
    image.save(directory/"source.png")
    mask.save(directory/"selection.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    make(p.parse_args().directory)
