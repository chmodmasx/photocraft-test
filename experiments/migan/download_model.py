#!/usr/bin/env python3
"""Opt-in download and verification of one immutable MI-GAN ONNX export."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

REV = "1538c135034b8cfe7a8472f34d09c8a5a45b17a7"
NAME = "migan_pipeline_v2.onnx"
SOURCE = f"https://huggingface.co/andraniksargsyan/migan/resolve/{REV}/{NAME}"
SHA256 = "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b"
SIZE = 28079181

def verify(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != SIZE:
        raise ValueError("MI-GAN weights missing or wrong size")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError("MI-GAN model SHA-256 mismatch")

def download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify(path)
        return
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".migan-", suffix=".partial")
    os.close(fd)
    tmp = Path(name)
    try:
        req = urllib.request.Request(SOURCE, headers={"User-Agent": "PhotoCraft-model-evaluation"})
        total = 0
        with urllib.request.urlopen(req, timeout=90) as response, tmp.open("wb") as output:
            if not response.geturl().startswith("https://"):
                raise ValueError("Refusing insecure model redirect")
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > SIZE:
                    raise ValueError("Pinned model size exceeded")
                output.write(chunk)
        verify(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path(__file__).parent / "models" / NAME)
    p.add_argument("--download", action="store_true", help="Explicitly download 28.1 MB of pinned weights")
    args = p.parse_args()
    if args.download:
        download(args.model)
    else:
        verify(args.model)
    print("Verified pinned MI-GAN model:", args.model)
