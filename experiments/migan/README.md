# MI-GAN experimental CPU probe

Status: **isolated evaluation only**. This is not yet a PhotoCraft command or a desktop model dependency.

This probe measures real inference with the compact 28,079,181-byte MI-GAN ONNX model.
It uses a pinned Hugging Face revision and SHA-256 checksum; neither the editor nor
the tests download weights automatically. Runtime dependencies remain in a separate
Python virtual environment; production integration should be native Rust + optional ONNX
Runtime, not this Python harness.

## Try on Linux (Python 3.11)

From the fork checkout:

1. python3 -m venv .venv-migan
2. . .venv-migan/bin/activate
3. pip install -r experiments/migan/requirements.txt
4. python -m unittest discover -s experiments/migan -p 'test_*.py' -v
5. python experiments/migan/download_model.py --download
6. python experiments/migan/make_fixture.py --directory /tmp/migan-eval
7. python experiments/migan/probe.py --model experiments/migan/models/migan_pipeline_v2.onnx --image /tmp/migan-eval/source.png --mask /tmp/migan-eval/selection.png --output /tmp/migan-eval/result.png --metrics /tmp/migan-eval/metrics.json

You can substitute real source and selection images. The selection is a grayscale
PNG with **white = change / remove** and **black = preserve**. Mask dimensions must
match the source. This probe currently converts the image to RGB, so alpha, CMYK and
16/32-bit documents are NOT yet covered. Input limits: 2048x2048, up to 4MP.

The underlying ONNX pipeline expects:
- image: uint8 [1,3,H,W], RGB channel-first
- mask: uint8 [1,1,H,W], 0=erase / 255=keep (inverted by the adapter)
- output: uint8 [1,3,H,W], composited inpainting

The output is additionally blended using PhotoCraft selection convention to guarantee
byte-identical RGB outside the selection. This guarantees **integrity**, not visual quality.

## Evidence and acceptance criteria

CI runs offline unit tests and a live CPU test on a synthetic 384x384 fixture.
It records model load/inference times, changed pixels and mask preservation, and
uploads the before/mask/after PNGs. Synthetic success does not validate quality on
photographs; compare licensed real examples against PhotoCraft's classical Remove
tool before integrating a native model.

Source: https://github.com/Picsart-AI-Research/MI-GAN
Pinned export: https://huggingface.co/andraniksargsyan/migan/tree/1538c135034b8cfe7a8472f34d09c8a5a45b17a7
