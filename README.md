# AutoCuts

Local CLI that splits compilation videos into scenes. The default **last-occurrence** mode is holistic and under-splits: each lasting look becomes one file from the first time it appears to the last time it appears, and the middle stays in the file. Export uses ffmpeg. Everything runs on your machine.

## Last-occurrence (default)

1. Sample the video about once a second.
2. Match samples into visual looks (a hue/saturation prototype; brightness is ignored).
3. For each look that lasts, export **one** file from its first sample to its last sample anywhere later in the video.
4. Leave the middle in that span. Digressions are not cut out here.
5. Scenes may be long. That is the point.

Artistic junk — a fountain, a mountain, a brief camera move — stays inside the parent span when it never lasts long enough to be its own look. A nested scene that does last is different: it gets its own file from its own first sample to its own last sample, even when that range sits inside the parent. Those files overlap in time on purpose. A later pass can cut the junk.

```bash
autocuts /path/to/video.mp4
# writes to /path/to/AutoCuts/

autocuts video.mp4 -o ./out --min-look 15 --similarity 0.5
```

`--min-look` (default 30s) is how long a look must run contiguously before it gets a file. A shorter nest stays inside the parent unless you lower the flag. `--similarity` (default 0.55) is the cosine match against the look's prototype. Lower it to under-split more.

## DINOv2 dry-run

`--mode dino` reports where lasting looks would be cut. It samples about one frame a second, embeds each frame with DINOv2, and groups those embeddings with the same first-to-last rules as last-occurrence. It writes a JSON report of frame ranges and does not encode scene files.

A look still needs `--min-look` contiguous seconds (default 30). Each lasting look spans its first sample to its last. A nested lasting look is its own overlapping span. Short transitions stay inside the parent and are listed as skipped looks.

GPU is optional. An NVIDIA GPU such as an RTX 4070 is used when PyTorch can see it; CPU works and is slower. The first run downloads the weights. The default backbone is the small `dinov2_vits14` model. `dinov2_vitb14` is the base model. `--similarity` defaults to 0.70 in this mode (0.55 for the histogram modes) and is a starting point to tune against a ground-truth timeline.

```bash
pip install -e ".[dino]"
# CUDA example for an RTX 4070 (install the matching wheel, then the extra):
# pip install torch --index-url https://download.pytorch.org/whl/cu124
autocuts video.mp4 --mode dino -o ./out
autocuts video.mp4 --mode dino --dino-model dinov2_vitb14 --similarity 0.65
```

## Sticky mode

`--mode sticky` is the local alternative. It keeps one scene open until the picture stays unlike the current prototype for a hold-out window (default 20s, allowed range 15–30s). A blip that reverts before then stays inside the scene. It does not reach back to pair a look with a later return.

```bash
autocuts video.mp4 --mode sticky --hold-out 20
```

## Legacy PySceneDetect path

`--mode adaptive` or `--mode content` drops bursts shorter than `--min-keep` (default 4s) and stitches the long stretches on either side. `--detector` is an alias of `--mode`.

```bash
autocuts video.mp4 --mode adaptive --min-keep 4
autocuts video.mp4 --mode content --no-stitch
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# needs ffmpeg on PATH
```

## License

MIT
