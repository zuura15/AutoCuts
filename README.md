# AutoCuts

Local CLI that splits compilation videos into scenes. The default **sticky** mode under-splits on purpose: a brief zoom or lighting blip that comes back does not end a scene. Leaving a little junk inside a scene is fine. Export uses ffmpeg. Everything runs on your machine.

## Sticky mode (default)

Sticky mode samples the picture about once a second and keeps a cheap hue/saturation signature of the current stretch (brightness is ignored). That prototype is a slow average of samples that still match it, so a pan or a gradual zoom can drift without a cut. A scene closes only when samples stay unlike the prototype for a **hold-out** window (default 20s, allowed range 15–30s).

```bash
autocuts /path/to/video.mp4
# writes to /path/to/AutoCuts/

autocuts video.mp4 -o ./out --hold-out 25 --similarity 0.5
```

`--similarity` is the cosine threshold (default 0.55). Lower it to under-split more. Two stretches with a similar palette may stay one file; that is the tradeoff.

## Legacy PySceneDetect path

`--mode adaptive` or `--mode content` keeps the previous detector: drop bursts shorter than `--min-keep` (default 4s) and stitch the long stretches on either side. `--detector` is an alias of `--mode`.

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
