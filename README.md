# AutoCuts

Local CLI that splits compilation videos into individual scenes, **drops short pan-outs** (a few seconds away from the main action), and keeps the main stretches. Built on [PySceneDetect](https://www.scenedetect.com/) + ffmpeg. Everything runs on your machine.

## Why not only PySceneDetect?

PySceneDetect already detects cuts and can enforce a minimum scene length. AutoCuts adds the workflow we wanted for messy compilations (including dark / night-vision footage):

- treat sub-~4s digressions as **bursts to remove**, not as their own clips
- when a short burst sits between two longer stretches, **excise it and stitch** those stretches into one output file
- sensible defaults for dark footage (`adaptive` detector)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# needs ffmpeg on PATH
```

## Usage

```bash
autocuts /path/to/video.mp4
# writes to /path/to/AutoCuts/ by default

autocuts video.mp4 -o ./out --min-keep 4
autocuts video.mp4 --detector content --no-stitch
```

## License

MIT
