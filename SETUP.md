# Setting up CineForge on a new machine

This repository contains the **pipeline code only**. Model weights, voice recordings, renders and projects aren't included. Download the models yourself (below) and record your own voice sample.

Tested on Windows 11 with an RTX 4060 Laptop GPU (8 GB) and 64 GB RAM.

## 1. Three Python environments
The parts depend on conflicting libraries, so each runs in its own environment. Package lists are in `requirements/`.

| Environment | Used for | Install |
|---|---|---|
| `venv` | App, explainer engine, rendering | `python -m venv venv` then `venv\Scripts\pip install -r requirements\main.txt` |
| `tts_venv` | Qwen3-TTS voices + Whisper voice check | `python -m venv tts_venv` then `tts_venv\Scripts\pip install -r requirements\tts.txt` |
| `cb_venv` | Chatterbox voices (optional engine) | `python -m venv cb_venv` then `cb_venv\Scripts\pip install -r requirements\chatterbox.txt` |

PyTorch CUDA builds (the `+cu126` versions) come from PyTorch's own package index. If pip can't find them, install torch first with the command from pytorch.org for your CUDA version, then install the rest. You also need **FFmpeg** on your PATH.

## 2. ComfyUI (images, motion, lip-sync, sound effects)
Install ComfyUI next to this folder (`..\ComfyUI`) with the **ComfyUI-GGUF** custom node. CineForge starts it automatically on `127.0.0.1:8188`. The model files go in `ComfyUI\models\…`:

| File | Folder | Used for |
|---|---|---|
| `Wan2.2-TI2V-5B-Q6_K.gguf` | `unet` | Illustrations and motion clips (Wan 2.2) |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders` | Wan text encoder |
| `wan2.2_vae.safetensors` | `vae` | Wan VAE |
| `stable-audio-open-1.0.safetensors` | `checkpoints` | Sound effects |
| `t5_base.safetensors` | `text_encoders` | Stable Audio text encoder |
| LTX-2.3 files (unet, Gemma text encoder, projection, video/audio VAEs, upscaler) | see `download_ltx.bat` | Talking-character lip-sync (optional) |

`download_ltx.bat` downloads the LTX files and can resume interrupted downloads. Check each model's licence before commercial use.

## 3. Voice models (downloaded automatically on first use, from Hugging Face)
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and `…-VoiceDesign` (narration and designed voices)
- `openai/whisper-small.en` and `openai/whisper-medium.en` (voice quality check)
- Chatterbox (optional engine)

## 4. Your voice
Record yourself, or extract a clean sample from a video of you speaking:
```bat
tts_venv\Scripts\python voices\make_my_voice.py path\to\your_video.mp4 --seconds 30
tts_venv\Scripts\python voices\transcribe.py voices\recordings\me.wav
```
Then point a script at it with `[ref Narrator: voices\recordings\me.wav]`. **Only clone voices you have permission to use.**

## 5. Run
- Web app: `run.bat`, which opens http://127.0.0.1:7860
- Explainer videos: `venv\Scripts\python -m cineforge.explainer_render path\to\script.txt`
- Voice check: `venv\Scripts\python -m cineforge.voice_qa path\to\script.txt [--fix]`

The script format is described in `README.md` and in the docstring at the top of `cineforge/explainer.py`.

## Optional assets (not included)
- Kids thumbnails use a character cutout at `projects/nisa_baba/images/02_open_mouth_cutout.png`. Without it, they render without characters.
- Renders go to `outputs/`, which git ignores.
