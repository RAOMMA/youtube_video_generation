# CineForge

A local pipeline that turns a text script into a narrated YouTube video. It powers the
[ML wali Sarkar](https://www.youtube.com/@mlwalisarkar) channel.

It has two modes:

- **Explainer videos.** You write a script with tags such as `[code:]`, `[nn:]`, `[quiz:]` and `[flow:]`.
  CineForge voices the script in a cloned voice, checks every line with Whisper, draws animated
  visuals in sync with the narration, and writes `final.mp4` with chapters and a thumbnail.
- **Cinematic film (web app).** You upload images and a script. Each image becomes a scene with a
  Ken Burns move or AI motion (Wan 2.2), plus narration and subtitles.

Everything runs locally and nothing is uploaded. It was tested on Windows 11 with an RTX 4060 Laptop
GPU (8 GB) and 64 GB of RAM.

> This repository is **code only**. Model weights, voice recordings, projects and renders are not
> included. Download the models (step 3) and record your own voice (step 5).

---

## Requirements

- Windows 10 or 11, with an NVIDIA GPU (8 GB VRAM or more) and a recent driver
- Python 3.12 ([python.org](https://www.python.org/downloads/))
- Git
- FFmpeg on your PATH (`winget install Gyan.FFmpeg`, then open a new terminal and run `ffmpeg -version` to check)
- Around 60 GB of free disk space for models

## Installation

### 1. Clone

Clone the repo into its own folder. ComfyUI will go **next to** that folder, not inside it.

```bat
mkdir D:\AI
cd /d D:\AI
git clone https://github.com/RAOMMA/youtube_video_generation.git CineForge
cd CineForge
```

Target layout:

```
D:\AI\
  CineForge\     this repo
  ComfyUI\       AI image/video engine (step 3)
```

### 2. Python environments

The parts need conflicting library versions, so each one gets its own virtual environment.

| Environment | Used for | Required? |
|---|---|---|
| `venv` | Web app, explainer engine, rendering, stock voice (Kokoro) | Yes |
| `tts_venv` | Qwen3-TTS voice cloning (`[engine: qwen]`, used by the channel) and the Whisper voice check | Yes, for explainer videos |
| `cb_venv` | Chatterbox voices, the default when a script has no `[engine:]` tag | Only if you use Chatterbox |

```bat
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\pip install -r requirements\main.txt

python -m venv tts_venv
tts_venv\Scripts\python -m pip install --upgrade pip
tts_venv\Scripts\pip install -r requirements\tts.txt

rem optional
python -m venv cb_venv
cb_venv\Scripts\pip install -r requirements\chatterbox.txt
```

The requirement files pin CUDA builds of PyTorch (`+cu126`). If pip says it can't find `torch`, first
install torch from the PyTorch index into that environment, then run the `-r` command again:

```bat
venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Check that the GPU is visible:

```bat
venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3. ComfyUI and models (illustrations, motion, sound effects, lip-sync)

You can skip this step if you only use Ken Burns motion and explainer scenes without
`[art:]`, `[image:]`, `[show:]`, `[motion:]` or `[host:]`.

Install ComfyUI in the sibling folder, with its own `venv`. CineForge starts it automatically on
`127.0.0.1:8188`.

```bat
cd /d D:\AI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python -m venv venv
venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
venv\Scripts\pip install -r requirements.txt
git clone https://github.com/city96/ComfyUI-GGUF.git custom_nodes\ComfyUI-GGUF
venv\Scripts\pip install -r custom_nodes\ComfyUI-GGUF\requirements.txt
```

Download these files into `ComfyUI\models\...` (search the file name on Hugging Face):

| File | Folder | Used for |
|---|---|---|
| `Wan2.2-TI2V-5B-Q6_K.gguf` | `unet` | Illustrations and motion clips |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders` | Wan text encoder |
| `wan2.2_vae.safetensors` | `vae` | Wan VAE |
| `stable-audio-open-1.0.safetensors` | `checkpoints` | Sound effects |
| `t5_base.safetensors` | `text_encoders` | Stable Audio text encoder |
| LTX-2.3 files | run `download_ltx.bat` | Talking characters, `[host:]` (optional) |

`download_ltx.bat` fetches the LTX files and resumes interrupted downloads. Check each model's
licence before commercial use.

### 4. Voice models (automatic)

On first use these models download from Hugging Face by themselves:

- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `openai/whisper-small.en` and `openai/whisper-medium.en`
- `hexgrad/Kokoro-82M`

### 5. Your voice

Record about 30 seconds of yourself speaking clearly, or cut a clean sample from a video of you:

```bat
tts_venv\Scripts\python voices\make_my_voice.py path\to\your_video.mp4 --seconds 30
tts_venv\Scripts\python voices\transcribe.py voices\recordings\me.wav
```

Then point a script at it with `[ref Narrator: voices\recordings\me.wav]`.
**Only clone voices you have permission to use.**

---

## How to run

### A. Explainer video (command line)

1. Save a script as a `.txt` file, for example `my_video.txt`:

   ````
   [engine: qwen]
   [ref Narrator: voices\recordings\me.wav]
   [thumb: GRADIENT DESCENT | loss.backward() | blue | BEGINNER]
   [chapter: Intro]
   [title: How Neural Networks Learn | in 8 minutes]
   Every neural network starts out completely wrong. Let's see how it fixes itself.
   ---
   [bullets: Guess | Measure the error | Nudge the weights]
   First it makes a guess.
   Then it measures how wrong that guess was.
   Then it nudges every weight a tiny bit in the right direction.
   ---
   [code: python | train.py]
   ```
   w = w - lr * grad
   print(round(w, 2))
   > 0.42
   ```
   That one line is the whole idea of gradient descent.
   ---
   [quiz: If the error goes up, the step was... | Too big | Too small | Too big]
   Quick quiz. If the error goes up, was the step too big or too small?
   It was too big.
   ````

2. Render it:

   ```bat
   venv\Scripts\python -m cineforge.explainer_render my_video.txt
   ```

   Add `--kokoro` for a quick draft with a stock voice (no cloning, much faster).

3. The output goes to `outputs\<timestamp>_<name>\`. It contains `final.mp4`, `chapters.txt`
   (timestamps for the description), `thumbnail.png`, `subtitles.srt` and `narration.wav`.

4. Optionally, re-check the voice and respeak bad lines. `--fix` moves the flagged lines out of the
   cache, so the next render voices them again:

   ```bat
   venv\Scripts\python -m cineforge.voice_qa my_video.txt --fix
   venv\Scripts\python -m cineforge.explainer_render my_video.txt
   ```

Renders are cached by line and by scene, so a re-render only redoes what changed.

#### Explainer tags

Scenes are separated by `---`. Plain lines are narration, and each tag sets that scene's visual.

| Tag | What it shows |
|---|---|
| `[title: Big title \| subtitle]` | Title card |
| `[chapter: Name]` | Starts a YouTube chapter (badge plus a timestamp in the description) |
| `[bullets: A \| B \| C]` | One card per narration sentence |
| `[steps: row \| => result]` | Worked example, one row per sentence |
| `[flow: A -> B -> C \| loop]` | Boxes that appear one per sentence |
| `[equation: y = w·x + b \| caption]` | Equation card |
| `[code: python \| file.py]` + a fenced block | Code typed line by line; lines starting with `> ` are shown as output |
| `[nn: 3,4,4,2]` | Neural-network diagram with signal flowing |
| `[plot: sigmoid, relu]` | Function plots |
| `[fit: both]` | Good fit vs overfitting |
| `[rnn: ...]`, `[gates: lstm]`, `[conv: edge]` | RNN, LSTM/GRU and CNN diagrams |
| `[tokens: ...]`, `[next: ...]`, `[attention: ...]` | LLM token, next-token and attention visuals |
| `[quiz: Question \| A \| B \| answer]` | Quiz; the answer is revealed on the scene's last line |
| `[art: prompt]`, `[image: prompt]` | AI illustration (needs ComfyUI) |
| `[show: prompt \| WORD]`, `[motion: prompt \| WORD]` | Picture or animated clip of what is being said (needs ComfyUI) |
| `[host: characters.png \| caption]` | Talking characters with lip-sync (needs LTX) |

These global tags apply to the whole video:

| Tag | Effect |
|---|---|
| `[ref Name: voice.wav]` | Clone this voice for the character `Name` (`Narrator` is the default) |
| `[design Name: description]` | Design a new voice from a text description |
| `[engine: qwen]` | Voice engine: `qwen` (needs `tts_venv`) or `chatterbox` (the default, needs `cb_venv`) |
| `[theme: kids]` | Bright kids look, with bigger code and pop-in animations |
| `[thumb: WORDS \| snippet \| color \| badge]` | Generates the thumbnail |

Dialogue is written as `Name (emotion): "line"`. The full tag reference is in the docstring at the
top of [`cineforge/explainer.py`](cineforge/explainer.py).

### B. Cinematic film (web app)

1. Double-click `run.bat`. It opens http://127.0.0.1:7860 in your browser.
2. Upload your images in scene order.
3. Write the script (format below) and click **Preview scenes** to check how it splits.
4. Choose a voice, a motion mode and a look, then click **Generate film**.
5. The film is saved to `outputs\<timestamp>\final.mp4`, along with `subtitles.srt`, `narration.wav`
   and every clip.

Script format:

```
[camera: zoom_in]
[motion: mist rolls across the valley, birds circling]
In the age before memory, the valley slept beneath a sea of mist.
---
[mode: ai]
[image: 3]
Until tonight.
```

| Tag | Values | Effect |
|---|---|---|
| `[camera: ...]` | zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down, auto | Ken Burns camera move |
| `[mode: ...]` | ai, kenburns | Overrides the motion mode for this scene |
| `[motion: ...]` | free text | What should move in the AI shot (describe motion, not story) |
| `[image: N]` | 1-based number | Use image N instead of the next one |
| `[duration: N]` | seconds | Minimum scene length |

Motion modes:

- **Ken Burns** is a smooth zoom and pan that takes seconds per scene.
- **AI motion (Wan 2.2)** takes about 2 to 6 minutes per 2-second segment on an RTX 4060.
  - Segments are chained to fill long narration.
  - Clips are cached in `outputs\_ai_cache`.
  - A failed shot falls back to Ken Burns.

A good workflow is to render with Ken Burns first to check pacing, then switch only the key shots to `[mode: ai]`.

---

## Project layout

```
app.py                  Gradio web app (run.bat starts it)
cineforge\
  explainer.py          explainer script parser + visual scenes
  explainer_render.py   explainer renderer (CLI)
  qwen_voices.py        Qwen3-TTS voices with automatic retakes (runs in tts_venv)
  cb_voices.py          Chatterbox voices (runs in cb_venv)
  asr_check.py          Whisper check that each line says what the script says
  voice_qa.py           re-check a finished video's voice lines (CLI)
  comfy_client.py       drives ComfyUI: Wan 2.2, LTX lip-sync, Stable Audio
  pipeline.py, script_parser.py, kenburns.py, tts.py, ffmpeg_utils.py   web-app film pipeline
voices\                 helper scripts to prepare your voice sample
requirements\           pinned packages for venv / tts_venv / cb_venv
download_ltx.bat        downloads the LTX-2.3 lip-sync models
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ComfyUI is not installed at ...` | ComfyUI must be in the folder next to this repo, with `ComfyUI\venv` (step 3). |
| ComfyUI fails to start | Read `ComfyUI\cineforge_comfy.log`. |
| `torch.cuda.is_available()` is False | Reinstall torch from the cu126 index (step 2) and update the NVIDIA driver. |
| `ffmpeg` not found | Install FFmpeg and open a new terminal. |
| A word in the narration sounds wrong | Run `cineforge.voice_qa ... --fix` and render again. |
| Out of VRAM | Close other GPU apps. Renders run one model at a time to fit in 8 GB. |
