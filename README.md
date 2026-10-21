# CineForge

Turn your images and a script into a narrated cinematic video. Everything runs locally on your laptop
(RTX 4060 8 GB, 64 GB RAM). Nothing is uploaded anywhere.

## Start

Double-click `run.bat`. A browser tab opens at http://127.0.0.1:7860.

1. Upload your images in scene order.
2. Write the script (format below) and click **Preview scenes** to check how it splits.
3. Choose a voice, a motion mode, and a look, then click **Generate film**.
4. The finished film is saved to `outputs/<timestamp>/final.mp4`, along with `subtitles.srt`,
   `narration.wav`, and every individual clip.

## Script format

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
| `[mode: ...]` | ai, kenburns | Overrides the default motion mode for this scene |
| `[motion: ...]` | free text | What should move in the AI shot. Describe motion, not the story |
| `[image: N]` | 1-based number | Use image N instead of the next image in order |
| `[duration: N]` | seconds | Minimum length of the scene (useful for silent shots) |

Scenes are separated by `---`. If there is no `---`, blank lines separate them instead.
Any text that is not a tag becomes narration, and the narration length sets the scene length.

## Motion modes

* **Ken Burns:** smooth sub-pixel zoom and pan. Takes a few seconds per scene.
* **AI motion (Wan 2.2 TI2V-5B, Q6_K GGUF):** real animation generated from your image, through
  ComfyUI, which the app starts automatically. Takes roughly 2-6 minutes per 2-second segment
  on the 4060. When the narration is longer than one segment, the app chains up to *N* segments
  (each one continues from the previous last frame), then uses smooth motion-interpolated slow-mo
  and holds the final frame to fill the rest.
  * Generated segments are cached in `outputs/_ai_cache`. Re-running the same image, prompt and
    settings is instant. To get a new take, change the seed or delete the cache.
  * If an AI shot fails, that scene falls back to Ken Burns so the film still renders.

A good workflow: render the whole film with Ken Burns first to check pacing, then tag only the key
shots with `[mode: ai]`.

## Layout

```
D:\New folder\
  CineForge\            this app (its own venv)
    app.py              Gradio UI
    cineforge\          pipeline: script_parser, tts (Kokoro), kenburns, comfy_client (Wan 2.2), pipeline
    outputs\
  ComfyUI\              AI video engine (its own venv) + ComfyUI-GGUF node
    models\unet\Wan2.2-TI2V-5B-Q6_K.gguf
    models\text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors
    models\vae\wan2.2_vae.safetensors
```

ComfyUI's log is in `ComfyUI\cineforge_comfy.log`.
