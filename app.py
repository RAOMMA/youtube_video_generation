"""CineForge - turn images + a script into a narrated cinematic video, fully local."""
import os
import traceback

import gradio as gr

from cineforge.comfy_client import missing_models, missing_sfx_models
from cineforge.pipeline import GRADES, OUTPUT_DIR, Pipeline, Settings
from cineforge.script_parser import parse_script
from cineforge.tts import VOICES

EXAMPLE_SCRIPT = """[camera: zoom_in]
[motion: mist rolls slowly across the valley, birds circling in the distance]
In the age before memory, the valley slept beneath a sea of mist.
---
[camera: pan_right]
No one remembered who built the old tower. Only that its light had never gone out.
---
[camera: zoom_out]
[mode: ai]
[motion: the hero's cloak ripples in the wind, embers drift past]
Until tonight.
"""

AI_PRESETS = {
    "Fast preview (672x384, 2s)": (672, 384, 49, 15),
    "Balanced (832x480, 2s)": (832, 480, 49, 20),
    "Long shot (832x480, 3s)": (832, 480, 73, 20),
    "Long take (832x480, 5s, ~5 min)": (832, 480, 121, 20),
    "High quality (1280x704, 2s, slow)": (1280, 704, 49, 25),
}


def _paths(files):
    return [f if isinstance(f, str) else f.name for f in (files or [])]


def preview(script, files):
    imgs = _paths(files)
    try:
        scenes = parse_script(script, max(1, len(imgs)))
    except Exception as e:
        return [[f"Error: {e}", "", "", "", ""]]
    rows = []
    for s in scenes:
        name = os.path.basename(imgs[s.image_index]) if imgs else f"#{s.image_index + 1}"
        rows.append([s.index + 1, name, s.camera, s.mode or "(global)", s.motion, s.text])
    return rows


def status():
    missing, sfx = missing_models(), missing_sfx_models()
    motion = ("⏳ AI motion models missing: " + ", ".join(missing)) if missing else "✅ AI motion ready (Wan 2.2)"
    sound = ("⏳ Sound-effect models missing: " + ", ".join(sfx)) if sfx else "✅ Sound effects ready (Stable Audio Open)"
    return f"{motion} · {sound}"


def generate(script, files, voice, speed, motion, style, preset, max_seg, seed,
             grade, letterbox, vignette, grain, subs, crossfade, kb_strength, music, music_vol,
             progress=gr.Progress()):
    imgs = _paths(files)
    if not imgs:
        raise gr.Error("Upload at least one image.")
    if not script.strip():
        raise gr.Error("Write a script.")
    w, h, frames, steps = AI_PRESETS[preset]
    s = Settings(voice=VOICES[voice], speed=speed,
                 motion_mode="ai" if motion.startswith("AI") else "kenburns",
                 style_prompt=style, ai_width=w, ai_height=h, ai_frames=frames, ai_steps=steps,
                 ai_max_segments=int(max_seg), seed=int(seed), kb_strength=kb_strength,
                 crossfade=crossfade, grade=grade, letterbox=letterbox, vignette=vignette,
                 grain=grain, subtitles=subs, music_path=_paths([music])[0] if music else "",
                 music_volume=music_vol)
    logs = []

    def log(msg):
        print(msg)
        logs.append(msg)

    try:
        out = Pipeline(log=log, progress=lambda f, d="": progress(f, desc=d)).run(script, imgs, s)
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"{e}") from e
    return out, "\n".join(logs)


with gr.Blocks(title="CineForge") as demo:
    gr.Markdown("# 🎬 CineForge\nImages + script → narrated cinematic film. Runs 100% on your laptop.")
    status_md = gr.Markdown(status())
    with gr.Row():
        with gr.Column(scale=3):
            files = gr.File(label="Images (in scene order)", file_count="multiple", file_types=["image"])
            gallery = gr.Gallery(label="Images", columns=6, height=160)
            script = gr.Textbox(label="Script", lines=14, value=EXAMPLE_SCRIPT,
                                info="Scenes separated by ---. Tags: [camera: zoom_in|zoom_out|pan_left|pan_right|"
                                     "pan_up|pan_down] [mode: ai|kenburns] [motion: ...] [image: N] [duration: s] "
                                     "[sfx: sound prompt | seconds]. Dialogue: Name: \"line\". "
                                     "Cast: [voice Name: af_heart, pitch=1.3, speed=0.9]")
            with gr.Row():
                preview_btn = gr.Button("Preview scenes")
                go = gr.Button("🎬 Generate film", variant="primary")
            table = gr.Dataframe(headers=["#", "Image", "Camera", "Mode", "Motion prompt", "Narration"],
                                 wrap=True, interactive=False)
        with gr.Column(scale=2):
            with gr.Tab("Voice"):
                voice = gr.Dropdown(list(VOICES), value="George (British male, deep)", label="Narrator")
                speed = gr.Slider(0.7, 1.3, 0.95, step=0.05, label="Speaking speed")
            with gr.Tab("Motion"):
                motion = gr.Radio(["Ken Burns (fast)", "AI motion - Wan 2.2 (slow, ~2-6 min/shot)"],
                                  value="Ken Burns (fast)", label="Default motion",
                                  info="Scenes tagged [mode: ai] or [mode: kenburns] override this.")
                kb_strength = gr.Slider(0.3, 2.0, 1.0, step=0.1, label="Ken Burns movement amount")
                preset = gr.Dropdown(list(AI_PRESETS), value="Balanced (832x480, 2s)", label="AI quality preset")
                max_seg = gr.Slider(1, 4, 2, step=1, label="Max AI segments per shot",
                                    info="Long narration: chain segments (each continues the last). More = slower.")
                style = gr.Textbox(label="Style prompt (added to every AI shot)",
                                   value="cinematic, film grain, dramatic lighting, shallow depth of field")
                seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
            with gr.Tab("Look & Sound"):
                grade = gr.Dropdown(list(GRADES), value="Teal & Orange", label="Color grade")
                with gr.Row():
                    letterbox = gr.Checkbox(True, label="2.39:1 letterbox")
                    vignette = gr.Checkbox(True, label="Vignette")
                    grain = gr.Checkbox(True, label="Film grain")
                    subs = gr.Checkbox(True, label="Subtitles")
                crossfade = gr.Slider(0.0, 2.0, 0.8, step=0.1, label="Crossfade (s)")
                music = gr.File(label="Background music (optional, auto-ducks under voice)", file_types=["audio"])
                music_vol = gr.Slider(0.05, 0.6, 0.25, step=0.05, label="Music volume")
            video = gr.Video(label="Your film", height=360)
            log = gr.Textbox(label="Log", lines=8)
            gr.Markdown(f"Outputs are saved in `{OUTPUT_DIR}`")

    files.change(lambda f: _paths(f), files, gallery)
    preview_btn.click(preview, [script, files], table)
    go.click(generate, [script, files, voice, speed, motion, style, preset, max_seg, seed,
                        grade, letterbox, vignette, grain, subs, crossfade, kb_strength, music, music_vol],
             [video, log])
    demo.load(status, None, status_md)

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True, allowed_paths=[OUTPUT_DIR],
                        theme=gr.themes.Soft(primary_hue="amber"))
