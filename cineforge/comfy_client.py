"""Drive a local ComfyUI server to animate images with Wan 2.2 TI2V-5B (GGUF)."""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import time
import uuid

import numpy as np
import requests
import websocket
from PIL import Image

from .ffmpeg_utils import run as ffmpeg
from .kenburns import cover_crop

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMFY_DIR = os.path.join(ROOT, "ComfyUI")
COMFY_PY = os.path.join(COMFY_DIR, "venv", "Scripts", "python.exe")
HOST = "127.0.0.1:8188"

UNET = "Wan2.2-TI2V-5B-Q6_K.gguf"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan2.2_vae.safetensors"

DEFAULT_NEGATIVE = ("static, frozen, still image, slideshow, blurry, low quality, worst quality, jpeg artifacts, "
                    "distorted, deformed, morphing, melting, flicker, overexposed, watermark, text, subtitles, "
                    "logo, ugly, bad anatomy, extra limbs, extra fingers, fused fingers, messy background")


AUDIO_CKPT = "stable-audio-open-1.0.safetensors"
AUDIO_TEXT_ENCODER = "t5_base.safetensors"
SFX_NEGATIVE = "music, melody, speech, talking, voice, singing, low quality, distorted, muffled, noise"


LTX_UNET = "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
LTX_GEMMA = "gemma_3_12B_it_fp8_scaled.safetensors"
LTX_PROJECTION = "ltx-2.3_text_projection_bf16.safetensors"
LTX_VIDEO_VAE = "LTX23_video_vae_bf16.safetensors"
LTX_AUDIO_VAE = "LTX23_audio_vae_bf16.safetensors"
LTX_UPSCALER = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LTX_NEGATIVE = ("blurry, low quality, worst quality, distorted face, deformed, disfigured, extra limbs, extra fingers, "
                "morphing, melting, flicker, jitter, still frame, static, text, watermark, subtitles, logo")
# Distilled-model sigma schedules from the official LTX-2.3 ComfyUI template.
LTX_SIGMAS_STAGE1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX_SIGMAS_STAGE2 = "0.85, 0.7250, 0.4219, 0.0"


def missing_ltx_models() -> list[str]:
    return _missing([("unet", LTX_UNET), ("text_encoders", LTX_GEMMA), ("text_encoders", LTX_PROJECTION),
                     ("vae", LTX_VIDEO_VAE), ("vae", LTX_AUDIO_VAE), ("latent_upscale_models", LTX_UPSCALER)])


def _missing(paths) -> list[str]:
    return [name for folder, name in paths
            if not os.path.isfile(os.path.join(COMFY_DIR, "models", folder, name))]


def missing_models() -> list[str]:
    return _missing([("unet", UNET), ("text_encoders", TEXT_ENCODER), ("vae", VAE)])


def missing_sfx_models() -> list[str]:
    return _missing([("checkpoints", AUDIO_CKPT), ("text_encoders", AUDIO_TEXT_ENCODER)])


class ComfyClient:
    def __init__(self, log=print):
        self.log = log
        self.client_id = uuid.uuid4().hex
        self.proc: subprocess.Popen | None = None

    # ---------- server ----------
    def is_running(self) -> bool:
        try:
            return requests.get(f"http://{HOST}/system_stats", timeout=2).ok
        except requests.RequestException:
            return False

    def ensure_running(self, timeout: int = 300) -> None:
        if self.is_running():
            return
        if not os.path.isfile(COMFY_PY):
            raise RuntimeError(f"ComfyUI is not installed at {COMFY_DIR}")
        self.log("Starting ComfyUI server (first start takes ~30-60 s)...")
        logf = open(os.path.join(COMFY_DIR, "cineforge_comfy.log"), "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [COMFY_PY, "main.py", "--listen", "127.0.0.1", "--port", "8188",
             "--preview-method", "none", "--disable-auto-launch"],
            cwd=COMFY_DIR, stdout=logf, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        start = time.time()
        while time.time() - start < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("ComfyUI exited during startup. See ComfyUI/cineforge_comfy.log")
            if self.is_running():
                self.log("ComfyUI is ready.")
                return
            time.sleep(2)
        raise RuntimeError("ComfyUI did not start in time. See ComfyUI/cineforge_comfy.log")

    # ---------- workflow ----------
    @staticmethod
    def build_workflow(image_name: str, prompt: str, negative: str, width: int, height: int,
                       length: int, steps: int, cfg: float, seed: int, prefix: str) -> dict:
        return {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
            "6": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "7": {"class_type": "Wan22ImageToVideoLatent",
                  "inputs": {"vae": ["3", 0], "width": width, "height": height, "length": length,
                             "batch_size": 1, "start_image": ["6", 0]}},
            "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 8.0}},
            "9": {"class_type": "KSampler",
                  "inputs": {"model": ["8", 0], "seed": seed, "steps": steps, "cfg": cfg,
                             "sampler_name": "uni_pc", "scheduler": "simple",
                             "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["7", 0],
                             "denoise": 1.0}},
            "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
            "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": prefix}},
        }

    def upload_image(self, path: str, width: int, height: int) -> str:
        img = cover_crop(Image.open(path).convert("RGB"), width, height)
        name = f"cineforge_{uuid.uuid4().hex[:10]}.png"
        tmp = os.path.join(os.path.dirname(path), name)
        img.save(tmp)
        with open(tmp, "rb") as fh:
            r = requests.post(f"http://{HOST}/upload/image", files={"image": (name, fh, "image/png")},
                              data={"overwrite": "true"}, timeout=60)
        os.remove(tmp)
        r.raise_for_status()
        return r.json()["name"]

    def _run(self, workflow: dict, progress=None) -> dict:
        ws = websocket.create_connection(f"ws://{HOST}/ws?clientId={self.client_id}", timeout=30)
        try:
            r = requests.post(f"http://{HOST}/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=30)
            if not r.ok:
                raise RuntimeError(f"ComfyUI rejected the workflow: {r.text[:2000]}")
            pid = r.json()["prompt_id"]
            ws.settimeout(3600)
            while True:
                msg = ws.recv()
                if not isinstance(msg, str):
                    continue
                data = json.loads(msg)
                kind, body = data.get("type"), data.get("data", {})
                if body.get("prompt_id") not in (None, pid):
                    continue
                if kind == "progress" and progress:
                    progress(body["value"], body["max"])
                elif kind == "execution_error":
                    raise RuntimeError(f"ComfyUI error in {body.get('node_type')}: {body.get('exception_message')}")
                elif kind == "execution_interrupted" and body.get("prompt_id") == pid:
                    raise RuntimeError("ComfyUI job was interrupted")
                elif kind == "executing" and body.get("node") is None and body.get("prompt_id") == pid:
                    break
                elif kind == "execution_success" and body.get("prompt_id") == pid:
                    break
        finally:
            ws.close()
        # the "done" message can arrive just before ComfyUI writes the job to its history
        for _ in range(30):
            hist = requests.get(f"http://{HOST}/history/{pid}", timeout=30).json()
            if pid in hist:
                return hist[pid]
            time.sleep(1)
        raise RuntimeError(f"ComfyUI finished job {pid} but it never appeared in /history")

    def animate(self, image_path: str, out_path: str, prompt: str, negative: str = "",
                width: int = 832, height: int = 480, frames: int = 49, steps: int = 20,
                cfg: float = 5.0, seed: int | None = None, fps: int = 24, progress=None) -> None:
        """Generate a clip from an image and write it to out_path as an mp4 at `fps`."""
        self.ensure_running()
        width, height = width // 32 * 32, height // 32 * 32
        frames = (max(5, frames) - 1) // 4 * 4 + 1
        seed = random.randint(0, 2**31) if seed is None else seed
        name = self.upload_image(image_path, width, height)
        prefix = f"cineforge/{uuid.uuid4().hex[:8]}"
        wf = self.build_workflow(name, prompt, negative or DEFAULT_NEGATIVE, width, height,
                                 frames, steps, cfg, seed, prefix)
        hist = self._run(wf, progress)
        images = hist["outputs"]["11"]["images"]

        frame_dir = out_path + "_frames"
        os.makedirs(frame_dir, exist_ok=True)
        for i, im in enumerate(images):
            r = requests.get(f"http://{HOST}/view", params=im, timeout=60)
            r.raise_for_status()
            with open(os.path.join(frame_dir, f"{i:05d}.png"), "wb") as fh:
                fh.write(r.content)
        ffmpeg(["-framerate", str(fps), "-i", os.path.join(frame_dir, "%05d.png"),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", out_path])

    def text_to_image(self, prompt: str, out_path: str, width: int = 704, height: int = 1280,
                      steps: int = 30, cfg: float = 5.0, seed: int | None = None, negative: str = "",
                      progress=None) -> None:
        """Single still frame from Wan 2.2 TI2V-5B (no start image, length 1)."""
        self.ensure_running()
        seed = random.randint(0, 2**31) if seed is None else seed
        wf = self.build_workflow("", prompt, negative or DEFAULT_NEGATIVE, width // 32 * 32, height // 32 * 32,
                                 1, steps, cfg, seed, f"cineforge_t2i/{uuid.uuid4().hex[:8]}")
        del wf["6"]
        del wf["7"]["inputs"]["start_image"]
        hist = self._run(wf, progress)
        im = hist["outputs"]["11"]["images"][0]
        r = requests.get(f"http://{HOST}/view", params=im, timeout=60)
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            fh.write(r.content)

    def _upload(self, path: str, name: str) -> str:
        with open(path, "rb") as fh:
            r = requests.post(f"http://{HOST}/upload/image", files={"image": (name, fh)},
                              data={"overwrite": "true"}, timeout=120)
        r.raise_for_status()
        return r.json()["name"]

    def _download_frames(self, images: list, frame_dir: str) -> None:
        os.makedirs(frame_dir, exist_ok=True)
        for i, im in enumerate(images):
            r = requests.get(f"http://{HOST}/view", params=im, timeout=60)
            r.raise_for_status()
            with open(os.path.join(frame_dir, f"{i:05d}.png"), "wb") as fh:
                fh.write(r.content)

    def ltx_talking(self, image_path: str, audio_path: str, out_path: str, prompt: str,
                    seconds: float, width: int = 1280, height: int = 704, fps: int = 24,
                    seed: int | None = None, first_frame_strength: float = 0.9, progress=None) -> None:
        """LTX-2.3 image+audio -> video. The given audio is kept fixed (noise mask 0) so the
        model animates lips/motion to it. Two passes: half resolution, then 2x latent upscale + refine."""
        self.ensure_running()
        width, height = width // 64 * 64, height // 64 * 64
        frames = int(np.ceil(seconds * fps / 8)) * 8 + 1
        seed = random.randint(0, 2**31) if seed is None else seed
        tag = uuid.uuid4().hex[:10]
        img = cover_crop(Image.open(image_path).convert("RGB"), width, height)
        tmp_img = out_path + f"_{tag}.png"
        img.save(tmp_img)
        try:
            image_name = self._upload(tmp_img, f"cineforge_{tag}.png")
        finally:
            os.remove(tmp_img)
        audio_name = self._upload(audio_path, f"cineforge_{tag}.wav")

        wf = {
            # loaders
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": LTX_UNET}},
            "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": LTX_GEMMA, "clip_name2": LTX_PROJECTION,
                                                             "type": "ltxv", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": LTX_VIDEO_VAE}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": LTX_AUDIO_VAE}},
            "5": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LTX_UPSCALER}},
            # prompt
            "10": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "11": {"class_type": "CLIPTextEncode", "inputs": {"text": LTX_NEGATIVE, "clip": ["2", 0]}},
            "12": {"class_type": "LTXVConditioning", "inputs": {"positive": ["10", 0], "negative": ["11", 0],
                                                               "frame_rate": float(fps)}},
            # image
            "20": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "21": {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": 18}},
            # our audio, frozen with a zero noise mask
            "30": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
            "31": {"class_type": "LTXVAudioVAEEncode", "inputs": {"audio": ["30", 0], "audio_vae": ["4", 0]}},
            "32": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 64, "height": 64}},
            "33": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["31", 0], "mask": ["32", 0]}},
            # stage 1: half resolution
            "40": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": width // 2, "height": height // 2,
                                                                   "length": frames, "batch_size": 1}},
            "41": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["3", 0], "image": ["21", 0],
                                                                    "latent": ["40", 0],
                                                                    "strength": first_frame_strength,
                                                                    "bypass": False}},
            "42": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["41", 0], "audio_latent": ["33", 0]}},
            "43": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["12", 0],
                                                         "negative": ["12", 1], "cfg": 1.0}},
            "44": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "45": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
            "46": {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX_SIGMAS_STAGE1}},
            "47": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["44", 0], "guider": ["43", 0],
                                                                    "sampler": ["45", 0], "sigmas": ["46", 0],
                                                                    "latent_image": ["42", 0]}},
            "48": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["47", 0]}},
            # stage 2: 2x latent upscale + short refine
            "50": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["48", 0], "upscale_model": ["5", 0],
                                                                  "vae": ["3", 0]}},
            "51": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["3", 0], "image": ["21", 0],
                                                                    "latent": ["50", 0], "strength": 1.0,
                                                                    "bypass": False}},
            "52": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["51", 0], "audio_latent": ["48", 1]}},
            "53": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["12", 0], "negative": ["12", 1],
                                                             "latent": ["48", 0]}},
            "54": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["53", 0],
                                                         "negative": ["53", 1], "cfg": 1.0}},
            "55": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 1}},
            "56": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_cfg_pp"}},
            "57": {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX_SIGMAS_STAGE2}},
            "58": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["55", 0], "guider": ["54", 0],
                                                                    "sampler": ["56", 0], "sigmas": ["57", 0],
                                                                    "latent_image": ["52", 0]}},
            "59": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["58", 0]}},
            "60": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["59", 0], "vae": ["3", 0], "tile_size": 512,
                                                             "overlap": 64, "temporal_size": 64,
                                                             "temporal_overlap": 8}},
            "61": {"class_type": "SaveImage", "inputs": {"images": ["60", 0],
                                                         "filename_prefix": f"cineforge_ltx/{tag}"}},
        }
        hist = self._run(wf, progress)
        frame_dir = out_path + "_frames"
        self._download_frames(hist["outputs"]["61"]["images"], frame_dir)
        ffmpeg(["-framerate", str(fps), "-i", os.path.join(frame_dir, "%05d.png"),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", out_path])
        shutil.rmtree(frame_dir, ignore_errors=True)

    def sound_effect(self, prompt: str, seconds: float, out_path: str, seed: int | None = None,
                     steps: int = 60, progress=None, negative: str = SFX_NEGATIVE) -> None:
        """Generate a sound effect with Stable Audio Open and save it to out_path (.wav)."""
        self.ensure_running()
        seed = random.randint(0, 2**31) if seed is None else seed
        prefix = f"cineforge_sfx/{uuid.uuid4().hex[:8]}"
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": AUDIO_CKPT}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": AUDIO_TEXT_ENCODER, "type": "stable_audio",
                                                         "device": "default"}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
            "5": {"class_type": "EmptyLatentAudio", "inputs": {"seconds": float(seconds), "batch_size": 1}},
            "6": {"class_type": "KSampler",
                  "inputs": {"model": ["1", 0], "seed": seed, "steps": steps, "cfg": 6.0,
                             "sampler_name": "dpmpp_3m_sde_gpu", "scheduler": "exponential",
                             "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0],
                             "denoise": 1.0}},
            "7": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
            "8": {"class_type": "SaveAudio", "inputs": {"audio": ["7", 0], "filename_prefix": prefix}},
        }
        hist = self._run(wf, progress)
        item = hist["outputs"]["8"]["audio"][0]
        r = requests.get(f"http://{HOST}/view", params=item, timeout=60)
        r.raise_for_status()
        raw = out_path + ".flac"
        with open(raw, "wb") as fh:
            fh.write(r.content)
        # trim leading silence, gentle fade-out, normalise loudness
        ffmpeg(["-i", raw, "-af",
                f"silenceremove=start_periods=1:start_threshold=-45dB,atrim=duration={seconds:.2f},"
                f"afade=t=out:st={max(0.1, seconds - 0.6):.2f}:d=0.6,loudnorm=I=-16:TP=-1.5",
                "-ar", "48000", out_path])
        os.remove(raw)

    def shutdown(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
