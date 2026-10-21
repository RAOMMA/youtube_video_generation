"""Find glitchy narration lines in an episode: transcribe every cached voice line with Whisper and flag lines
whose words don't match the script, or that are much longer than they should be.

  venv\\Scripts\\python -m cineforge.voice_qa projects\\mlwalisarkar\\ep03_rnn.txt [--fix]

--fix deletes the flagged lines from the voice cache, so the next render speaks them again (new seed)."""
import argparse
import json
import os
import re
import subprocess
import time

from .explainer import parse_explainer, parse_theme, set_theme
from .explainer_render import VOICE_CACHE_DIR
from .script_parser import parse_cast
from .tts import QWEN_PY, Narrator, split_sentences


def lines_of(script_path):
    script = open(script_path, encoding="utf-8").read()
    set_theme(parse_theme(script))
    scenes = [x.scene for x in parse_explainer(script)]
    cast = parse_cast(script)
    voice = {"voice": "am_michael", "speed": 1.0, "pitch": 1.0, **cast.get("narrator", {})}
    m = re.search(r"^\s*\[engine\s*:\s*(\w+)\s*\]", script, flags=re.M | re.I)
    n = Narrator()
    # only build the line -> cache file map; never start a voice job here
    n._run_voice_tool = lambda *a, **k: None
    n.prerender_designed(scenes, cast, voice, VOICE_CACHE_DIR, lambda *a: None,
                         engine=m.group(1).lower() if m else "chatterbox")
    scene_of = {}
    for sc in scenes:
        for ev in sc.events:
            if ev.kind == "speech":
                for s in split_sentences(ev.text):
                    scene_of.setdefault(s, sc.index)
    return [{"scene": scene_of.get(s, -1), "text": s, "path": p} for (_, s, _), p in n._designed.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    items = [it for it in lines_of(a.script) if os.path.isfile(it["path"])]
    job = os.path.join(VOICE_CACHE_DIR, "qa_job.json")
    with open(job, "w", encoding="utf-8") as fh:
        json.dump(items, fh)
    proc = subprocess.run([QWEN_PY, os.path.join(os.path.dirname(__file__), "voice_qa_asr.py"), job],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(proc.stderr[-2000:])
    res = json.load(open(job + ".out.json", encoding="utf-8"))
    bad = [r for r in res if r["flag"]]
    for r in res:
        mark = "BAD " if r["flag"] else "ok  "
        print(f"{mark} scene {r['scene']:2d}  match {r['match']:.2f}  {r['secs']:4.1f}s/{r['expect']:4.1f}s  "
              f"{r['text'][:60]}" + (f"\n        heard: {r['heard'][:90]}" if r["flag"] else ""))
    print(f"\n{len(bad)} of {len(res)} lines flagged")
    if a.fix:
        for r in bad:
            os.replace(r["path"], r["path"] + f".{int(time.time())}.bad")
        print(f"moved {len(bad)} flagged lines out of the cache; re-render the episode to respeak them")


if __name__ == "__main__":
    main()
