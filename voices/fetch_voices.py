"""Build reference voice samples from an openly licensed speech corpus (consenting adult speakers).

Source: CSTR VCTK Corpus (University of Edinburgh), CC BY 4.0, via the parquet mirror
sanchit-gandhi/vctk on Hugging Face. Only the needed rows are read (HTTP range reads), not the 12 GB.
Each chosen speaker's clips are joined into a ~12 s sample; attribution goes to CREDITS.txt.

venv\\Scripts\\python voices\\fetch_voices.py
"""
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "datasets/sanchit-gandhi/vctk"
TARGET_SECONDS = 12.0

PICKS = {  # VCTK speaker -> why it's a candidate
    "p251": "South Asian (Indian) man", "p376": "South Asian (Indian) man", "p248": "South Asian (Indian) woman",
    "p227": "English man, 38", "p226": "English man", "p311": "American man", "p345": "American man",
    "p225": "English woman", "p294": "American woman", "p318": "American woman", "p330": "American woman",
}


def save_sample(spk, rows, out_dir):
    meta = rows[0]
    pieces, total, sr = [], 0.0, 48000
    for r in rows[2:]:  # skip the very short opening prompts
        audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) / sr < 1.5:
            continue
        pieces += [audio, np.zeros(int(0.3 * sr), dtype=np.float32)]
        total += len(audio) / sr
        if total >= TARGET_SECONDS:
            break
    wav = np.concatenate(pieces)
    wav = wav / max(1e-6, float(np.abs(wav).max())) * 0.9
    name = f"{meta['gender']}_{meta['accent']}_{spk}_age{meta['age']}"
    sf.write(os.path.join(out_dir, name + ".wav"), wav, sr)
    json.dump({k: meta[k] for k in ("gender", "age", "accent", "region")},
              open(os.path.join(out_dir, name + ".json"), "w"))
    print(f"saved {name}  {total:.1f}s", flush=True)


def main():
    out_dir = os.path.join(HERE, "candidates")
    os.makedirs(out_dir, exist_ok=True)
    have = {f.split("_")[2] for f in os.listdir(out_dir) if f.endswith(".wav")}
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"{REPO}/data/*.parquet"))
    for shard in shards:
        todo = [s for s in PICKS if s not in have]
        if not todo:
            break
        with fs.open(shard, "rb", block_size=2 * 1024 * 1024) as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.num_row_groups):
                ids = set(pf.read_row_group(rg, columns=["speaker_id"]).column(0).to_pylist())
                hits = [s for s in todo if s in ids and s not in have]
                if not hits:
                    if ids and max(ids) > max(todo):
                        break
                    continue
                table = pf.read_row_group(rg, columns=["speaker_id", "audio", "text", "gender", "age",
                                                       "accent", "region"]).to_pylist()
                for spk in hits:
                    save_sample(spk, [r for r in table if r["speaker_id"] == spk], out_dir)
                    have.add(spk)

    credits, index = [], []
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".wav"):
            continue
        name, spk = f[:-4], f.split("_")[2]
        meta = json.load(open(os.path.join(out_dir, name + ".json")))
        index.append({"name": name, "speaker": spk, **meta, "note": PICKS.get(spk, ""),
                      "file": os.path.join(out_dir, f)})
        credits.append(f"{name}: VCTK corpus speaker {spk}")
    json.dump(index, open(os.path.join(HERE, "candidates.json"), "w"), indent=2)
    with open(os.path.join(HERE, "CREDITS.txt"), "w", encoding="utf-8") as fh:
        fh.write("Reference voices are built from the CSTR VCTK Corpus (Yamagishi, Veaux, MacDonald),\n"
                 "Centre for Speech Technology Research, University of Edinburgh, licensed CC BY 4.0.\n"
                 "https://doi.org/10.7488/ds/2645\n\nSuggested credit for video descriptions:\n"
                 "  \"Voices based on the CSTR VCTK Corpus (University of Edinburgh), CC BY 4.0.\"\n\n")
        fh.write("\n".join(credits) + "\n")


if __name__ == "__main__":
    main()
