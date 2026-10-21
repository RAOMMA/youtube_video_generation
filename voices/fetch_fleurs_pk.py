"""Pakistani (Urdu, ur_pk) male reference voices from Google FLEURS (CC BY 4.0, consenting speakers).

FLEURS has no speaker ids, so each candidate is a single clip (one clip = one speaker).
Reads only the needed parquet rows from the Hugging Face parquet branch.
venv\\Scripts\\python voices\\fetch_fleurs_pk.py  ->  voices/candidates/M_Pakistani_fleursNN.wav
"""
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfFileSystem

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "datasets/google/fleurs@~parquet"
WANT = 8              # candidates
MIN_S, MAX_S = 9.0, 16.0
MALE = 0              # FLEURS gender label: 0 = male, 1 = female


def main():
    out_dir = os.path.join(HERE, "candidates")
    os.makedirs(out_dir, exist_ok=True)
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{REPO}/ur_pk/train/*.parquet"))
    picked, seen_text = [], set()
    for f in files:
        with fs.open(f, "rb", block_size=2 * 1024 * 1024) as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.num_row_groups):
                meta = pf.read_row_group(rg, columns=["gender", "num_samples", "transcription"]).to_pylist()
                idx = [i for i, m in enumerate(meta) if m["gender"] == MALE
                       and MIN_S <= m["num_samples"] / 16000 <= MAX_S and m["transcription"] not in seen_text]
                if not idx:
                    continue
                rows = pf.read_row_group(rg, columns=["audio", "transcription", "num_samples"]).to_pylist()
                for i in idx[: WANT - len(picked)]:
                    seen_text.add(rows[i]["transcription"])
                    audio, sr = sf.read(io.BytesIO(rows[i]["audio"]["bytes"]), dtype="float32")
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    audio = audio / max(1e-6, float(np.abs(audio).max())) * 0.9
                    name = f"M_Pakistani_fleurs{len(picked) + 1:02d}"
                    sf.write(os.path.join(out_dir, name + ".wav"), audio, sr)
                    json.dump({"gender": "M", "age": "", "accent": "Pakistani (Urdu)", "region": "Pakistan",
                               "source": "Google FLEURS ur_pk, CC BY 4.0"},
                              open(os.path.join(out_dir, name + ".json"), "w"))
                    picked.append(name)
                    print(f"saved {name}  {len(audio) / sr:.1f}s", flush=True)
                if len(picked) >= WANT:
                    break
        if len(picked) >= WANT:
            break
    with open(os.path.join(HERE, "CREDITS.txt"), "a", encoding="utf-8") as fh:
        fh.write("\nPakistani voices (M_Pakistani_fleurs*): Google FLEURS dataset, Urdu (Pakistan) subset, "
                 "CC BY 4.0. https://huggingface.co/datasets/google/fleurs\n"
                 "Suggested credit: \"Voice based on Google FLEURS, CC BY 4.0.\"\n")


if __name__ == "__main__":
    main()
