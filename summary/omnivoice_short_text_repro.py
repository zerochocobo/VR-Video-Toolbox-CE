#!/usr/bin/env python3
"""Minimal reproduction: OmniVoice cross-lingual voice cloning fails for SHORT
target text.

Setup
-----
* Model:     k2-fsa/OmniVoice  (loaded from a local snapshot), fp16, CUDA.
* Reference: ONE fixed Japanese speech clip (~3.7 s) + its Japanese transcript.
             This is the recommended 3-10 s reference length.
* Target:    Chinese sentences of decreasing length, same reference for all.

Observation
-----------
* LONG Chinese target  -> cloned correctly  (Whisper round-trip == target).
* SHORT Chinese target -> garbage / wrong content / wrong language
                          (Whisper round-trip != target, often another language).

Each generated clip is transcribed back with Whisper (faster-whisper) so the
output is judged objectively, not by ear. Results are printed and also written
to ``omnivoice_short_text_repro_result.json`` next to this script.

Run:
    python summary/omnivoice_short_text_repro.py
"""
import json
import os

import numpy as np
import torch
from faster_whisper import WhisperModel

from omnivoice import OmniVoice

# --- paths (edit if your layout differs) ---
OMNIVOICE_DIR = "models/OmniVoice"                     # local snapshot of k2-fsa/OmniVoice
ASR_DIR = "models/faster-whisper-large-v3"             # any Whisper works; used only to read back the output
REF_AUDIO = "videos/sub_35s.clone/ref_SPEAKER_00.wav"  # fixed ~3.7s Japanese speech reference
REF_TEXT = "仕事できない人に対してどう教えていいか分かんないんだよね"  # transcript of REF_AUDIO (Japanese)

# Chinese targets, long -> short. Same reference, same params for all.
TARGETS = [
    ("long",   "你好，今天天气真不错，我们一起出去散步聊聊天吧。"),
    ("medium", "后辈一下子多了好多。"),
    ("short",  "加班多吧。"),
    ("short2", "厉害啊。"),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OmniVoice.from_pretrained(OMNIVOICE_DIR, device_map=device, dtype=torch.float16)
    asr = WhisperModel(ASR_DIR, device=device, compute_type="float16" if device == "cuda" else "int8")
    sr = model.sampling_rate

    def round_trip(audio):
        clip = np.asarray(audio, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(clip))) or 1.0
        clip = clip / peak * 0.6  # OmniVoice output is quiet; normalize so ASR can read it
        segs, info = asr.transcribe(clip, language="zh", beam_size=1)
        return round(len(clip) / sr, 2), info.language, "".join(s.text for s in segs).strip()

    results = []
    # 1) length sweep, identical generate() call for each target
    for name, text in TARGETS:
        audio = model.generate(
            text=text,
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            language="Chinese",
            num_step=32,
            guidance_scale=2.0,
        )[0]
        dur, lang, rec = round_trip(audio)
        row = {"case": name, "target": text, "gen_dur_s": dur, "asr_lang": lang, "asr_text": rec}
        results.append(row)
        print(f"[{name:7}] target={text!r}")
        print(f"          gen={dur}s  asr_lang={lang}  asr={rec!r}\n")

    # 2) the short target with language passed as 'zh' / 'Chinese' / None
    for lang_arg in ("zh", "Chinese", None):
        audio = model.generate(
            text="加班多吧。", ref_audio=REF_AUDIO, ref_text=REF_TEXT,
            language=lang_arg, num_step=32, guidance_scale=2.0,
        )[0]
        dur, lang, rec = round_trip(audio)
        results.append({"case": f"short_lang={lang_arg}", "target": "加班多吧。",
                        "gen_dur_s": dur, "asr_lang": lang, "asr_text": rec})
        print(f"[short lang={str(lang_arg):8}] asr_lang={lang}  asr={rec!r}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omnivoice_short_text_repro_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
