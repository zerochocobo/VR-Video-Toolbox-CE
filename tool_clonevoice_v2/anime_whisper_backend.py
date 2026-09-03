"""litagin/anime-whisper text generation for the dub.

A Whisper large-v2 fine-tune on anime and drama-style Japanese dialogue -- the
register this material is actually in, which neither the base model nor kotoba
was trained on. It produces text only: no segment boundaries and no word
timestamps. That is workable here because the WhisperSeg split already frames
one utterance per window, so the timing comes from the audio and the model only
has to supply the words. It is the arrangement WhisperJAV calls ChronosJAV.

Japanese only. Any other source language has to use kotoba or large-v3.

Two constraints come from the model author's own demo app and model card, and
both are easy to get wrong:

* **No initial prompt.** Whisper's ``prompt_ids`` conditioning makes this model
  hallucinate. There is deliberately no way to pass context here.
* **Greedy decoding** (``do_sample=False``, ``num_beams=1``).

And one from WhisperJAV's adapter, which is a Windows landmine: the HuggingFace
``pipeline()`` wrapper crashes the process with 0xC0000409 on torch 2.9 /
transformers 4.57. The low-level processor + model API is used instead. We have
hit the same class of hard crash with CTranslate2 and cuDNN, so the warning is
taken at face value rather than re-tested.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

LogCallback = Callable[[str], None]

SAMPLE_RATE = 16000
MODEL_KEY = "anime-whisper"

# 448 target positions minus the four special tokens. The shipped
# generation_config asks for 4096, which exceeds what the model can attend to.
MAX_NEW_TOKENS = 444


def model_dir(models_root: str) -> Path:
    """Where the weights live. One table for every ASR model, in whisperx_backend,
    so the GUI's presence check, size query and download button all reach this one
    the same way they reach kotoba."""
    from tool_clonevoice_v2 import whisperx_backend as wx

    return wx.model_dir(MODEL_KEY, models_root)


def check_model_files(models_root: str) -> bool:
    from tool_clonevoice_v2 import whisperx_backend as wx

    return wx.check_model_files(MODEL_KEY, models_root)


class AnimeWhisperGenerator:
    """Turns one window of 16 kHz mono audio into a line of Japanese text."""

    def __init__(self, model_path: str, log: LogCallback = print,
                 use_gpu: bool = True):
        self.model_path = model_path
        self.log = log
        self.use_gpu = use_gpu
        self.processor = None
        self.model = None
        self.device = "cpu"
        self.dtype = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.log(f"[anime] loading anime-whisper on {self.device}")
        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=self.dtype,
        ).to(self.device).eval()

    def unload(self) -> None:
        self.model = None
        self.processor = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, audio) -> str:
        """Transcribe one window. Returns '' when the model produces nothing."""
        import torch

        self.load()
        features = self.processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
        ).input_features.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            tokens = self.model.generate(
                features,
                language="ja",
                task="transcribe",
                do_sample=False,
                num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS,
                # The model card's default. Whisper repetition loops are handled
                # downstream by is_repetition_noise, which can also tell a real
                # repeated cry from a stuck decoder; an n-gram ban cannot.
                no_repeat_ngram_size=0,
            )
        text = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
        return (text or "").strip()

    def generate_batch(self, windows: list, batch_size: int = 8) -> list[str]:
        """Transcribe several windows per forward pass.

        Whisper pads every input to 30 seconds regardless of its real length, so
        a three-second window costs the same as a thirty-second one and batching
        is where the time goes.
        """
        import torch

        self.load()
        results: list[str] = []
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            features = self.processor(
                batch, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            ).input_features.to(self.device, dtype=self.dtype)
            with torch.no_grad():
                tokens = self.model.generate(
                    features,
                    language="ja",
                    task="transcribe",
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=MAX_NEW_TOKENS,
                    no_repeat_ngram_size=0,
                )
            results.extend(
                (text or "").strip()
                for text in self.processor.batch_decode(tokens, skip_special_tokens=True)
            )
        return results
