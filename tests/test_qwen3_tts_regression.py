"""End-to-end regression test for the in-process Qwen3-TTS path.

Two well-known failure modes have been observed when running the vendored Qwen3-TTS
runtime under transformers >= 5.x:

1. ``Qwen3TTSTalkerRotaryEmbedding.inv_freq`` ends up as a zero buffer after meta-device
   materialisation, which removes all positional information and collapses the talker
   into a language-model prior (audio output is unintelligible filler / "感谢观看").
2. ``talker.text_projection.linear_fc{1,2}.bias`` are silently dropped by HF's loader
   and left at their ``_init_weights`` zero baseline, which destroys text→talker
   projection and weakens text conditioning so the model never emits EOS.

This test guards against both regressions by:
  * verifying ``inv_freq`` recomputes to a non-zero vector on first forward,
  * verifying the bias tensors load with non-zero values,
  * (optionally, when ASR is available) synthesising a short Chinese sentence and
    asserting the transcript matches the input text.

The model and ASR weights are large, so the test is opt-in via the
``VRTB_TEST_RUN_TTS_MODEL=1`` environment variable.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path


MODEL_DIR_ENV = "VRTB_TEST_TTS_MODEL_DIR"
RUN_FLAG = "VRTB_TEST_RUN_TTS_MODEL"


def _resolve_model_dir() -> Path | None:
    raw = os.environ.get(MODEL_DIR_ENV)
    candidates = []
    if raw:
        candidates.append(Path(raw))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "models" / "Qwen3-TTS-12Hz-0.6B-CustomVoice")
    for path in candidates:
        if path.is_dir() and (path / "config.json").is_file():
            return path
    return None


@unittest.skipUnless(os.environ.get(RUN_FLAG) == "1", "set VRTB_TEST_RUN_TTS_MODEL=1 to enable")
class Qwen3TTSInProcessRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_dir = _resolve_model_dir()
        if cls.model_dir is None:
            raise unittest.SkipTest(
                "Qwen3-TTS model directory not found; set VRTB_TEST_TTS_MODEL_DIR"
            )
        import torch

        from tool_si._vendor.qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        has_cuda = torch.cuda.is_available()
        cls.tts_model = Qwen3TTSModel.from_pretrained(
            str(cls.model_dir),
            device_map="cuda:0" if has_cuda else "cpu",
            dtype=torch.bfloat16 if has_cuda else torch.float32,
            attn_implementation="sdpa" if has_cuda else "eager",
        )

    def test_text_projection_biases_are_loaded(self) -> None:
        """Regression for the silently-dropped text_projection biases under transformers 5.x."""
        b1 = self.tts_model.model.talker.text_projection.linear_fc1.bias
        b2 = self.tts_model.model.talker.text_projection.linear_fc2.bias
        b1_rms = (b1.float() ** 2).mean().sqrt().item()
        b2_rms = (b2.float() ** 2).mean().sqrt().item()
        # Checkpoint values are ~5e-3 / ~3e-2; anything below 1e-4 means the loader silently
        # zeroed them again.
        self.assertGreater(b1_rms, 1e-4, "talker.text_projection.linear_fc1.bias did not load")
        self.assertGreater(b2_rms, 1e-4, "talker.text_projection.linear_fc2.bias did not load")

    def test_rotary_inv_freq_recomputes_on_forward(self) -> None:
        """Regression for the zeroed inv_freq buffer after meta-device materialisation."""
        import torch

        talker_re = self.tts_model.model.talker.model.rotary_emb
        # Force a forward to trigger lazy recompute.
        device = next(self.tts_model.model.parameters()).device
        dtype = next(self.tts_model.model.parameters()).dtype
        hidden = torch.zeros(1, 4, talker_re.config.hidden_size, device=device, dtype=dtype)
        pos = torch.arange(4, device=device).view(1, 1, -1).expand(3, 1, -1)
        talker_re(hidden, pos)
        inv = talker_re.inv_freq
        self.assertGreater(inv.float().abs().max().item(), 0.0,
                           "RoPE inv_freq still zero after forward — positional encoding is broken")

    def test_chinese_synthesis_matches_input(self) -> None:
        """Synthesise a short Chinese sentence and verify ASR transcript matches the input."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            self.skipTest("faster_whisper not installed; skipping ASR verification")

        import tempfile

        import numpy as np
        import soundfile as sf
        import torch

        torch.manual_seed(0)
        text = "最近肩膀疼"
        wavs, sr = self.tts_model.generate_custom_voice(
            text=text + "。",
            speaker="Vivian",
            language="Chinese",
            max_new_tokens=150,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
        )
        wav = np.asarray(wavs[0], dtype=np.float32)
        duration = wav.shape[0] / sr
        # The talker collapsed-state failure mode runs the codec generator all the way to
        # ``max_new_tokens`` (~12 s+ at 12 Hz). A healthy response for this prompt is ~1-3 s.
        self.assertLess(
            duration, 6.0,
            f"audio duration {duration:.2f}s looks degenerate (model ran past EOS)",
        )
        self.assertGreater(duration, 0.5, f"audio duration {duration:.2f}s is suspiciously short")

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "out.wav"
            sf.write(str(wav_path), wav, sr)

            asr = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu",
                               compute_type="float16" if torch.cuda.is_available() else "int8")
            segs, _ = asr.transcribe(str(wav_path), language="zh", beam_size=5)
            transcript = "".join(s.text for s in segs).replace(" ", "")

        # Loose match: the ASR text should at least share most characters with the input —
        # the talker collapse mode produces unrelated phrases like "请订阅" / "感谢观看".
        common = sum(1 for ch in text if ch in transcript)
        self.assertGreaterEqual(
            common, max(2, len(text) - 1),
            f"ASR transcript {transcript!r} does not match input {text!r}",
        )


if __name__ == "__main__":
    unittest.main()
