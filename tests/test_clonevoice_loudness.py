from __future__ import annotations

from pathlib import Path

import numpy as np

from tool_clonevoice import omnivoice_backend as ov


def test_match_sentence_loudness_raises_quiet_clip_toward_source():
    clip = np.full(1600, 0.02, dtype=np.float32)
    source = np.full(1600, 0.08, dtype=np.float32)

    out, gain, source_db, synth_db = ov._match_sentence_loudness(clip, source)

    assert np.isclose(gain, 4.0)
    assert np.isclose(ov._rms(out), ov._rms(source), rtol=1e-5)
    assert source_db > synth_db


def test_match_sentence_loudness_lowers_loud_clip_to_quiet_source():
    clip = np.full(1600, 0.2, dtype=np.float32)
    source = np.full(1600, 0.02, dtype=np.float32)

    out, gain, source_db, synth_db = ov._match_sentence_loudness(clip, source)

    assert np.isclose(gain, 0.1, rtol=1e-5)
    assert np.isclose(ov._rms(out), ov._rms(source), rtol=1e-5)
    assert source_db < synth_db


def test_match_sentence_loudness_limits_gain_by_peak():
    clip = np.array([0.5, -0.5, 0.25], dtype=np.float32)
    source = np.full(3, 1.0, dtype=np.float32)

    out, gain, _source_db, _synth_db = ov._match_sentence_loudness(
        clip, source, target_peak_limit=0.8, max_gain=10.0
    )

    assert gain <= 1.6
    assert float(np.max(np.abs(out))) <= 0.800001


def test_match_sentence_loudness_falls_back_when_source_missing():
    clip = np.full(800, 0.1, dtype=np.float32)

    out, gain, source_db, synth_db = ov._match_sentence_loudness(clip, None)

    assert gain == 1.0
    assert np.array_equal(out, clip)
    assert source_db == synth_db


def _ramp(n, lo, hi):
    return np.linspace(lo, hi, n).astype(np.float32)


def test_follow_envelope_noop_without_source():
    clip = _ramp(24000, 0.2, 0.2)
    assert np.array_equal(ov._follow_energy_envelope(clip, None, 24000, 16000, 0.6), clip)
    assert np.array_equal(ov._follow_energy_envelope(clip, clip, 24000, 16000, 0.0), clip)


def test_follow_envelope_transfers_rising_contour():
    # flat synth, rising source -> output should end louder than it starts.
    rng = np.random.default_rng(0)
    clip = (rng.standard_normal(24000) * 0.1).astype(np.float32)
    source = (rng.standard_normal(16000) * _ramp(16000, 0.05, 0.5)).astype(np.float32)

    out = ov._follow_energy_envelope(clip, source, 24000, 16000, 0.6)

    head = ov._rms(out[: out.size // 4])
    tail = ov._rms(out[-out.size // 4 :])
    assert tail > head * 1.3


def test_follow_envelope_respects_db_clamp():
    rng = np.random.default_rng(1)
    clip = (rng.standard_normal(24000) * 0.1).astype(np.float32)
    # extreme source contour: silent start, loud end
    source = (rng.standard_normal(16000) * _ramp(16000, 0.0, 1.0)).astype(np.float32)

    out = ov._follow_energy_envelope(clip, source, 24000, 16000, 1.0, max_db=6.0)

    # per-sample gain stays within (1-a)+a*[1/lim, lim] = [0.5, 2.0] for a=1, 6 dB
    gain = np.divide(out, clip, out=np.ones_like(clip), where=np.abs(clip) > 1e-4)
    assert gain.min() >= 0.5 - 1e-3
    assert gain.max() <= 2.0 + 1e-3


def test_builtin_generic_ref_texts_fit_reference_duration_window():
    chinese = ov._GENERIC_REF_TEXTS["chinese"]
    english = ov._GENERIC_REF_TEXTS["english"]
    korean = ov._GENERIC_REF_TEXTS["korean"]
    thai = ov._GENERIC_REF_TEXTS["thai"]

    for language, text in ov._GENERIC_REF_TEXTS.items():
        assert 45 <= len("".join(text.split())) <= 130, language
    assert 35 <= len(ov._CJK_RE.findall(chinese)) <= 55
    assert 45 <= len(ov._CJK_RE.findall(korean)) <= 65
    assert len(ov._LATIN_WORD_RE.findall(english)) >= 18
    assert len("".join(english.split())) <= 115
    assert 80 <= len([ch for ch in thai if not ch.isspace()]) <= 125
    for language in ("english", "german", "french", "spanish", "portuguese", "italian"):
        assert 18 <= len(ov._LATIN_WORD_RE.findall(ov._GENERIC_REF_TEXTS[language])) <= 28
    for duration in ov._GENERIC_REF_DURATION.values():
        assert 7.0 <= duration <= 8.5
    assert 7.0 <= ov._resolve_generic_ref("Thai")[1] <= 8.5


class _RetryGenerateModel:
    sampling_rate = 24000
    device = "cpu"

    def __init__(self):
        self.generate_kwargs = []

    def generate(self, **kwargs):
        self.generate_kwargs.append(kwargs)
        if kwargs.get("postprocess_output") is False:
            return [np.full(2400, 0.05, dtype=np.float32)]
        raise ValueError("zero-size array to reduction operation maximum which has no identity")


def test_synthesize_retries_empty_omnivoice_postprocess(tmp_path: Path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    manifest = {
        "speakers": {},
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 1.0,
                "tgt_text": "你好",
            }
        ],
    }
    model = _RetryGenerateModel()

    out = ov.synthesize(
        model,
        manifest,
        str(video),
        tmp_path,
        text_field="tgt_text",
        language="Chinese",
        loudness_mode="flat",
        log=lambda _m: None,
    )

    assert Path(out).is_file()
    assert any(kwargs.get("postprocess_output") is False for kwargs in model.generate_kwargs)
