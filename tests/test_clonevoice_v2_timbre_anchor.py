"""One timbre reference per speaker, the sentence only for emotion.

Cloning every sentence from itself makes IndexTTS re-derive the voice each
time, so the timbre drifts line to line. The anchor path pins the voice on one
sentence per speaker and passes the current sentence as ``emo_audio_prompt``.
The shapes below come from the profiles ``analyze_reference_quality`` produces
on the real titles.
"""
from __future__ import annotations

import inspect

import pytest

from tool_clonevoice_v2 import backend


def _profile(duration, *, active=None, rms=-20.0, active_rms=-18.0, periodic=0.6):
    return {
        "duration": float(duration),
        "active_seconds": float(duration * 0.8 if active is None else active),
        "rms_dbfs": float(rms),
        "active_rms_dbfs": float(active_rms),
        "periodic_ratio": float(periodic),
        "periodic_seconds": 0.0,
    }


def _segment(seg_id, start, end, speaker=None):
    seg = {"id": seg_id, "start": float(start), "end": float(end), "text": "x"}
    if speaker is not None:
        seg["speaker"] = speaker
    return seg


# --- picking the anchor ---

def test_the_anchor_is_the_best_scoring_sentence_not_the_longest():
    """A 9.5s sentence that is mostly silence loses to a clean 6s one."""
    entries = [_segment(1, 0, 9.5), _segment(2, 20, 26)]
    profiles = [
        _profile(9.5, active=2.0, periodic=0.35),
        _profile(6.0, active=5.2, periodic=0.85),
    ]
    assert backend.select_speaker_timbre_anchors(entries, profiles) == {"": 1}


def test_a_sentence_outside_the_window_can_never_anchor():
    """Under 3s is too thin to fix a voice; over 10s dilutes it."""
    assert backend._anchor_score(_profile(2.5), -18.0) == float("-inf")
    assert backend._anchor_score(_profile(12.0), -18.0) == float("-inf")
    assert backend._anchor_score(_profile(6.0), -18.0) > float("-inf")


def test_loudness_is_judged_against_the_title_not_in_absolute_dbfs():
    """The whisper-heavy titles sit ~25 dB under the absolute stable-reference
    floor: on sivr-434 part1 an absolute gate rejected 174 of 182 sentences and
    the video got no anchor at all. Quiet in absolute terms must still qualify
    when it is the title's own normal level."""
    quiet = _profile(6.0, active_rms=-44.0, rms=-46.0)
    assert backend._anchor_score(quiet, -44.0) > float("-inf")
    # ...but a line far under its own title's median is still rejected.
    assert backend._anchor_score(quiet, -20.0) == float("-inf")


def test_a_speaker_whose_lines_are_all_short_simply_gets_no_anchor():
    """sivr-314 part1: 80 sentences, longest 2.7s. Degrading to per-sentence
    cloning is correct; inventing an anchor from a 1s line is not."""
    entries = [_segment(1, 0, 2.7), _segment(2, 5, 6.2)]
    profiles = [_profile(2.7, periodic=0.9), _profile(1.2, periodic=0.9)]
    assert backend.select_speaker_timbre_anchors(entries, profiles) == {}


def test_each_speaker_gets_its_own_anchor():
    entries = [
        _segment(1, 0, 6, "SPEAKER_00"), _segment(2, 10, 16, "SPEAKER_01"),
        _segment(3, 20, 26, "SPEAKER_00"),
    ]
    profiles = [_profile(6.0, periodic=0.5), _profile(6.0, periodic=0.9),
                _profile(6.0, periodic=0.9)]
    anchors = backend.select_speaker_timbre_anchors(entries, profiles)
    assert anchors == {"SPEAKER_00": 2, "SPEAKER_01": 1}


def test_unlabelled_segments_share_one_anchor():
    """The complaint came from single-speaker titles, which run without
    diarization and therefore carry no speaker key at all."""
    entries = [_segment(1, 0, 6), _segment(2, 10, 16)]
    profiles = [_profile(6.0, periodic=0.9), _profile(6.0, periodic=0.4)]
    assert backend.select_speaker_timbre_anchors(entries, profiles) == {"": 0}


def test_a_title_with_no_stable_sentence_yields_no_anchor():
    """Must degrade to the old per-sentence behaviour, not crash or pick junk."""
    entries = [_segment(1, 0, 1.2), _segment(2, 5, 6.0)]
    profiles = [_profile(1.2, active=0.3, periodic=0.05),
                _profile(1.0, active=0.2, periodic=0.02)]
    assert backend.select_speaker_timbre_anchors(entries, profiles) == {}


def test_an_unvoiced_sentence_is_never_the_anchor():
    """A moan or breath that bandit routed to the sfx stem carries no voice."""
    entries = [_segment(1, 0, 6), _segment(2, 10, 16)]
    profiles = [_profile(6.0, periodic=0.05), _profile(6.0, periodic=0.7)]
    assert backend.select_speaker_timbre_anchors(entries, profiles) == {"": 1}


# --- emo_alpha, the part that decides whether the anchor survives ---

def test_the_anchor_alpha_is_below_one():
    """IndexTTS-2's emotion prompt leaks timbre; at 1.0 the sentence's own
    voice comes back and cancels the anchor."""
    assert 0.0 < backend.SPEAKER_ANCHOR_EMOTION_ALPHA < 1.0


def test_the_model_default_is_one_so_alpha_must_be_passed_explicitly():
    """The premise behind the constant: never let emo_alpha fall through.

    Read from source rather than imported: infer_v2 pulls in librosa and the
    whole model stack, which the test environment does not carry.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(
        Path("tool_clonevoice_v2/vendor/indextts/infer_v2.py").read_text(encoding="utf-8")
    )
    infer = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "infer"
    )
    names = [arg.arg for arg in infer.args.args]
    defaults = dict(zip(names[-len(infer.args.defaults):], infer.args.defaults))
    assert ast.literal_eval(defaults["emo_alpha"]) == 1.0


def test_an_unvoiced_sentence_keeps_its_downweight_under_the_anchor():
    """The pre-existing protection must not be lost to the new default path:
    a breath is a bad emotion prompt as much as a bad timbre one."""
    assert backend.UNVOICED_EMOTION_ALPHA < backend.SPEAKER_ANCHOR_EMOTION_ALPHA


# --- the switch ---

def test_synthesize_defaults_to_anchoring():
    params = inspect.signature(backend.synthesize_manifest).parameters
    assert params["timbre_anchor"].default is True


@pytest.mark.parametrize(
    "func_name", ["run_synthesize", "run_full", "run_batch"]
)
def test_the_switch_reaches_every_entry_point(func_name):
    from tool_clonevoice_v2 import logic

    params = inspect.signature(getattr(logic, func_name)).parameters
    assert "timbre_anchor" in params, func_name
    assert params["timbre_anchor"].default is True, func_name


# --- what synthesize_manifest actually sends to IndexTTS ---

@pytest.fixture
def fake_audio(monkeypatch, tmp_path):
    """Run synthesize_manifest without soundfile or a model.

    WAVs live in a dict keyed by path, so the reference extraction, the model
    stub, and the read-back of generated clips all round-trip through numpy.
    """
    import numpy as np
    from pathlib import Path

    store: dict[str, tuple] = {}

    def write(path, wav, sr):
        store[str(path)] = (np.asarray(wav, dtype=np.float32).copy(), int(sr))
        # The code under test checks Path.is_file() on what the model produced.
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()

    def read(path):
        return store[str(path)]

    monkeypatch.setattr(backend, "_write_audio", write)
    monkeypatch.setattr(backend, "_read_audio", read)

    sr = backend.SAMPLE_RATE
    t = np.arange(int(30 * sr), dtype=np.float32) / sr
    store[str(tmp_path / "audio16k.wav")] = (
        (0.3 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32), sr
    )
    return store, tmp_path


def _run_manifest(store, tmp_path, entries, **kwargs):
    calls = []

    class Model:
        def infer(self, **kw):
            calls.append(dict(kw))
            out = kw["output_path"]
            backend._write_audio(
                out,
                __import__("numpy").zeros(backend.SAMPLE_RATE, dtype="float32"),
                backend.SAMPLE_RATE,
            )
            return out

    backend.synthesize_manifest(
        Model(), tmp_path / "audio16k.wav", entries, tmp_path / "out.wav",
        language="ja", fit_duration=False, log=lambda _m: None,
        intermediate_dir=tmp_path / "work", level_match=False, **kwargs,
    )
    return calls


def test_a_weak_sentence_clones_its_voice_from_the_anchor(fake_audio):
    """seg 2 is a 1s line: too short to be a voice print of its own."""
    store, tmp_path = fake_audio
    entries = [
        {"id": 1, "start": 0.0, "end": 6.0, "text": "anchor"},
        {"id": 2, "start": 8.0, "end": 9.0, "text": "short"},
    ]
    calls = _run_manifest(store, tmp_path, entries)

    assert calls[1]["spk_audio_prompt"].endswith("sentence_ref_00001.wav")
    assert calls[1]["emo_audio_prompt"].endswith("sentence_ref_00002.wav")
    assert calls[1]["emo_alpha"] == backend.SPEAKER_ANCHOR_EMOTION_ALPHA


def test_the_anchor_sentence_still_clones_from_itself(fake_audio):
    store, tmp_path = fake_audio
    entries = [
        {"id": 1, "start": 0.0, "end": 6.0, "text": "anchor"},
        {"id": 2, "start": 8.0, "end": 9.0, "text": "short"},
    ]
    calls = _run_manifest(store, tmp_path, entries)

    assert calls[0]["spk_audio_prompt"].endswith("sentence_ref_00001.wav")
    assert "emo_audio_prompt" not in calls[0]


def test_turning_the_anchor_off_restores_per_sentence_cloning(fake_audio):
    store, tmp_path = fake_audio
    entries = [
        {"id": 1, "start": 0.0, "end": 6.0, "text": "anchor"},
        {"id": 2, "start": 8.0, "end": 9.0, "text": "short"},
    ]
    calls = _run_manifest(store, tmp_path, entries, timbre_anchor=False)

    assert calls[1]["spk_audio_prompt"].endswith("sentence_ref_00002.wav")


def test_an_anchored_call_never_leaves_emo_alpha_to_the_model(fake_audio):
    """The whole point of the constant: 1.0 would cancel the anchor."""
    store, tmp_path = fake_audio
    entries = [
        {"id": 1, "start": 0.0, "end": 6.0, "text": "anchor"},
        {"id": 2, "start": 8.0, "end": 9.0, "text": "short"},
        {"id": 3, "start": 12.0, "end": 13.2, "text": "another"},
    ]
    calls = _run_manifest(store, tmp_path, entries)

    for call in calls:
        if "emo_audio_prompt" in call:
            assert 0.0 < call["emo_alpha"] < 1.0


def test_speakers_do_not_borrow_each_others_voices(fake_audio):
    store, tmp_path = fake_audio
    entries = [
        {"id": 1, "start": 0.0, "end": 6.0, "text": "a", "speaker": "SPEAKER_00"},
        {"id": 2, "start": 8.0, "end": 14.0, "text": "b", "speaker": "SPEAKER_01"},
        {"id": 3, "start": 16.0, "end": 17.0, "text": "c", "speaker": "SPEAKER_01"},
    ]
    calls = _run_manifest(store, tmp_path, entries)

    assert calls[2]["spk_audio_prompt"].endswith("sentence_ref_00002.wav")


# --- stitched anchors: titles whose longest line is too short ---

def test_a_title_of_only_short_lines_stitches_an_anchor():
    """sivr-314 part1: 80 lines, longest 2.68s, median 1.2s. Before stitching
    it got no anchor at all and every line cloned from itself."""
    entries = [_segment(i, i * 4.0, i * 4.0 + 2.2) for i in range(1, 6)]
    profiles = [_profile(2.2, periodic=0.8) for _ in entries]

    pieces = backend.select_stitch_pieces(entries, profiles, "", -18.0)

    assert pieces, "should stitch rather than give up"
    assert sum(profiles[i]["duration"] for i in pieces) >= backend.ANCHOR_MIN_SECONDS
    assert pieces == sorted(pieces), "pieces must stay in time order"


def test_a_whisper_is_never_stitched_into_the_anchor():
    """The anchor fixes the voice for the whole title; pinning it at whisper
    level is the one outcome the feature exists to avoid."""
    loud = _profile(1.5, active_rms=-18.0, periodic=0.8)
    whisper = _profile(1.5, active_rms=-40.0, periodic=0.8)
    assert backend._stitch_candidate(loud, -18.0)
    assert not backend._stitch_candidate(whisper, -18.0)


def test_an_unvoiced_piece_is_never_stitched_in():
    breath = _profile(1.5, active_rms=-18.0, periodic=0.05)
    assert not backend._stitch_candidate(breath, -18.0)


def test_a_speaker_of_only_whispers_gets_no_anchor():
    """Refusing is correct: cloning per sentence beats a whisper anchor."""
    entries = [_segment(i, i * 4.0, i * 4.0 + 2.0) for i in range(1, 6)]
    profiles = [_profile(2.0, active_rms=-45.0, periodic=0.8) for _ in entries]
    assert backend.select_stitch_pieces(entries, profiles, "", -18.0) == []


def test_scattered_lines_still_get_an_anchor():
    """Adjacency is a preference, not a requirement: hnvr-152's SPEAKER_00 had
    7.4s of usable material but spread far apart, and a hard gap limit refused
    it an anchor even though the material was there."""
    entries = [_segment(i, i * 600.0, i * 600.0 + 1.6) for i in range(1, 5)]
    profiles = [_profile(1.6, periodic=0.8) for _ in entries]

    assert backend.select_stitch_pieces(entries, profiles, "", -18.0)


def test_neighbours_are_preferred_when_both_are_possible():
    entries = [_segment(1, 0.0, 1.6), _segment(2, 2.0, 3.6), _segment(3, 4.0, 5.6),
               _segment(4, 900.0, 901.6)]
    profiles = [_profile(1.6, periodic=0.8) for _ in entries]

    pieces = backend.select_stitch_pieces(entries, profiles, "", -18.0)

    assert 3 not in pieces, "the far line must not be spliced in unnecessarily"


def test_only_the_speaker_s_own_lines_are_stitched():
    entries = [_segment(1, 0.0, 1.6, "A"), _segment(2, 2.0, 3.6, "B"),
               _segment(3, 4.0, 5.6, "A"), _segment(4, 6.0, 7.6, "A")]
    profiles = [_profile(1.6, periodic=0.8) for _ in entries]

    pieces = backend.select_stitch_pieces(entries, profiles, "A", -18.0)

    assert all(entries[i]["speaker"] == "A" for i in pieces)


def test_the_stitched_wav_is_faded_at_every_join(tmp_path, monkeypatch):
    """A butt join clicks, and a click is the kind of transient the model would
    take for a consonant."""
    import numpy as np

    written = {}
    monkeypatch.setattr(backend, "_write_audio",
                        lambda path, wav, sr: written.update(wav=np.asarray(wav), sr=sr))
    sr = 22050
    source = np.ones(int(20 * sr), dtype=np.float32)
    entries = [_segment(1, 0.0, 1.6), _segment(2, 2.0, 3.6), _segment(3, 4.0, 5.6)]

    backend.write_stitched_anchor(source, sr, entries, [0, 1, 2], tmp_path / "a.wav")

    out = written["wav"]
    assert out.size == pytest.approx(int(1.6 * sr) * 3, abs=3)
    assert out[0] == pytest.approx(0.0, abs=1e-6), "must fade in"
    assert out[-1] == pytest.approx(0.0, abs=1e-6), "must fade out"
    fade = int(backend.STITCH_JOIN_FADE_SECONDS * sr)
    assert out[int(1.6 * sr) - 1] < 0.2, "join must be faded, not butt-spliced"


def test_the_stitched_anchor_never_exceeds_the_window(tmp_path, monkeypatch):
    import numpy as np

    written = {}
    monkeypatch.setattr(backend, "_write_audio",
                        lambda path, wav, sr: written.update(wav=np.asarray(wav)))
    sr = 22050
    entries = [_segment(i, i * 10.0, i * 10.0 + 8.0) for i in range(1, 5)]

    backend.write_stitched_anchor(np.ones(int(60 * sr), dtype=np.float32), sr,
                                  entries, [0, 1, 2, 3], tmp_path / "a.wav")

    assert written["wav"].size <= int(backend.ANCHOR_MAX_SECONDS * sr)
