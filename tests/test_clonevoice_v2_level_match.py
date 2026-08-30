"""Source-loudness matching, level shaping and log hygiene."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import pytest
import soundfile as sf

from tool_clonevoice_v2 import backend, log_redirect


class ConstantModel:
    """Stand-in for IndexTTS: always returns the same level, as the real one does."""

    def __init__(self, amplitude: float = 0.02):
        self.amplitude = amplitude

    def infer(self, **kwargs):
        wav = np.full(backend.SAMPLE_RATE // 2, self.amplitude, dtype=np.float32)
        sf.write(kwargs["output_path"], wav, backend.SAMPLE_RATE)
        return kwargs["output_path"]


# --- gain computation ---

def test_absolute_mode_keeps_the_order_of_the_source_levels():
    """Quiet lines stay quieter than loud ones -- the point of the feature --
    but at a reduced ratio, and lifted as a set if the title is quiet.
    Reproducing the source's dB values literally is what put lines at
    -61 dBFS on sivr-314, where nobody could hear them."""
    targets = [-40.0, -50.0, -45.0]
    measured = [-34.0, -34.0, -34.0]
    gains = backend.compute_level_gains(measured, targets, "absolute")
    out = [m + g for m, g in zip(measured, gains)]
    assert out[1] < out[2] < out[0], "louder source must stay louder"
    spread = out[0] - out[1]
    assert spread == pytest.approx(10.0 * backend.LEVEL_DYNAMIC_RANGE_RATIO, abs=0.01)


def test_a_loud_title_is_not_lifted():
    """The lift is one-directional, so titles where matching already worked
    keep the levels they had."""
    targets = [-20.0, -30.0, -25.0]
    gains_loud = backend.compute_level_gains([-34.0] * 3, targets, "absolute")
    median = -25.0
    shaped = [median + backend.LEVEL_DYNAMIC_RANGE_RATIO * (t - median) for t in targets]
    assert [round(-34.0 + g, 2) for g in gains_loud] == [round(v, 2) for v in shaped]


def test_a_quiet_title_is_lifted_until_its_median_is_audible():
    """sivr-314 part1: source median -52 dBFS, lines matched down to -61 and
    -64 dBFS and were inaudible."""
    targets = [-52.0, -61.6, -64.5, -58.3]
    gains = backend.compute_level_gains([-47.0] * 4, targets, "absolute")
    out = [-47.0 + g for g in gains]
    assert max(out) > -50.0, "the whole set must come up"
    assert min(out) > -55.0, f"quietest line still inaudible: {min(out):.1f} dBFS"


def test_the_floor_keeps_a_whisper_audible_instead_of_reproducing_it_literally():
    """A -95 dBFS whisper is held at median-18 dB rather than cut all the way."""
    targets = [-40.0, -42.0, -44.0, -95.0]
    measured = [-34.0, -34.0, -34.0, -50.0]
    gains = backend.compute_level_gains(measured, targets, "absolute")
    # The whisper is shaped and floored rather than reproduced: it must stay
    # clearly the quietest line without being cut into inaudibility.
    out = [m + g for m, g in zip(measured, gains)]
    assert out[-1] == min(out), "the whisper must stay the quietest line"
    assert out[-1] > -60.0, f"whisper cut to {out[-1]:.1f} dBFS"
    assert gains[-1] > -backend.LEVEL_MATCH_MAX_CUT_DB


def test_gains_are_clamped_in_both_directions():
    # Relative mode is where both ends can be reached: the audibility lift in
    # absolute mode holds every desired level at or above median-12, so nothing
    # there can ask for a 40 dB cut.
    gains = backend.compute_level_gains([-10.0, -80.0], [-90.0, -10.0], "relative")
    assert gains[0] == -backend.LEVEL_MATCH_MAX_CUT_DB
    assert gains[1] == backend.LEVEL_MATCH_MAX_BOOST_DB


def test_absolute_mode_never_needs_the_cut_clamp():
    """The lift places the median at the anchor and the floor is 12 dB under it,
    so no absolute-mode line can be asked for a cut anywhere near the clamp."""
    gains = backend.compute_level_gains([-6.0] * 3, [-90.0, -50.0, -10.0], "absolute")
    assert min(gains) > -backend.LEVEL_MATCH_MAX_CUT_DB


def test_a_noisy_clip_is_not_boosted_past_its_own_hiss():
    """Two clips wanting the same gain, one with a floor 40 dB down and one with
    a floor only 10 dB down: the clean one gets the lift, the noisy one is held
    to what keeps its hiss within reach of the film's bed."""
    measured, targets = [-45.0, -45.0], [-45.0, -45.0]
    floors, beds = [-85.0, -55.0], [-46.0, -46.0]
    uncapped = backend.compute_level_gains(measured, targets, "absolute")
    capped = backend.compute_level_gains(measured, targets, "absolute", floors, beds)
    assert capped[0] == pytest.approx(uncapped[0])
    assert capped[1] < uncapped[1]
    # the held-back clip's floor lands exactly on the allowance, not above it
    assert floors[1] + capped[1] == pytest.approx(
        beds[1] + backend.NOISE_FLOOR_MAX_ABOVE_BACKGROUND_DB
    )


def test_an_unmeasurable_floor_falls_back_to_the_flat_cap():
    assert backend.noise_floor_boost_cap(None, -46.0) == backend.LEVEL_MATCH_MAX_BOOST_DB
    assert backend.noise_floor_boost_cap(-80.0, None) == backend.LEVEL_MATCH_MAX_BOOST_DB


def test_a_clip_noisier_than_the_bed_is_not_boosted_at_all():
    assert backend.noise_floor_boost_cap(-20.0, -60.0) == 0.0


def test_speaker_medians_are_levelled_without_touching_each_speaker_dynamics():
    """A near-mic speaker and a distant one, each with their own spread."""
    entries = (
        [{"speaker": "A"} for _ in range(6)] + [{"speaker": "B"} for _ in range(6)]
    )
    near = [-40.0, -38.0, -36.0, -34.0, -32.0, -30.0]   # median -35
    far = [-60.0, -58.0, -56.0, -54.0, -52.0, -50.0]    # median -55
    adjusted, shifts = backend.match_speaker_medians(entries, near + far)
    assert shifts["A"] == pytest.approx(-10.0)
    assert shifts["B"] == pytest.approx(10.0)
    assert float(np.median(adjusted[:6])) == pytest.approx(float(np.median(adjusted[6:])))
    # each speaker's own 10 dB spread survives the shift
    assert max(adjusted[:6]) - min(adjusted[:6]) == pytest.approx(10.0)
    assert max(adjusted[6:]) - min(adjusted[6:]) == pytest.approx(10.0)


def test_a_speaker_with_too_few_lines_is_left_alone():
    entries = [{"speaker": "A"} for _ in range(6)] + [{"speaker": "B"}, {"speaker": "B"}]
    targets = [-40.0] * 6 + [-60.0, -60.0]
    adjusted, shifts = backend.match_speaker_medians(entries, targets)
    assert shifts == {}
    assert adjusted == targets


def test_a_single_speaker_title_is_left_alone():
    entries = [{"speaker": "A"} for _ in range(8)]
    targets = [-50.0, -40.0] * 4
    adjusted, shifts = backend.match_speaker_medians(entries, targets)
    assert shifts == {}
    assert adjusted == targets


def test_relative_mode_corrects_the_shape_without_moving_the_overall_level():
    """Without a speech stem the mixture level is inflated, so only the
    deviation from the median is corrected."""
    targets = [-40.0, -50.0, -45.0, -35.0]
    measured = [-20.0, -20.0, -20.0, -20.0]
    gains = backend.compute_level_gains(measured, targets, "relative")
    after = [m + g for m, g in zip(measured, gains)]
    assert abs(float(np.median(after)) - float(np.median(measured))) < 1e-6
    # Ranking now follows the source instead of being flat.
    assert np.corrcoef(targets, after)[0, 1] > 0.99


def test_silent_clips_are_left_alone():
    assert backend.compute_level_gains([-240.0], [-40.0], "absolute") == [0.0]


def test_the_floor_keeps_a_quiet_line_audible():
    """18 dB below the median landed lines at -67 dBFS, inaudible over a bed."""
    assert backend.LEVEL_FLOOR_BELOW_MEDIAN_DB == 12.0
    targets = [-45.0, -48.0, -51.0, -95.0]
    gains = backend.compute_level_gains([-34.0] * 4, targets, "absolute")
    floor = float(np.median(targets)) - backend.LEVEL_FLOOR_BELOW_MEDIAN_DB
    # -95 dBFS is far past the floor, so the floor -- not the source -- decides.
    assert -34.0 + gains[-1] >= floor - 0.01
    assert -34.0 + gains[-1] > -62.0


def test_too_few_measured_segments_falls_back_to_the_mixture():
    """One stem in ten is too thin to establish the stem-minus-mixture offset,
    so the mixture is used on its own. It used to fall to ``relative`` here,
    which also gave up the audibility anchor; the mixture is an absolute
    reading, so only the shape is less trustworthy, not the scale."""
    entries = [{"src_level": {"active_rms_dbfs": -40.0, "active_seconds": 0.5}}] + [{}] * 9
    profiles = [{"active_rms_dbfs": -30.0}] * 10
    targets, mode = backend.segment_level_targets(entries, profiles)
    assert mode == "mixture"
    assert targets == [-30.0] * 10


def test_apply_level_gain_backs_off_instead_of_clipping():
    clip = np.full(64, 0.9, dtype=np.float32)
    out = backend.apply_level_gain(clip, 12.0)
    assert abs(float(np.max(np.abs(out))) - backend.LEVEL_MATCH_PEAK_CEILING) < 1e-6


# --- end to end through the synthesizer ---

def _write_source(path, duration_seconds: float = 3.0):
    sf.write(path, np.zeros(int(backend.SAMPLE_RATE * duration_seconds), dtype=np.float32),
             backend.SAMPLE_RATE)


def test_synthesize_manifest_scales_a_clip_onto_its_measured_source_level(tmp_path):
    source, output = tmp_path / "source.wav", tmp_path / "out.wav"
    # A source with an audible bed: a digitally silent one leaves the dub's own
    # hiss with nothing to hide under, and the noise-floor cap correctly refuses
    # to boost into that silence. This test is about the matching arithmetic.
    sf.write(source, np.full(int(backend.SAMPLE_RATE * 3.0), 0.03, dtype=np.float32),
             backend.SAMPLE_RATE)

    backend.synthesize_manifest(
        ConstantModel(0.02), source,
        [{"id": 1, "start": 0.0, "end": 0.5, "text": "line",
          "src_level": {"active_rms_dbfs": -40.0}}],
        output, language="en", fit_duration=False, log=lambda _m: None,
    )

    wav = sf.read(output, dtype="float32")[0]
    # The single target of -40 dBFS is also the median, so the audibility lift
    # carries it to the -27 dBFS anchor. The 0.02 constant measures -33.98 dBFS,
    # so the clip is raised by 6.98 dB rather than cut by 6.02 as it was when
    # the anchor sat at -42 and left this title below it.
    assert abs(float(np.max(np.abs(wav))) - 0.02 * 10 ** (6.98 / 20)) < 5e-4


def test_synthesize_manifest_level_match_can_be_turned_off(tmp_path):
    source, output = tmp_path / "source.wav", tmp_path / "out.wav"
    _write_source(source)

    backend.synthesize_manifest(
        ConstantModel(0.02), source,
        [{"id": 1, "start": 0.0, "end": 0.5, "text": "line",
          "src_level": {"active_rms_dbfs": -40.0}}],
        output, language="en", fit_duration=False, level_match=False, log=lambda _m: None,
    )

    wav = sf.read(output, dtype="float32")[0]
    assert abs(float(np.max(np.abs(wav))) - 0.02) < 5e-5


# --- the slot no longer crushes the delivery ---

def test_the_target_is_the_natural_reading_not_the_slot():
    """Two failures, opposite directions, one rule.

    sivr-314 line 5: a 0.98s slot sized for the Japanese held a 9-character
    Chinese line needing 1.64s, and crushing it to fit made it unintelligible.
    Line 6: the model dragged 6 characters over 2.84s, and letting that play
    in full sounded just as wrong. Aim at the natural reading in both cases.
    """
    import inspect

    source = inspect.getsource(backend._fit_to_slot)
    assert "natural_reading_seconds" in source
    assert "MAX_HARD_COMPRESSION" in source
    assert "overflow_end" in inspect.getsource(backend.synthesize_manifest)
    # A slot shorter than the line needs: extend towards natural.
    assert backend.natural_reading_seconds("回来的时候都淋湿了", "zh") > 1.5
    # A dragged reading is compressed back, not played out in full.
    assert backend.natural_reading_seconds("雨一直没停嘛", "zh") < 1.3


def test_the_overflow_never_reaches_the_next_line(tmp_path):
    """Running over is only safe into silence."""
    source, output = tmp_path / "source.wav", tmp_path / "out.wav"
    sf.write(source, np.zeros(int(backend.SAMPLE_RATE * 30), dtype=np.float32),
             backend.SAMPLE_RATE)

    class LongModel:
        def infer(self, **kwargs):
            sf.write(kwargs["output_path"],
                     np.full(int(backend.SAMPLE_RATE * 3.0), 0.05, dtype=np.float32),
                     backend.SAMPLE_RATE)
            return kwargs["output_path"]

    backend.synthesize_manifest(
        LongModel(), source,
        [{"id": 1, "start": 0.0, "end": 0.5, "text": "a"},
         {"id": 2, "start": 1.0, "end": 1.5, "text": "b"}],
        output, language="en", fit_duration=True, log=lambda _m: None,
        level_match=False, timbre_anchor=False,
    )

    wav = sf.read(output, dtype="float32")[0]
    sr = backend.SAMPLE_RATE
    # "a" is one character: its natural reading is the 0.3s floor, so the line
    # is compressed back rather than allowed to play its full 3s. Either way it
    # must never reach line 2 at 1.0s.
    quiet_zone = wav[int(1.0 * sr) - 1:int(1.02 * sr)]
    assert float(np.max(np.abs(quiet_zone))) < 0.06, "line 1 must not reach line 2"


def test_the_model_is_never_asked_to_slow_down():
    """Measured on sivr-314: asked for 1.42s on a 6-character line, IndexTTS
    returned 2.84s -- 2.6x the natural reading. Over 80 lines it overshoots the
    request by 1.40x at the median and 4.15x at worst, always on short lines.
    A slot longer than the line needs is not a reason to drag the delivery."""
    long_slot = backend.duration_factor_for_slot("雨一直没停嘛", 5.0, "zh", "moderate")
    assert long_slot <= 1.0, f"asked the model to slow to {long_slot:.2f}x"
    for fit in ("moderate", "strong"):
        assert backend.duration_factor_for_slot("短", 9.0, "zh", fit) <= 1.0, fit
    # Speeding up must still work: a slot really shorter than the line needs.
    assert backend.duration_factor_for_slot("回来的时候都淋湿了", 0.98, "zh", "moderate") < 1.0


def test_a_line_shorter_than_its_slot_is_not_stretched_to_fill_it(tmp_path, monkeypatch):
    """Once translations are written to a time budget, most come in under it.
    Padding them out to the slot dragged 18 of 80 lines past 1.8x their natural
    reading -- finishing early and leaving silence costs nothing."""
    source, output = tmp_path / "source.wav", tmp_path / "out.wav"
    sf.write(source, np.zeros(int(backend.SAMPLE_RATE * 30), dtype=np.float32),
             backend.SAMPLE_RATE)

    class Model:
        def infer(self, **kwargs):
            sf.write(kwargs["output_path"],
                     np.full(int(backend.SAMPLE_RATE * 0.9), 0.05, dtype=np.float32),
                     backend.SAMPLE_RATE)
            return kwargs["output_path"]

    # 5 Chinese characters read naturally in ~0.91s, inside a 4s slot.
    backend.synthesize_manifest(
        Model(), source, [{"id": 1, "start": 0.0, "end": 4.0, "text": "淋湿回来了"}],
        output, language="zh", fit_duration=True, log=lambda _m: None,
        level_match=False, timbre_anchor=False,
    )

    wav = sf.read(output, dtype="float32")[0]
    spoken = float(np.count_nonzero(np.abs(wav) > 1e-4)) / backend.SAMPLE_RATE
    assert spoken == pytest.approx(0.91, abs=0.15), f"stretched to {spoken:.2f}s"


# --- log volume: the GUI window holds a limited number of lines ---

def test_the_vendor_chatter_is_filtered_out():
    """IndexTTS prints ~14 lines per sentence; 1096 of the 1254 lines a single
    80-line title produced said nothing the pipeline did not already say, and
    pushed real progress out of the GUI window."""
    from tool_clonevoice_v2.log_redirect import LogWriter

    kept = []
    writer = LogWriter(lambda text, _progress: kept.append(text))
    for line in (">> starting inference...", "torch.Size([1, 2])",
                 "Use the specified emotion vector", ">> gpt_gen_time: 1.2 seconds",
                 "origin text:hello", "[indextts-v2] 4/80 ...", "RuntimeError: real problem"):
        writer.write(line + "\n")

    assert kept == ["[indextts-v2] 4/80 ...", "RuntimeError: real problem"]
    assert writer.dropped == 5


def test_real_errors_are_never_filtered():
    from tool_clonevoice_v2.log_redirect import LogWriter

    kept = []
    writer = LogWriter(lambda text, _progress: kept.append(text))
    for line in ("Traceback (most recent call last):", "  File \"x.py\", line 1",
                 "torch.cuda.OutOfMemoryError: CUDA out of memory"):
        writer.write(line + "\n")
    assert len(kept) == 3


def test_the_filter_can_be_turned_off():
    from tool_clonevoice_v2.log_redirect import LogWriter

    kept = []
    LogWriter(lambda text, _p: kept.append(text), drop_vendor_noise=False).write(">> x\n")
    assert kept == [">> x"]


def test_one_log_line_per_synthesized_sentence(tmp_path):
    """Two lines per sentence plus the vendor's own made an 80-line title
    unreadable in a window that holds a few hundred lines."""
    source, output = tmp_path / "source.wav", tmp_path / "out.wav"
    sf.write(source, np.zeros(int(backend.SAMPLE_RATE * 20), dtype=np.float32),
             backend.SAMPLE_RATE)
    messages: list[str] = []

    backend.synthesize_manifest(
        ConstantModel(0.02), source,
        [{"id": i, "start": i * 2.0, "end": i * 2.0 + 1.0, "text": "线"} for i in range(1, 6)],
        output, language="zh", fit_duration=False, log=messages.append,
        level_match=False, timbre_anchor=False,
    )

    per_sentence = [m for m in messages if m.startswith("[indextts-v2] ") and "/5 " in m]
    assert len(per_sentence) == 5, per_sentence


# --- progress-bar noise and degenerate spans -------------------------------

def _collect(sample: str, interval: float = 0.0):
    out: list[tuple[bool, str]] = []
    writer = log_redirect.LogWriter(
        lambda text, is_progress: out.append((is_progress, text)),
        min_progress_interval=interval,
    )
    writer.write(sample)
    return out, writer.dropped


def test_a_bare_progress_bar_is_dropped():
    """The vendored IndexTTS opens a 25-step bar per sentence; on a 352-line
    title that is 704 lines of nothing."""
    sample = (
        " 84%|####################8    | 21/25 [00:02<00:00, 17.17it/s]\n"
        "100%|#########################| 25/25 [00:02<00:00, 10.78it/s]\n"
    )
    out, dropped = _collect(sample)
    assert out == []
    assert dropped == 2


def test_a_described_progress_bar_survives_but_replaces_itself():
    """"Span 112/195" is the only progress the separation stage reports, so it
    is kept -- as a progress line, which the GUI overwrites in place."""
    sample = (
        "Span 112/195:   0%|          | 0/1 [00:00<?, ?blk/s]\n"
        "Span 112/195: 100%|##########| 1/1 [00:09<00:00,  9.14s/blk]\n"
    )
    out, dropped = _collect(sample)
    assert dropped == 0
    assert [is_progress for is_progress, _ in out] == [True, True]


def test_the_per_sentence_normalization_echo_is_dropped():
    out, dropped = _collect("text after normalization: 好紧张啊\n")
    assert out == []
    assert dropped == 1


def test_a_traceback_is_never_mistaken_for_a_progress_bar():
    sample = (
        "Traceback (most recent call last):\n"
        '  File "x.py", line 3, in <module>\n'
        "CUDA out of memory. Tried to allocate 2.00 GiB\n"
        "100% done, everything fine\n"
    )
    out, dropped = _collect(sample)
    assert dropped == 0
    assert all(not is_progress for is_progress, _ in out)
    assert len(out) == 4


def test_a_zero_length_slot_is_widened_into_a_usable_prompt():
    """hnvr-174 part1 id=139 arrived as 1109.08-1109.08 and the one-sample WAV
    it produced killed the run 134 lines into 352."""
    start, end = backend.widen_degenerate_span(1109.08, 1109.08, 3467.5)
    assert end - start == pytest.approx(backend.MIN_REFERENCE_SECONDS)
    assert start < 1109.08 < end


def test_widening_stays_inside_the_file():
    start, end = backend.widen_degenerate_span(0.0, 0.0, 3467.5)
    assert start == 0.0 and end == pytest.approx(backend.MIN_REFERENCE_SECONDS)
    start, end = backend.widen_degenerate_span(3467.5, 3467.5, 3467.5)
    assert end == pytest.approx(3467.5)
    assert end - start == pytest.approx(backend.MIN_REFERENCE_SECONDS)


def test_a_long_enough_span_is_left_exactly_alone():
    assert backend.widen_degenerate_span(10.0, 14.0, 100.0) == (10.0, 14.0)


def test_every_degenerate_prompt_failure_triggers_the_retry():
    """Catching only the mel-kernel wording let the resampler's ValueError
    abort a whole title instead of retrying with a wider reference."""
    for message in (
        "Calculated padded input size per channel: (2 x 1024). Kernel size: (3 x 3)",
        "Input waveform must have only one dimension, shape is ()",
    ):
        assert backend._short_prompt_runtime_error(ValueError(message))
        assert backend._short_prompt_runtime_error(RuntimeError(message))
    assert not backend._short_prompt_runtime_error(RuntimeError("CUDA out of memory"))


def test_every_stage_reports_memory_not_just_the_all_in_one_entry_point():
    """run_batch drives the stages directly, so memory logging placed only in
    run_full left the batch mode -- the one actually used -- reporting nothing."""
    import inspect

    from tool_clonevoice_v2 import logic

    transcribe = inspect.getsource(logic.run_transcribe_diarize)
    assert 'log_memory("before transcription"' in transcribe
    assert 'log_memory("after transcription"' in transcribe

    synthesize = inspect.getsource(logic.run_synthesize)
    assert 'log_memory("before synthesis"' in synthesize
    assert 'log_memory("after synthesis"' in synthesize

    # ...and run_full must not repeat what the stages now report themselves.
    full = inspect.getsource(logic.run_full)
    assert 'log_memory("before transcription"' not in full
    assert 'log_memory("before synthesis"' not in full


def test_memory_line_survives_a_missing_psutil(monkeypatch):
    """Reporting is diagnostics; it must never take the run down with it."""
    from tool_clonevoice_v2 import logic

    lines: list[str] = []
    monkeypatch.setitem(__import__("sys").modules, "psutil", None)
    logic.log_memory("smoke", lines.append)
    assert all("Traceback" not in line for line in lines)


# --- separation is opt-in ---------------------------------------------------

def test_mixture_only_targets_still_get_the_audibility_anchor():
    """Without the separation stage there are no speech stems, and the old code
    dropped to relative mode -- which leaves the level where IndexTTS put it,
    i.e. -45.9 dBFS on this material. The mixture is an absolute reading, so it
    keeps the anchor."""
    entries = [{"id": i} for i in range(8)]
    profiles = [{"active_rms_dbfs": -45.0} for _ in entries]
    targets, mode = backend.segment_level_targets(entries, profiles)
    assert mode == "mixture"

    gains = backend.compute_level_gains([-45.9] * 8, targets, mode)
    out = [-45.9 + g for g in gains]
    assert float(np.median(out)) == pytest.approx(backend.LEVEL_AUDIBLE_MEDIAN_DBFS, abs=0.1)


def test_relative_mode_is_still_reachable_when_nothing_is_measurable():
    entries = [{"id": i} for i in range(4)]
    profiles = [{"active_rms_dbfs": -240.0} for _ in entries]
    _targets, mode = backend.segment_level_targets(entries, profiles)
    assert mode == "relative"


# --- one model per run, not one per video ----------------------------------

def test_the_refined_clone_run_loads_indextts_once_for_every_video(tmp_path, monkeypatch):
    """run_synthesize loads its own model when none is passed, and nothing
    released the previous one: on hnvr-174 VRAM allocated climbed 0.01 -> 5.60
    -> 11.18 GB across a three-part title."""
    from tool_clonevoice_v2 import backend as backend_module
    from tool_clonevoice_v2 import logic as logic_module
    from tool_clonevoice_v2 import single_clone

    loads: list[str] = []
    models_seen: list[object] = []

    monkeypatch.setattr(
        backend_module, "load_model",
        lambda models_root, **kw: loads.append(models_root) or object(),
    )
    monkeypatch.setattr(single_clone, "ensure_translated", lambda *a, **k: None)
    monkeypatch.setattr(
        logic_module, "run_synthesize",
        lambda video, **kw: models_seen.append(kw.get("model")) or f"{video}.si.wav",
    )

    videos = [str(tmp_path / f"v{i}.mp4") for i in range(3)]
    result = single_clone.translate_and_synthesize(
        videos, target_language="Chinese", models_root=str(tmp_path),
        skip_existing=False, log=lambda _m: None,
    )

    assert len(result["written"]) == 3
    assert len(loads) == 1, f"loaded the model {len(loads)} times for 3 videos"
    assert all(m is not None for m in models_seen), "a video fell back to its own load"
    assert len(set(id(m) for m in models_seen)) == 1, "videos did not share one model"


def test_a_fully_skipped_run_never_loads_the_model(tmp_path, monkeypatch):
    from tool_clonevoice_v2 import backend as backend_module
    from tool_clonevoice_v2 import single_clone
    from tool_si import logic as si

    loads: list[str] = []
    monkeypatch.setattr(
        backend_module, "load_model",
        lambda models_root, **kw: loads.append(models_root) or object(),
    )
    from tool_clonevoice_v2 import logic as logic_module

    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    Path(si.default_si_audio_path(str(video))).write_bytes(b"")
    # Skipping now requires the export to be recorded as up to date, not just
    # for the file to be sitting there.
    manifest = {"segments": [], "target_language": "Chinese"}
    manifest["synthesis"] = {
        "signature": logic_module.synthesis_signature(manifest, language="Chinese")
    }
    logic_module.save_manifest(video, manifest)

    result = single_clone.translate_and_synthesize(
        [str(video)], target_language="Chinese", models_root=str(tmp_path),
        skip_existing=True, log=lambda _m: None,
    )

    assert result["skipped"] and not result["written"]
    assert loads == []


# --- the floor is measured on silence, not on quiet speech ------------------

def _speech_like(seconds: float, pause: float = 0.0, floor_amp: float = 3e-4,
                 sr: int = backend.SAMPLE_RATE) -> np.ndarray:
    """A tone at speech level, optionally with a silent gap carrying only hiss."""
    rng = np.random.default_rng(0)
    total = int(seconds * sr)
    t = np.arange(total) / sr
    clip = (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    clip += (rng.standard_normal(total) * floor_amp).astype(np.float32)
    if pause > 0:
        lo = total // 2
        hi = min(total, lo + int(pause * sr))
        clip[lo:hi] = (rng.standard_normal(hi - lo) * floor_amp).astype(np.float32)
    return clip


def test_the_floor_reads_the_silence_not_the_speech():
    quiet = float(20 * np.log10(3e-4))
    measured = backend.clip_noise_floor_dbfs(_speech_like(2.0, pause=0.5))
    assert measured is not None
    assert abs(measured - quiet) < 6.0, f"read {measured:.1f} dBFS, hiss is {quiet:.1f}"


def test_a_line_that_never_pauses_gets_no_floor_reading():
    """Rather than mistaking its quietest syllable for hiss. These are the
    short lines that were hardest to make out in the first place."""
    assert backend.clip_noise_floor_dbfs(_speech_like(0.4, pause=0.0)) is None


def test_the_floor_reading_no_longer_tracks_clip_length():
    """The old blanket 5th percentile fell 11 dB from sub-0.5s lines to 3s+
    ones (r=-0.35 against duration) while their speech level barely moved."""
    readings = [
        backend.clip_noise_floor_dbfs(_speech_like(seconds, pause=0.4 * seconds))
        for seconds in (1.0, 2.0, 4.0)
    ]
    assert all(value is not None for value in readings)
    assert max(readings) - min(readings) < 4.0


def test_a_genuinely_hissy_clip_still_reads_high():
    """Both hiss levels sit under the activity threshold, so both are silence;
    the noisier one must read as such rather than being averaged away."""
    loud_hiss = backend.clip_noise_floor_dbfs(
        _speech_like(2.0, pause=0.5, floor_amp=6e-4)
    )
    clean = backend.clip_noise_floor_dbfs(_speech_like(2.0, pause=0.5, floor_amp=1e-4))
    assert loud_hiss is not None and clean is not None
    assert loud_hiss > clean + 12.0


def test_the_reading_is_the_actual_hiss_level():
    for amplitude in (1e-4, 3e-4, 6e-4):
        measured = backend.clip_noise_floor_dbfs(
            _speech_like(2.0, pause=0.5, floor_amp=amplitude)
        )
        assert measured is not None
        assert abs(measured - 20 * np.log10(amplitude)) < 1.0


def test_hiss_loud_enough_to_count_as_speech_yields_no_reading():
    """Above the activity threshold there is no silence left to measure, and
    inventing a floor from speech is the mistake this rewrite removes."""
    assert backend.clip_noise_floor_dbfs(
        _speech_like(2.0, pause=0.5, floor_amp=6e-3)
    ) is None


# --- time fitting must not crush what the model produced --------------------

class TalkativeModel:
    """Renders every line as a fixed, deliberately long delivery."""

    def __init__(self, seconds: float = 2.0):
        self.seconds = seconds

    def infer(self, **kwargs):
        n = int(self.seconds * backend.SAMPLE_RATE)
        t = np.arange(n) / backend.SAMPLE_RATE
        wav = (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        sf.write(kwargs["output_path"], wav, backend.SAMPLE_RATE)
        return kwargs["output_path"]


def _fitted_seconds(tmp_path, text, start, end, next_start=None):
    source, output = tmp_path / "source.wav", tmp_path / "out.wav"
    sf.write(source, np.full(int(backend.SAMPLE_RATE * 200.0), 0.03, dtype=np.float32),
             backend.SAMPLE_RATE)
    segments = [{"id": 1, "start": start, "end": end, "text": text}]
    if next_start is not None:
        segments.append({"id": 2, "start": next_start, "end": next_start + 1.0, "text": "x"})
    backend.synthesize_manifest(
        TalkativeModel(2.0), source, segments, output,
        language="zh", fit_duration=True, level_match=False, timbre_anchor=False,
        log=lambda _m: None,
    )
    wav = sf.read(output, dtype="float32")[0]
    return wav, backend.SAMPLE_RATE


def test_a_one_character_line_is_not_crushed_to_its_estimated_reading_time(tmp_path):
    """ipvr-385: natural_reading_seconds("服") is 0.30s, the model rendered it
    as a 2s delivery, and the fit squeezed it 9.6x into a slot with 15s of
    silence after it. 97% of that title ran over 1.15x."""
    log_lines: list[str] = []
    source, output = tmp_path / "source.wav", tmp_path / "out.wav"
    sf.write(source, np.full(int(backend.SAMPLE_RATE * 200.0), 0.03, dtype=np.float32),
             backend.SAMPLE_RATE)
    backend.synthesize_manifest(
        TalkativeModel(2.0), source,
        [{"id": 1, "start": 10.0, "end": 11.0, "text": "服"}],
        output, language="zh", fit_duration=True, level_match=False,
        timbre_anchor=False, log=log_lines.append,
    )
    line = next(l for l in log_lines if l.startswith("[indextts-v2] 1/1"))
    model_s = float(line.split("model=")[1].split("s")[0])
    final_s = float(line.split("-> ")[1].split("s")[0])
    assert model_s / final_s <= backend.MAX_HARD_COMPRESSION + 0.01, line


def test_the_bound_yields_when_the_next_line_leaves_no_room(tmp_path):
    """The cap is on gratuitous crushing, not on physics: with the next line
    right behind it the clip still has to fit."""
    log_lines: list[str] = []
    source, output = tmp_path / "source.wav", tmp_path / "out.wav"
    sf.write(source, np.full(int(backend.SAMPLE_RATE * 200.0), 0.03, dtype=np.float32),
             backend.SAMPLE_RATE)
    backend.synthesize_manifest(
        TalkativeModel(2.0), source,
        [{"id": 1, "start": 10.0, "end": 10.5, "text": "服"},
         {"id": 2, "start": 10.6, "end": 11.6, "text": "好"}],
        output, language="zh", fit_duration=True, level_match=False,
        timbre_anchor=False, log=log_lines.append,
    )
    line = next(l for l in log_lines if l.startswith("[indextts-v2] 1/2"))
    final_s = float(line.split("-> ")[1].split("s")[0])
    assert final_s <= 0.5, line


def test_a_line_the_model_reads_briskly_is_still_not_stretched(tmp_path):
    """The anti-drag rule survives: a short rendition is left short rather than
    padded out to fill a long slot."""
    log_lines: list[str] = []
    source, output = tmp_path / "source.wav", tmp_path / "out.wav"
    sf.write(source, np.full(int(backend.SAMPLE_RATE * 200.0), 0.03, dtype=np.float32),
             backend.SAMPLE_RATE)
    backend.synthesize_manifest(
        TalkativeModel(0.4), source,
        [{"id": 1, "start": 10.0, "end": 20.0, "text": "服"}],
        output, language="zh", fit_duration=True, level_match=False,
        timbre_anchor=False, log=log_lines.append,
    )
    line = next(l for l in log_lines if l.startswith("[indextts-v2] 1/1"))
    final_s = float(line.split("-> ")[1].split("s")[0])
    assert final_s < 1.0, line


def test_the_chosen_anchor_is_written_where_a_person_can_hear_it(tmp_path):
    """Ticking "lock timbre" used to leave the anchor inside
    indextts_v2_manifest/ under a sentence_ref_NNNNN.wav name, one of a hundred
    identical-looking files -- nothing to listen to and nothing to argue with.
    It now lands beside the manifest under the same <speaker>.basis.wav name
    the refined-clone tab writes a hand-picked basis to."""
    clone = tmp_path / "video.clone"
    clone.mkdir()
    source, output = clone / "audio16k.wav", tmp_path / "out.wav"
    rng = np.random.default_rng(3)
    sr = backend.SAMPLE_RATE
    t = np.arange(int(40.0 * sr)) / sr
    audio = (0.08 * np.sin(2 * np.pi * 190 * t)).astype(np.float32)
    audio += (rng.standard_normal(audio.size) * 1e-4).astype(np.float32)
    sf.write(source, audio, sr)

    segments = [
        {"id": i, "start": 2.0 + i * 7.0, "end": 2.0 + i * 7.0 + 5.0,
         "text": "这是一句足够长的台词", "src_text": "これは十分に長い台詞です",
         "speaker": "SPEAKER_00"}
        for i in range(4)
    ]
    backend.synthesize_manifest(
        ConstantModel(0.05), source, segments, output,
        language="zh", fit_duration=False, level_match=False,
        timbre_anchor=True, log=lambda _m: None,
    )

    wav = clone / "SPEAKER_00.basis.wav"
    assert wav.is_file(), sorted(p.name for p in clone.iterdir())
    assert (clone / "SPEAKER_00.basis.txt").is_file()
    meta = json.loads((clone / "SPEAKER_00.basis.meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "auto-timbre-anchor"
    assert meta["basis_wav"] == "SPEAKER_00.basis.wav"
    assert sf.info(wav).frames > 0
