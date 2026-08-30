"""Ducking only the frequencies that carry the original voice.

Broadband ducking gives up a dB of background for every dB it takes off the
original dialogue, which is why the ducked mix sounds dead. Measured on
HNVR-174 over one 30s window, speech against the gaps between it:

    50-80 Hz    +0.5 dB voice, 12.8% of the background   -> keep
    80-300 Hz   +12..17 dB voice, ~10% of the background -> male fundamental
    300-400 Hz  +5.0 dB voice, 41.4% of the background   -> costliest to cut
    400-8000 Hz +7..19 dB voice, ~3% of the background

A listening test settled the 300-400 question: keeping it left a voice you
could still make out. So one span, 80-8000 Hz, is ducked and the rest is not.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tool_si import logic as sl


def _filter(**kwargs):
    return sl.build_si_mix_filter("both", 100, 50, 0.0, duck_original=True,
                                  duck_key_input=True, **kwargs)


def test_band_ducking_is_off_by_default_everywhere():
    """v1 and tool_subtitle mix through these too and must not change."""
    for func in (sl.build_si_mix_filter, sl.build_si_audio_mix_command,
                 sl.mix_si_audio_track, sl.batch_mix_si_audio_tracks):
        assert inspect.signature(func).parameters["duck_band"].default is False, func.__name__


def test_the_default_graph_is_the_old_one():
    graph = _filter()
    assert "asplit=3" not in graph
    assert "[orig_base][si_key]sidechaincompress=" in graph


def test_band_ducking_splits_ducks_the_middle_and_sums_back():
    graph = _filter(duck_band=True)
    assert "asplit=3" in graph
    # only the middle band reaches the compressor
    assert "[duck_mid_band][si_key]sidechaincompress=" in graph
    assert "[orig_base][si_key]sidechaincompress=" not in graph
    assert "amix=inputs=3" in graph
    assert "normalize=0" in graph, "summing must not rescale the bands"


def test_the_kept_bands_are_the_ones_without_voice_in_them():
    graph = _filter(duck_band=True)
    assert f"lowpass=f={sl.SI_DUCK_BAND_LOW_HZ:.0f}" in graph
    assert f"highpass=f={sl.SI_DUCK_BAND_HIGH_HZ:.0f}" in graph
    assert sl.SI_DUCK_BAND_LOW_HZ == 80.0, "below this is room tone, not voice"
    assert sl.SI_DUCK_BAND_HIGH_HZ == 8000.0


def test_the_low_crossover_is_steep_enough_to_catch_a_male_fundamental():
    """A single 12 dB/oct stage at 80 Hz is only 12 dB down at 160 Hz, so the
    un-ducked low branch passes the male fundamental straight through: measured
    through ffmpeg that capped suppression at -9.6 dB where the offline
    reference reached -23.9. Cascading gets it back to -15.2."""
    graph = _filter(duck_band=True)
    assert graph.count(f"lowpass=f={sl.SI_DUCK_BAND_LOW_HZ:.0f}:poles=2") >= 3
    assert graph.count(f"highpass=f={sl.SI_DUCK_BAND_LOW_HZ:.0f}:poles=2") >= 3


@pytest.mark.parametrize("channel", ["both", "left", "right"])
@pytest.mark.parametrize("key", [True, False])
def test_every_ducking_route_supports_the_band_split(channel, key):
    graph = sl.build_si_mix_filter(channel, 100, 50, 0.0, duck_original=True,
                                   duck_key_input=key, duck_band=True)
    assert "asplit=3" in graph
    assert "[orig_base][si_key]sidechaincompress=" not in graph


def test_band_ducking_does_nothing_when_ducking_is_off():
    graph = sl.build_si_mix_filter("both", 100, 50, 0.0, duck_original=False, duck_band=True)
    assert "asplit=3" not in graph
    assert "sidechaincompress" not in graph


def test_the_command_passes_the_flag_down():
    cmd = sl.build_si_audio_mix_command(
        "in.mp4", "in.si.wav", "out.mp4", "both", 100, 50,
        duck_original=True, duck_key_path="in.si.duck.wav", duck_band=True)
    assert any("asplit=3" in str(part) for part in cmd)


def test_the_v2_mix_page_ducks_voice_bands_by_default():
    """Chosen by ear on HNVR-174. It is only worth it at 'strongest': the
    three-way split costs ~1.8 dB of background whatever the preset, so at
    'light' broadband is the better trade. So the page pins the strength and
    drops the selector rather than leaving a setting that quietly makes the
    default worse."""
    source = Path("tool_clonevoice_v2/gui.py").read_text(encoding="utf-8")
    assert "self.single_mix_duck_voice_var = tk.BooleanVar(value=True)" in source
    assert 'self.single_mix_duck_preset_var = tk.StringVar(value=si("opt_duck_preset_strongest"))' in source
    assert "single_mix_duck_preset_combo" not in source, "the strength selector must be gone"


def test_the_two_halves_are_one_switch():
    """The key track decides when the original is pushed down, the band split
    decides which frequencies. Running one without the other is never wanted."""
    source = Path("tool_clonevoice_v2/gui.py").read_text(encoding="utf-8")
    assert "use_duck_key = duck_band = self.single_mix_duck_voice_var.get()" in source
    assert "single_mix_duck_key_var" not in source
    assert "single_mix_duck_band_var" not in source


def test_the_library_default_stays_off_for_v1_and_the_subtitle_tool():
    """Only the v2 page opts in; nothing else changes behaviour."""
    assert inspect.signature(sl.mix_si_audio_track).parameters["duck_band"].default is False
    v1 = Path("tool_clonevoice/gui.py").read_text(encoding="utf-8")
    subtitle = Path("tool_subtitle/gui.py").read_text(encoding="utf-8")
    assert "duck_band" not in v1
    assert "duck_band" not in subtitle
