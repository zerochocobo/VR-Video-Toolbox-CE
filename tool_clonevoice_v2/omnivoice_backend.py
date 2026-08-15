"""Compatibility guard for OmniVoice-only candidate features.

The migrated v2 pipeline deliberately has no OmniVoice dependency. A few
legacy GUI candidate/design actions still resolve these names; fail with an
actionable message instead of importing the old backend or crashing with a
missing-module error.
"""

from __future__ import annotations


def _unsupported(*_args, **_kwargs):
    raise RuntimeError(
        "IndexTTS-2.5 uses reference audio directly and does not support "
        "OmniVoice text-based voice design or two-pass candidate previews. "
        "Choose a longer reference sentence for the speaker instead."
    )


_generate_target_reference_takes_with_model = _unsupported
process_target_reference_batch = _unsupported
prepare_prompt_reference_audio = _unsupported
generate_voice_design_sample_with_model = _unsupported
_seed_generation = _unsupported
_stable_seed = _unsupported
_normalize_peak = _unsupported
_match_sentence_loudness = _unsupported
_read_wav_mono_f32 = _unsupported

