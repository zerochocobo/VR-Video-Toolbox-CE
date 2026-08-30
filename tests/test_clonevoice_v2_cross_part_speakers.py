"""Lining up speaker labels across the parts of one title.

The prescan used to concatenate every part and diarize the lot, so labels
matched across a title for free. On 3dsvr-1911 that clustered by *part*
instead of by person: two clusters at 1015.5s and 1007.4s, each part dominated
by a different one, and the man who introduces the new hire in part 1 folded
into the woman's cluster. Diarized alone, part 1 separates him correctly
(39.7s against 26.3s concatenated).
"""
from __future__ import annotations

import numpy as np
import pytest

from tool_clonevoice_v2 import diarize as diar


def _unit(*values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / float(np.linalg.norm(vector))


MAN = _unit(1.0, 0.0, 0.0)
WOMAN = _unit(0.0, 1.0, 0.0)
# The same woman in another scene: close, but not identical.
WOMAN_LATER = _unit(0.12, 1.0, 0.05)
THIRD = _unit(0.0, 0.0, 1.0)


def test_the_same_voice_gets_one_label_across_parts():
    part1 = {"SPEAKER_00": MAN, "SPEAKER_01": WOMAN}
    # The diarizer numbers each part independently, so she may come back second
    # in one part and first in the next.
    part2 = {"SPEAKER_00": WOMAN_LATER, "SPEAKER_01": MAN}
    first, second = diar.match_speakers_across_parts([part1, part2])
    assert first["SPEAKER_00"] != first["SPEAKER_01"]
    assert second["SPEAKER_00"] == first["SPEAKER_01"], "the woman must keep her label"
    assert second["SPEAKER_01"] == first["SPEAKER_00"], "and so must the man"


def test_a_voice_only_one_part_has_keeps_its_own_label():
    """Part 2 has a third person. Folding them onto someone else would send
    their lines out in that person's voice."""
    first, second = diar.match_speakers_across_parts([
        {"SPEAKER_00": MAN, "SPEAKER_01": WOMAN},
        {"SPEAKER_00": WOMAN_LATER, "SPEAKER_01": MAN, "SPEAKER_02": THIRD},
    ])
    assert len({*first.values(), *second.values()}) == 3
    assert second["SPEAKER_02"] not in first.values()


def test_two_people_in_one_part_never_collapse_onto_each_other():
    """Greedy matching must not hand the same global label to two speakers of
    the same part, however close they sound."""
    close_a = _unit(1.0, 0.02, 0.0)
    close_b = _unit(1.0, 0.03, 0.0)
    first, = diar.match_speakers_across_parts([{"A": close_a, "B": close_b}])
    assert first["A"] != first["B"]


def test_a_weak_match_is_not_merged():
    first, second = diar.match_speakers_across_parts(
        [{"SPEAKER_00": MAN}, {"SPEAKER_00": THIRD}], min_similarity=0.55
    )
    assert first["SPEAKER_00"] != second["SPEAKER_00"]


def test_a_part_without_embeddings_does_not_break_the_rest():
    first, second, third = diar.match_speakers_across_parts([
        {"SPEAKER_00": MAN}, {}, {"SPEAKER_00": MAN},
    ])
    assert second == {}
    assert third["SPEAKER_00"] == first["SPEAKER_00"]


def test_centroids_are_skipped_when_the_bundle_is_absent(tmp_path):
    """No ECAPA bundle means no matching; the caller falls back to per-part
    labels rather than guessing."""
    assert diar.speaker_centroids(
        "nonexistent.wav", [(0.0, 2.0, "SPEAKER_00")],
        models_root=str(tmp_path), device="cpu", log=lambda _m: None,
    ) == {}


def test_the_prescan_no_longer_concatenates():
    """Concatenation is what made the clustering split by part."""
    import inspect

    from tool_clonevoice_v2 import multi_clone

    source = inspect.getsource(multi_clone.prescan_global_diarize)
    # The calls, not the word: the docstring explains what was removed.
    assert "np.concatenate" not in source
    assert "global_diarize_concat" not in source
    assert "split_turns_to_video" not in source
    assert "speaker_centroids" in source
    assert "match_speakers_across_parts" in source
