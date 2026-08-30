"""The speaker count means "voices to keep together", not "clusters to force".

Asked for an exact cluster count of 1, the diarizer put a whole title in one
bucket: the man's lines were cloned from the woman's anchor, and her bucket,
polluted with his voice, no longer matched her own other parts. These lock the
replacement -- over-split, then merge the extras back by voice similarity.

The numbers in the thresholds come from 3dsvr-1911; see diarize.py.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool_clonevoice_v2 import diarize as diar


def unit(*values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / float(np.linalg.norm(vector))


# Voices on their own axes so they stay independent of each other -- in two
# dimensions everything sits on one circle, and a "man" placed away from one
# woman lands right next to the other.
WOMAN = unit(1.0, 0.0, 0.0, 0.0)
WOMAN_AGAIN = unit(0.925, 0.380, 0.0, 0.0)  # cos 0.925: her, over-split
SIMILAR = unit(0.55, 0.835, 0.0, 0.0)  # cos 0.55: the ambiguous middle band
MAN = unit(0.0, 0.0, 1.0, 0.0)  # cos 0.0 against every voice above
OTHER = unit(0.0, 0.0, 0.0, 1.0)


@pytest.fixture
def stub(monkeypatch):
    """Drive diarize_primary_speakers off canned turns and centroids."""

    state = types.SimpleNamespace(asked=None, turns=[], centroids={})

    def fake_diarize(_path, *, backend, num_speakers, models_root, device, log):
        state.asked = num_speakers
        return state.turns

    def fake_centroids(_path, turns, *, models_root, device, log, **kwargs):
        return {name: vector for name, vector in state.centroids.items()
                if any(str(t[2]) == name for t in turns)}

    monkeypatch.setattr(diar, "diarize", fake_diarize)
    monkeypatch.setattr(diar, "speaker_centroids", fake_centroids)
    monkeypatch.setattr(diar, "resolve_backend", lambda *_a, **_k: "pyannote")
    monkeypatch.setattr(diar, "ecapa_available", lambda *_a, **_k: True)
    return state


def run(requested, **kwargs):
    return diar.diarize_primary_speakers(
        "audio.wav", backend="pyannote", num_speakers=requested,
        models_root="models", device="cpu", log=lambda _m: None, **kwargs,
    )


def seconds_by_speaker(turns):
    out = {}
    for start, end, speaker in turns:
        out[speaker] = out.get(speaker, 0.0) + (end - start)
    return out


def test_one_voice_still_separates_a_clearly_different_one(stub):
    """The complaint this exists for: one female voice, one short male line."""
    stub.turns = [
        (0.0, 200.0, "A"),  # her
        (200.0, 260.0, "B"),  # her again, over-split
        (260.0, 280.0, "C"),  # him
    ]
    stub.centroids = {"A": WOMAN, "B": WOMAN_AGAIN, "C": MAN}

    turns = run(1)

    seconds = seconds_by_speaker(turns)
    assert len(seconds) == 2, "the man must not be folded into her voice"
    # Numbered by speaking time, so the voice the title is made of is 00.
    assert seconds["SPEAKER_00"] == pytest.approx(260.0)
    assert seconds["SPEAKER_01"] == pytest.approx(20.0)
    assert [t[2] for t in turns] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]


def test_the_count_asks_the_diarizer_for_headroom(stub):
    stub.turns = [(0.0, 10.0, "A")]
    stub.centroids = {"A": WOMAN}

    run(1, headroom=2)
    assert stub.asked == 3, "1 voice is looked for in 3 clusters"

    run(2, headroom=2)
    assert stub.asked == 4


def test_headroom_is_capped_so_a_high_count_stays_sane(stub):
    stub.turns = [(0.0, 10.0, "A")]
    stub.centroids = {"A": WOMAN}
    run(7, headroom=5)
    assert stub.asked == diar.MAX_SPLIT_CLUSTERS


def test_one_voice_over_split_comes_back_as_one(stub):
    """The headroom must cost nothing when the extra voices are not there."""
    stub.turns = [(0.0, 100.0, "A"), (100.0, 180.0, "B"), (180.0, 240.0, "C")]
    stub.centroids = {"A": WOMAN, "B": WOMAN_AGAIN, "C": unit(0.9, 0.2, 0.1, 0.0)}

    turns = run(1)

    assert {t[2] for t in turns} == {"SPEAKER_00"}
    assert seconds_by_speaker(turns)["SPEAKER_00"] == pytest.approx(240.0)


def test_a_kept_extra_does_not_eat_into_the_requested_count(stub):
    """Two requested voices stay two; the man is a third on top of them."""
    stub.turns = [
        (0.0, 300.0, "A"),
        (300.0, 380.0, "B"),  # A over-split
        (380.0, 560.0, "C"),  # the second voice asked for
        (560.0, 580.0, "D"),  # the man
    ]
    stub.centroids = {"A": WOMAN, "B": WOMAN_AGAIN, "C": OTHER, "D": MAN}

    seconds = seconds_by_speaker(run(2))

    assert len(seconds) == 3
    assert seconds["SPEAKER_00"] == pytest.approx(380.0)
    assert seconds["SPEAKER_01"] == pytest.approx(180.0)
    assert seconds["SPEAKER_02"] == pytest.approx(20.0)


def test_asking_for_two_does_not_clone_one_person_twice(stub):
    """Measured on 3dsvr-1911 part 1: asked for 2, the two longest clusters were
    both the woman at cos 0.794, and taking them as two people cloned her in two
    different voices. The count is a bound, not a number to hit."""
    stub.turns = [
        (0.0, 425.0, "A"),
        (425.0, 556.0, "B"),  # her again -- cos 0.925 with A
        (556.0, 584.0, "C"),  # the man
    ]
    stub.centroids = {"A": WOMAN, "B": WOMAN_AGAIN, "C": MAN}

    seconds = seconds_by_speaker(run(2))

    assert len(seconds) == 2, "her two clusters are one voice, the man is the other"
    assert seconds["SPEAKER_00"] == pytest.approx(556.0)
    assert seconds["SPEAKER_01"] == pytest.approx(28.0)


def test_the_count_still_separates_two_voices_that_are_merely_similar(stub):
    """Collapsing two requested voices takes better evidence than absorbing a
    leftover: pairs in the ambiguous 0.46-0.56 band stay apart, because there
    the caller's own count is the better evidence."""
    stub.turns = [(0.0, 400.0, "A"), (400.0, 700.0, "B")]
    stub.centroids = {"A": WOMAN, "B": SIMILAR}

    assert len(seconds_by_speaker(run(2))) == 2, "asked for two, and 0.55 is not proof of one"
    assert len(seconds_by_speaker(run(1))) == 1, "asked for one, 0.55 is close enough to rejoin"


def test_a_voice_that_frees_a_slot_lets_a_later_one_take_it(stub):
    """When two of the leading clusters turn out to be one person, the count is
    not spent on them -- the next distinct voice still gets held."""
    stub.turns = [
        (0.0, 400.0, "A"),
        (400.0, 700.0, "B"),  # A again
        (700.0, 900.0, "C"),  # a second voice, merely similar
        (900.0, 920.0, "D"),  # the man
    ]
    stub.centroids = {
        "A": WOMAN, "B": WOMAN_AGAIN, "C": SIMILAR, "D": MAN,
    }

    seconds = seconds_by_speaker(run(2))

    assert seconds["SPEAKER_00"] == pytest.approx(700.0), "A and B are one voice"
    assert seconds["SPEAKER_01"] == pytest.approx(200.0), "C took the freed slot"
    assert seconds["SPEAKER_02"] == pytest.approx(20.0), "the man is still kept apart"


def test_primaries_are_the_longest_speaking_clusters(stub):
    """A brief voice must not take the one slot from the title's own voice."""
    stub.turns = [(0.0, 20.0, "A"), (20.0, 400.0, "B")]
    stub.centroids = {"A": MAN, "B": WOMAN}

    turns = run(1)

    kept = {t[2] for t in turns if t[1] - t[0] > 100}
    assert kept == {"SPEAKER_00"}
    assert seconds_by_speaker(turns)["SPEAKER_00"] == pytest.approx(380.0)


def test_a_borderline_cluster_merges_below_the_threshold_and_splits_above(stub):
    stub.turns = [(0.0, 200.0, "A"), (200.0, 240.0, "B")]
    stub.centroids = {"A": WOMAN, "B": unit(0.5, 0.866, 0.0, 0.0)}  # cos 0.5

    assert len({t[2] for t in run(1, merge_similarity=0.45)}) == 1
    assert len({t[2] for t in run(1, merge_similarity=0.55)}) == 2


def test_a_cluster_with_no_embedding_is_left_alone(stub):
    """Better an extra label than one person's lines in another's voice."""
    stub.turns = [(0.0, 200.0, "A"), (200.0, 260.0, "B")]
    stub.centroids = {"A": WOMAN}

    assert len({t[2] for t in run(1)}) == 2


def test_without_embeddings_it_asks_for_the_exact_count(monkeypatch, stub):
    """No centroids means nothing to merge on, so do not over-split."""
    monkeypatch.setattr(diar, "ecapa_available", lambda *_a, **_k: False)
    stub.turns = [(0.0, 10.0, "A")]

    turns = run(2)

    assert stub.asked == 2
    assert turns == stub.turns, "labels are left exactly as the diarizer gave them"


def test_no_count_is_left_to_the_diarizer(stub):
    stub.turns = [(0.0, 10.0, "A")]
    stub.centroids = {"A": WOMAN}

    run(None)

    assert stub.asked is None


def test_turn_boundaries_are_never_moved(stub):
    """Only labels change here; the timing belongs to the diarizer."""
    stub.turns = [(0.0, 200.0, "A"), (200.5, 260.25, "B"), (260.25, 280.0, "C")]
    stub.centroids = {"A": WOMAN, "B": WOMAN_AGAIN, "C": MAN}

    turns = run(1)

    assert [(t[0], t[1]) for t in turns] == [(0.0, 200.0), (200.5, 260.25), (260.25, 280.0)]
