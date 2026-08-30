"""Reassigning a line's speaker during proofreading.

Diarization mixes up who is talking, most visibly between a man and a woman in
the same exchange, and a wrong label sends the line to the wrong cloned voice.
The proofread pass is where a person can see the mistake, so it has to be
fixable there -- and the fix has to reach the manifest, which is what the
synthesizer reads.
"""
from __future__ import annotations

import json

import pytest

from tool_clonevoice_v2 import logic, proofread


def _manifest(tmp_path, speakers):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    cdir = logic.clone_dir(video)
    cdir.mkdir(parents=True, exist_ok=True)
    segments = [
        {"id": i + 1, "start": float(i), "end": float(i) + 0.9,
         "src_text": f"src{i}", "tgt_text": f"tgt{i}", "speaker": spk}
        for i, spk in enumerate(speakers)
    ]
    logic.save_manifest(video, {
        "video": str(video), "language": "ja", "target_language": "Chinese",
        "segments": segments, "speakers": {},
    })
    return video


def test_a_reassigned_speaker_reaches_the_manifest(tmp_path):
    video = _manifest(tmp_path, ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"])
    rows = proofread.load_rows(video)["rows"]
    seg_rows = [r for r in rows if r.get("kind") == "seg"]
    assert [r["speaker"] for r in seg_rows] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]

    seg_rows[1]["speaker"] = "SPEAKER_01"
    proofread.save_rows(video, rows)

    saved = logic.load_manifest(video)["segments"]
    assert [s["speaker"] for s in saved] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]


def test_the_other_edits_still_go_through_alongside_it(tmp_path):
    video = _manifest(tmp_path, ["SPEAKER_00", "SPEAKER_01"])
    rows = proofread.load_rows(video)["rows"]
    seg_rows = [r for r in rows if r.get("kind") == "seg"]
    seg_rows[0]["speaker"] = "SPEAKER_01"
    seg_rows[0]["tgt_text"] = "改过的译文"
    proofread.save_rows(video, rows)

    saved = logic.load_manifest(video)["segments"]
    assert saved[0]["speaker"] == "SPEAKER_01"
    assert saved[0]["tgt_text"] == "改过的译文"


def test_a_blank_speaker_never_wipes_the_existing_label(tmp_path):
    """Rows that carry no speaker (a reference-only row, or a video that was
    never diarized) must not blank out what the manifest already has."""
    video = _manifest(tmp_path, ["SPEAKER_00"])
    rows = proofread.load_rows(video)["rows"]
    for row in rows:
        if row.get("kind") == "seg":
            row["speaker"] = "   "
    proofread.save_rows(video, rows)
    assert logic.load_manifest(video)["segments"][0]["speaker"] == "SPEAKER_00"


# --- the control itself ---

@pytest.fixture
def dialog(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no Tk display: {exc}")
    root.geometry("1200x700")
    from tool_clonevoice_v2 import gui_proofread

    video = _manifest(tmp_path, ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"])
    dlg = gui_proofread.ProofreadDialog(root, str(video), show_speaker=True)
    root.update_idletasks()
    root.update()
    yield dlg
    dlg.destroy()
    root.destroy()


def test_the_speaker_control_is_visible_and_lists_this_video_speakers(dialog):
    assert dialog.speaker_combo is not None
    assert dialog.speaker_combo.winfo_ismapped()
    assert list(dialog.speaker_combo["values"]) == ["SPEAKER_00", "SPEAKER_01"]


def test_applying_to_a_multi_row_selection_changes_every_one(dialog):
    """Diarization flips a run of lines in one exchange, so a single-row tool
    would be the wrong unit of work."""
    seg_iids = [
        iid for iid in dialog.tree.get_children()
        if dialog._row(iid) is not None and dialog._row(iid).get("kind") == "seg"
    ]
    dialog.tree.selection_set(seg_iids[0], seg_iids[1])
    dialog.speaker_var.set("SPEAKER_01")
    dialog.apply_speaker_to_selection()

    assert [dialog._row(i)["speaker"] for i in seg_iids[:2]] == ["SPEAKER_01"] * 2
    assert dialog.tree.set(seg_iids[0], "speaker") == "SPEAKER_01"


def test_the_control_follows_the_highlighted_row(dialog):
    seg_iids = [
        iid for iid in dialog.tree.get_children()
        if dialog._row(iid) is not None and dialog._row(iid).get("kind") == "seg"
    ]
    dialog._load_row(seg_iids[2])
    assert dialog.speaker_var.get() == "SPEAKER_01"
    dialog._load_row(seg_iids[0])
    assert dialog.speaker_var.get() == "SPEAKER_00"


# --- auditioning the clone against the original -----------------------------

def _with_cloned_track(tmp_path, seconds=12.0):
    """A manifest plus an exported .si.wav, as a proofread-after-export state."""
    import numpy as np
    import soundfile as sf
    from tool_si import logic as si

    video = _manifest(tmp_path, ["SPEAKER_00", "SPEAKER_01"])
    cdir = logic.clone_dir(video)
    sr = 22050
    t = np.arange(int(seconds * sr)) / sr
    sf.write(cdir / logic.AUDIO16K_NAME,
             (0.05 * np.sin(2 * np.pi * 180 * t)).astype("float32"), sr)
    sf.write(si.default_si_audio_path(str(video)),
             (0.05 * np.sin(2 * np.pi * 320 * t)).astype("float32"), sr)
    return video


def test_the_clone_is_auditioned_from_the_exported_track_not_regenerated(tmp_path):
    """No synthesis: export already wrote the whole dub, so the row's slice of
    it is that row's cloned reading."""
    import soundfile as sf

    video = _with_cloned_track(tmp_path)
    track = proofread.cloned_track_path(video)
    assert track is not None and track.is_file()

    clip = proofread.cut_segment_preview(
        video, 1.0, 2.0, tail_pad=1.5, source=track, out_name="pf_preview_cloned.wav"
    )
    assert clip.name == "pf_preview_cloned.wav"
    assert sf.info(clip).frames > 0
    # ...and it did not overwrite the original-audio preview
    assert clip != logic.clone_dir(video) / "pf_preview.wav"


def test_the_cloned_preview_keeps_the_overflow_tail(tmp_path):
    """The dub may run past its slot into the silence behind it; cutting at
    `end` would clip a correct rendition short."""
    import soundfile as sf

    video = _with_cloned_track(tmp_path)
    track = proofread.cloned_track_path(video)
    short = proofread.cut_segment_preview(
        video, 1.0, 2.0, source=track, out_name="a.wav")
    long = proofread.cut_segment_preview(
        video, 1.0, 2.0, tail_pad=1.5, source=track, out_name="b.wav")
    assert sf.info(long).frames > sf.info(short).frames


def test_no_exported_track_means_no_offer_to_make_one(tmp_path):
    video = _manifest(tmp_path, ["SPEAKER_00"])
    assert proofread.cloned_track_path(video) is None


def _dialog_for(tmp_path, video):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no Tk display: {exc}")
    from tool_clonevoice_v2 import gui_proofread

    dlg = gui_proofread.ProofreadDialog(root, str(video), show_speaker=True)
    root.update()
    return root, dlg


def test_the_button_stays_live_before_any_export(tmp_path):
    """A greyed-out button explains nothing; this one says what it is for."""
    video = _manifest(tmp_path, ["SPEAKER_00", "SPEAKER_01"])
    root, dlg = _dialog_for(tmp_path, video)
    try:
        assert dlg.btn_play_cloned.winfo_ismapped()
        assert str(dlg.btn_play_cloned.cget("state")) == "normal"
    finally:
        dlg.destroy()
        root.destroy()


def test_clicking_it_without_an_export_explains_rather_than_plays(tmp_path, monkeypatch):
    video = _manifest(tmp_path, ["SPEAKER_00", "SPEAKER_01"])
    root, dlg = _dialog_for(tmp_path, video)
    try:
        from tool_clonevoice_v2 import gui_proofread

        shown: list[str] = []
        monkeypatch.setattr(
            gui_proofread.messagebox, "showinfo",
            lambda _title, message, **kw: shown.append(message),
        )
        played: list[str] = []
        monkeypatch.setattr(
            gui_proofread.proofread, "cut_segment_preview",
            lambda *a, **k: played.append(a) or tmp_path / "x.wav",
        )
        dlg.play_row_cloned()
        assert shown and "" != shown[0]
        assert played == [], "must not try to cut a track that does not exist"
    finally:
        dlg.destroy()
        root.destroy()


def test_it_picks_up_an_export_that_happened_since_the_dialog_opened(tmp_path):
    """The path is resolved per click, so exporting while proofreading is open
    does not require reopening the dialog."""
    import numpy as np
    import soundfile as sf
    from tool_si import logic as si

    video = _manifest(tmp_path, ["SPEAKER_00", "SPEAKER_01"])
    cdir = logic.clone_dir(video)
    sr = 22050
    t = np.arange(int(12.0 * sr)) / sr
    sf.write(cdir / logic.AUDIO16K_NAME,
             (0.05 * np.sin(2 * np.pi * 180 * t)).astype("float32"), sr)

    root, dlg = _dialog_for(tmp_path, video)
    try:
        assert proofread.cloned_track_path(video) is None
        sf.write(si.default_si_audio_path(str(video)),
                 (0.05 * np.sin(2 * np.pi * 320 * t)).astype("float32"), sr)
        assert proofread.cloned_track_path(video) is not None
    finally:
        dlg.destroy()
        root.destroy()


# --- a saved proofread must not invite a re-translation ---------------------

def _translated(tmp_path, rows):
    """A manifest as it looks after transcription + translation."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")
    cdir = logic.clone_dir(video)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "translated.srt").write_text("1\n", encoding="utf-8")
    segments = [
        {"id": i + 1, "start": float(i), "end": float(i) + 0.9,
         "src_text": src, "tgt_text": tgt, "speaker": "SPEAKER_00"}
        for i, (src, tgt) in enumerate(rows)
    ]
    logic.save_manifest(video, {
        "video": str(video), "language": "ja", "target_language": "Chinese",
        "segments": segments,
    })
    return video


def test_a_line_the_translator_gave_up_on_does_not_force_a_retranslation(tmp_path):
    """ipvr-385 part1 had five: ASR fragments the model could make nothing of.
    Saving a proofread removed the legacy escape hatch, so from then on those
    five declared the whole video untranslated -- and run_translate blanks
    every tgt_text before it starts, so the just-saved edits went with it."""
    from tool_clonevoice_v2 import single_clone

    video = _translated(tmp_path, [("a", "甲"), ("b", ""), ("c", "丙")])
    assert single_clone.manifest_has_target_translation(video, "Chinese"), (
        "precondition: before any proofread the old translated.srt counts"
    )

    rows = proofread.load_rows(video)["rows"]
    proofread.save_rows(video, rows)

    assert single_clone.manifest_has_target_translation(video, "Chinese")


def test_the_edits_survive_the_round_trip(tmp_path):
    from tool_clonevoice_v2 import single_clone

    video = _translated(tmp_path, [("a", "甲"), ("b", ""), ("c", "丙")])
    rows = proofread.load_rows(video)["rows"]
    seg_rows = [r for r in rows if r.get("kind") == "seg"]
    seg_rows[0]["tgt_text"] = "手工改过"
    proofread.save_rows(video, rows)

    assert single_clone.manifest_has_target_translation(video, "Chinese")
    assert logic.load_manifest(video)["segments"][0]["tgt_text"] == "手工改过"


def test_a_proofread_saved_before_translating_still_asks_for_one(tmp_path):
    """The guard is "the translator has already run", not "the box is empty":
    opening proofreading on a freshly transcribed video and saving must not
    mark every line as reviewed and skip translation forever."""
    from tool_clonevoice_v2 import single_clone

    video = _translated(tmp_path, [("a", ""), ("b", ""), ("c", "")])
    rows = proofread.load_rows(video)["rows"]
    proofread.save_rows(video, rows)

    assert not single_clone.manifest_has_target_translation(video, "Chinese")


def test_a_line_emptied_on_purpose_stays_cleared(tmp_path):
    from tool_clonevoice_v2 import single_clone

    video = _translated(tmp_path, [("a", "甲"), ("b", "乙")])
    rows = proofread.load_rows(video)["rows"]
    [r for r in rows if r.get("kind") == "seg"][1]["tgt_text"] = ""
    proofread.save_rows(video, rows)

    assert single_clone.manifest_has_target_translation(video, "Chinese")
    assert 2 in proofread.effective_cleared_segment_ids(logic.load_manifest(video))
