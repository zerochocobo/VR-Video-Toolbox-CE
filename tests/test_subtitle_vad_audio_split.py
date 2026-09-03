"""Separate feeds for the speech detector and the decoder (2026-09-02).

WhisperJAV's "enhance for VAD only": a cleaned copy of the audio decides *where*
the speech is, while the decoder still hears the original. Denoising the ASR
feed lifts quiet speech over the gate but dulls the consonant detail the model
needs; splitting the feeds takes only the first half of that trade.

The contract worth pinning is which array each stage reads, so a later refactor
cannot quietly send the denoised audio to the decoder.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from tool_subtitle import logic as subl


class SplitFeedTests(unittest.TestCase):
    def setUp(self):
        rate = 16000
        # Two one-second bursts with a second of silence between them.
        self.decode_audio = np.zeros(3 * rate, dtype=np.float32)
        self.decode_audio[:rate] = 0.5
        self.decode_audio[2 * rate:] = 0.5
        # The "cleaned" copy is deliberately a different signal, so a stage
        # reading the wrong one is visible rather than merely equal.
        self.vad_audio = self.decode_audio * 0.25

        self.generator = subl.SubtitleGenerator.__new__(subl.SubtitleGenerator)
        self.generator.log_callback = lambda _message: None
        self.generator.set_vad_sensitivity("high")
        self.generator.last_speech_probs = None

        self.loaded = {}
        self.prob_input = []

    def _decode(self, path, sampling_rate=16000):
        return self.loaded[path]

    def _probs(self, audio, sampling_rate=16000):
        self.prob_input.append(np.asarray(audio).copy())
        # One frame per 20 ms, speech everywhere: the regions then span the
        # whole clip and the chunk arrays are easy to check.
        return np.ones(len(audio) // (sampling_rate // 50), dtype=np.float32)

    def run_split(self, vad_path=None):
        self.loaded = {"decode.wav": self.decode_audio, "vad.wav": self.vad_audio}
        with patch.object(subl, "_ensure_faster_whisper",
                          return_value=(None, self._decode, None)), \
             patch.object(subl.SubtitleGenerator, "whisperseg_speech_probs",
                          side_effect=self._probs, autospec=False):
            return self.generator.split_audio_whisperseg("decode.wav", 28.0, vad_path)

    def test_without_a_vad_copy_both_stages_read_the_same_audio(self):
        chunks = self.run_split()
        np.testing.assert_allclose(self.prob_input[0], self.decode_audio)
        self.assertTrue(chunks)
        self.assertAlmostEqual(float(np.max(chunks[0]["array"])), 0.5, places=5)

    def test_the_detector_reads_the_clean_copy_and_the_decoder_the_original(self):
        chunks = self.run_split("vad.wav")
        np.testing.assert_allclose(self.prob_input[0], self.vad_audio)
        self.assertTrue(chunks)
        # The chunk handed to the ASR must carry the original amplitude.
        self.assertAlmostEqual(float(np.max(chunks[0]["array"])), 0.5, places=5)

    def test_a_length_mismatch_is_trimmed_rather_than_indexed_past_the_end(self):
        # A filter pass can return a slightly different length; region times
        # must stay valid indices into the audio the chunks are cut from.
        self.vad_audio = np.concatenate([self.vad_audio, np.zeros(4000, np.float32)])
        chunks = self.run_split("vad.wav")
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(
                chunk["offset_sec"] + chunk["duration_sec"],
                len(self.decode_audio) / 16000.0 + 1e-6,
            )



class OverlongWindowTests(unittest.TestCase):
    """A single unbroken speech region must still be cut to the window size.

    The merge loop caps *combined* regions, but one continuous region can exceed
    the window on its own. Whisper pads or truncates its input to 30 seconds, so
    an over-long window silently loses everything past that: on the benchmark
    corpus one region ran 65 seconds and 35 of them never reached the model.
    """

    RATE = 16000

    def setUp(self):
        self.generator = subl.SubtitleGenerator.__new__(subl.SubtitleGenerator)
        self.generator.log_callback = lambda _message: None
        self.generator.set_vad_sensitivity("high")
        self.generator.last_speech_probs = None

    def split(self, audio, window_seconds):
        loaded = {"a.wav": audio}
        probs = np.ones(len(audio) // (self.RATE // 50), dtype=np.float32)
        with patch.object(subl, "_ensure_faster_whisper",
                          return_value=(None, lambda p, sampling_rate=16000: loaded[p], None)), \
             patch.object(subl.SubtitleGenerator, "whisperseg_speech_probs",
                          return_value=probs, autospec=False):
            return self.generator.split_audio_whisperseg("a.wav", window_seconds)

    def test_one_long_region_becomes_several_windows(self):
        audio = np.full(30 * self.RATE, 0.4, dtype=np.float32)
        chunks = self.split(audio, 3.0)
        self.assertGreaterEqual(len(chunks), 10)
        for chunk in chunks:
            self.assertLessEqual(chunk["duration_sec"], 3.0 + 1e-6)

    def test_the_pieces_tile_the_region_without_gaps(self):
        audio = np.full(20 * self.RATE, 0.4, dtype=np.float32)
        chunks = sorted(self.split(audio, 3.0), key=lambda c: c["offset_sec"])
        for before, after in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(
                before["offset_sec"] + before["duration_sec"],
                after["offset_sec"], places=3)
        self.assertAlmostEqual(chunks[-1]["offset_sec"] + chunks[-1]["duration_sec"],
                               20.0, places=2)

    def test_the_cut_lands_in_the_quiet_part(self):
        # Loud, a dip at 2.6s, loud again. The cut should take the dip rather
        # than the window limit at 3.0s.
        audio = np.full(6 * self.RATE, 0.4, dtype=np.float32)
        dip = int(2.6 * self.RATE)
        audio[dip:dip + int(0.1 * self.RATE)] = 0.0
        chunks = sorted(self.split(audio, 3.0), key=lambda c: c["offset_sec"])
        self.assertAlmostEqual(chunks[0]["duration_sec"], 2.6, delta=0.15)

    def test_a_region_inside_the_window_is_untouched(self):
        audio = np.full(2 * self.RATE, 0.4, dtype=np.float32)
        chunks = self.split(audio, 3.0)
        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
