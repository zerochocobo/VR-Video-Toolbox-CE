import unittest

from tool_clonevoice import logic as v1
from tool_clonevoice_v2 import logic as v2

MODULES = (("v1", v1), ("v2", v2))


def seg(index, start, end, text, speaker="A"):
    return {
        "id": index, "srt_index": index,
        "start": start, "end": end, "dur": round(end - start, 3),
        "src_text": text, "tgt_text": "", "emotion_ref": "",
        "words": [{"w": text, "start": start, "end": end}],
        "speaker": speaker,
    }


def lines(result):
    return [(round(s["start"], 2), round(s["end"], 2), s["src_text"]) for s in result]


class DubFragmentMergeTests(unittest.TestCase):
    """Both clonevoice packages carry the same dub-merge implementation."""

    def merge(self, module, segments):
        return module._merge_dub_fragments([dict(s) for s in segments], lambda message: None)

    def test_joins_near_contiguous_fragments(self):
        segments = [seg(1, 0.0, 1.5, "通常コースで"), seg(2, 1.6, 3.0, "よろしかったですか")]
        for name, module in MODULES:
            with self.subTest(name):
                self.assertEqual(
                    lines(self.merge(module, segments)),
                    [(0.0, 3.0, "通常コースでよろしかったですか")],
                )

    def test_keeps_real_pauses_and_speaker_changes(self):
        pause = [seg(1, 0.0, 1.5, "はいどうぞ"), seg(2, 2.4, 3.5, "ありがとう")]
        speakers = [seg(1, 0.0, 1.5, "はいどうぞ"), seg(2, 1.6, 3.0, "ありがとう", speaker="B")]
        for name, module in MODULES:
            with self.subTest(name):
                self.assertEqual(len(self.merge(module, pause)), 2)
                self.assertEqual(len(self.merge(module, speakers)), 2)

    def test_merged_line_stays_within_the_slot_budget(self):
        # Joining would span 9s, past DUB_MERGE_MAX_DURATION_SECONDS.
        long_pair = [seg(1, 0.0, 4.5, "ながいながいながい"), seg(2, 4.6, 9.0, "つづきのながいながい")]
        # Four 0.3s gaps would swallow 0.9s of silence, past the gap budget:
        # the tempo fit would pay for it by slowing the whole line down.
        chain = [seg(1, 0.0, 1.0, "いち"), seg(2, 1.3, 2.0, "にい"),
                 seg(3, 2.3, 3.0, "さん"), seg(4, 3.3, 4.0, "しい")]
        for name, module in MODULES:
            with self.subTest(name):
                self.assertEqual(len(self.merge(module, long_pair)), 2)
                merged = self.merge(module, chain)
                self.assertEqual(len(merged), 2)
                self.assertLessEqual(
                    max(s["end"] - s["start"] for s in merged),
                    module.DUB_MERGE_MAX_DURATION_SECONDS,
                )

    def test_hanging_fragment_joins_the_utterance_it_heads(self):
        # 本 is the head of 本日は…, stamped onto the tail of the previous burst.
        # Folding it backwards would split the word across two inferences.
        segments = [seg(1, 9.02, 9.96, "失礼いたします"),
                    seg(2, 9.96, 10.68, "本"),
                    seg(3, 11.64, 15.18, "日はご来店いただきありがとうございます")]
        for name, module in MODULES:
            with self.subTest(name):
                self.assertEqual(
                    lines(self.merge(module, segments)),
                    [(9.02, 9.96, "失礼いたします"),
                     # The fragment's own timestamp is the unreliable part, so the
                     # merged line keeps the slot of the utterance it joined.
                     (11.64, 15.18, "本日はご来店いただきありがとうございます")],
                )

    def test_hanging_fragment_folds_back_when_the_next_line_is_out_of_reach(self):
        segments = [seg(1, 0.0, 1.5, "そうです"), seg(2, 2.2, 2.4, "ね"), seg(3, 9.0, 10.0, "つぎ")]
        for name, module in MODULES:
            with self.subTest(name):
                self.assertEqual(
                    lines(self.merge(module, segments)),
                    [(0.0, 1.5, "そうですね"), (9.0, 10.0, "つぎ")],
                )

    def test_isolated_fragment_is_left_alone(self):
        segments = [seg(1, 0.0, 1.0, "こんにちは"), seg(2, 5.0, 5.2, "お"), seg(3, 9.0, 10.0, "客様です")]
        for name, module in MODULES:
            with self.subTest(name):
                self.assertEqual(len(self.merge(module, segments)), 3)

    def test_result_is_renumbered_and_carries_no_scratch_state(self):
        segments = [seg(1, 0.0, 1.5, "いち"), seg(2, 1.6, 3.0, "にい"), seg(3, 5.0, 6.0, "さん")]
        for name, module in MODULES:
            with self.subTest(name):
                merged = self.merge(module, segments)
                self.assertEqual([s["id"] for s in merged], [1, 2])
                self.assertEqual([s["srt_index"] for s in merged], [1, 2])
                self.assertEqual([s["dur"] for s in merged], [3.0, 1.0])
                self.assertEqual([len(s["words"]) for s in merged], [2, 1])
                for item in merged:
                    self.assertNotIn(module.GAP_SPENT_KEY, item)

    def test_short_input_passes_through(self):
        for name, module in MODULES:
            with self.subTest(name):
                self.assertEqual(self.merge(module, []), [])
                self.assertEqual(len(self.merge(module, [seg(1, 0.0, 0.2, "あ")])), 1)


if __name__ == "__main__":
    unittest.main()
