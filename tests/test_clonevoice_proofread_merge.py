import unittest

from tool_clonevoice.proofread import merge_adjacent_rows


class ProofreadMergeTests(unittest.TestCase):
    def _rows(self):
        return [
            {"kind": "seg", "seg_id": 1, "start": 0.0, "end": 1.0, "speaker": "A", "src_text": "a", "tgt_text": "甲", "original_tgt_text": "甲", "ref_text": ""},
            {"kind": "seg", "seg_id": 2, "start": 1.2, "end": 2.0, "speaker": "A", "src_text": "b", "tgt_text": "乙", "original_tgt_text": "乙", "ref_text": ""},
            {"kind": "seg", "seg_id": 3, "start": 2.1, "end": 6.0, "speaker": "A", "src_text": "c", "tgt_text": "丙", "original_tgt_text": "丙", "ref_text": ""},
        ]

    def test_merges_with_gap_and_duration_limits(self):
        rows = self._rows()
        result = merge_adjacent_rows(rows, max_duration=2.5, max_gap_ms=250)
        self.assertEqual(result["merged_count"], 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["tgt_text"], "甲乙")
        self.assertEqual(rows[0]["_merged_ids"], [2])

    def test_does_not_merge_different_speakers_or_reference_rows(self):
        rows = self._rows()
        rows[1]["speaker"] = "B"
        result = merge_adjacent_rows(rows, max_duration=10, max_gap_ms=500)
        self.assertEqual(result["merged_count"], 0)
        rows.insert(1, {"kind": "ref_only", "start": 1.1, "end": 1.2, "text": "ref"})
        result = merge_adjacent_rows(rows, max_duration=10, max_gap_ms=500, same_speaker=False)
        self.assertEqual(result["merged_count"], 1)
        self.assertEqual(rows[0]["seg_id"], 1)
        self.assertEqual(rows[1]["kind"], "ref_only")

    def test_selected_scope_only_merges_selected_rows_and_neighbors(self):
        rows = self._rows()
        result = merge_adjacent_rows(
            rows,
            max_duration=10,
            max_gap_ms=500,
            selected_seg_ids={"2"},
        )
        self.assertEqual(result["merged_pairs"], [(1, 2)])
        self.assertEqual([row["seg_id"] for row in rows], [1, 3])

    def test_empty_selected_scope_does_not_fall_back_to_all(self):
        rows = self._rows()
        result = merge_adjacent_rows(rows, max_duration=10, max_gap_ms=500, selected_seg_ids=set())
        self.assertEqual(result["merged_count"], 0)
        self.assertEqual(len(rows), 3)

    def test_empty_subtitle_is_not_merged(self):
        rows = self._rows()
        rows[1]["tgt_text"] = ""
        result = merge_adjacent_rows(rows, max_duration=10, max_gap_ms=500)
        self.assertEqual(result["merged_count"], 0)
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
