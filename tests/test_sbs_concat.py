from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpu_engine.probe import ColorMetadata, VideoMetadata
from utils.keyframe_cutter import TimelineEntry
from utils import sbs_concat


def _meta(
    path: Path,
    *,
    width: int = 4096,
    height: int = 4096,
    codec: str = "hevc",
    audio_codec: str = "aac",
) -> VideoMetadata:
    return VideoMetadata(
        path=str(path),
        codec_name=codec,
        profile="Main",
        pix_fmt="yuv420p",
        width=width,
        height=height,
        bit_depth=8,
        duration=10.0,
        source_fps=30.0,
        color=ColorMetadata("tv", "bt709", "bt709", "bt709"),
        audio_codec=audio_codec,
    )


class SbsConcatTests(unittest.TestCase):
    def test_concat_uses_copy_when_clip_params_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            gap = root / "研究" / "gap_seg000.mp4"
            mosaic = root / "研究" / "mosaic_seg000.restored.mp4"
            gap.parent.mkdir()
            gap.write_bytes(b"gap")
            mosaic.write_bytes(b"mosaic")
            output = root / "out.mp4"
            timeline = [
                TimelineEntry(0.0, 10.0, gap, "gap"),
                TimelineEntry(10.0, 20.0, mosaic, "mosaic", 0.9),
            ]
            seen = {}

            def fake_run(cmd, **_kwargs):
                seen["cmd"] = cmd
                list_path = Path(cmd[cmd.index("-i") + 1])
                seen["list_path"] = list_path
                seen["list"] = list_path.read_text(encoding="utf-8")
                output.write_bytes(b"out")

            with (
                patch("utils.sbs_concat.shutil.which", return_value="ffmpeg"),
                patch("utils.sbs_concat.probe.probe_video", side_effect=lambda path: _meta(Path(path))),
                patch("utils.sbs_concat._run", side_effect=fake_run),
            ):
                sbs_concat.concat_timeline(timeline, output)

            self.assertIn("-c:v", seen["cmd"])
            self.assertIn("copy", seen["cmd"])
            self.assertIn("-an", seen["cmd"])
            self.assertIn("ffconcat version 1.0", seen["list"])
            self.assertIn(str(gap.resolve()).replace("\\", "/"), seen["list"])
            self.assertIn(str(mosaic.resolve()).replace("\\", "/"), seen["list"])
            self.assertEqual(seen["list_path"].parent, output.parent)

    def test_concat_ignores_segment_audio_and_muxes_source_audio_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "研究" / "source.mp4"
            mosaic = root / "研究" / "mosaic_seg000.restored.mp4"
            source.parent.mkdir()
            source.write_bytes(b"source")
            mosaic.write_bytes(b"mosaic")
            output = root / "out.mp4"
            timeline = [
                TimelineEntry(0.0, 10.0, source, "gap", inpoint_s=0.0, outpoint_s=10.0),
                TimelineEntry(10.0, 20.0, mosaic, "mosaic", 0.9),
            ]
            seen_cmds: list[list[str]] = []
            seen = {}

            def fake_run(cmd, **_kwargs):
                seen_cmds.append(cmd)
                if "-f" in cmd and "concat" in cmd:
                    list_path = Path(cmd[cmd.index("-i") + 1])
                    seen["list"] = list_path.read_text(encoding="utf-8")
                Path(cmd[-1]).write_bytes(b"out")

            metas = {
                str(source): _meta(source, audio_codec="aac"),
                str(mosaic): _meta(mosaic, audio_codec=""),
            }
            with (
                patch("utils.sbs_concat.shutil.which", return_value="ffmpeg"),
                patch("utils.sbs_concat.probe.probe_video", side_effect=lambda path: metas[str(Path(path))]),
                patch("utils.sbs_concat._run", side_effect=fake_run),
            ):
                sbs_concat.concat_timeline(timeline, output, audio_source=source)

            self.assertEqual(len(seen_cmds), 2)
            self.assertIn("-c:v", seen_cmds[0])
            self.assertIn("copy", seen_cmds[0])
            self.assertNotIn("hevc_nvenc", seen_cmds[0])
            self.assertIn("inpoint 0.000000", seen["list"])
            self.assertIn("outpoint 10.000000", seen["list"])
            self.assertIn("-map", seen_cmds[1])
            self.assertIn("1:a:0", seen_cmds[1])
            self.assertNotIn("1:a:0?", seen_cmds[1])
            self.assertNotIn("-shortest", seen_cmds[1])
            self.assertEqual(seen_cmds[1][-1], str(output))

    def test_concat_reencodes_when_clip_params_do_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "a.mp4"
            second = root / "b.mp4"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            output = root / "out.mp4"
            timeline = [
                TimelineEntry(0.0, 10.0, first, "gap"),
                TimelineEntry(10.0, 20.0, second, "mosaic", 0.9),
            ]
            metas = {
                str(first): _meta(first, codec="hevc"),
                str(second): _meta(second, codec="h264"),
            }
            seen = {}

            def fake_run(cmd, **_kwargs):
                seen["cmd"] = cmd
                output.write_bytes(b"out")

            with (
                patch("utils.sbs_concat.shutil.which", return_value="ffmpeg"),
                patch("utils.sbs_concat.probe.probe_video", side_effect=lambda path: metas[str(Path(path))]),
                patch("utils.sbs_concat._run", side_effect=fake_run),
            ):
                sbs_concat.concat_timeline(timeline, output, bitrate_bps=1_000_000)

            self.assertIn("hevc_nvenc", seen["cmd"])
            self.assertIn("-b:v", seen["cmd"])

    def test_fast_hevc_merge_extracts_raw_parts_and_muxes_once(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.mp4"
            restored = root / "mosaic_seg000.restored.mp4"
            output = root / "out.mp4"
            source.write_bytes(b"source")
            restored.write_bytes(b"restored")
            timeline = [
                TimelineEntry(0.0, 10.0, source, "gap", inpoint_s=0.0, outpoint_s=10.0),
                TimelineEntry(10.0, 20.0, restored, "mosaic", 0.9),
            ]
            commands: list[list[str]] = []

            def fake_run(cmd, **_kwargs):
                commands.append(cmd)
                Path(cmd[-1]).write_bytes(b"hevc-part")

            def fake_probe(path):
                p = Path(path)
                if p == output:
                    return _meta(p, width=4096, height=4096)
                return _meta(p, width=4096, height=4096)

            def fake_mux(raw_hevc, out_path, **_kwargs):
                self.assertTrue(Path(raw_hevc).exists())
                Path(out_path).write_bytes(b"muxed")

            with (
                patch("utils.sbs_concat.shutil.which", return_value="ffmpeg"),
                patch("utils.sbs_concat.probe.probe_video", side_effect=fake_probe),
                patch("utils.sbs_concat._run", side_effect=fake_run),
                patch("gpu_engine.mux.mux_hevc_with_audio", side_effect=fake_mux) as mux,
            ):
                sbs_concat.concat_timeline_hevc_fast(
                    timeline,
                    output,
                    source_src=source,
                    audio_source=source,
                )

            self.assertEqual(len(commands), 2)
            self.assertIn("hevc_mp4toannexb", commands[0])
            self.assertIn("-t", commands[0])
            self.assertIn("10.000000", commands[0])
            self.assertIn("hevc_mp4toannexb", commands[1])
            mux.assert_called_once()
            self.assertEqual(mux.call_args.args[1], output)
            self.assertEqual(mux.call_args.kwargs["audio_source"], str(source))


if __name__ == "__main__":
    unittest.main()
