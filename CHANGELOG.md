# Changelog

## English

- [May 29, 2026] v12.0 Ported the DLNA server functionality from another project of mine, refactored the multilingual functionality, and added Japanese translation.
- [May 19, 2026] v11.5 One-click mode output video bitrate further adjusted, subtitle hard embedding corrected output bitrate and option errors.
- [May 15, 2026] v11.4 Released. Lada and Jasna support custom additional parameters. All log windows now include execution completion time statistics.
- [May 14, 2026] v11.3 released. Added support for Jasna as an alternative AI mosaic removal engine (switchable from the main screen). Fixed a bug where the merge tool in one-click mode produced no log output.
- [May 6, 2026] v11.2 released, officially renamed VR Video Toolbox, with a redesigned homepage interface. The hard subtitle tool now includes subtitle direction selection, allowing for vertical subtitles. The UI for image preview has been optimized with improved waiting prompts.
- [May 5, 2026] Version 11.1 adds VR subtitle hard subtitle embedding functionality. All imaginable subtitle features have been developed. I can now focus on developing VR video passthrough tools.
- [May 4, 2026] v11.0 Official Version: Multi-threaded subtitle extraction, added user prompts when ffmpeg encounters errors.
- [May 2, 2026] Released v11.0 RC2, introducing WhisperSeg as the default VAD recommended by the new version of the WhisperJAV project, and optimizing the judgment of CUDA installation.
- [April 30, 2026] Released v11.0 RC1, primarily for non-technical users, adding tutorials and documentation to help them when necessary dependencies are missing.
- [April 28, 2026] Added a batch JAV subtitle processing tool, which summarizes my long-standing needs for subtitles, research, and tool development. This is released as a beta version of 11.0. My goal is to make the software lightweight, the interface simplified, and the operations batch-processable, minimizing the technical requirements for users.
- [April 15, 2026] The new training restoration model has been officially abandoned. The test model and intermediate model can be downloaded from https://huggingface.co/zelefans/vrmr/tree/main. Future updates will add some useful tools; version 10.8 adds a batch function to add soft srt files to videos. The development of this feature is because DLNA cannot load subtitle files in the same directory, and VR videos are really not suitable for hard subtitles.
- [February 5, 2026] To ensure the clarity of the output video during VR video processing, most of the bitrate control in the intermediate process of the previous two versions has been removed, and an option to "keep the output video bitrate consistent with the original video" has been added to some interfaces.
- [February 1, 2026] Added the ability for ffmpeg to handle splitting/merging and fisheye conversion simultaneously in a single command, further improving processing efficiency. Thanks to @joseff_joester for providing this feature. Features involved: One-click mode, VR Split/Combine Tool, VR projection conversion tool.
- [January 29, 2026] Added the ability to simultaneously cut and output left and right eye video files using ffmpeg, thanks to @joseff_joester for providing this feature. Increased control over the output video bitrate and reduced the size of intermediate and output files significantly.
- [January 28, 2026] Added a standalone video splitting and merging tool, as well as a VR hemispherical and fisheye conversion tool. Both of these new tools and the one-click mode attempt to add automatic bitrate matching functionality to the original video, striving to match the bitrate of the new video with the original file as closely as possible.
- [January 7, 2026] Added an option to convert to fisheye mode before processing in one-click mode. I only learned from @AlcoholicOverfitter's submission to the lada project that manufacturers like SAVR add mosaic effects in fisheye mode. Fisheye mode can handle mosaic effects along the central axis very well, although it still cannot handle severe deformation on the sides.
- [January 4, 2026] lada0.10 changed many parameters, thank you @tofupi submit. The restoration model is still training, the first stage is almost done, entering the second stage.
- [December 15, 2025] A proof-of-concept 0.1 restoration model was released, simply to verify that VR restore is feasible. It needs to be used in conjunction with the previous detection model; see details: [issue#10](https://codeberg.org/zelefans/vr_remove_mosaic/issues/10).
- [December 6, 2025] A new mosaic recognition model for VR-to-2D conversion was trained using 30,000 images. It achieves almost 100% accuracy in recognizing new VR videos. However, it was found that the Lada restore model can only restore mosaics without tilt distortion, requiring further investigation. The scripts I used, as well as the GUI that allows model selection, have been released. The newly generated model has also been released at https://huggingface.co/zelefans/vrmr/tree/main and on this project's release page. Simply place the model in the models directory.
- [December 2, 2025] Version 10.0 can be released now. Implemented a feature in selection mode where only a specific time interval is selected for decoding. By cutting the first and last videos and setting it to overwrite the original video, it quickly processes decoding of different areas of the screen at different times, greatly improving processing efficiency. From an efficiency perspective, this should be the ultimate method, because the mosaic position in VR videos is often fixed within a certain time interval, making it unnecessary to process the entire video. Added a VR to flat tool. I can't think of any more features or areas for improvement at the moment. Next, I will try to research the Lada mosaic recognition problem, hoping to solve the mosaic recognition problem under tilt and distortion as soon as possible.

## 中文
- 【2026年5月29日】v12.0 移植了我另一个项目的DLNA服务器功能，重构了多语言功能，增加了日语翻译。
- 【2026年5月19日】v11.5 一键模式输出视频比特率进一步调整，字幕硬嵌入修正输出的比特率和选项错误。
- 【2026年5月15日】v11.4 Lada、Jasna支持自定义额外参数。所有日志窗口增加执行完成后的时间统计。
- 【2026年5月14日】v11.3 发布，支持 Jasna 作为 AI 去马赛克引擎的平替选项（可在主界面切换）。修正一键模式下合并工具单独使用时无日志输出的 BUG。
- 【2026年5月6日】v11.2 发布，正式更名为VR视频工具箱，重构首页界面。硬字幕工具增加字幕方向选择，可以垂直字幕了。优化了软件中图像预览的UI界面等待提示。
- 【2026年5月5日】v11.1 发布，增加VR字幕硬字幕嵌入功能，字幕功能能想到的都已经开发完成了。后续可以安心开发VR视频透视处理工具了。
- 【2026年5月4日】v11.0 正式版本，多线程运行字幕提取，增加ffmpeg出错后的用户提示。
- 【2026年5月2日】发布v11.0 RC2版本，引入WhisperJAV项目新版本推荐的WhisperSeg作为默认VAD，优化了cuda安装的判断。
- 【2026年4月30日】发布v11.0 RC1版本，主要针对非技术用户增加缺少必要依赖项时的引导教程和说明文档。
- 【2026年4月28日】增加了JAV批量字幕处理工具，将我一直以来对字幕的需求、研究和开发工具汇总在一起。作为11.0的beta版本发布吧。我的目的还是让软件轻量化、界面精简化，操作批量化，尽量希望减少对使用者对技术的要求。
- 【2026年4月15日】正式放弃了新训练恢复模型，测试模型和中间模型在 https://huggingface.co/zelefans/vrmr/tree/main 可以下载。今后新更新程序会加一些有用的小工具上。10.8版本增加了批量给视频添加软字幕功能，开发这个功能是因为在用DLNA等情况下是无法载入同目录下的字幕文件的，并且VR视频实在不适合硬字幕。
- 【2026年2月5日】为了确保VR视频处理过程中成中的清晰度，移除了大部分之前两个版本在中间过程对比特率的控制，在部分界面增加了“输出视频码率和原视频保持一致”的可选项。
- 【2026年2月1日】增加了ffmpeg在一条命令中同时处理切割/合并和鱼眼转换的功能，进一步增加处理执行效率。感谢@joseff_joester提供。涉及到的功能：一键模式、VR分屏合并工具、VR投影转换工具。
- 【2026年1月29日】增加了一次ffmpeg处理同时切割输出左右眼视频文件的功能，感谢@joseff_joester提供。增加了更多的对输出视频比特率的控制，减少很多中间和输出文件的大小。
- 【2026年1月28日】增加了独立的视频分割合并工具，半球形等距和鱼眼的转换工具。这两个新工具和一键模式都尝试增加了原视频比特率自动匹配的功能，尽量让新视频的比特率匹配原文件。
- 【2026年1月7日】在一键模式中增加了处理前转换成鱼眼的选项。根据@AlcoholicOverfitter在lada项目中的提交才知道SAVR等厂家是在鱼眼模式下添加的马赛克。在鱼眼模式下可以很好地处理中轴线上的马赛克（虽然侧面的形变严重还是无法处理）。
- 【2026年1月4日】lada0.10变动了很多参数，感谢@tofupi提交。恢复模型还在训练中，第一阶段差不多了，进入第二阶段。
- 【2025年12月15日】发布一个概念验证用的0.1恢复模型，只是验证一下VR恢复是可行的，配合之前的识别模型，具体见[issue#10](https://codeberg.org/zelefans/vr_remove_mosaic/issues/10)。
- 【2025年12月6日】训练了新的针对VR转平面之后的马赛克识别模型，生成了3万张图片做的训练，识别新的VR视频几乎100%准确。但发现lada修复模型只能修复没有倾斜畸变的马赛克，还要继续研究…… 我用到的脚本，以及可以自己选择模型的GUI已经发布。新生成的模型也发布到了：https://huggingface.co/zelefans/vrmr/tree/main，以及本项目的版本发布页面，模型放到models目录下即可。
- 【2025年12月2日】版本可以作为10.0发布了。实现了选区模式下如果只选择某个时间区间解码，通过切割首尾视频，并设置覆盖原视频，快速处理不同时间段不同画面区域的解码，极大提升了处理效率，从效率角度来说应该是终极方法了（因为VR视频马赛克位置往往在某个时间区间位置固定，没必要处理整个视频）。增加了VR视频转平面功能。暂时想不到更多功能和改进地方了，接下去试图研究一下lada马赛克识别问题，希望尽快解决倾斜和畸变下的马赛克识别问题。
