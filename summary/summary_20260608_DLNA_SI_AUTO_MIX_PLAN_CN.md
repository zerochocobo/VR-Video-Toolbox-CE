# DLNA 服务器集成同声传译(SI)音轨 — 开发计划 v2（实时流式版）

> v1 的"预混缓存"方案已被用户否决；v2 改为**实时 ffmpeg 转码 + seek-by-time**，并新增**配置热重载**。

## 1. 用户需求（已确认）

1. 主程序"DLNA 配置服务器"对话框，**"自动关联外部字幕"下方**新增 checkbox："**如果同声传译(SI)文件存在，自动增加[SI]播放入口**"。
2. 勾选时**展开 4 个选项**（默认与 `tool_si` 一致）：
   - 叠加声道（左/右；默认 left）
   - 原声音量（70/80/90/100，默认 100）
   - SI 音量（50/60/70/80/90/100，默认 50）
   - SI 延迟（0/0.3/0.5/0.7/1/1.2/1.5/2，默认 1.0）
3. 保存到 `vr_toolbox_config.json`，key 前缀 `dlna_si_`。
4. DLNA 浏览时，若 `<stem>.si.wav` 旁邻文件存在，在原视频条目旁多生成一条 `[SI] <title>` 入口。
5. 客户端点 `[SI]` 入口：**实时**混音播放；**视频流保持原样**（不重编码），音频流按配置混入 SI。

## 2. 用户已确认的关键决策

| 项 | 决策 |
|---|---|
| 缓存策略 | **不预混不落盘**，全实时流式 |
| 等待 | **不可接受任何阻塞** |
| 预热 | **不预热**；改为支持 seek：视频拖动到位置 X，wav 也从位置 X 开始混合 |
| 配置变更生效 | **热重载**，无需重启 DLNA 子进程 |
| 独立音轨开关 | **不暴露**（DLNA 单流，不需要） |

---

## 3. 技术方案：实时流式转码

### 3.1 核心思路

DLNA 客户端通过 HTTP Range 请求字节区间。要支持"实时 + seek"，必须解决两个矛盾：

- **客户端发的是字节偏移**（`Range: bytes=N-`），不是时间偏移
- **输出大小未知**（边转边出）

业界成熟做法（Plex / Jellyfin / Universal Media Server 都用类似套路）：

1. **预估总大小**：用 `ffprobe` 拿到视频时长 + 视频流大小 + 音频目标码率，估算总字节数作为 `Content-Length`
2. **字节↔时间换算**：`time = (byte_offset / total_bytes) * duration`
3. **seek 即重启 ffmpeg**：客户端发 `Range: bytes=N-` 时，反算出 `t = (N/total)*duration`，杀掉旧的 ffmpeg，用 `-ss t` 起一个新的，从字节 N 开始把 ffmpeg stdout 直接 pipe 给 HTTP body
4. **顺序流式**：当 Range 紧接着上次流末尾（或无 Range / `Range: bytes=0-`）时，**复用**正在跑的 ffmpeg，不重启

视频 `-c:v copy`，所以 seek 必须落在关键帧上（用 `-ss <t> -i video` 把 ffmpeg 自身的 seek 对齐到最近关键帧），客户端体验等同普通 DLNA 视频的拖动行为。

### 3.2 ffmpeg 命令模板

```
ffmpeg -hide_banner -loglevel error \
    -ss {start_time} -i {video} \
    -ss {start_time} -i {si_wav} \
    -filter_complex "{si_mix_filter}" \
    -map 0:v -c:v copy \
    -map "[si_track]" -c:a aac -b:a 192k -ar 48000 -ac 2 \
    -movflags +frag_keyframe+empty_moov+default_base_moof \
    -f mp4 pipe:1
```

关键点：
- **两个 `-ss` 都放在 `-i` 前**：fast seek，效果是在 demuxer 层跳到最近关键帧附近，对 `-c:v copy` 必需
- **fragmented MP4**：`frag_keyframe+empty_moov+default_base_moof` 让 mp4 不需要后端写 moov，可以纯流式输出
- **不要 `+faststart`**：那个 flag 要求两遍处理，与流式互斥
- **滤镜复用** `tool_si.logic.build_si_mix_filter`：已经处理好 `aformat`、`aresample`、`alimiter`、`adelay`，直接调用

### 3.3 Content-Length 估算

```python
def estimate_output_size(video_path, audio_bitrate_bps=192000):
    probe = probe_cached(video_path)            # 已有
    duration = probe["duration"]
    video_size = probe["size"]                  # 整个原视频
    # video stream 大约占整个文件的 95%，audio 重编码后码率固定
    # 更准确：用 ffprobe -show_streams 拿到 video stream size
    audio_size = int(audio_bitrate_bps * duration / 8)
    # mp4 容器开销 ~2%
    return int((video_size + audio_size) * 1.02)
```

更精确做法（推荐）：用 `ffprobe` 拿到 video stream 的精确 `nb_read_packets * avg_packet_size` 或 `stream.size`，避免把原 audio stream 大小也算进去。`probe_cached` 已经有 bitrate，但没有分流 size。需要在 `_probe_video` 里多取 `stream=size` 字段。

估算精度只要在 ±5% 内，DLNA 客户端就能正常拖动；播放到接近末尾时 ffmpeg 自然结束流，HTTP 也自然 EOF，客户端会停止。

### 3.4 Range 请求处理流程

```
HTTP GET /media_si/<key>  Range: bytes=N-M (or N-)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ 1. 解析 N，估算 t = (N / total_bytes) * duration         │
│ 2. 查询该视频的 LiveStreamSession（若存在）              │
│ 3a. 若 session 当前 byte_cursor 在 N 附近 (容差 ~1MB)：   │
│     → 复用，从 session.stdout 读 bytes [N..M]            │
│ 3b. 否则（seek 或新连接）：                              │
│     → 终止旧 ffmpeg（若有）                              │
│     → 用 -ss t 启动新 ffmpeg                             │
│     → 把 byte_cursor 重置到 N                            │
│     → 流 ffmpeg.stdout 到 HTTP body                      │
│ 4. 维护 byte_cursor += 已写出字节                        │
└──────────────────────────────────────────────────────────┘
```

注：因为多数客户端 seek 后会立刻发 `Range: bytes=N-` 然后顺序读到尾，并发"两段 Range"在 DLNA 场景几乎不会出现，**每个 video key 最多维护一个活跃 session 即可**。

---

## 4. 涉及文件与改动概览

| 文件 | 改动 | 行数估计 |
|---|---|---|
| `main.py` | DLNA 配置对话框新增 checkbox + 4 选项动态显示；保存 5 个新 config key；保存后调用 `_dlna_reload_si_config()` 热推送 | +100 |
| `tool_dlna/main.py` | 读 5 个新 config，启动时连同 `SIMixConfig` 注入 `create_app`；监听热重载触发 | +30 |
| `tool_dlna/dlna_server.py` | `create_app` 接受 SI 配置 holder；新增 `/media_si/{name:path}` 路由 + Range 处理；新增 `/admin/reload_si_config` 热重载端点 | +120 |
| `tool_dlna/si_stream.py` *(新建)* | 实时流式核心：`SIMixConfig` dataclass、`LiveStreamSession` 类、`SIStreamService` 协调器 | +280 |
| `tool_dlna/content_directory.py` | `_get_items_for_dir` / `BrowseMetadata` 加 `[SI]` 平行条目；新增 `VIDEO_SI_PREFIX="vs_"` | +60 |
| `tool_dlna/content_directory.py` | `_probe_video` 多取 `stream=size`，便于估算 | +5 |
| `tool_si/logic.py` | 抽出一个**纯函数版** `build_si_mix_filter`（已是纯函数，可直接复用），无改动 | 0 |
| `i18n/{zh,en,ja}.json` | 新增 `dlna_si_*` 命名空间下 6 个 key | +18 |
| `utils/app_config.py` | 5 个新 key 默认值 | +6 |
| `tests/test_tool_dlna_si_stream.py` *(新建)* | 见第 7 节 | +200 |

**总计：约 +820 行（含测试），比 v1 多约 50%，主要在流式状态机和测试。**

---

## 5. 关键模块设计

### 5.1 `tool_dlna/si_stream.py` 新建

```python
@dataclass(frozen=True)
class SIMixConfig:
    enabled: bool
    mix_channel: str          # "left" | "right"
    original_volume_percent: int
    si_volume_percent: int
    si_delay_seconds: float

    @classmethod
    def from_app_config(cls, getter) -> "SIMixConfig": ...
    def filter_string(self) -> str:
        # 直接 return tool_si.logic.build_si_mix_filter(...)
        ...


class LiveStreamSession:
    """单个 (video, si_wav) 的活跃 ffmpeg 流；持有 stdout pipe 和 byte_cursor。"""
    def __init__(self, video: Path, si_wav: Path, config: SIMixConfig,
                 start_time: float, estimated_total: int):
        self.proc: subprocess.Popen | None = None
        self.byte_cursor: int = 0   # 起始字节偏移（对应 start_time 在估算文件中的位置）
        self.lock = threading.Lock()
        self._start_ffmpeg(start_time)

    def _start_ffmpeg(self, t: float): ...
    def read(self, n: int) -> bytes: ...     # 阻塞读 n 字节或到 EOF
    def close(self): ...                      # _terminate_process


class SIStreamService:
    """协调所有 active sessions；config 热重载；缓存估算。"""
    def __init__(self, media_library, config_holder: ConfigHolder):
        self._sessions: dict[str, LiveStreamSession] = {}   # key = abs_path
        self._sessions_lock = threading.Lock()
        self._config_holder = config_holder

    def current_config(self) -> SIMixConfig:
        return self._config_holder.get()

    def has_si_source(self, video: Path) -> Path | None:
        sibling = video.with_suffix(".si.wav")
        return sibling if sibling.is_file() else None

    def estimate_output_size(self, video: Path) -> int: ...

    def open_stream(self, video: Path, range_start: int, range_end: int | None) \
            -> tuple[Iterator[bytes], int, int, int]:
        """
        返回 (字节迭代器, content_length, total_size, status_code)。
        status: 200 完整 | 206 Range
        """
        ...

    def reload_config(self, new_config: SIMixConfig) -> None:
        """终止所有活跃 session，下次请求按新配置重启。"""
        ...

    def shutdown(self) -> None: ...
```

**ConfigHolder**：包一个 `threading.RLock()` 保护的 `SIMixConfig`。`/admin/reload_si_config` 收到新配置 → 写入 holder → 调 `service.reload_config()` 终止旧 sessions。

### 5.2 `dlna_server.py` 中的路由

```python
@app.get("/media_si/{name:path}")
async def media_si_get(request: Request, name: str):
    path = _safe_video_path(name)
    config = si_service.current_config()
    if not config.enabled:
        raise HTTPException(404, "SI entries disabled")
    if si_service.has_si_source(path) is None:
        raise HTTPException(404, "No SI source")

    range_header = request.headers.get("range", "")
    range_start, range_end = _parse_range(range_header)   # bytes=N- or bytes=N-M
    total = si_service.estimate_output_size(path)

    chunks, content_length, status = si_service.open_stream(
        path, range_start, range_end, total
    )

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "transferMode.dlna.org": "Streaming",
        "contentFeatures.dlna.org": f"DLNA.ORG_PN=AVC_MP4_HP_HD_AAC;DLNA.ORG_OP=01;DLNA.ORG_CI=1;DLNA.ORG_FLAGS={DLNA_FLAGS_BASE}",
    }
    if status == 206:
        headers["Content-Range"] = f"bytes {range_start}-{range_start + content_length - 1}/{total}"
    return StreamingResponse(chunks, status_code=status, headers=headers, media_type="video/mp4")
```

注意 `DLNA.ORG_CI=1`：标记此流为"转码后内容"，DLNA 客户端会调整对该流的 timing/buffering 行为。

### 5.3 `content_directory.py` 浏览修改

要点（与 v1 一致，做了简化）：

```python
VIDEO_SI_PREFIX = "vs_"

# 在 _get_items_for_dir 的 video 分支末尾：
if si_service is not None:
    config = si_service.current_config()
    if config.enabled and si_service.has_si_source(child) is not None:
        si_total = si_service.estimate_output_size(child)
        items.append({
            "id": f"{VIDEO_SI_PREFIX}{rel_key}",
            "parent_id": parent_id,
            "title": f"[SI] {title}",
            "url": f"{base_url}/media_si/{quoted_key}",
            "thumb": f"{base_url}/thumb/{quoted_key}",   # 缩略图复用原视频的
            "size": si_total,
            "duration": meta["duration"],
            "resolution": f"{meta['width']}x{meta['height']}" if meta["width"] > 0 else "",
            "bitrate": meta["bitrate"],
            "mime": "video/mp4",
            "dlna_pn": "AVC_MP4_HP_HD_AAC",
            "subtitles": sub_list,    # SRT 仍然挂上
        })
```

`BrowseMetadata` 分支同样要识别 `VIDEO_SI_PREFIX`，剥前缀回查原视频 key。

### 5.4 热重载链路

```
GUI 保存按钮
    │
    ▼
main.py: save_config()
    │ app_config.set('dlna_si_*', ...)
    ▼
main.py: _dlna_push_si_reload()
    │ POST http://127.0.0.1:{dlna_port}/admin/reload_si_config
    │ body: {mix_channel, original_volume_percent, si_volume_percent, si_delay_seconds, enabled}
    ▼
dlna_server.py: /admin/reload_si_config
    │ 校验 source IP 是 127.0.0.1（拒绝外网）
    │ config_holder.set(new_config)
    │ si_service.reload_config(new_config)
    ▼
SIStreamService.reload_config():
    │ 用 sessions_lock 取所有 active sessions
    │ session.close() 一一终止
    │ 清空 self._sessions
    │ 下次客户端请求自然按新配置起新 ffmpeg
```

**为何安全**：终止活跃 session 后，客户端下一次 Range 请求会触发新 ffmpeg 启动，从客户端当前播放位置开始 seek，体验上是"播到一半声音变了"，与电视上换音轨的行为一致。

热重载端点用 `127.0.0.1` 校验，不暴露给局域网，避免被外部修改配置。

---

## 6. 边界与风险

| 场景 | 处理 |
|---|---|
| 客户端 seek 频繁（快速拖动） | 防抖：`open_stream` 内对同 video 的 ffmpeg 启动加 200ms 冷却；旧 ffmpeg `_terminate_process` 等返回后再启新的 |
| ffmpeg startup 延迟（~500ms～1.5s） | 首帧延迟正常；DLNA 客户端可接受 |
| video 是 H.265/HEVC | `-c:v copy` 兼容；但客户端若不支持 HEVC 会失败 — 这是原视频本身的兼容性问题，与 SI 无关 |
| 视频关键帧间隔大（>10s） | `-ss` seek 后会跳到最近关键帧，跟点可能偏几秒。与电视上拖动 mp4 行为一致 |
| Content-Length 估算偏差大 | 误差 ±5% 内大多数客户端可接受；超过则客户端可能在末尾报错。给 `estimate_output_size` 加一个保守上浮 `*1.05` |
| 客户端不支持 `DLNA.ORG_CI=1`（转码标记） | 改为 `CI=0`（兼容性更高）；体验上无差异 |
| SI wav 比视频短 | `amix=duration=first` 已处理（[tool_si/logic.py:402-405](tool_si/logic.py:402)）：SI 结束后只剩原声 |
| ffprobe 测不出 duration | `has_si_source` 返回 None 时不列 SI 条目；或列出但 estimate_size 用 fallback |
| 多客户端同时播同一 [SI] | 各自有独立 `LiveStreamSession`，互不干扰（每个 session 一个 ffmpeg 进程）|
| 同一客户端发并发 Range（少见） | 简化：以最后一次 Range 为准，旧的 ffmpeg 直接终止；HTTP 早期的 chunk 客户端会丢弃 |
| 热重载时正在流出的客户端 | 客户端会收到 ffmpeg 终止后的连接关闭；正常情况下会自动重连发新 Range（多数 DLNA 客户端如此），表现为"音轨切换 0.5~1s 卡顿" |
| 路径含 unicode（中日韩） | ffmpeg Windows 子进程要确保 `subprocess.Popen` 不被 `text=True` 解码 stdout（**保留二进制 stdout**） |
| `-ss` 在 stdin 是 pipe 的情况下 | 两路 `-i` 都是文件，不存在 pipe stdin 问题 |
| 视频极短 (<10s) 或 si.wav 极短 | 估算 size 时给下界保护，避免除零 |

---

## 7. 测试计划

新建 `tests/test_tool_dlna_si_stream.py`：

1. **`test_si_mix_config_from_app_config_defaults`** — 不读任何 key 时返回默认值
2. **`test_si_mix_config_filter_string_matches_tool_si`** — 与 `tool_si.logic.build_si_mix_filter` 输出一致
3. **`test_has_si_source_detects_sibling_wav`** — 临时目录 `a.mp4` + `a.si.wav` → 返回 wav 路径
4. **`test_estimate_output_size_within_bounds`** — mock probe 返回已知 video size / duration，验证估算在合理范围
5. **`test_browse_lists_si_entry_when_enabled`** — 含 .si.wav 的 video，browse 输出多一条 `vs_` 条目
6. **`test_browse_omits_si_entry_when_disabled`** — `config.enabled=False` 时不列
7. **`test_browse_metadata_for_si_entry`** — BrowseMetadata `vs_<rel>` 返回 `[SI]` title
8. **`test_open_stream_starts_ffmpeg_with_seek`** — mock `subprocess.Popen`，验证传入的 `-ss <t>` 参数与 range_start 对应
9. **`test_open_stream_returns_206_for_range`** — Range 请求返回 status=206 + Content-Range
10. **`test_open_stream_reuses_session_on_sequential_read`** — 连续两次 Range（无间隙）只启一次 ffmpeg
11. **`test_reload_config_terminates_active_sessions`** — 模拟 active session → reload_config → session.close 被调
12. **`test_reload_endpoint_rejects_non_loopback`** — FastAPI TestClient 模拟非 127.0.0.1 → 403
13. **`test_reload_endpoint_updates_filter_string`** — POST 新 config → 下次 open_stream 用新滤镜参数
14. **`test_range_parser_handles_open_ended`** — `bytes=1024-` 正确解析为 (1024, None)
15. **`test_range_parser_rejects_malformed`** — `bytes=abc` 返回 (0, None) 兜底
16. i18n 三语 key 完整性（由现有 `test_i18n.py` 自动覆盖）

---

## 8. 实施步骤

| 步骤 | 工作量 | 关键交付 |
|---|---|---|
| 1. `SIMixConfig` + `ConfigHolder` + 单测 1/2/15 | 0.5d | dataclass + 配置载入读取闭环 |
| 2. `LiveStreamSession` + ffmpeg 启动/读取/终止 | 1d | mock 单测 + 真 ffmpeg 联调（短 mp4 + 短 wav） |
| 3. `SIStreamService.open_stream` + Range 解析 | 0.5d | 单测 8/9/10/14 |
| 4. FastAPI 路由 `/media_si/...` + StreamingResponse | 0.5d | curl 验证字节正确流出 + Range 行为 |
| 5. `content_directory` 浏览扩展 + 新 prefix | 0.5d | 单测 5/6/7 |
| 6. 热重载端点 + GUI 推送 | 0.5d | 单测 11/12/13 + 手工 GUI 验证 |
| 7. GUI 对话框 + i18n | 0.5d | 手工对话框开关验证 |
| 8. 真实链路（电视/Quest）联调 | 0.5d | 拖动 seek 测试、热重载测试、多客户端测试 |

**总计：约 4.5 个工作日。** 比 v1 多一天，主要是状态机和真机联调。

---

## 9. 与 v1 方案的对比

| 维度 | v1（预混缓存） | v2（实时流式） |
|---|---|---|
| 首次播放延迟 | 30s ~ 5min | ~1s（ffmpeg startup） |
| seek 体验 | 缓存就绪后秒级 | ~500ms（ffmpeg restart） |
| 磁盘占用 | 每视频额外 ~100% | 0 |
| CPU 占用 | 一次性，后续 0 | 每次播放都跑 ffmpeg（音频重编码）|
| 多客户端 | 共享缓存 | 各自一份 ffmpeg |
| 实现复杂度 | 中（180 行） | 高（280 行 + 状态机） |
| 配置变更生效 | 重启 / 删缓存 | 热重载，下次 Range 自然切 |
| 兼容性风险 | 低（标准 FileResponse） | 中（Content-Length 估算误差、客户端转码标志兼容性） |

---

## 10. 待用户在开工前最终确认

1. ✅ 已确认：缓存目录用 `runtime_cache/`（v2 实际不落盘，但 ffmpeg stderr/调试日志仍可放这里）
2. ✅ 已确认：不阻塞、纯实时
3. ✅ 已确认：不预热，靠 seek
4. ✅ 已确认：热重载（127.0.0.1 内部端点）
5. ✅ 已确认：不暴露独立音轨开关

**剩余开工前小决策（默认值已选好，不反对就照做）：**

- **A**. Content-Length 估算上浮系数：默认 `×1.05`（5% 保守上界）。如电视客户端报"文件损坏"再调到 `×1.10`。
- **B**. DLNA.ORG_CI 标志：默认 `CI=1`（转码标记）。如某些老电视播不动再改 `CI=0`。
- **C**. ffmpeg seek 防抖：默认 `200ms`。
- **D**. Range 复用容差：默认 `1MB`（连续 Range 落在 [cursor, cursor+1MB] 内复用 session）。

以上 4 项均可在 `dlna_si_*` 隐藏配置或 module 顶层常量调，首发用上面默认值。

---

## 11. 风险声明

**与 v1 相比，v2 的实际播放兼容性需要在真机上验证**。已知风险：

- **某些 DLNA 客户端不接受 Content-Length 误差**：若发现 ≥5% 误差就拒绝，需要调上浮系数或改为 chunked-transfer（牺牲一些 DLNA 兼容性）
- **Quest / Pico VR 浏览器**：对 fragmented MP4 的支持参差不齐，需要测
- **拖动到尾部 1% 区域**：估算偏差被放大，可能出现"看着拖到 99% 实际跳到 95%"

若真机联调发现 v2 在主力设备上行为不可接受，**保留回滚到 v1（预混缓存）的余地**：两版本的 GUI / 配置 / DLNA browse 改动完全相同，仅 `si_stream.py` 替换为 `si_mixer.py` 即可。建议在 PR 里把流式核心隔离在单一模块，方便回滚。
