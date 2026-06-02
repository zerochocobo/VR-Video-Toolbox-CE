# CuPy RawKernel / GPU "卡死" 排查手册

- 文档日期：2026-05-31
- 适用：VR_Video_Toolbox_NE（gpu_engine / native_mosaic）
- 环境：RTX 5060 Ti（Blackwell sm_120）+ 驱动 581.57 + CUDA 12.8 全栈（cupy-cuda12x + torch cu128 + nvrtc/runtime 12.8 wheel）
- 用途：**遇到 GPU kernel launch 后卡住 / 程序无响应时，先对照本手册自查，再动手改代码。**

---

## 0. 一句话结论（先看这个）

> **本栈下 `cp.RawKernel` 是稳定可用的。绝大多数"卡死"是 (a) 上一个 kernel 出错把 CUDA 上下文搞坏了、后面全跟着卡，或 (b) kernel 标量参数 ABI 传错。不是 CuPy/驱动坏了。**

历史教训：曾有人误判"`cp.RawKernel` 首次 launch 卡死"，改成 `NVRTC→PTX→cp.RawModule(path=...)`，结果那条路才真卡，又因上下文污染让 `cp.RawKernel` 也跟着卡，绕了一大圈。**别再走 PTX→RawModule。**

---

## 1. 关键概念：CUDA 上下文污染（最容易误判的坑）

**一旦某个 GPU kernel 发生非法访问（越界 / 错误指针 / faults），整个进程的 CUDA 上下文就进入错误态，之后该进程里的所有 kernel launch（包括完全正常的 cp.RawKernel）都会卡住或报错。**

后果：你以为"连最简单的 add1 都卡 → 一定是环境/CuPy 坏了"，其实是**前面某个 kernel already 把上下文弄坏了**。

> **铁律：诊断 GPU 卡死，永远在 fresh 进程（新开 `uv run python`）里复现，且一次只测一个 kernel。** 不要在同一个已经卡过的进程里继续测别的 kernel 下结论。

自查命令（fresh 进程，带超时，先测最简单的）：
```bash
timeout 60 uv run python -c "
import gpu_engine  # 必须先 import 它跑 _cuda_env.configure()
import cupy as cp
k = cp.RawKernel(r'extern \"C\" __global__ void a(unsigned char* x){x[threadIdx.x]+=1;}','a')
x = cp.zeros(8, dtype=cp.uint8); k((1,),(8,),(x,)); cp.cuda.Stream.null.synchronize()
print('OK', x[:3].tolist())
"
```
- 输出 `OK [1,1,1]` → cp.RawKernel 正常，问题在你后面的某个 kernel。
- 卡到 timeout → 见 §4（环境层面，少见）。

---

## 2. 第二大坑：kernel 标量参数 ABI（传错 → 行为诡异 / 越界 / 卡）

`cp.RawKernel` 调用 `k(grid, block, (args...))` 时，**Python 标量的字节宽度必须和 kernel 形参一致**，否则按字节错位读：

| kernel 形参 | 必须传 | 传 Python 原生标量的后果 |
|---|---|---|
| `float` | `np.float32(v)` | Python `float` = C double(8B) 塞进 4B 槽 → 读低 4 字节，`1.0`→`0.0` |
| `int` | `np.int32(v)` | Windows 上 Python int→32位 long 恰好匹配，**侥幸不炸**；但跨平台/混排会错位 |
| `double` | `np.float64(v)` 或 Python float | — |
| 指针 | CuPy ndarray | — |

**症状**：颜色全错、输出常数、或后续参数错位导致越界 → 卡死。
**真实案例**：`nv_to_bgr(... float norm)` 传了 Python `float(1.0)` → norm 实际=0.0 → 灰色 Y=120 出 `[0,77,0]`。改 `np.float32(norm)` 后逐像素 0 误差。

> **规矩：所有标量 kernel 参数一律用 `np.int32 / np.float32 / np.uint8` 等显式 dtype 包一层。** 项目里 `nv12_kernels.py` 的 remap kernel 只传 int（Windows 侥幸 OK），但新写 kernel 务必显式。

---

## 3. 第三坑：CuPy ↔ torch 同进程的整设备同步（不是崩，是慢/串行）

不是"卡死"，但表现像"卡住不动"（fps 极低）：在 lada 多线程流水线里每帧 `cp.cuda.Device().synchronize()`（**整设备**同步）会等别的线程的 GPU 工作（如 BasicVSR++），把并行打回串行 → 看起来像卡。
- 详见 `summary/summary_20260531_NATIVE_STREAM_FUSION_PLAN_CN.md` §1.3。
- 单线程下用 `cp.cuda.get_current_stream().synchronize()`（当前流），不要整设备同步。

---

## 4. 环境层面（少见，CUDA 12.8 全栈对齐后基本不会再遇到）

如果 fresh 进程里**最简单的 add1 都卡/报错**，才查这层（详见 memory `cuda-env-blackwell-coexistence`）：
- `import gpu_engine` 必须先于 `import cupy`（`_cuda_env.configure()` 设 PTX、清系统 CUDA_PATH）。
- `import torch` 与 CuPy 共存：已靠 **CUDA 12.8 全栈对齐**根治（cupy-cuda12x + nvrtc/runtime 12.8 wheel + torch cu128），不再有 nvrtc 版本冲突。
- 自检：`uv run python -c "import gpu_engine; from gpu_engine import runtime; print(runtime.warmup(verbose=True))"`，看 nvrtc 版本应是 (12,8)、available=True。
- CuPy 警告 `CUDA path could not be detected` 是**无害的**（我们故意清了系统 CUDA_PATH，让它用 wheel 的 12.8 头），不要为了消警告去设 CUDA_PATH（设错反而崩，见 memory）。

---

## 5. 排查流程图（照着走）

```
GPU 卡死 / 无响应
   │
   ├─ 1. 新开 fresh 进程，timeout 60，只跑最简单 add1 (§1)
   │     ├─ add1 OK   → 上下文污染或后续 kernel 的问题，往下
   │     └─ add1 卡   → 环境层 (§4)；查 warmup / nvrtc 版本 / import 顺序
   │
   ├─ 2. fresh 进程单独跑"出事的那个 kernel"（别和别的混）
   │     ├─ 卡       → 看该 kernel：标量参数 ABI (§2)？越界索引？LUT/输入是否含 NaN/越界坐标？
   │     └─ 不卡     → 是上下文污染：前面某 kernel 先炸了，去找它
   │
   ├─ 3. 怀疑标量 ABI：把所有标量参数改 np.int32/np.float32 (§2)，再跑
   │
   ├─ 4. 怀疑越界：用 compute-sanitizer 定位（见 §6）
   │
   └─ 5. fps 极低但没真卡 → 整设备同步串行化 (§3)，换当前流同步
```

---

## 6. 有用的工具/命令

- **fresh 进程 + 超时**（防止把自己也挂住）：`timeout 60 uv run python -c "..."`
- **逐像素对照参考实现**（验证 kernel 正确性，而不是只看跑没跑）：和 torch / numpy 等价实现比 PSNR，应 99dB / 0 误差。见本次 `nv12_to_bgr` 验证。
- **几何回归**：`uv run python tests/verify_lut_geometry.py`（remap kernel 正确性，62-75dB）。
- **越界定位**：`compute-sanitizer --tool memcheck uv run python xxx.py`（NVIDIA 自带，能指出哪个 kernel 哪行越界访问）。
- **warmup 自检**：`runtime.warmup(verbose=True)` 看 available/nvrtc 版本。（注意：当前 warmup 不覆盖"任意 kernel 可执行性"，available=True 不保证某个具体 kernel 不越界。）

---

## 7. 不要做的事（避免重蹈覆辙）

1. **不要**因为"看起来 cp.RawKernel 卡"就改成 `NVRTC→PTX→cp.RawModule(path=...)`——本栈 cp.RawKernel 正常，那条路反而坑。
2. **不要**在同一个已经卡过的进程里继续测别的 kernel 然后下"全都卡"的结论（上下文已污染）。
3. **不要**给 kernel 传 Python 原生 `float` 标量（用 np.float32）。
4. **不要**为了消 `CUDA path could not be detected` 警告去设系统 CUDA_PATH（会引入头文件版本不匹配，见 memory）。
5. **不要**在多线程 GPU 流水线里用 `cp.cuda.Device().synchronize()`（整设备同步会串行化）。

---

## 8. 相关文档
- memory `cuda-env-blackwell-coexistence`（CUDA 12.8 全栈对齐、RawKernel 稳定性、标量 ABI）
- memory `gpu-engine-architecture`
- `prompt/HANDOVER_20260531.md`（本次卡点的完整经过与解决）
- `summary/summary_20260531_NATIVE_STREAM_FUSION_PLAN_CN.md`（整设备同步串行化 / Option A）
