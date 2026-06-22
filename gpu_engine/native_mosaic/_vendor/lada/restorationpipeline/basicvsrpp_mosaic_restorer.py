from contextlib import nullcontext

import torch

from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.utils import ImageTensor
from gpu_engine._profile import get_active_profile
from gpu_engine.native_mosaic import _torch_tuning
from gpu_engine.native_mosaic._cuda_graph_runner import CudaGraphRunner

class BasicvsrppMosaicRestorer:
    def __init__(self, model: BasicVSRPlusPlusGan, device: torch.device, fp16: bool):
        self.model = model
        self.device: torch.device = torch.device(device)
        self.dtype = torch.float16 if fp16 else torch.float32
        self._channels_last_failed = False
        self._graph_runner = CudaGraphRunner(
            model,
            self.device,
            enabled=torch.cuda.is_available(),
        )

    def _prepare_inference_view(self, video: list[ImageTensor]):
        inference_view = torch.stack(video, dim=0).permute(0, 3, 1, 2).contiguous()
        inference_view = inference_view.to(device=self.device, dtype=self.dtype).div_(255.0).unsqueeze(0)
        if not self._channels_last_failed:
            tuned = _torch_tuning.to_channels_last_5d(inference_view)
            if tuned is not inference_view:
                inference_view = tuned
        return inference_view

    def _forward_model(self, inputs):
        profile = get_active_profile()
        section = profile.section("model.forward", torch_module=torch, cuda=True) if profile else nullcontext()
        with section:
            try:
                return self._graph_runner(inputs)
            except Exception:
                if not self._channels_last_failed:
                    self._channels_last_failed = True
                    return self._graph_runner(inputs.contiguous())
                raise

    def warmup_graph(self, clip_length: int, *, size: int = 256) -> bool:
        if not self._graph_runner.enabled:
            return False
        clip_length = max(1, int(clip_length))
        inputs = torch.zeros((1, clip_length, 3, size, size), device=self.device, dtype=self.dtype)
        inputs = _torch_tuning.to_channels_last_5d(inputs)
        # Capture is only permitted here (single-threaded warmup), never during
        # concurrent restore/encode. See CudaGraphRunner._capture_allowed.
        with self._graph_runner.allow_capture(), torch.no_grad():
            _ = self._forward_model(inputs)
        return True

    def restore(self, video: list[ImageTensor], max_frames=-1) -> list[ImageTensor]:
        input_frame_count = len(video)
        input_frame_shape = video[0].shape
        with torch.inference_mode():
            result = []
            inference_view = self._prepare_inference_view(video)

            if max_frames > 0:
                for i in range(0, inference_view.shape[1], max_frames):
                    output = self._forward_model(inference_view[:, i:i + max_frames])
                    result.append(output)
                result = torch.cat(result, dim=1)
            else:
                result = self._forward_model(inference_view)

            # (H, W, C[BGR]) uint8 images to (B, T, C, H, W) float in [0,1]
            result = result.squeeze(0)[:input_frame_count] # -> (T, C, H, W)
            result = result.mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1) # (T, H, W, C)
            result = list(torch.unbind(result, 0)) # (T, H, W, C) to list of (H, W, C)
            output_frame_count = len(result)
            output_frame_shape = result[0].shape
            assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape

        return result
