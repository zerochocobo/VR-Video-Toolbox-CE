# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local-only Depth Anything 3 API subset for 2D->Depth VR.

This intentionally omits DA3's export, app, service, and Hugging Face Hub code.
It loads the local config.json + model.safetensors already placed under
models/DA3/Small and runs depth inference only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor


class DepthAnything3(nn.Module):
    """Minimal local DA3 wrapper with the same inference shape as the upstream API."""

    def __init__(self, model_name: str = "da3-small", config: dict | None = None):
        super().__init__()
        self.model_name = model_name
        self.config = OmegaConf.create(config) if config is not None else load_config(MODEL_REGISTRY[model_name])
        self.model = create_object(self.config)
        self.model.eval()
        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()
        self.device = None

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "DepthAnything3":
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.safetensors"
        if not config_path.exists():
            raise FileNotFoundError(f"DA3 config not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"DA3 weights not found: {weights_path}")

        with config_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        model = cls(model_name=str(data.get("model_name") or "da3-small"), config=data.get("config"))

        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("Missing dependency: safetensors. Run `uv sync` after updating pyproject.toml.") from exc

        state_dict = load_file(str(weights_path), device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) > 20 or len(unexpected) > 20:
            raise RuntimeError(
                f"DA3 weights do not match the model: missing={len(missing)} unexpected={len(unexpected)}"
            )
        return model

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "middle",
        skip_camera: bool = False,
        skip_sky: bool = False,
    ) -> dict[str, torch.Tensor]:
        export_feat_layers = export_feat_layers or []
        if image.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                return self.model(
                    image, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy,
                    skip_camera=skip_camera, skip_sky=skip_sky,
                )
        return self.model(
            image, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy,
            skip_camera=skip_camera, skip_sky=skip_sky,
        )

    def inference(
        self,
        image: list[np.ndarray],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "middle",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: Sequence[int] | None = None,
        **_unused,
    ) -> Prediction:
        imgs_cpu, ex_t, in_t = self._preprocess_inputs(
            image, extrinsics, intrinsics, process_res, process_res_method
        )
        imgs, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, ex_t, in_t)
        ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
        raw_output = self.forward(
            imgs,
            ex_t_norm,
            in_t,
            list(export_feat_layers or []),
            infer_gs,
            use_ray_pose,
            ref_view_strategy,
        )
        prediction = self.output_processor(raw_output)
        prediction = self._align_to_input_extrinsics_intrinsics(
            ex_t, in_t, prediction, align_to_input_ext_scale
        )
        prediction.processed_images = self._processed_images_to_numpy(imgs_cpu)
        return prediction

    def inference_depth_only(
        self,
        image,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        use_gpu_preprocess: bool | None = None,
    ) -> torch.Tensor:
        """Return DA3 depth maps on the model device without CPU post-processing.

        ``image`` may be either a ``list[np.ndarray]`` of CPU RGB frames or a
        ``torch.Tensor`` of shape ``(B, H, W, 3)`` uint8 already on CUDA (PyNv
        zero-copy path).
        """
        device = self._get_model_device()

        is_tensor_batch = isinstance(image, torch.Tensor)
        batch_count = int(image.shape[0]) if is_tensor_batch else len(image)
        if batch_count <= 0:
            return torch.empty((0, 0, 0), device=device, dtype=torch.float32)

        if use_gpu_preprocess is None:
            raw = os.environ.get("TOOL_2DVR_GPU_PREPROCESS", "1").strip().lower()
            use_gpu_preprocess = raw not in {"0", "false", "no", "off", "cpu"}
        imgs = None
        cpu_list_path = isinstance(image, list) and all(isinstance(item, np.ndarray) for item in image)
        if use_gpu_preprocess and device.type == "cuda" and (is_tensor_batch or cpu_list_path):
            try:
                from depth_anything_3.utils.io.input_processor_gpu import gpu_preprocess

                imgs = gpu_preprocess(
                    image,
                    device=device,
                    target_res=process_res,
                    process_res_method=process_res_method,
                )
            except Exception:
                imgs = None

        if imgs is None:
            if is_tensor_batch:
                # Fallback for tensor input when GPU preprocess fails: go through CPU pipeline.
                cpu_frames = [np.ascontiguousarray(frame.detach().cpu().numpy()) for frame in image]
            else:
                cpu_frames = list(image)
            imgs_cpu, ex_t, in_t = self._preprocess_inputs(
                cpu_frames, None, None, process_res, process_res_method
            )
            imgs, _, _ = self._prepare_model_inputs(imgs_cpu, ex_t, in_t)

        raw_output = self.forward(
            imgs,
            None,
            None,
            [],
            False,
            False,
            "middle",
            skip_camera=True,
            skip_sky=True,
        )
        depth = self._depth_tensor_from_output(raw_output)
        if depth.shape[0] != batch_count:
            raise RuntimeError(f"DA3 returned {depth.shape[0]} depth maps for {batch_count} inputs")
        return depth.contiguous()

    def _preprocess_inputs(
        self,
        image: list[np.ndarray],
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
        process_res: int,
        process_res_method: str,
    ):
        return self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
            num_workers=1,
            sequential=True,
            print_progress=False,
        )

    def _prepare_model_inputs(self, imgs_cpu, extrinsics, intrinsics):
        device = self._get_model_device()
        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()
        ex_t = extrinsics.to(device, non_blocking=True)[None].float() if extrinsics is not None else None
        in_t = intrinsics.to(device, non_blocking=True)[None].float() if intrinsics is not None else None
        return imgs, ex_t, in_t

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        dists = c2ws[..., :3, 3].norm(dim=-1)
        median_dist = torch.clamp(torch.median(dists), min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def _align_to_input_extrinsics_intrinsics(
        self,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        prediction: Prediction,
        align_to_input_ext_scale: bool = True,
    ) -> Prediction:
        if extrinsics is None:
            return prediction
        from depth_anything_3.utils.pose_align import align_poses_umeyama

        prediction.intrinsics = intrinsics.cpu().numpy() if intrinsics is not None else None
        _, _, scale, aligned = align_poses_umeyama(
            prediction.extrinsics,
            extrinsics.cpu().numpy(),
            ransac=len(extrinsics) >= 10,
            return_aligned=True,
            random_state=42,
        )
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].cpu().numpy()
            prediction.depth /= scale
        else:
            prediction.extrinsics = aligned
        return prediction

    @staticmethod
    def _processed_images_to_numpy(imgs_cpu: torch.Tensor) -> np.ndarray:
        imgs = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        imgs = np.clip(imgs * std + mean, 0, 1)
        return (imgs * 255).astype(np.uint8)

    @staticmethod
    def _depth_tensor_from_output(model_output: dict[str, torch.Tensor]) -> torch.Tensor:
        depth = model_output["depth"]
        if depth.ndim == 5 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim == 4 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        if depth.ndim != 3:
            raise RuntimeError(f"Unexpected DA3 depth shape: {tuple(depth.shape)}")
        return depth.float()

    def _get_model_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        for param in self.parameters():
            self.device = param.device
            return param.device
        raise ValueError("No tensor found in DA3 model")
