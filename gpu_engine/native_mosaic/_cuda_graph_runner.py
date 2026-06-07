"""CUDA Graph wrapper for fixed-shape BasicVSR++ forward calls."""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


_FALSE_VALUES = {"0", "false", "no", "off"}


def cuda_graph_enabled() -> bool:
    # Default OFF: BasicVSR++ (deform_conv2d, SPyNet) has triggered native
    # fast-fail (Windows STATUS_STACK_BUFFER_OVERRUN 0xc0000409) during graph
    # capture/replay. The Python try/except in CudaGraphRunner cannot catch
    # those — they bypass the interpreter. Opt in with VRVT_CUDA_GRAPH=1 once
    # the supported-op surface is confirmed stable for the current torch build.
    return str(os.environ.get("VRVT_CUDA_GRAPH", "0")).strip().lower() not in _FALSE_VALUES


@dataclass
class _Entry:
    graph: Any
    static_input: Any
    static_output: Any


class CudaGraphRunner:
    """Capture model forward once per input shape, replay on subsequent calls."""

    def __init__(self, model, device, *, warmup_iters: int = 3,
                 enabled: bool = True, max_entries: int = 2):
        self.model = model
        self.device = device
        self.warmup_iters = max(0, int(warmup_iters))
        self.enabled = bool(enabled) and cuda_graph_enabled()
        self.max_entries = max(1, int(max_entries))
        self._cache: OrderedDict[tuple, _Entry] = OrderedDict()
        self._disabled_keys: set[tuple] = set()
        self._lock = threading.Lock()
        self._graph_stream = None
        self.capture_failures = 0
        self.captures = 0
        self.replays = 0

    @staticmethod
    def cache_key(inputs) -> tuple:
        return (
            tuple(inputs.shape),
            str(inputs.dtype),
            tuple(inputs.stride()),
            str(inputs.device),
        )

    def __call__(self, inputs):
        if not self.enabled or not getattr(inputs, "is_cuda", False):
            return self.model(inputs=inputs)

        import torch

        key = self.cache_key(inputs)
        with self._lock:
            if key in self._disabled_keys:
                return self.model(inputs=inputs)
            entry = self._cache.get(key)
            if entry is None:
                try:
                    entry = self._capture(inputs)
                    self._cache[key] = entry
                    self._cache.move_to_end(key)
                    self.captures += 1
                    while len(self._cache) > self.max_entries:
                        self._cache.popitem(last=False)
                except Exception:
                    self.capture_failures += 1
                    self._disabled_keys.add(key)
                    self._cache.pop(key, None)
                    return self.model(inputs=inputs)
            else:
                self._cache.move_to_end(key)

            try:
                graph_stream = self._get_graph_stream(torch)
                current_stream = torch.cuda.current_stream(self.device)
                graph_stream.wait_stream(current_stream)
                with torch.cuda.stream(graph_stream):
                    entry.static_input.copy_(inputs, non_blocking=True)
                    entry.graph.replay()
                current_stream.wait_stream(graph_stream)
                self.replays += 1
                return entry.static_output.clone()
            except Exception:
                self._disabled_keys.add(key)
                self._cache.pop(key, None)
                return self.model(inputs=inputs)

    def _get_graph_stream(self, torch):
        if self._graph_stream is None:
            self._graph_stream = torch.cuda.Stream(device=self.device)
        return self._graph_stream

    def _capture(self, inputs):
        import torch

        if not hasattr(torch.cuda, "CUDAGraph"):
            raise RuntimeError("torch.cuda.CUDAGraph is unavailable")

        static_input = torch.empty_strided(
            tuple(inputs.shape),
            tuple(inputs.stride()),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        static_input.copy_(inputs, non_blocking=True)

        graph_stream = self._get_graph_stream(torch)
        current_stream = torch.cuda.current_stream(self.device)
        graph_stream.wait_stream(current_stream)
        with torch.cuda.stream(graph_stream):
            with torch.inference_mode(False), torch.no_grad():
                for _ in range(self.warmup_iters):
                    _ = self.model(inputs=static_input)
        current_stream.wait_stream(graph_stream)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(graph_stream):
            with torch.inference_mode(False), torch.no_grad():
                with torch.cuda.graph(graph, stream=graph_stream):
                    static_output = self.model(inputs=static_input)
        return _Entry(graph=graph, static_input=static_input, static_output=static_output)
