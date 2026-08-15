"""Vendored base class for bandit-v2 models.

Upstream subclassed ``pytorch_lightning.LightningModule`` here, but the
inference path only needs plain ``nn.Module`` behaviour, so we drop the
Lightning dependency to keep the frozen build lean.
"""
from torch import nn


class BaseEndToEndModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
