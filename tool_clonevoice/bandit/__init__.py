"""Vendored bandit-v2 (cinematic source separation) inference code.

Trimmed to the inference path only: the Bandit model and the chunked tensor
inference handler. Used by ``tool_clonevoice.separate`` to strip the original
dialogue from a video so the cloned/translated voice can be dubbed over the
remaining music+sfx bed.

Upstream: https://github.com/kwatcharasupat/bandit-v2 (Apache-2.0).
"""
