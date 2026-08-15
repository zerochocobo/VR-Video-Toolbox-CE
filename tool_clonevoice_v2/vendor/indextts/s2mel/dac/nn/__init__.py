"""Neural network components used by the vendored IndexTTS inference path.

Submodules are intentionally not imported eagerly: ``loss`` depends on the
optional descript-audiotools training stack, while inference imports
``layers`` and ``quantize`` directly.
"""
