__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

# IndexTTS inference imports only dac.nn.quantize.  The upstream package
# eagerly imports its training and standalone codec APIs here, which makes
# descript-audiotools mandatory even though those APIs are never used by the
# s2mel inference path.  Keep this package initializer lightweight so the
# vendored inference code can share the application's audio dependency stack.
