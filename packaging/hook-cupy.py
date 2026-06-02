"""PyInstaller hook that fully collects CuPy and its Cython extension .pyd files.

CuPy has many _core / cuda backend extension modules imported lazily at runtime.
PyInstaller misses them by default, causing frozen imports such as
cupy._core._carray to fail.
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = collect_all("cupy")
hiddenimports += collect_submodules("cupy")
hiddenimports += collect_submodules("cupy_backends")
hiddenimports += collect_submodules("fastrlock")
