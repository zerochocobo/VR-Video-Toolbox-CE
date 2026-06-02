"""
utils/engine_runner.py
Mosaic-removal engine command builder layer.

Supported engines:
  - lada   : lada-cli  (https://codeberg.org/ladaapp/lada)
  - jasna  : jasna-cli (https://github.com/Kruk2/jasna)

Public API
----------
build_engine_cmd(input_file, output_file,
                 mosaic_model_path=None,
                 mosaic_restoration_model_path=None) -> list[str]

get_engine_executable() -> str   returns the current engine executable name

Parameter mapping differences (Lada -> Jasna)
---------------------------------------------
Lada parameter                    Jasna counterpart        Notes
--encoder hevc_nvenc              (removed)                Jasna fixes the encoder internally
--encoding-preset ...             (removed)                Jasna has no preset system
--encoder-options "-cq 18 ..."    --encoder-settings cq=18 keep only the cq value
--mosaic-detection-model <p>    --detection-model-path <p>
--mosaic-restoration-model <p>  --restoration-model-path <p>
"""
import re
import shlex
from . import app_config


def get_engine_executable() -> "str | None":
    """Return the current engine CLI executable name; native_gpu is in-process and returns None."""
    engine = app_config.get_engine()
    if engine == 'native_gpu':
        return None
    return 'jasna-cli' if engine == 'jasna' else 'lada-cli'


def is_native_engine() -> bool:
    """Return whether the current engine is the built-in native_gpu in-process engine."""
    return app_config.get_engine() == 'native_gpu'


def _extract_cq(encoder_options_str: str) -> str | None:
    """Extract the cq value from Lada's --encoder-options string."""
    m = re.search(r'-cq\s+(\d+)', encoder_options_str or '')
    return m.group(1) if m else None


def get_configured_nvenc_preset(default: str = "P7") -> str:
    """Return configured NVENC preset as P1..P7."""
    preset = str(app_config.get("gpu_encode_preset", default) or default).upper()
    return preset if preset in {f"P{i}" for i in range(1, 8)} else default


def build_lada_encoder_options(cq: int | str = 18) -> str:
    """Build Lada --encoder-options using the shared UI NVENC preset."""
    preset = get_configured_nvenc_preset().lower()
    return f" -cq {cq} -preset {preset}"


def build_engine_cmd(
    input_file: str,
    output_file: str,
    mosaic_model_path: str | None = None,
    mosaic_restoration_model_path: str | None = None,
    encoder_options: str | None = None,
) -> list | None:
    """
    Build the mosaic-removal CLI command list from the current engine config.

    Parameters
    ----------
    input_file : str
    output_file : str
    mosaic_model_path : str | None
        Lada: --mosaic-detection-model
        Jasna: --detection-model-path
    mosaic_restoration_model_path : str | None
        Lada: --mosaic-restoration-model
        Jasna: --restoration-model-path
    encoder_options : str | None
        Lada-style encoding options string, for example " -cq 18 -preset p4 ...".
        In Jasna mode, the cq value is extracted and converted automatically.

    Returns
    -------
    list[str] | None
    """
    engine = app_config.get_engine()

    if engine == 'native_gpu':
        # native_gpu is an in-process engine with no CLI command; callers should use gpu_engine.native_mosaic.
        return None
    if engine == 'jasna':
        return _build_jasna_cmd(input_file, output_file,
                                mosaic_model_path,
                                mosaic_restoration_model_path,
                                encoder_options)
    else:
        return _build_lada_cmd(input_file, output_file,
                               mosaic_model_path,
                               mosaic_restoration_model_path,
                               encoder_options)


def _build_lada_cmd(input_file, output_file,
                    mosaic_model_path, mosaic_restoration_model_path,
                    encoder_options) -> list:
    cmd = [
        'lada-cli',
        '--input', input_file,
        '--output', output_file,
        '--encoder', 'hevc_nvenc',
    ]
    if encoder_options:
        cmd.extend(['--encoder-options', encoder_options])
    else:
        # Default high-quality preset.
        cmd.extend(['--encoding-preset', 'hevc-nvidia-gpu-hq'])

    if mosaic_model_path:
        cmd.extend(['--mosaic-detection-model', mosaic_model_path])
    if mosaic_restoration_model_path:
        cmd.extend(['--mosaic-restoration-model', mosaic_restoration_model_path])

    custom_args = app_config.get_custom_args('lada')
    if custom_args:
        cmd.extend(shlex.split(custom_args))
        
    return cmd


def _build_jasna_cmd(input_file, output_file,
                     mosaic_model_path, mosaic_restoration_model_path,
                     encoder_options) -> list:
    cmd = [
        'jasna-cli',
        '--input', input_file,
        '--output', output_file,
    ]

    # Extract the cq value and convert it to Jasna's --encoder-settings.
    cq = _extract_cq(encoder_options) if encoder_options else '18'
    if cq:
        cmd.extend(['--encoder-settings', f'cq={cq}'])

    # Model path mapping.
    if mosaic_model_path:
        cmd.extend(['--detection-model-path', mosaic_model_path])
    if mosaic_restoration_model_path:
        cmd.extend(['--restoration-model-path', mosaic_restoration_model_path])

    custom_args = app_config.get_custom_args('jasna')
    if custom_args:
        cmd.extend(shlex.split(custom_args))

    return cmd


def get_mosaic_tool_name() -> str:
    """Return the current engine display name for logs."""
    engine = app_config.get_engine()
    if engine == 'native_gpu':
        return 'NativeGPU'
    return 'Jasna' if engine == 'jasna' else 'Lada'
