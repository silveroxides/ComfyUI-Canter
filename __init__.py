if __package__:
    from .nodes import comfy_entrypoint
else:
    import importlib
    import sys
    import types
    from pathlib import Path

    _package_name = "_comfyui_canter_runtime"
    _package = types.ModuleType(_package_name)
    _package.__path__ = [str(Path(__file__).parent)]
    sys.modules.setdefault(_package_name, _package)
    comfy_entrypoint = importlib.import_module(f"{_package_name}.nodes").comfy_entrypoint

__all__ = ["comfy_entrypoint"]
