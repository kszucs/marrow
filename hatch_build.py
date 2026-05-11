"""
Hatchling build hook for marrow.

Compilation is driven by pixi (via cibuildwheel's before-build step or
`pixi run build_python` locally). This hook copies the pre-built shared
library into the wheel alongside the Python package, with the correct
CPython-suffixed name.

Local usage:
    pixi run build_python
    pip wheel --no-build-isolation .
Or via the pixi task:
    pixi run -e wheel build_wheel
"""

import shutil
import subprocess
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PREBUILT_SO = Path("python/libmarrow.so")


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        # Binary wheel: hatchling infers platform tag from running interpreter.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        if not PREBUILT_SO.exists():
            pixi = shutil.which("pixi")
            if pixi:
                subprocess.check_call([pixi, "run", "build_python"])

        if not PREBUILT_SO.exists():
            raise RuntimeError(
                f"{PREBUILT_SO} not found. Run `pixi run build_python` first."
            )

        # Place both the __init__.py and the compiled extension inside
        # the marrow/ package in the wheel.
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        build_data["force_include"]["python/__init__.py"] = "marrow/__init__.py"
        build_data["force_include"][str(PREBUILT_SO)] = f"marrow/libmarrow{suffix}"
