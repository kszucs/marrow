import shutil
import subprocess
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ROOT = Path(__file__).parent.parent
SO = Path(__file__).parent / "marrow" / "libmarrow.so"
BUILD_CMD = [
    "mojo",
    "build",
    "-O3",
    "-g0",
    "-I",
    str(ROOT),
    str(Path(__file__).parent / "bindings" / "lib.mojo"),
    "--emit",
    "shared-lib",
    "-o",
    str(SO),
]


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        if not SO.exists():
            if shutil.which("mojo"):
                subprocess.check_call(BUILD_CMD)
            else:
                raise RuntimeError(
                    f"{SO} not found and mojo not in PATH. "
                    "Run `pixi run build_python` first, or install mojo-compiler."
                )

        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        # Every module in the package, not just `__init__.py`: it ends with
        # `from . import compute`, so shipping it alone made `import marrow`
        # raise ImportError in the built wheel. Globbing keeps a new module from
        # being forgotten the same way.
        for module in sorted(SO.parent.glob("*.py")):
            build_data["force_include"][str(module)] = f"marrow/{module.name}"
        build_data["force_include"][str(SO)] = f"marrow/libmarrow{suffix}"
