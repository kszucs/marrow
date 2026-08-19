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

        # `marrow compile` needs marrow's own Mojo source to pass as `-I` to
        # `mojo build` for an installed (pip) user — resolve_marrow_path()'s
        # third resolution step looks for it at `marrow/_mojo/marrow/...`
        # inside the installed package. Ship source, not a precompiled
        # `.mojoc`: it is smaller (1.68 MB vs. 5.6 MB) and, unlike a
        # `.mojoc`, tolerates a compiler version that has drifted from the
        # exact pin. Tests, benchmarks and profiles are excluded — they are
        # not needed to build a user's query and only add weight.
        mojo_root = ROOT / "marrow"
        for source in sorted(mojo_root.rglob("*.mojo")):
            rel = source.relative_to(ROOT)
            if "tests" in rel.parts:
                continue
            if source.name.startswith("bench_") or source.name.startswith("profile_"):
                continue
            build_data["force_include"][str(source)] = f"marrow/_mojo/{rel}"
