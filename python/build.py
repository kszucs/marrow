"""The hatchling build hook: compile `libmarrow.so` and lay out the wheel.

The shared library is built by `devkit`, not by a recipe of its own -- the flags
and the paths have exactly one definition, in `BuildOptions.for_shared_lib` and
`Repo`, and this hook is one more caller of them.  That is why `devkit.mojo`
imports nothing but the standard library: cibuildwheel builds each wheel in a
fresh environment holding `hatchling` and the Mojo compiler and nothing else, so
anything reaching for `rich` or `psutil` at import time could not be used here.

The repository root is available because cibuildwheel copies the whole checkout
and builds `python/` inside it -- the same reason `mojo build -I <root>` can find
marrow's Mojo sources at all.
"""

import importlib.util
import shutil
import sys
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from devkit.mojo import (  # noqa: E402 - must follow the path insertion
    BuildOptions,
    MojoToolchain,
    ProcessRunner,
    Repo,
)

try:
    from devkit.progress import ConsoleProgress as Progress  # noqa: E402
except ImportError:  # no rich, no psutil -- the cibuildwheel environment
    from devkit.mojo import SilentProgress as Progress  # noqa: E402


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        repo = Repo(ROOT)
        if not repo.libmarrow.exists():
            self._build(repo)

        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        # Every module in the package, not just `__init__.py`: it ends with
        # `from . import compute`, so shipping it alone made `import marrow`
        # raise ImportError in the built wheel. Globbing keeps a new module from
        # being forgotten the same way.
        for module in sorted(repo.libmarrow.parent.glob("*.py")):
            build_data["force_include"][str(module)] = f"marrow/{module.name}"
        build_data["force_include"][str(repo.libmarrow)] = f"marrow/libmarrow{suffix}"

        # The `dlopen`-ed libraries, beside the extension. `delocate`/`auditwheel`
        # cannot find these for us: they walk load commands, and a `dlopen`-ed
        # library has none -- the same blind spot `compile.py` documents for a
        # `--bundle` directory. Without this a pip-installed marrow cannot read a
        # zstd-compressed Parquet file, let alone an `s3://` one.
        # `marrow/_dylibs.py` is the other half: it points the Mojo loader here.
        for lib in self._dlopen_libs():
            build_data["force_include"][str(lib)] = f"marrow/{lib.name}"

        # `marrow compile` needs marrow's own Mojo source to pass as `-I` to
        # `mojo build` for an installed (pip) user — resolve_marrow_path()'s
        # third resolution step looks for it at `marrow/_mojo/marrow/...`
        # inside the installed package. Ship source, not a precompiled
        # `.mojoc`: it is smaller (1.68 MB vs. 5.6 MB) and, unlike a
        # `.mojoc`, tolerates a compiler version that has drifted from the
        # exact pin. Tests, benchmarks and profiles are excluded — they are
        # not needed to build a user's query and only add weight.
        for source in sorted(repo.package_dir.rglob("*.mojo")):
            rel = source.relative_to(repo.root)
            if "tests" in rel.parts:
                continue
            if source.name.startswith("bench_") or source.name.startswith("profile_"):
                continue
            build_data["force_include"][str(source)] = f"marrow/_mojo/{rel}"

    @staticmethod
    def _dlopen_libs():
        """Every optional C library marrow may `dlopen`, with its own
        dependency closure, resolved from the build environment.

        `compile.py` is loaded from its path rather than imported as
        `marrow.compile`, which would run the package `__init__` and with it
        `from . import libmarrow`. The extension exists by now, but importing
        it here would make laying out the wheel depend on the extension being
        loadable by the *building* interpreter -- which under cross-compilation
        it is not. `compile.py` imports nothing but the standard library, so
        loading it alone is well defined.

        A missing library is a warning inside those helpers, never a raise: a
        wheel built without OpenDAL is a wheel that reads local files, which is
        the supported configuration today.
        """
        spec = importlib.util.spec_from_file_location(
            "_marrow_compile", ROOT / "python" / "marrow" / "compile.py"
        )
        compile_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compile_mod)

        staged = {}
        libs = (
            compile_mod.stage_codec_libs(compile_mod.codec_lib_dir())
            + compile_mod.optional_lib_paths()
        )
        for lib in libs:
            # manylinux guarantees these two, and a wheel carrying a conda
            # copy would hide from `auditwheel` whether the codecs fit the
            # policy's GLIBCXX. `devkit wheel check` refuses them.
            if lib.name.split(".", 1)[0] in ("libstdc++", "libgcc_s"):
                continue
            staged.setdefault(lib.name, lib)
        return list(staged.values())

    @staticmethod
    def _build(repo):
        """Compile the bindings, the same way `devkit build lib` does.

        `bench=True` because a published wheel is the optimized artifact: the
        development build takes -O1 to keep the edit-compile loop short, and
        that trade is wrong for something a user installs.
        """
        if not shutil.which("mojo"):
            raise RuntimeError(
                f"{repo.libmarrow} not found and mojo not in PATH. "
                "Run `pixi run build_python` first, or install mojo-compiler."
            )
        toolchain = MojoToolchain(ProcessRunner(repo.root, Progress()))
        result = toolchain.build_shared_lib(
            repo.bindings_entry,
            repo.libmarrow,
            BuildOptions.for_shared_lib(bench=True),
            "building libmarrow.so",
        )
        if not result.ok:
            raise RuntimeError(result.failure(f"Failed to build {repo.libmarrow}"))
