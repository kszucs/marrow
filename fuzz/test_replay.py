# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The committed fuzz corpus, replayed: every input must do what
`fuzz/corpus/<target>/expected.toml` says it does.

One `mojo build` of every harness behind a `main()`, then one process per
input. Building it is also the compile check for `fuzz/`, which `precompile`
does not reach. Under `--asan` (the `asan` environment) the driver links the
AddressSanitizer runtime and the entries only a sanitizer catches run too.
See `devkit/fuzzing.py`.
"""

import pytest

from devkit.fuzzing import Expectations, Replay, Targets, judge
from devkit.mojo import AsanRuntime, MojoToolchain, ProcessRunner, Repo, SilentProgress

REPO = Repo.locate(__file__)
TARGETS = Targets(REPO)
ENTRIES = list(Replay(REPO, None).entries())


@pytest.fixture(scope="module")
def replay(request):
    asan = bool(request.config.getoption("--asan", default=False))
    runtime = AsanRuntime.locate() if asan else None
    if asan and runtime is None:
        pytest.fail("--asan needs the ASAN runtime: use the `asan` environment")
    toolchain = MojoToolchain(
        ProcessRunner(REPO.root, SilentProgress(), timeout=1800), runtime
    )
    replay = Replay(REPO, toolchain, asan=asan)
    result = replay.build()
    if toolchain.reports_errors(result):
        pytest.fail(result.failure("the fuzz harnesses do not compile"), pytrace=False)
    return replay


def test_fuzz_harnesses_compile(replay):
    assert replay.binary.exists()


@pytest.mark.parametrize("target", TARGETS.names())
def test_fuzz_corpus_has_verdicts(target):
    assert Expectations.load(TARGETS.corpus(target)).problems() == []


@pytest.mark.parametrize(
    "target,entry", ENTRIES, ids=[f"{t}/{e.name}" for t, e in ENTRIES]
)
def test_fuzz_replay(replay, target, entry):
    reason = entry.skip_reason(replay.asan)
    if reason is not None:
        pytest.skip(reason)
    problem = judge(entry, replay.run(target, entry))
    if problem is not None:
        pytest.fail(problem, pytrace=False)
