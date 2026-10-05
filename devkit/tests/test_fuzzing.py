# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The fuzzing tooling: targets, generated entry points, crash signatures and
the corpus verdicts. Nothing here compiles or fuzzes."""

import pytest

from devkit.fuzzing import (
    Clang,
    BitcodeVersionError,
    PLATFORM,
    Entry,
    Expectations,
    Fuzzer,
    Outcome,
    Replay,
    RunOptions,
    Targets,
    entry_source,
    judge,
    replay_source,
    signature,
)
from devkit.mojo import BuildOptions, CommandResult, Repo


def scratch_repo(tmp_path):
    (tmp_path / "pytest.ini").write_text("")
    fuzz = tmp_path / "fuzz"
    fuzz.mkdir()
    (fuzz / "alpha.mojo").write_text(
        "def fuzz_one(data: Span[UInt8, _]) raises:\n    pass\n"
    )
    (fuzz / "common.mojo").write_text("def helper():\n    pass\n")
    return Repo(tmp_path)


def result(returncode=0, stdout="", stderr=""):
    return CommandResult(("x",), returncode, stdout, stderr, 0.0)


def test_targets_are_files_defining_fuzz_one(tmp_path):
    targets = Targets(scratch_repo(tmp_path))
    assert targets.names() == ["alpha"]
    with pytest.raises(ValueError, match="targets are: alpha"):
        targets.resolve("common")


def test_the_tree_has_targets():
    assert "ipc_stream" in Targets(Repo.locate()).names()


def test_entry_routes_mojo_heap_through_libc_before_any_input():
    source = entry_source("ipc_stream")
    assert "from ipc_stream import fuzz_one" in source
    initialize = source.index("LLVMFuzzerInitialize")
    assert source.index("KGEN_CompilerRT_SetAsanAllocators") > initialize
    assert source.index("KGEN_CompilerRT_SetAsanAllocators") < source.index(
        "LLVMFuzzerTestOneInput"
    )


def test_replay_dispatches_to_every_target():
    source = replay_source(["a", "b"])
    assert "from a import fuzz_one as a" in source
    assert 'if target == "a":' in source
    assert 'elif target == "b":' in source


def test_binaries_are_named_for_their_platform(tmp_path):
    """A host and a container sharing one tree must not run each other's."""
    repo = scratch_repo(tmp_path)
    plain, asan = Replay(repo, None).binary, Replay(repo, None, asan=True).binary
    fuzzer = Fuzzer(repo, None, None, None).binary("alpha")
    assert plain != asan
    assert all(path.name.endswith(PLATFORM) for path in (plain, asan, fuzzer))


def test_fuzzing_bitcode_is_asserted_and_marked_for_asan():
    flags = BuildOptions.for_fuzzing().flags()
    assert "ASSERT=all" in flags
    assert flags[flags.index("--sanitize") + 1] == "address"
    assert flags.count("-I") == 2 and "fuzz" in flags
    # Marked, not linked: no runtime is needed to emit bitcode.
    assert "--shared-libasan" not in flags


def test_replay_build_is_asserted_and_unsanitized():
    flags = BuildOptions.for_fuzz_replay().flags()
    assert "ASSERT=all" in flags
    assert "--sanitize" not in flags


def test_fork_mode_survives_crashes():
    assert "-ignore_crashes=1" in RunOptions(jobs=4).flags("out")
    assert not any(flag.startswith("-fork") for flag in RunOptions().flags("out"))


def test_a_too_old_clang_is_reported_as_such(tmp_path):
    class Runner:
        def run(self, argv, label, env=None):
            return result(
                1,
                stderr="error: Unknown attribute kind (102) (Producer: "
                "'LLVM24.0.0git' Reader: 'LLVM APPLE_1_1700.6.4.2_0')",
            )

    with pytest.raises(BitcodeVersionError, match="LLVM24.0.0git"):
        Clang("/usr/bin/clang", Runner()).link_fuzzer("a.bc", "a", tmp_path, "x")


ASSERT_OUTPUT = """\
At: ./marrow/ipc.mojo:2007:30: Assert Error: index 0 is out of bounds, valid range is 0 to -1
==1== ERROR: libFuzzer: deadly signal
    #7 0x1 in marrow::ipc::_BatchDecoder::read_array(marrow::ipc::_BatchDecoder) dtypes.mojo
SUMMARY: libFuzzer: deadly signal
"""

ASAN_OUTPUT = """\
==1==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6 at pc 0x1
READ of size 1 at 0x6 thread T0
    #0 0x1 in std::memory::load(x) memory.mojo:12
    #1 0x2 in marrow::arrays::BinaryLikeArray::write_to[W](a) arrays.mojo:1145
SUMMARY: AddressSanitizer: heap-buffer-overflow arrays.mojo:1145 in write_to
"""


def test_an_assert_is_bucketed_by_message_and_line():
    assert signature(ASSERT_OUTPUT) == (
        "index N is out of bounds, valid range is N to N @ marrow/ipc.mojo:2007"
    )


def test_a_sanitizer_report_is_bucketed_by_kind_and_first_marrow_frame():
    assert signature(ASAN_OUTPUT) == (
        "asan: heap-buffer-overflow @ marrow::arrays::BinaryLikeArray::write_to "
        "arrays.mojo:1145"
    )


def write_corpus(directory, manifest, files):
    directory.mkdir(parents=True)
    (directory / "expected.toml").write_text(manifest)
    for name in files:
        (directory / name).write_bytes(b"x")


def test_every_corpus_file_needs_a_verdict(tmp_path):
    write_corpus(
        tmp_path / "c",
        '["a.bin"]\nverdict = "accept"\n["gone.bin"]\nverdict = "reject"\n',
        ["a.bin", "stray.bin"],
    )
    assert Expectations.load(tmp_path / "c").problems() == [
        "gone.bin: listed in expected.toml but missing",
        "stray.bin: no verdict in expected.toml",
    ]


def test_an_unknown_verdict_is_refused(tmp_path):
    write_corpus(tmp_path / "c", '["a.bin"]\nverdict = "maybe"\n', ["a.bin"])
    with pytest.raises(ValueError, match="maybe"):
        Expectations.load(tmp_path / "c")


def test_outcomes_follow_the_replay_driver():
    assert Outcome.of(result(0, "accept\n")).verdict == "accept"
    assert Outcome.of(result(0, "reject: CorruptError: x\n")).verdict == "reject"
    assert Outcome.of(result(-6, "", "ABORT: boom")).verdict == "crash"


def test_a_fixed_bug_fails_the_replay_until_its_verdict_moves(tmp_path):
    entry = Entry(tmp_path / "b1.bin", "crash", bug="B1", match="out of bounds")
    assert judge(entry, Outcome("crash", "index 9 out of bounds")) is None
    assert "B1" in judge(entry, Outcome("reject", "reject: CorruptError"))
    assert "no longer" in judge(entry, Outcome("crash", "ABORT: other"))


def test_an_anchor_that_crashes_is_a_regression(tmp_path):
    entry = Entry(tmp_path / "ok.arrows", "accept")
    assert judge(entry, Outcome("accept", "accept")) is None
    assert "expected accept, got crash" in judge(entry, Outcome("crash", ASSERT_OUTPUT))
