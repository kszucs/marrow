# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""`Glob`: matching keys, and expanding a pattern over a local directory and
through OpenDAL's listing."""

from std.os import makedirs, remove, rmdir
from std.os.path import join
from std.testing import assert_equal, assert_false, assert_raises, assert_true

from ...utils.testing import ScratchDir
from ..glob import Glob
from ..opendal import OpenDalStore
from ..uri import Uri


def _touch(path: String) raises:
    with open(path, "w") as f:
        f.write("")


def test_glob_star_stays_in_one_segment() raises:
    var g = Glob("main/train-*")
    assert_true(g.matches("main/train-00000-of-00001.parquet"))
    assert_false(g.matches("main/sub/train-0.parquet"))
    assert_false(Glob("*.parquet").matches("data/a.parquet"))
    assert_true(Glob("data/*.parquet").matches("data/a.parquet"))


def test_glob_double_star_spans_segments() raises:
    assert_true(Glob("**/*.parquet").matches("a.parquet"))
    assert_true(Glob("**/*.parquet").matches("x/y/a.parquet"))
    assert_true(Glob("data/**").matches("data/x/y.json"))
    assert_false(Glob("data/**/*.json").matches("other/x.json"))


def test_glob_question_mark_is_one_character() raises:
    assert_true(Glob("part-?.json").matches("part-1.json"))
    assert_false(Glob("part-?.json").matches("part-12.json"))
    assert_false(Glob("a?b").matches("a/b"))


def test_glob_escape_matches_only_that_path() raises:
    var g = Glob(Glob.escape("a*b?.parquet"))
    assert_true(g.matches("a*b?.parquet"))
    assert_false(g.matches("axxbz.parquet"))


def test_glob_expands_a_local_directory() raises:
    with ScratchDir() as dir:
        makedirs(join(dir, "data", "nested"))
        _touch(join(dir, "data", "b.parquet"))
        _touch(join(dir, "data", "a.parquet"))
        _touch(join(dir, "data", "notes.txt"))
        _touch(join(dir, "data", "nested", "c.parquet"))
        var flat = Glob(join(dir, "data", "*.parquet")).expand()
        assert_equal(len(flat), 2)
        assert_equal(flat[0], join(dir, "data", "a.parquet"))
        assert_equal(flat[1], join(dir, "data", "b.parquet"))
        var deep = Glob(join(dir, "data", "**", "*.parquet")).expand()
        assert_equal(len(deep), 3)
        assert_equal(deep[2], join(dir, "data", "nested", "c.parquet"))
        # `ScratchDir` removes one level only.
        for ref f in deep:
            remove(f)
        remove(join(dir, "data", "notes.txt"))
        rmdir(join(dir, "data", "nested"))
        rmdir(join(dir, "data"))


def test_glob_expands_through_opendal() raises:
    """`fs://` goes through OpenDAL's listing, which every remote scheme
    shares; skipped when the library or its `fs` service is absent. A name
    with `%` comes back escaped, so parsing the URI reads that file."""
    try:
        _ = OpenDalStore("fs", {"root": "/"})
    except:
        return
    with ScratchDir() as dir:
        _touch(join(dir, "b.parquet"))
        _touch(join(dir, "a%20.parquet"))
        _touch(join(dir, "c.json"))
        var got = Glob(String("fs://", dir, "/*.parquet")).expand()
        assert_equal(len(got), 2)
        assert_equal(got[0], String("fs://", dir, "/a%2520.parquet"))
        assert_equal(
            Uri.parse(got[0]).path, String(dir[byte=1:], "/a%20.parquet")
        )
        assert_equal(got[1], String("fs://", dir, "/b.parquet"))


def test_glob_question_mark_expands_in_a_path() raises:
    with ScratchDir() as dir:
        _touch(join(dir, "part-1.json"))
        _touch(join(dir, "part-12.json"))
        var got = Glob(join(dir, "part-?.json")).expand()
        assert_equal(len(got), 1)
        assert_equal(got[0], join(dir, "part-1.json"))


def test_glob_url_with_a_query_is_one_file() raises:
    var url = String("https://example.com/x.parquet?download=true")
    var got = Glob(url).expand()
    assert_equal(len(got), 1)
    assert_equal(got[0], url)


def test_glob_literal_is_returned_unchecked() raises:
    var got = Glob("/nowhere/at/all.parquet").expand()
    assert_equal(len(got), 1)
    assert_equal(got[0], "/nowhere/at/all.parquet")


def test_glob_matching_nothing_raises() raises:
    with ScratchDir() as dir:
        with assert_raises(contains="no files match"):
            _ = Glob(join(dir, "*.parquet")).expand()
