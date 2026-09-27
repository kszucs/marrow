# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Test errors.mojo: typed errors, the `DynError` box, and bare frames."""
from std.testing import assert_equal, assert_false, assert_true

from ..errors import (
    ArrowError,
    CorruptError,
    DynError,
    IndexError,
    IOError,
    InternalError,
    InvalidError,
    KeyError,
    NotImplementedError,
    TypeError,
)


def _find(name: String) raises KeyError -> Int:
    raise KeyError(t"column '{name}' not found")


def _check(n: Int) raises TypeError -> Int:
    if n < 0:
        raise TypeError(t"expected a count, got {n}")
    return n


def _plan(name: String, n: Int) raises DynError -> Int:
    return _check(n) + _find(name)


def _bare(name: String) raises -> Int:
    return _find(name)


def _decode(text: String) raises CorruptError -> Int:
    try:
        return atol(text)
    except e:
        raise CorruptError(e)


def _round_trips[E: ArrowError]() raises:
    try:
        raise E(t"row {7} is bad")
    except e:
        var err = DynError(e)
        assert_true(err.isa[E]())
        assert_equal(err.message, "row 7 is bad")
        assert_equal(String(e), String(E.kind, ": row 7 is bad"))


def test_error_typed_catch_reads_the_message() raises:
    try:
        _ = _find("x")
    except e:
        assert_equal(e.message(), "column 'x' not found")
        assert_equal(String(e), "KeyError: column 'x' not found")


def test_error_box_tells_kinds_apart() raises:
    try:
        _ = _plan("x", 1)
    except e:
        assert_true(e.isa[KeyError]())
        assert_false(e.isa[TypeError]())
    try:
        _ = _plan("x", -1)
    except e:
        assert_true(e.isa[TypeError]())
        assert_equal(e.message, "expected a count, got -1")


def test_error_kind_survives_a_bare_frame() raises:
    try:
        _ = _bare("y")
    except e:
        assert_true(DynError(e).isa[KeyError]())
        assert_equal(DynError(e).message, "column 'y' not found")


def test_error_every_kind_round_trips() raises:
    _round_trips[InvalidError]()
    _round_trips[TypeError]()
    _round_trips[KeyError]()
    _round_trips[IndexError]()
    _round_trips[NotImplementedError]()
    _round_trips[IOError]()
    _round_trips[CorruptError]()
    _round_trips[InternalError]()


def test_error_untagged_is_kindless() raises:
    var err = DynError(Error("no tag here: at all"))
    assert_equal(err.kind, "")
    assert_equal(String(err), "no tag here: at all")


def test_error_wraps_a_stdlib_error() raises:
    try:
        _ = _decode("x")
    except e:
        assert_true("convertible" in e.message())
