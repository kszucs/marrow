# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""C Data Interface import: valid exports round-trip, malformed ones must not
crash.

The valid half is a property: any PyArrow array, sliced or not, crosses into
marrow and back unchanged. The malformed half exports a PyArrow array into C
structs this module owns, corrupts one field, and asks marrow to import the
result. Marrow may import it (where the spec cannot tell) or raise, but it must
not take the process down — so every malformed case runs in a child process,
and a case that kills its child is a finding.

PyArrow's release callbacks walk ``n_children``, ``children`` and
``dictionary``; each export therefore swaps in a release callback that restores
the original fields before calling PyArrow's, so a corrupted struct is still
released correctly by whoever owns it.
"""

import ctypes
import gc
import os
import subprocess
import sys

import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st

import marrow as ma
from marrow.tests.strategies import any_arrays, assert_same

# ── valid exports ──────────────────────────────────────────────────────────


@given(any_arrays())
def test_array_roundtrips_through_c_data(arr):
    """PyArrow -> marrow -> PyArrow over every type, offset or not."""
    assert_same(pa.array(ma.array(arr)), arr)


@given(any_arrays(), st.data())
def test_marrow_slice_exports_like_pyarrow(arr, data):
    """A slice taken in marrow exports the same values as one taken in
    PyArrow — the export has to carry marrow's offset."""
    offset = data.draw(st.integers(0, len(arr)))
    length = data.draw(st.integers(0, len(arr) - offset))
    assert_same(
        pa.array(ma.array(arr).slice(offset, length)), arr.slice(offset, length)
    )


@given(st.lists(any_arrays(size=st.just(5)), min_size=1, max_size=4))
def test_record_batch_roundtrips_through_c_data(columns):
    batch = pa.record_batch({f"c{i}": c for i, c in enumerate(columns)})
    back = pa.record_batch(ma.record_batch(batch))
    assert back.schema.names == batch.schema.names
    for i in range(batch.num_columns):
        assert_same(back.column(i), batch.column(i))


# ── malformed exports ──────────────────────────────────────────────────────

_KEEP = []  # cffi objects that must outlive marrow's copy of a struct


def _capsule(ptr, name):
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return new(ptr, name, None)


class Export:
    """A PyArrow array exported into C structs owned here, open to mutation
    before marrow imports them through ``__arrow_c_array__``."""

    def __init__(self, arr):
        from pyarrow.cffi import ffi

        self.ffi = ffi
        self.schema = ffi.new("struct ArrowSchema*")
        self.array = ffi.new("struct ArrowArray*")
        arr._export_to_c(self._addr(self.array), self._addr(self.schema))
        self._guard(self.schema, "struct ArrowSchema*")
        self._guard(self.array, "struct ArrowArray*")

    def _addr(self, ptr):
        return int(self.ffi.cast("uintptr_t", ptr))

    def _guard(self, struct, ctype):
        """Wrap `struct`'s release so it restores the fields PyArrow walks."""
        saved = (struct.n_children, struct.children, struct.dictionary)
        original = struct.release

        @self.ffi.callback(f"void({ctype})")
        def release(p):
            p.n_children, p.children, p.dictionary = saved
            p.release = original
            original(p)

        _KEEP.append(release)
        struct.release = release

    def new(self, ctype, init):
        obj = self.ffi.new(ctype, init)
        _KEEP.append(obj)
        return obj

    def __arrow_c_array__(self, requested_schema=None):
        return (
            _capsule(self._addr(self.schema), b"arrow_schema"),
            _capsule(self._addr(self.array), b"arrow_array"),
        )

    def __arrow_c_schema__(self):
        return _capsule(self._addr(self.schema), b"arrow_schema")


def _ints():
    return pa.array([1, None, 3, 4], pa.int32())


def _strings():
    return pa.array(["a", None, "ccc"])


def _struct():
    return pa.array(
        [{"a": 1, "b": "x"}, None], pa.struct([("a", pa.int32()), ("b", pa.string())])
    )


def _list():
    return pa.array([[1, 2], None, []], pa.list_(pa.int32()))


def _fixed_size_list():
    return pa.array([[1, 2], None, [5, 6]], pa.list_(pa.int32(), 2))


def _dictionary():
    return pa.array(["x", None, "y", "x"]).dictionary_encode()


def _set_format(fmt):
    def mutate(e):
        e.schema.format = e.new("char[]", fmt)

    return mutate


def _set_array(field, value):
    def mutate(e):
        setattr(e.array, field, value)

    return mutate


def _null_buffer(index):
    def mutate(e):
        e.array.buffers[index] = e.ffi.NULL

    return mutate


def _null_buffers(e):
    e.array.buffers = e.ffi.NULL


def _released(which):
    def mutate(e):
        getattr(e, which).release = e.ffi.NULL

    return mutate


def _null_format(e):
    e.schema.format = e.ffi.NULL


def _drop_schema_children(e):
    e.schema.n_children = 0


def _null_array_children(e):
    e.array.children = e.ffi.NULL


def _drop_schema_dictionary(e):
    e.schema.dictionary = e.ffi.NULL


def _drop_array_dictionary(e):
    e.array.dictionary = e.ffi.NULL


def _null_child_format(e):
    e.schema.children[0].format = e.new("char[]", b"zz")


def _release_child(e):
    e.array.children[0].release = e.ffi.NULL


def _child_length(e):
    e.array.children[0].length = 1


def _misaligned():
    # A data buffer one byte into a bytes object: not 8-, let alone 64-, aligned.
    data = pa.py_buffer(b"\x00" + (1).to_bytes(4, "little") * 2)[1:]
    return pa.Array.from_buffers(pa.int32(), 2, [None, data])


# name -> (array factory, mutation, what marrow may do)
#   "raise"  — invalid by the spec and detectable at import: marrow must raise
#   "either" — the spec cannot be checked without trusting the producer:
#              importing or raising are both fine, crashing is not
CASES = {
    "valid_null_count_unknown": (_ints, _set_array("null_count", -1), "import"),
    "valid_offset": (lambda: _ints().slice(1, 2), lambda e: None, "import"),
    "valid_empty_null_data_buffer": (
        lambda: pa.array([], pa.int32()),
        _null_buffer(1),
        "either",
    ),
    "format_empty": (_ints, _set_format(b""), "raise"),
    "format_unknown": (_ints, _set_format(b"zz"), "raise"),
    "format_string_on_int_buffers": (_ints, _set_format(b"u"), "raise"),
    "format_struct_without_children": (_ints, _set_format(b"+s"), "raise"),
    "format_decimal_garbage": (_ints, _set_format(b"d:abc"), "raise"),
    "format_fixed_binary_negative": (_ints, _set_format(b"w:-1"), "raise"),
    "format_timestamp_bad_unit": (_ints, _set_format(b"tsx:"), "raise"),
    "format_null": (_ints, _null_format, "raise"),
    "format_decimal_bad_bit_width": (_ints, _set_format(b"d:5,2,100"), "raise"),
    "format_decimal_zero_precision": (_ints, _set_format(b"d:0,2"), "raise"),
    "format_decimal_precision_too_wide": (_ints, _set_format(b"d:10,2,32"), "raise"),
    "format_fixed_list_negative": (_list, _set_format(b"+w:-1"), "raise"),
    "format_map_over_list_child": (_list, _set_format(b"+m"), "raise"),
    "n_buffers_too_few": (_ints, _set_array("n_buffers", 1), "raise"),
    "n_buffers_zero": (_ints, _set_array("n_buffers", 0), "raise"),
    "n_buffers_too_many": (_ints, _set_array("n_buffers", 3), "raise"),
    "buffers_null": (_ints, _null_buffers, "raise"),
    "null_count_below_unknown": (_ints, _set_array("null_count", -2), "raise"),
    "length_negative": (_ints, _set_array("length", -1), "raise"),
    "offset_negative": (_ints, _set_array("offset", -1), "raise"),
    "null_count_above_length": (_ints, _set_array("null_count", 10), "either"),
    "validity_null_with_nulls": (_ints, _null_buffer(0), "raise"),
    "data_null_with_length": (_ints, _null_buffer(1), "raise"),
    "string_offsets_null": (_strings, _null_buffer(1), "raise"),
    "string_data_null": (_strings, _null_buffer(2), "raise"),
    "struct_array_n_children_zero": (_struct, _set_array("n_children", 0), "raise"),
    "struct_schema_n_children_zero": (_struct, _drop_schema_children, "raise"),
    "struct_array_children_null": (_struct, _null_array_children, "raise"),
    "struct_child_bad_format": (_struct, _null_child_format, "raise"),
    "struct_child_too_short": (_struct, _child_length, "raise"),
    "struct_child_released": (_struct, _release_child, "raise"),
    "fixed_size_list_child_too_short": (_fixed_size_list, _child_length, "raise"),
    "list_schema_n_children_zero": (_list, _drop_schema_children, "raise"),
    "list_array_children_null": (_list, _null_array_children, "raise"),
    "dictionary_schema_missing": (_dictionary, _drop_schema_dictionary, "raise"),
    "dictionary_array_missing": (_dictionary, _drop_array_dictionary, "raise"),
    "schema_released": (_ints, _released("schema"), "raise"),
    "array_released": (_ints, _released("array"), "raise"),
    "length_oversized": (_ints, _set_array("length", 2**40), "either"),
    "offset_oversized": (_ints, _set_array("offset", 2**40), "either"),
    # Not 64-byte aligned, as a numpy-backed export may not be.
    "valid_misaligned_data_buffer": (_misaligned, lambda e: None, "import"),
}

# Oversized lengths and offsets cannot be validated against buffers the C Data
# Interface does not size, so reading the values would read past them; those
# cases only import.
_IMPORT_ONLY = {"length_oversized", "offset_oversized", "null_count_above_length"}


def _run_case(name):
    """Import one export, read it back, and drop it — so a crash in the
    release path is charged to this case too. Returns "raise <type>" when the
    import raised, else "import"."""
    factory, mutate, expected = CASES[name]
    original = factory()
    export = Export(original)
    mutate(export)
    try:
        imported = ma.array(export)
    except Exception as exc:  # noqa: BLE001 - any Python exception is a pass
        return f"raise {type(exc).__name__}"
    outcome = "import"
    if name not in _IMPORT_ONLY:
        try:
            values = imported.to_pylist()
        except Exception:  # noqa: BLE001 - raising on read is not a crash
            values = None
        if expected == "import" and values != original.to_pylist():
            outcome = f"import wrong-values {values!r:.100}"
    del imported
    gc.collect()
    return outcome


def _run_all(names):
    """Run `names` in child processes, restarting after a crash; return
    {name: outcome}, where a crash is "crash <signal or code>"."""
    results = {}
    pending = list(names)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    while pending:
        proc = subprocess.run(
            [sys.executable, __file__, *pending],
            capture_output=True,
            text=True,
            env=env,
        )
        started = None
        for line in proc.stdout.splitlines():
            kind, _, rest = line.partition(" ")
            if kind == "START":
                started = rest
            elif kind == "RESULT":
                case, _, outcome = rest.partition(" ")
                results[case] = outcome
                started = None
        if started is not None:
            tail = proc.stderr.strip().splitlines()[-1:] or [""]
            results[started] = f"crash {proc.returncode} {tail[0][:200]}"
        elif proc.returncode != 0:
            raise RuntimeError(f"case runner failed:\n{proc.stderr[-2000:]}")
        pending = [n for n in pending if n not in results]
    return results


@pytest.fixture(scope="module")
def outcomes():
    return _run_all(sorted(CASES))


@pytest.mark.parametrize("name", sorted(CASES))
def test_malformed_import_does_not_crash(outcomes, name):
    outcome = outcomes[name]
    assert not outcome.startswith("crash"), outcome
    expected = CASES[name][2]
    if expected == "raise":
        assert outcome.startswith("raise"), f"imported a malformed export: {outcome}"
        # A marrow error kind, not the TypeError meant for a non-producer or
        # the untagged ArrowException a stray stdlib error becomes.
        assert outcome in ("raise ArrowInvalid", "raise ArrowNotImplementedError"), (
            outcome
        )
    elif expected == "import":
        assert outcome == "import", outcome


def test_malformed_import_raises_the_importer_kind(outcomes):
    """A malformed export raises the kind the importer chose, not the
    TypeError meant for an object that is no Arrow producer at all."""
    assert outcomes["n_buffers_too_few"] == "raise ArrowInvalid"
    assert outcomes["format_unknown"] == "raise ArrowNotImplementedError"


def test_import_of_a_non_producer_is_a_type_error():
    # The public functions read a non-producer as a Python sequence first, so
    # the binding is called directly.
    from marrow import libmarrow

    with pytest.raises(TypeError, match="cannot convert"):
        libmarrow.concat([object()], None)


if __name__ == "__main__":
    for case in sys.argv[1:]:
        print("START", case, flush=True)
        print("RESULT", case, _run_case(case), flush=True)
