"""filter / take / drop_null over nested & complex array types, verified against
PyArrow as the conformance oracle (round-tripped through the Arrow C-Data
interface)."""

import pyarrow as pa
import pytest

import marrow as ma

_MASK = [True, False, True, True, False, True]
_IDX = [5, 0, 2, 0, 4]

# (id, pyarrow array) — one representative array per supported type family.
CASES = [
    ("list", pa.array([[1, 2], None, [3], [], [4, 5, 6], [7]])),
    (
        "large_list",
        pa.array([[1, 2], None, [3], [], [4, 5], [6]], pa.large_list(pa.int64())),
    ),
    (
        "struct",
        pa.array(
            [
                {"a": 1, "b": "x"},
                None,
                {"a": 3, "b": None},
                {"a": 4, "b": "z"},
                {"a": 5, "b": "w"},
                {"a": 6, "b": "q"},
            ],
            pa.struct([("a", pa.int64()), ("b", pa.string())]),
        ),
    ),
    (
        "fixed_size_list",
        pa.array(
            [[1, 2], None, [3, 4], [5, 6], [7, 8], [9, 10]], pa.list_(pa.int64(), 2)
        ),
    ),
    ("dictionary", pa.array(["a", "b", "a", None, "c", "b"]).dictionary_encode()),
    ("binary", pa.array([b"aa", None, b"c", b"dddd", b"e", b"ff"], pa.binary())),
    ("large_string", pa.array(["aa", None, "c", "dddd", "e", "ff"], pa.large_string())),
    (
        "fixed_size_binary",
        pa.array([b"ab", None, b"cd", b"ef", b"gh", b"ij"], pa.binary(2)),
    ),
    (
        "map",
        pa.array(
            [[("k1", 1), ("k2", 2)], None, [("k3", 3)], [], [("k4", 4)], [("k5", 5)]],
            pa.map_(pa.string(), pa.int64()),
        ),
    ),
    ("null", pa.array([None] * 6, pa.null())),
    (
        "list_of_struct",
        pa.array(
            [[{"a": 1}], None, [{"a": 2}, {"a": 3}], [], [{"a": 4}], [{"a": 5}]],
            pa.list_(pa.struct([("a", pa.int64())])),
        ),
    ),
]

IDS = [c[0] for c in CASES]
ARRAYS = [c[1] for c in CASES]


@pytest.mark.parametrize("arr", ARRAYS, ids=IDS)
def test_filter_matches_pyarrow(arr):
    got = pa.array(ma.compute.filter(ma.array(arr), ma.array(_MASK))).to_pylist()
    assert got == arr.filter(pa.array(_MASK)).to_pylist()


@pytest.mark.parametrize("arr", ARRAYS, ids=IDS)
def test_take_matches_pyarrow(arr):
    idx = ma.array(_IDX, type=ma.int32())
    got = pa.array(ma.compute.take(ma.array(arr), idx)).to_pylist()
    assert got == arr.take(pa.array(_IDX, type=pa.int32())).to_pylist()


@pytest.mark.parametrize("arr", ARRAYS, ids=IDS)
def test_drop_null_matches_pyarrow(arr):
    got = pa.array(ma.compute.drop_null(ma.array(arr))).to_pylist()
    assert got == arr.drop_null().to_pylist()


# ── group-by on a nested key column (exercises nested rapidhash) ──────────────


def test_group_by_list_key():
    rb = ma.record_batch(
        {
            "k": ma.array(
                pa.array([[1, 2], [3], [1, 2], [3], [1, 2], [4]], pa.list_(pa.int64()))
            ),
            "v": ma.array([10, 20, 30, 40, 50, 60]),
        }
    )
    out = rb.group_by("k").aggregate([("v", "sum")]).to_pylist()
    got = {tuple(r["k"]): r["v_sum"] for r in out}
    assert got == {(1, 2): 90, (3,): 60, (4,): 60}


def test_group_by_struct_key():
    sk = pa.array(
        [{"a": 1, "b": "x"}, {"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
        pa.struct([("a", pa.int64()), ("b", pa.string())]),
    )
    rb = ma.record_batch({"k": ma.array(sk), "v": ma.array([1, 2, 3])})
    out = rb.group_by("k").aggregate([("v", "sum")]).to_pylist()
    assert sorted(r["v_sum"] for r in out) == [3, 3]
