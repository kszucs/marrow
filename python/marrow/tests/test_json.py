# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The NDJSON reader, verified against ``pyarrow.json`` as the oracle.

Every case reads the same bytes with both libraries and compares column names,
types and values; the error cases require both to raise. The fixed cases follow
pyarrow's own ``test_json.py``, and a seeded generator adds rows neither
suite spells out.
"""

import json
import random

import pyarrow as pa
import pyarrow.json as pj
import pytest

import marrow
import marrow.json as mj


def _to_pa(marrow_table):
    """marrow Table -> pyarrow Table via the Arrow C stream interface."""
    return pa.RecordBatchReader.from_stream(marrow_table).read_all()


def _write(tmp_path, text, name="rows.jsonl"):
    path = tmp_path / name
    path.write_text(text)
    return path


def _assert_equiv(got, want):
    assert got.column_names == want.column_names
    for i in range(want.num_columns):
        assert str(got.column(i).type) == str(want.column(i).type)
        assert got.column(i).to_pylist() == want.column(i).to_pylist()


def _check(tmp_path, text, read_options=None, parse_options=None):
    """Read `text` with both libraries and require the same table.

    pyarrow always reads with its default block size: it rejects a row longer
    than a block (see `test_row_longer_than_a_block`), and a column's values do
    not depend on how it is chunked.
    """
    path = _write(tmp_path, text)
    pa_parse = None
    if parse_options is not None:
        pa_parse = pj.ParseOptions(
            explicit_schema=parse_options.explicit_schema,
            unexpected_field_behavior=parse_options.unexpected_field_behavior,
        )
    want = pj.read_json(path, parse_options=pa_parse)
    got = _to_pa(mj.read_json(path, read_options, parse_options))
    _assert_equiv(got, want)
    return got


# ---------------------------------------------------------------------------
# pyarrow's cases
# ---------------------------------------------------------------------------


def test_simple_ints(tmp_path):
    _check(tmp_path, '{"a": 1, "b": 2, "c": 3}\n{"a": 4, "b": 5, "c": 6}\n')


def test_simple_varied(tmp_path):
    _check(
        tmp_path,
        '{"a": 1, "b": 2.0, "c": "3", "d": false}\n'
        '{"a": 4.0, "b": -5, "c": "foo", "d": true}\n',
    )


def test_simple_nulls(tmp_path):
    _check(
        tmp_path,
        '{"a": 1, "b": 2, "c": null, "d": null, "e": null}\n'
        '{"a": null, "b": -5, "c": "foo", "d": null, "e": true}\n'
        '{"a": 4.5, "b": null, "c": "nan", "d": null, "e": false}\n',
    )


def test_empty_lists(tmp_path):
    _check(tmp_path, '{"a": []}\n{"a": []}\n')


def test_nested(tmp_path):
    _check(
        tmp_path,
        '{"s": {"x": 1, "l": [1, 2]}, "t": [{"k": "a"}]}\n'
        '{"s": {"y": "z", "l": [3.5]}, "t": []}\n'
        '{"s": null, "t": null}\n',
    )


def test_inference_timestamps(tmp_path):
    _check(
        tmp_path,
        '{"t": "1970-01-01", "u": "2018-11-13T17:11:10Z", "v": "x"}\n'
        '{"t": "2000-02-29 12:00", "u": "2018-11-13 17:11:10+01:00", "v": "1970-01-01"}\n',
    )


def test_absent_keys(tmp_path):
    _check(tmp_path, '{"a": 1}\n{"b": "x"}\n{"a": 2, "c": [true]}\n')


@pytest.mark.parametrize("behavior", ["ignore", "infer"])
def test_explicit_schema_with_unexpected_behaviour(tmp_path, behavior):
    schema = pa.schema([("a", pa.int32()), ("b", pa.string())])
    _check(
        tmp_path,
        '{"a": 1, "c": true}\n{"b": "x", "d": 2.5}\n',
        parse_options=mj.ParseOptions(
            explicit_schema=schema, unexpected_field_behavior=behavior
        ),
    )


def test_explicit_schema_error_on_unexpected(tmp_path):
    path = _write(tmp_path, '{"a": 1, "c": true}\n')
    schema = pa.schema([("a", pa.int32())])
    with pytest.raises(Exception, match="unexpected field"):
        pj.read_json(
            path,
            parse_options=pj.ParseOptions(
                explicit_schema=schema, unexpected_field_behavior="error"
            ),
        )
    with pytest.raises(Exception, match="unexpected field"):
        mj.read_json(
            path,
            parse_options=mj.ParseOptions(
                explicit_schema=schema, unexpected_field_behavior="error"
            ),
        )


def test_reconcile_across_blocks(tmp_path):
    # A column null for the first blocks and typed in a later one: the whole
    # file agrees on the type in the end (ARROW-12065).
    rows = ['{"a": null}'] * 20 + ['{"a": 1}']
    got = _check(
        tmp_path, "\n".join(rows) + "\n", read_options=mj.ReadOptions(block_size=32)
    )
    assert got.column(0).num_chunks > 1


@pytest.mark.parametrize("block_size", [16, 64, 1 << 20])
def test_block_sizes(tmp_path, block_size):
    rows = [json.dumps({"a": i, "b": str(i) * (i % 5)}) for i in range(50)]
    _check(
        tmp_path,
        "\n".join(rows) + "\n",
        read_options=mj.ReadOptions(block_size=block_size),
    )


def test_row_longer_than_a_block(tmp_path):
    # A deliberate difference: pyarrow fails, marrow grows the read.
    path = _write(tmp_path, '{"a": "' + "x" * 100 + '"}\n{"a": "y"}\n')
    with pytest.raises(pa.ArrowInvalid, match="straddling"):
        pj.read_json(path, read_options=pj.ReadOptions(block_size=16))
    got = _to_pa(mj.read_json(path, read_options=mj.ReadOptions(block_size=16)))
    assert got.column(0).to_pylist() == ["x" * 100, "y"]


def test_no_newline_at_end(tmp_path):
    _check(tmp_path, '{"a": 1}\n{"a": 2}')


def test_objects_across_lines_and_whitespace(tmp_path):
    _check(tmp_path, '{"a": 1} {"a": 2}\n\n{"a":\n3}\r\n  {"a" : 4 }  \n')


@pytest.mark.parametrize(
    "text",
    [
        "",
        '{"a": 1}\n{"a": "x"}\n',
        '{"a": 1, "a": 2}\n',
        "1\n",
        '{"a": 1} x\n',
    ],
)
def test_both_raise(tmp_path, text):
    path = _write(tmp_path, text)
    with pytest.raises(Exception):
        pj.read_json(path)
    with pytest.raises(Exception):
        mj.read_json(path)


# ---------------------------------------------------------------------------
# generated rows
# ---------------------------------------------------------------------------


def _value(rng, kind):
    if rng.random() < 0.15:
        return None
    if kind == "int":
        return rng.randint(-(2**40), 2**40)
    if kind == "float":
        return rng.choice([rng.uniform(-1e6, 1e6), float(rng.randint(-5, 5))])
    if kind == "str":
        return "".join(rng.choice('abcé😀 "\\\n') for _ in range(rng.randint(0, 6)))
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "list":
        return [rng.randint(0, 9) for _ in range(rng.randint(0, 3))]
    return {"x": rng.randint(0, 9)} if rng.random() < 0.5 else {"y": "s"}


@pytest.mark.parametrize("seed", range(5))
def test_small_random_json(tmp_path, seed):
    rng = random.Random(seed)
    kinds = {"i": "int", "f": "float", "s": "str", "b": "bool", "l": "list", "o": "obj"}
    lines = []
    for _ in range(rng.randint(1, 60)):
        row = {k: _value(rng, t) for k, t in kinds.items() if rng.random() < 0.8}
        lines.append(json.dumps(row, ensure_ascii=rng.random() < 0.5))
    _check(
        tmp_path,
        "\n".join(lines) + "\n",
        read_options=mj.ReadOptions(block_size=rng.choice([64, 256, 1 << 20])),
    )


# ---------------------------------------------------------------------------
# the streaming reader and the lazy scan
# ---------------------------------------------------------------------------


def test_open_json_streams_one_batch_per_block(tmp_path):
    path = _write(tmp_path, '{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    reader = mj.open_json(path, read_options=mj.ReadOptions(block_size=9))
    assert reader.schema.names == ["a"]
    batches = list(reader)
    assert [b.num_rows for b in batches] == [1, 1, 1]


def test_open_json_is_strict_after_the_first_block(tmp_path):
    path = _write(tmp_path, '{"a": 1}\n{"a": 2, "b": 3}\n')
    reader = mj.open_json(path, read_options=mj.ReadOptions(block_size=9))
    reader.read_next_batch()
    with pytest.raises(Exception, match="unexpected field"):
        reader.read_next_batch()


def test_lazy_read_json_filters_and_projects(tmp_path):
    path = _write(tmp_path, '{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n{"a": 3}\n')
    got = marrow.read_json(path).filter(marrow.col("a") > 1).select("b").collect()
    assert got.column_names == ["b"]
    assert got.column(0).to_pylist() == ["y", None]


# ---------------------------------------------------------------------------
# the writer, read back by pyarrow
# ---------------------------------------------------------------------------


def test_write_json_reads_back_in_pyarrow(tmp_path):
    table = pa.table(
        {
            "i": [1, None, 3],
            "f": [0.1, None, -2.5],
            "s": ["a", 'q"u\\o é', None],
            "b": [True, False, None],
            "l": [[1, 2], [], None],
            "st": [{"x": 1, "y": "p"}, None, {"x": 3, "y": None}],
            "t": pa.array([0, None, 1_600_000_000], pa.timestamp("s")),
        }
    )
    path = tmp_path / "out.jsonl"
    mj.write_json(table, path)
    _assert_equiv(pj.read_json(path), table)
    _assert_equiv(_to_pa(mj.read_json(path)), table)


def test_write_json_non_finite_is_null(tmp_path):
    table = pa.table({"f": [float("nan"), float("inf"), 1.5]})
    path = tmp_path / "out.jsonl"
    mj.write_json(table, path)
    assert pj.read_json(path).column("f").to_pylist() == [None, None, 1.5]


@pytest.mark.parametrize("seed", range(5))
def test_write_json_random_round_trip(tmp_path, seed):
    rng = random.Random(seed)
    kinds = {"i": "int", "f": "float", "s": "str", "b": "bool", "l": "list", "o": "obj"}
    lines = []
    for _ in range(rng.randint(1, 60)):
        row = {k: _value(rng, t) for k, t in kinds.items() if rng.random() < 0.8}
        lines.append(json.dumps(row))
    source = _write(tmp_path, "\n".join(lines) + "\n", name="in.jsonl")
    table = pj.read_json(source)
    out = tmp_path / "out.jsonl"
    mj.write_json(table, out)
    _assert_equiv(pj.read_json(out), table)
