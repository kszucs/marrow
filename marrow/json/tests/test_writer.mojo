# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The NDJSON writer: the exact text it writes, and that `read_json` reads it
back to the same values and types."""

from std.os.path import join
from std.testing import assert_equal, assert_raises, assert_true

from ...arrays import DynArray
from ...builders import array
from ...dtypes import binary, field, int64, millisecond, second, timestamp
from ...io import BufferSource
from ...schema import Schema
from ...tabular import RecordBatch, record_batch
from ...utils.testing import ScratchDir

from ..reader import ParseOptions, read_json
from ..writer import render_json, write_json


def _strings(var values: List[Optional[String]]) raises -> DynArray:
    return array(values^)


def _read(text: String) raises -> RecordBatch:
    return read_json(BufferSource(text.as_bytes())).combine_chunks()


def test_render_json_flat_rows() raises:
    # The second string is q"u\o: a quote and a backslash to escape.
    var batch = record_batch(
        [
            array([1, None, 3], int64).to_dyn(),
            _strings([String("a"), String('q"u\\o'), None]),
        ],
        names=["n", "s"],
    )
    assert_equal(
        render_json(batch),
        '{"n":1,"s":"a"}\n{"n":null,"s":"q\\"u\\\\o"}\n{"n":3,"s":null}\n',
    )


def test_render_json_floats_are_shortest() raises:
    var batch = _read('{"f": 0.1}\n{"f": -0.0}\n{"f": 1e300}\n{"f": 2.5}\n')
    var out = render_json(batch)
    # Each float reads back to the very same double.
    assert_true(_read(out).column("f") == batch.column("f"))
    assert_equal(out.split("\n")[0], '{"f":0.1}')
    assert_equal(out.split("\n")[3], '{"f":2.5}')


def test_render_json_round_trips_nested_text() raises:
    """Compact text goes in and the same text comes out: lists, structs, a
    list of structs, and nulls at every level. The unicode escape is decoded
    on the way in and written back as UTF-8, after which it is stable."""
    var text = String(
        '{"a":1,"l":[1,2],"s":{"x":"\\u00e9","y":null},"ls":[{"k":true}]}\n'
        + '{"a":null,"l":[],"s":null,"ls":null}\n'
    )
    var out = render_json(_read(text))
    assert_equal(
        out,
        '{"a":1,"l":[1,2],"s":{"x":"é","y":null},"ls":[{"k":true}]}\n'
        + '{"a":null,"l":[],"s":null,"ls":null}\n',
    )
    assert_equal(render_json(_read(out)), out)


def test_render_json_timestamps_are_iso8601() raises:
    var batch = _read('{"t": "2020-01-01 10:00:00"}\n{"t": "1969-12-31"}\n')
    assert_true(batch.schema.fields[0].dtype == timestamp(second))
    var out = render_json(batch)
    assert_equal(
        out, '{"t":"2020-01-01 10:00:00"}\n{"t":"1969-12-31 00:00:00"}\n'
    )
    # And they infer back as timestamps.
    assert_true(_read(out).schema.fields[0].dtype == timestamp(second))


def test_render_json_subsecond_timestamps_keep_their_fraction() raises:
    var text = String('{"t": "2020-01-01 10:00:00.5"}\n{"t": "2020-01-01"}\n')
    var batch = read_json(
        BufferSource(text.as_bytes()),
        parse_options=ParseOptions(
            Schema(fields=[field("t", timestamp(millisecond))])
        ),
    ).combine_chunks()
    assert_equal(
        render_json(batch),
        '{"t":"2020-01-01 10:00:00.500"}\n{"t":"2020-01-01 00:00:00"}\n',
    )


def test_render_json_of_a_slice() raises:
    """A sliced batch writes its own rows, at every nesting level."""
    var batch = _read(
        '{"s":{"x":1},"l":[1]}\n{"s":{"x":2},"l":[2,3]}\n{"s":{"x":3},"l":[]}\n'
    )
    assert_equal(
        render_json(batch.slice(1, 2)),
        '{"s":{"x":2},"l":[2,3]}\n{"s":{"x":3},"l":[]}\n',
    )


def test_render_json_unsupported_type_raises() raises:
    var batch = record_batch([array(binary)], names=["b"])
    with assert_raises(contains="JSON writer: unsupported type"):
        _ = render_json(batch)


def test_write_json_to_a_file_reads_back() raises:
    var batch = _read('{"a":1,"b":"x"}\n{"a":2,"b":null}\n')
    with ScratchDir() as dir:
        var path = join(dir, "out.jsonl")
        write_json(batch, path)
        var back = read_json(path).combine_chunks()
        assert_equal(back.num_rows(), 2)
        assert_equal(render_json(back), render_json(batch))
