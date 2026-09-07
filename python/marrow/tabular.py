"""``RecordBatch`` and ``Table``.

A `RecordBatch` is one contiguous batch; a `Table` is a schema plus one
:class:`~marrow.ChunkedArray` per column. PyArrow's split, and PyArrow's
method names.

**`Table`'s verbs are thin Python over `RecordBatch`'s.** Mojo's `Table` has
`num_rows`, `column`, `combine_chunks`, `to_batches` and `from_batches` and
nothing else — no `select`, no `filter`, no `sort_by`. Rather than write a
second Mojo surface, each verb combines the chunks, uses the batch verb that
already exists, and rebuilds. That materialises a chunked table once per call,
which is the honest cost of not having chunk-aware kernels; it is written in
one place (`_via_batch`) so there is one thing to change when they arrive.
The pure metadata verbs do not pay it at all: they map across the chunks and
keep the table's chunking (`_per_batch`).
"""

from . import libmarrow as _ma
from ._wrapper import _Wrapper, unwrap
from .arrays import (
    Array,
    ChunkedArray,
    Scalar,
    _as_binding_array,
    _serial,
    array,
)
from .types import Schema, field as _field, schema as _schema

__all__ = [
    "RecordBatch",
    "Table",
    "concat_arrays",
    "concat_tables",
    "record_batch",
    "table",
]


class _Tabular(_Wrapper):
    """The surface `RecordBatch` and `Table` share."""

    __slots__ = ()

    def __arrow_c_schema__(self):
        return self._binding.__arrow_c_schema__()

    def __str__(self):
        return str(self._binding)

    def __repr__(self):
        return repr(self._binding)

    @property
    def num_rows(self):
        return self._binding.num_rows()

    @property
    def num_columns(self):
        return self._binding.num_columns()

    @property
    def schema(self):
        return Schema.wrap(self._binding.schema())

    @property
    def column_names(self):
        return self._binding.column_names()

    @property
    def shape(self):
        return self._binding.shape()

    def equals(self, other):
        return self._binding.equals(unwrap(other))

    def __eq__(self, other):
        return type(self) is type(other) and self.equals(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    def to_pydict(self):
        return {
            name: col.to_pylist()
            for name, col in zip(self.column_names, self.columns)
        }

    def to_pylist(self):
        names = list(self.column_names)
        cols = [c.to_pylist() for c in self.columns]
        return [dict(zip(names, row)) for row in zip(*cols)]


class RecordBatch(_Tabular):
    """One contiguous batch of columns."""

    def __arrow_c_array__(self, requested_schema=None):
        return self._binding.__arrow_c_array__(requested_schema)

    def __arrow_c_record_batch__(self, requested_schema=None):
        return self._binding.__arrow_c_record_batch__(requested_schema)

    @property
    def columns(self):
        return [Array.wrap(c) for c in self._binding.columns()]

    def column(self, key):
        return Array.wrap(self._binding.column(key))

    def __getitem__(self, key):
        return self.column(key)

    def select(self, columns):
        return RecordBatch.wrap(self._binding.select(list(columns)))

    def drop(self, columns):
        """Every column except these, in input order."""
        drop = set(columns)
        missing = drop - set(self.column_names)
        if missing:
            raise KeyError(f"drop: no such column(s): {sorted(missing)}")
        return self.select([n for n in self.column_names if n not in drop])

    # PyArrow spells it `drop_columns` on Table and `drop` on both. Both work.
    drop_columns = drop

    def slice(self, offset=0, length=None):
        if length is None:
            length = max(0, self.num_rows - offset)
        return RecordBatch.wrap(self._binding.slice(offset, length))

    def filter(self, mask):
        """Keep rows where `mask` is true.

        Per column rather than through a plan: the mask is already computed,
        so there is nothing for an operator to do that `filter` does not."""
        mask = _as_binding_array(mask)
        ctx = _serial()
        # Over the raw bindings: `self.columns` allocates an `Array` wrapper
        # per column and the next token would unwrap it again.
        return RecordBatch.wrap(
            _ma.record_batch(
                [_ma.filter(c, mask, ctx) for c in self._binding.columns()],
                self._binding.schema(),
                None,
            )
        )

    def take(self, indices):
        indices = _as_binding_array(indices)
        ctx = _serial()
        return RecordBatch.wrap(
            _ma.record_batch(
                [_ma.take(c, indices, ctx) for c in self._binding.columns()],
                self._binding.schema(),
                None,
            )
        )

    def sort_by(self, by, null_placement=None, num_threads=0):
        return RecordBatch.wrap(
            self._binding.sort_by(by, null_placement, num_threads)
        )

    def rename_columns(self, names):
        return RecordBatch.wrap(self._binding.rename_columns(list(names)))

    def add_column(self, i, field, column):
        return RecordBatch.wrap(
            self._binding.add_column(i, unwrap(field), unwrap(column))
        )

    def append_column(self, field, column):
        return RecordBatch.wrap(
            self._binding.append_column(unwrap(field), unwrap(column))
        )

    def remove_column(self, i):
        return RecordBatch.wrap(self._binding.remove_column(i))

    def set_column(self, i, field, column):
        return RecordBatch.wrap(
            self._binding.set_column(i, unwrap(field), unwrap(column))
        )

    def join(self, right, keys, right_keys=None, join_type="inner", num_threads=0):
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(right_keys, str):
            right_keys = [right_keys]
        return RecordBatch.wrap(
            self._binding.join(
                unwrap(right), keys, right_keys, join_type, num_threads
            )
        )

    # ── constructors ───────────────────────────────────────────────────────

    @classmethod
    def from_pydict(cls, mapping, schema=None):
        """``{name: values}`` — values may be lists or arrays."""
        names = list(mapping)
        return cls.from_arrays(
            [mapping[n] for n in names], names=names, schema=schema
        )

    @classmethod
    def from_pylist(cls, rows, schema=None):
        """A list of row dicts, PyArrow's spelling.

        Column order comes from the first row, then any key a later row
        introduces, so a ragged list is not silently truncated. A missing key
        is a null."""
        # `dict.fromkeys` preserves first-seen order in one pass; the `not in
        # names` list scan it replaces was quadratic in the key count.
        names = list(dict.fromkeys(key for row in rows for key in row))
        columns = [[row.get(name) for row in rows] for name in names]
        return cls.from_arrays(columns, names=names, schema=schema)

    @classmethod
    def from_arrays(cls, arrays, names=None, schema=None):
        columns = [a if isinstance(a, Array) else array(a) for a in arrays]
        if schema is None and names is None:
            raise ValueError("from_arrays: pass `names` or `schema`")
        if schema is None:
            schema = _schema(
                [_field(n, c.type) for n, c in zip(names, columns)]
            )
        return cls.wrap(
            _ma.record_batch(
                [unwrap(c) for c in columns], unwrap(schema), None
            )
        )


class Table(_Tabular):
    """A schema plus one :class:`~marrow.ChunkedArray` per column."""

    def __arrow_c_stream__(self, requested_schema=None):
        return self._binding.__arrow_c_stream__(requested_schema)

    @property
    def columns(self):
        return [ChunkedArray.wrap(c) for c in self._binding.columns()]

    def column(self, key):
        return ChunkedArray.wrap(self._binding.column(key))

    def __getitem__(self, key):
        return self.column(key)

    def to_batches(self):
        return [RecordBatch.wrap(b) for b in self._binding.to_batches()]

    def combine_chunks(self):
        """The whole table as one :class:`RecordBatch`."""
        return RecordBatch.wrap(self._binding.combine_chunks())

    def _per_batch(self, verb, *args, **kwargs):
        """One `RecordBatch` verb, mapped across the chunks.

        Nothing is combined and nothing is copied: `to_batches` refcount-bumps
        the chunk handles, and the verb runs per batch, so the table keeps its
        chunking.

        **Only for verbs whose arguments are column-shaped**, which is to say
        the pure metadata edits. Anything taking a *row*-shaped argument --
        `slice`'s offset, `filter`'s mask, `take`'s indices, the column a
        `set_column` installs -- is indexed against the whole table, and
        handing the same argument to each chunk would answer a different
        question per chunk. Those go through `_via_batch`.
        """
        return Table.from_batches(
            [getattr(b, verb)(*args, **kwargs) for b in self.to_batches()]
        )

    def _via_batch(self, verb, *args, **kwargs):
        """One `RecordBatch` verb, applied to the *combined* table.

        For everything `_per_batch` cannot do: the verbs whose answer depends
        on rows in other chunks (`sort_by`, `join`) and the verbs whose
        argument is indexed against the whole table (`slice`, `filter`, `take`,
        the column mutations). Materialising is the honest cost of having no
        chunk-aware kernels, and it is now confined to the verbs that need
        it rather than paid by every one."""
        return Table.from_batches(
            [getattr(self.combine_chunks(), verb)(*args, **kwargs)]
        )

    def select(self, columns):
        return self._per_batch("select", columns)

    def drop(self, columns):
        return self._per_batch("drop", columns)

    drop_columns = drop

    def slice(self, offset=0, length=None):
        return self._via_batch("slice", offset, length)

    def filter(self, mask):
        return self._via_batch("filter", mask)

    def take(self, indices):
        return self._via_batch("take", indices)

    def sort_by(self, by, null_placement=None, num_threads=0):
        return self._via_batch("sort_by", by, null_placement, num_threads)

    def rename_columns(self, names):
        return self._per_batch("rename_columns", names)

    def add_column(self, i, field, column):
        return self._via_batch("add_column", i, field, _as_array(column))

    def append_column(self, field, column):
        return self._via_batch("append_column", field, _as_array(column))

    def remove_column(self, i):
        return self._per_batch("remove_column", i)

    def set_column(self, i, field, column):
        return self._via_batch("set_column", i, field, _as_array(column))

    def join(self, right, keys, right_keys=None, join_type="inner", num_threads=0):
        return self._via_batch(
            "join",
            right.combine_chunks() if isinstance(right, Table) else right,
            keys,
            right_keys,
            join_type,
            num_threads,
        )

    # ── constructors ───────────────────────────────────────────────────────

    @classmethod
    def from_pydict(cls, mapping, schema=None):
        return cls.from_batches([RecordBatch.from_pydict(mapping, schema)])

    @classmethod
    def from_pylist(cls, rows, schema=None):
        return cls.from_batches([RecordBatch.from_pylist(rows, schema)])

    @classmethod
    def from_arrays(cls, arrays, names=None, schema=None):
        return cls.from_batches(
            [RecordBatch.from_arrays(arrays, names=names, schema=schema)]
        )

    @classmethod
    def from_batches(cls, batches, schema=None):
        """One chunk per batch — the only way to build a chunked column."""
        batches = list(batches)
        if not batches:
            raise ValueError("from_batches: needs at least one batch")
        return cls.wrap(_ma.table_from_batches([unwrap(b) for b in batches]))


def _as_array(column):
    """A `Table` column argument as one contiguous `Array`."""
    if isinstance(column, ChunkedArray):
        return column.combine_chunks()
    return column


# ── free functions ─────────────────────────────────────────────────────────


def record_batch(data, names=None, schema=None):
    if isinstance(data, dict):
        return RecordBatch.from_pydict(data, schema)
    if names is not None or schema is not None:
        return RecordBatch.from_arrays(data, names=names, schema=schema)
    return RecordBatch.wrap(_ma.record_batch(unwrap(data), None, None))


def table(data, names=None, schema=None):
    if isinstance(data, dict):
        return Table.from_pydict(data, schema)
    if names is not None or schema is not None:
        return Table.from_arrays(data, names=names, schema=schema)
    return Table.wrap(_ma.table(unwrap(data), None))


def concat_arrays(arrays):
    """One array from several of the same type."""
    arrays = [a if isinstance(a, Array) else array(a) for a in arrays]
    if not arrays:
        raise ValueError("concat_arrays: needs at least one array")
    return Array.wrap(_ma.concat([unwrap(a) for a in arrays], _serial()))


def concat_tables(tables):
    """One table from several sharing a schema — a chunk per input batch."""
    batches = []
    for t in tables:
        batches.extend(t.to_batches() if isinstance(t, Table) else [t])
    return Table.from_batches(batches)
