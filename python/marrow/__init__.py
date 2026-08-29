from collections.abc import Mapping

from . import libmarrow as _ma
from .libmarrow import (
    DataType,
    Field,
    Schema,
    Table,
    null,
    bool_,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    string,
    binary,
    fixed_size_binary,
    list_,
    fixed_size_list_,
    struct,
    date32,
    date64,
    time32,
    time64,
    timestamp,
    duration,
    year_month_interval,
    day_time_interval,
    month_day_nano_interval,
    infer_type,
)


# ── Base wrapper ───────────────────────────────────────────────────────────────


class _Wrapper:
    """Base for all Python wrappers around C extension binding objects."""

    __slots__ = ("_binding",)

    def __init__(self, binding):
        self._binding = binding

    @classmethod
    def wrap(cls, binding):
        obj = cls.__new__(cls)
        obj._binding = binding
        return obj

    def unwrap(self):
        return self._binding


# ── ExecContext ───────────────────────────────────────────────────────────


ExecContext = _ma.ExecContext


def _serial():
    return _ma.ExecContext.serial()


# ── Scalar ─────────────────────────────────────────────────────────────────────


class Scalar(_Wrapper):
    def as_py(self):
        result = self._binding.as_py()
        if isinstance(result, _ma.Array):
            return Array.wrap(result).to_pylist()
        return result

    def is_valid(self):
        return self._binding.is_valid()

    def is_null(self):
        return self._binding.is_null()

    def type(self):
        return self._binding.type()

    def __bool__(self):
        return bool(self._binding)

    def __str__(self):
        return str(self._binding)

    def __repr__(self):
        return repr(self._binding)

    def __eq__(self, other):
        py = other.as_py() if isinstance(other, Scalar) else other
        return self.as_py() == py

    def __ne__(self, other):
        py = other.as_py() if isinstance(other, Scalar) else other
        return self.as_py() != py

    def __lt__(self, other):
        py = other.as_py() if isinstance(other, Scalar) else other
        return self.as_py() < py

    def __le__(self, other):
        py = other.as_py() if isinstance(other, Scalar) else other
        return self.as_py() <= py

    def __gt__(self, other):
        py = other.as_py() if isinstance(other, Scalar) else other
        return self.as_py() > py

    def __ge__(self, other):
        py = other.as_py() if isinstance(other, Scalar) else other
        return self.as_py() >= py

    def __hash__(self):
        return hash(self.as_py())


# ── Array ──────────────────────────────────────────────────────────────────────


class Array(_Wrapper):
    def __arrow_c_array__(self, requested_schema=None):
        return self._binding.__arrow_c_array__(requested_schema)

    def __arrow_c_schema__(self):
        return self._binding.__arrow_c_schema__()

    def __len__(self):
        return self._binding.__len__()

    def __str__(self):
        return str(self._binding)

    def __repr__(self):
        return repr(self._binding)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        n = len(self)
        if isinstance(index, slice):
            start, stop, step = index.indices(n)
            if step != 1:
                raise NotImplementedError("slice step != 1 not supported")
            return Array.wrap(self._binding.slice(start, max(0, stop - start)))
        if index < 0:
            index += n
        return Scalar.wrap(self._binding.__getitem__(index))

    def null_count(self):
        return self._binding.null_count()

    def type(self):
        return self._binding.type()

    def is_valid(self):
        """Boolean array, True where the value is not null."""
        from . import compute

        return compute.is_valid(self)

    def is_null(self, *, nan_is_null=False):
        """Boolean array, True where the value is null."""
        from . import compute

        return compute.is_null(self, nan_is_null=nan_is_null)

    def slice(self, offset=0, length=None):
        if length is None:
            length = max(0, len(self) - offset)
        return Array.wrap(self._binding.slice(offset, length))

    def to_pylist(self):
        return [s.as_py() for s in self]

    def argsort(self, order="ascending", null_placement="at_end"):
        asc = order != "descending" if order is not None else True
        nulls_first = (
            null_placement != "at_end" if null_placement is not None else False
        )
        return Array.wrap(_ma.sort_indices(self._binding, asc, nulls_first, _serial()))

    def sort(self, order="ascending", null_placement="at_end"):
        asc = order != "descending" if order is not None else True
        nulls_first = (
            null_placement != "at_end" if null_placement is not None else False
        )
        return Array.wrap(_ma.sort(self._binding, asc, nulls_first, _serial()))

    def take(self, indices):
        return Array.wrap(_ma.take(self._binding, indices.unwrap(), _serial()))

    def filter(self, mask):
        return Array.wrap(_ma.filter(self._binding, mask.unwrap(), _serial()))

    def drop_null(self):
        return Array.wrap(_ma.drop_null(self._binding, _serial()))


# ── RecordBatch ────────────────────────────────────────────────────────────────


class _Tabular(_Wrapper):
    """The surface RecordBatch and Table share.

    Both wrap a binding exposing the same schema, column-access and conversion
    methods, so these were written out twice. Anything that differs — a
    RecordBatch's mutation and join methods, a Table's chunking — stays on the
    subclass.
    """

    __slots__ = ()

    def __arrow_c_schema__(self):
        return self._binding.__arrow_c_schema__()

    def __str__(self):
        return str(self._binding)

    def __repr__(self):
        return repr(self._binding)

    def num_rows(self):
        return self._binding.num_rows()

    def num_columns(self):
        return self._binding.num_columns()

    def schema(self):
        return self._binding.schema()

    def column_names(self):
        return self._binding.column_names()

    def shape(self):
        return self._binding.shape()

    def column(self, key):
        return Array.wrap(self._binding.column(key))

    def columns(self):
        return [Array.wrap(c) for c in self._binding.columns()]

    def __eq__(self, other):
        return self._binding.equals(other.unwrap())

    def equals(self, other):
        return self._binding.equals(other.unwrap())

    def to_pydict(self):
        return {
            name: col.to_pylist()
            for name, col in zip(self.column_names(), self.columns())
        }

    def to_pylist(self):
        names = list(self.column_names())
        cols = self.columns()
        return [
            {name: cols[j][i].as_py() for j, name in enumerate(names)}
            for i in range(self.num_rows())
        ]


class RecordBatch(_Tabular):
    def __arrow_c_array__(self, requested_schema=None):
        return self._binding.__arrow_c_array__(requested_schema)

    def __arrow_c_record_batch__(self, requested_schema=None):
        return self._binding.__arrow_c_record_batch__(requested_schema)

    def sort_by(self, by, null_placement=None, num_threads=0):
        return RecordBatch.wrap(self._binding.sort_by(by, null_placement, num_threads))

    def select(self, columns):
        return RecordBatch.wrap(self._binding.select(columns))

    def slice(self, offset=0, length=None):
        if length is None:
            length = max(0, self.num_rows() - offset)
        return RecordBatch.wrap(self._binding.slice(offset, length))

    def rename_columns(self, names):
        return RecordBatch.wrap(self._binding.rename_columns(names))

    def add_column(self, i, field, column):
        return RecordBatch.wrap(self._binding.add_column(i, field, column.unwrap()))

    def append_column(self, field, column):
        return RecordBatch.wrap(self._binding.append_column(field, column.unwrap()))

    def remove_column(self, i):
        return RecordBatch.wrap(self._binding.remove_column(i))

    def set_column(self, i, field, column):
        return RecordBatch.wrap(self._binding.set_column(i, field, column.unwrap()))

    def join(self, right, keys, right_keys=None, join_type="inner", num_threads=0):
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(right_keys, str):
            right_keys = [right_keys]
        return RecordBatch.wrap(
            self._binding.join(right.unwrap(), keys, right_keys, join_type, num_threads)
        )


# ── Table ──────────────────────────────────────────────────────────────────────


class Table(_Tabular):
    def __arrow_c_stream__(self, requested_schema=None):
        return self._binding.__arrow_c_stream__(requested_schema)

    def to_batches(self):
        return [RecordBatch.wrap(b) for b in self._binding.to_batches()]


# ── Factory functions ──────────────────────────────────────────────────────────


def array(obj, type=None):
    return Array.wrap(_ma.array(obj, type))


def field(name, type=None, nullable=True, metadata=None):
    """Create a marrow Field.

    Parameters
    ----------
    name : str
        Name of the field.
    type : DataType, default None
        Arrow datatype of the field.
    nullable : bool, default True
        Whether the field's values are nullable.
    metadata : dict, default None
        Optional field metadata.
    """
    if nullable is None:
        nullable = True
    return _ma.field(name, type, nullable, metadata)


def schema(fields):
    """Construct a marrow Schema from a collection of fields.

    Parameters
    ----------
    fields : iterable of Fields or tuples, or mapping of strings to DataTypes
        Can also pass an object that implements the Arrow PyCapsule Protocol
        for schemas (has an ``__arrow_c_schema__`` method).
    """
    if hasattr(fields, "__arrow_c_schema__") or isinstance(fields, Schema):
        return _ma.Schema(fields)

    if isinstance(fields, Mapping):
        fields = list(fields.items())

    coerced = []
    for item in fields:
        if isinstance(item, tuple):
            coerced.append(field(*item))
        else:
            coerced.append(item)

    return _ma.Schema(coerced)


def record_batch(data, names=None, schema=None):
    return RecordBatch.wrap(_ma.record_batch(data, schema, names))


def table(data, names=None):
    return Table.wrap(_ma.table(data, names))


# ── IPC ────────────────────────────────────────────────────────────────────────


def write_ipc_file(path, batches=None, schema=None):
    raw_batches = [b.unwrap() for b in batches] if batches is not None else None
    return _ma.write_ipc_file(
        path, raw_batches, schema.unwrap() if schema is not None else None
    )


def write_ipc_stream(path, batches=None, schema=None):
    raw_batches = [b.unwrap() for b in batches] if batches is not None else None
    return _ma.write_ipc_stream(
        path, raw_batches, schema.unwrap() if schema is not None else None
    )


def read_ipc_file(path):
    return [RecordBatch.wrap(b) for b in _ma.read_ipc_file(path)]


def read_ipc_stream(path):
    return [RecordBatch.wrap(b) for b in _ma.read_ipc_stream(path)]


def read_ipc_file_schema(path):
    return RecordBatch.wrap(_ma.read_ipc_file_schema(path))


def read_ipc_stream_schema(path):
    return RecordBatch.wrap(_ma.read_ipc_stream_schema(path))


# Imported last: `compute` pulls Array/Scalar/RecordBatch back out of this
# module, so it can only be bound once those exist.
from . import compute  # noqa: E402
