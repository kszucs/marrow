"""``Scalar``, ``Array`` and ``ChunkedArray``.

Property-shaped where PyArrow is property-shaped — `arr.type`, `arr.null_count`
— which the binding cannot be, since `tp_getset` is unexposed and
`def_property` does not exist. The wrapper is where that difference is paid.
"""

from . import libmarrow as _ma
from ._wrapper import _Wrapper, unwrap
from .types import DataType

__all__ = ["Array", "ChunkedArray", "Scalar", "array", "chunked_array"]


def _serial():
    return _ma.ExecContext.serial()


class Scalar(_Wrapper):
    """One value out of an :class:`Array`."""

    def as_py(self):
        result = self._binding.as_py()
        if isinstance(result, _ma.Array):
            return Array.wrap(result).to_pylist()
        return result

    def is_valid(self):
        return self._binding.is_valid()

    def is_null(self):
        return self._binding.is_null()

    @property
    def type(self):
        return DataType.wrap(self._binding.type())

    def __bool__(self):
        return bool(self._binding)

    def __str__(self):
        return str(self._binding)

    def __repr__(self):
        return repr(self._binding)

    # Comparison delegates to `as_py`, so a Scalar compares against a plain
    # Python value the way PyArrow's does.

    def __eq__(self, other):
        return self.as_py() == _as_py(other)

    def __ne__(self, other):
        return self.as_py() != _as_py(other)

    def __lt__(self, other):
        return self.as_py() < _as_py(other)

    def __le__(self, other):
        return self.as_py() <= _as_py(other)

    def __gt__(self, other):
        return self.as_py() > _as_py(other)

    def __ge__(self, other):
        return self.as_py() >= _as_py(other)

    def __hash__(self):
        return hash(self.as_py())


def _as_py(value):
    return value.as_py() if isinstance(value, Scalar) else value


class Array(_Wrapper):
    """A contiguous Arrow array."""

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
        # Straight to the binding: `__getitem__` re-reads `len(self)` on every
        # call for its slice and negative-index branches, which an ordered walk
        # needs neither of.
        for i in range(len(self)):
            yield Scalar.wrap(self._binding.__getitem__(i))

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

    @property
    def null_count(self):
        return self._binding.null_count()

    @property
    def type(self):
        return DataType.wrap(self._binding.type())

    def is_valid(self):
        """Boolean array, True where the value is not null.

        Always an array. The single-element question is `arr[i].is_valid()`,
        which is PyArrow's spelling and does not need this to answer two
        different types depending on its arity."""
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

    def cast(self, target_type, *, safe=True):
        from . import compute

        return compute.cast(self, target_type, safe=safe)

    def argsort(self, order="ascending", null_placement="at_end"):
        asc, nulls_first = _sort_flags(order, null_placement)
        return Array.wrap(_ma.sort_indices(self._binding, asc, nulls_first, _serial()))

    def sort(self, order="ascending", null_placement="at_end"):
        asc, nulls_first = _sort_flags(order, null_placement)
        return Array.wrap(_ma.sort(self._binding, asc, nulls_first, _serial()))

    def take(self, indices):
        """Rows at `indices`, which may be a list or any integer array."""
        return Array.wrap(
            _ma.take(self._binding, _as_binding_array(indices), _serial())
        )

    def filter(self, mask):
        return Array.wrap(_ma.filter(self._binding, _as_binding_array(mask), _serial()))

    def drop_null(self):
        return Array.wrap(_ma.drop_null(self._binding, _serial()))


def _as_binding_array(value):
    """A binding `Array` from a wrapper, a binding, or any sequence.

    `list()` rather than the value itself for the sequence case, so a set or a
    generator works where `array()` alone wants something indexable."""
    if isinstance(value, Array):
        return value.unwrap()
    if isinstance(value, _ma.Array):
        return value
    return unwrap(array(list(value)))


def _sort_flags(order, null_placement):
    asc = order != "descending" if order is not None else True
    nulls_first = null_placement != "at_end" if null_placement is not None else False
    return asc, nulls_first


class ChunkedArray(_Wrapper):
    """One logical column made of several contiguous :class:`Array` chunks.

    This is what a :class:`~marrow.Table` actually holds. It was unreachable
    from Python until the type was registered, so ``Table.column()`` used to
    combine the chunks and hand back a single ``Array`` — a copy whose shape
    said nothing about the table's own.
    """

    def __len__(self):
        return self._binding.__len__()

    @property
    def type(self):
        return DataType.wrap(self._binding.type())

    @property
    def num_chunks(self):
        return self._binding.num_chunks()

    @property
    def chunks(self):
        return [Array.wrap(c) for c in self._binding.chunks()]

    def chunk(self, index):
        return Array.wrap(self._binding.chunk(index))

    def combine_chunks(self):
        """One contiguous :class:`Array`. Copies."""
        return Array.wrap(self._binding.combine_chunks())

    @property
    def null_count(self):
        # Over the raw chunks: `self.chunks` allocates a wrapper per chunk and
        # every one is discarded after reading a single integer.
        return sum(c.null_count() for c in self._binding.chunks())

    def __iter__(self):
        for chunk in self.chunks:
            yield from chunk

    def __getitem__(self, index):
        # Chunk-aware rather than combining: indexing a column should not
        # copy the whole column.
        if isinstance(index, slice):
            return self.combine_chunks()[index]
        if index < 0:
            index += len(self)
        for chunk in self._binding.chunks():
            # `__len__()` explicitly: `def_method` fills `tp_dict`, not the
            # `sq_length` slot, so `len()` does not reach a raw binding.
            size = chunk.__len__()
            if index < size:
                return Array.wrap(chunk)[index]
            index -= size
        raise IndexError("index out of range")

    def to_pylist(self):
        out = []
        for chunk in self.chunks:
            out.extend(chunk.to_pylist())
        return out

    def __str__(self):
        return self._binding.__str__()

    def __repr__(self):
        return f"<marrow.ChunkedArray: {self}>"


def array(obj, type=None):
    """An :class:`Array` from a Python sequence, or from an Arrow producer."""
    return Array.wrap(_ma.array(unwrap(obj), unwrap(type)))


def chunked_array(arrays, type=None):
    """A :class:`ChunkedArray` from several arrays of one type.

    There is no binding constructor for it — `ChunkedArray` reaches Python out
    of a `Table` — so this routes through a one-column table, which is the
    same object a chunked column comes from anyway.
    """
    from .tabular import RecordBatch, Table

    chunks = [a if isinstance(a, Array) else array(a, type) for a in arrays]
    if not chunks:
        raise ValueError("chunked_array: needs at least one array")
    batches = [RecordBatch.from_arrays([c], names=["c"]) for c in chunks]
    return Table.from_batches(batches).column(0)
