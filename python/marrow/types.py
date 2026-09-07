"""The type system — ``DataType``, ``Field``, ``Schema``, and the factories.

These three are wrapped rather than re-exported raw, and that is the whole
point of the module. The bindings expose `equals` and `__str__` as ordinary
methods because `def_method` fills `tp_dict` rather than a CPython slot, so
`==` and `str()` never reach them: before this, `ma.int32() == ma.int32()` was
`False` — an identity comparison between two distinct binding objects — and
the golden corpus compared dtypes by rendering them to strings.

Naming follows PyArrow's, so `pa` muscle memory carries over: `schema.names`,
`schema.types`, `schema.field(i)`, `field.nullable`, `dtype.byte_width`.
"""

from collections.abc import Mapping

from . import libmarrow as _ma
from ._wrapper import _Wrapper, unwrap

__all__ = [
    "DataType",
    "Field",
    "Schema",
    "binary",
    "bool_",
    "date32",
    "date64",
    "day_time_interval",
    "duration",
    "field",
    "fixed_size_binary",
    "fixed_size_list_",
    "float16",
    "float32",
    "float64",
    "infer_type",
    "int16",
    "int32",
    "int64",
    "int8",
    "list_",
    "month_day_nano_interval",
    "null",
    "schema",
    "string",
    "struct",
    "time32",
    "time64",
    "timestamp",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "year_month_interval",
]


class DataType(_Wrapper):
    """An Arrow data type."""

    def equals(self, other):
        return self._binding.equals(unwrap(other))

    def __eq__(self, other):
        return isinstance(other, DataType) and self.equals(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # Hashable, unlike an expression: a dtype is a value, and dict keys
        # like `{int64(): ...}` are the natural way to write a dispatch table.
        return hash(str(self))

    @property
    def byte_width(self):
        """Bytes per value for a fixed-width type."""
        return self._binding.byte_width()

    def __str__(self):
        return self._binding.__str__()

    def __repr__(self):
        return f"DataType({self})"


class Field(_Wrapper):
    """One named, typed, nullable column in a :class:`Schema`."""

    @property
    def name(self):
        return self._binding.name()

    @property
    def type(self):
        return DataType.wrap(self._binding.type())

    @property
    def nullable(self):
        return self._binding.nullable()

    def equals(self, other):
        return self._binding.equals(unwrap(other))

    def __eq__(self, other):
        return isinstance(other, Field) and self.equals(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, str(self.type), self.nullable))

    def __str__(self):
        return self._binding.__str__()

    def __repr__(self):
        return f"Field({self})"


class Schema(_Wrapper):
    """An ordered collection of :class:`Field`."""

    def __len__(self):
        return self._binding.__len__()

    def __iter__(self):
        for i in range(len(self)):
            yield self.field(i)

    @property
    def names(self):
        return self._binding.names()

    @property
    def types(self):
        return [DataType.wrap(t) for t in self._binding.types()]

    def field(self, key):
        """One field, by position or by name."""
        return Field.wrap(self._binding.field(key))

    def get_field_index(self, name):
        """The field's position, or ``-1`` — PyArrow's sentinel and Mojo's."""
        return self._binding.get_field_index(name)

    def equals(self, other):
        return self._binding.equals(unwrap(other))

    def __eq__(self, other):
        return isinstance(other, Schema) and self.equals(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None  # mutable-ish and rarely a key; matches PyArrow

    def __arrow_c_schema__(self):
        return self._binding.__arrow_c_schema__()

    def __str__(self):
        return self._binding.__str__()

    def __repr__(self):
        return f"Schema({self})"


# ── factories ──────────────────────────────────────────────────────────────
#
# Each returns a wrapped `DataType`. Generated from one list rather than
# written out: they differ only in name, and a hand-written set is how
# `timestamp` came to be the only one whose dropped default was noticed.

_NULLARY = (
    "null bool_ int8 int16 int32 int64 uint8 uint16 uint32 uint64 "
    "float16 float32 float64 string binary"
).split()


def _wrap_factory(name, doc):
    """A binding dtype factory, wrapping whatever it answers.

    `*args` covers the nullary ones too — calling it with none is what a
    zero-argument factory does — so there is one generator here, not one per
    arity."""
    binding = getattr(_ma, name)

    def factory(*args):
        return DataType.wrap(binding(*args))

    factory.__name__ = name
    factory.__doc__ = doc
    return factory


for _name in _NULLARY:
    globals()[_name] = _wrap_factory(_name, f"The ``{_name.rstrip('_')}`` type.")
del _name


fixed_size_binary = _wrap_factory(
    "fixed_size_binary", "``fixed_size_binary(width)`` — `width` bytes per value."
)
date32 = _wrap_factory("date32", "Days since the Unix epoch.")
date64 = _wrap_factory("date64", "Milliseconds since the Unix epoch.")
time32 = _wrap_factory("time32", '``time32(unit)`` — ``"s"`` or ``"ms"``.')
time64 = _wrap_factory("time64", '``time64(unit)`` — ``"us"`` or ``"ns"``.')
duration = _wrap_factory("duration", "``duration(unit)`` — an elapsed span.")
year_month_interval = _wrap_factory("year_month_interval", "A month count.")
day_time_interval = _wrap_factory("day_time_interval", "Days and milliseconds.")
month_day_nano_interval = _wrap_factory(
    "month_day_nano_interval", "Months, days and nanoseconds."
)


def timestamp(unit, tz=None):
    """``timestamp("us")``, ``timestamp("us", "UTC")``.

    `dtypes.mojo` declares ``tz: PythonObject = None``, but the default does
    not survive `def_function`, so the binding wants both arguments. PyArrow's
    `pa.timestamp("us")` takes one and the Mojo lane's `timestamp(microsecond)`
    takes one, so this supplies what the binding drops."""
    return DataType.wrap(_ma.timestamp(unit, tz))


def list_(value_type):
    """A variable-length list of `value_type`."""
    return DataType.wrap(_ma.list_(unwrap(value_type)))


def fixed_size_list_(value_type, list_size):
    """A list of exactly `list_size` elements of `value_type`."""
    return DataType.wrap(_ma.fixed_size_list_(unwrap(value_type), list_size))


def struct(fields):
    """A struct of `fields`, given as Fields or ``(name, type)`` pairs."""
    return DataType.wrap(_ma.struct([unwrap(f) for f in _coerce_fields(fields)]))


def field(name, type=None, nullable=True, metadata=None):
    """One named, typed column.

    Parameters
    ----------
    name : str
    type : DataType
    nullable : bool, default True
    metadata : dict, default None
    """
    if nullable is None:
        nullable = True
    return Field.wrap(_ma.field(name, unwrap(type), nullable, metadata))


def _coerce_fields(fields):
    """Fields, ``(name, type)`` tuples or a mapping — all to Fields."""
    if isinstance(fields, Mapping):
        fields = list(fields.items())
    out = []
    for item in fields:
        out.append(field(*item) if isinstance(item, tuple) else item)
    return out


def schema(fields):
    """A :class:`Schema` from Fields, ``(name, type)`` pairs, or a mapping.

    Also accepts anything implementing the Arrow PyCapsule schema protocol."""
    if hasattr(fields, "__arrow_c_schema__") or isinstance(fields, Schema):
        return Schema.wrap(_ma.Schema(unwrap(fields)))
    return Schema.wrap(
        _ma.Schema([unwrap(f) for f in _coerce_fields(fields)])
    )


def infer_type(obj):
    """The type `marrow.array` would infer for `obj`."""
    return DataType.wrap(_ma.infer_type(obj))
