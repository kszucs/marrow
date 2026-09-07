"""The composition base every Python wrapper shares.

Wrappers hold a binding object rather than inheriting from it. The Mojo side
cannot carry the surface Python needs — `PythonTypeBuilder.bind` installs four
CPython slots (`tp_new`, `tp_init`, `tp_dealloc`, `tp_repr`) and `def_method`
fills the type's `tp_dict` rather than a slot, so operators, rich comparison,
properties and keyword arguments all have to live here. Composition keeps that
split visible: `._binding` is the strict, minimal Mojo object, and the class
around it is the friendly one.
"""

__all__ = ["_Wrapper", "unwrap"]


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


def unwrap(value):
    """The binding behind `value`, or `value` itself if it is already one.

    Every wrapper's methods take user values that may be wrapped or raw — a
    `marrow.Array` or the `libmarrow.Array` inside it — and the binding layer
    only accepts the latter. One helper rather than a `hasattr` check at each
    call site."""
    return value.unwrap() if isinstance(value, _Wrapper) else value
