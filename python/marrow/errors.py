# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The exception classes a marrow failure raises, named as PyArrow names them.

Every failure the Mojo library raises is an `ArrowError` of some kind
(`marrow/errors.mojo`). The binding layer cannot hand Python the kind: the
stdlib's `def_function` / `def_method` wrappers turn any Mojo error into a bare
`Exception(str(e))`. What survives is the message, which starts with the kind's
name (`"KeyError: drop: column 'x' not found"`), so `install` wraps every
binding callable once, when this module is imported, to re-raise that
`Exception` as the class the name maps to -- with the name stripped, since the
class now says it.

Each class also derives from the builtin a Python caller already catches:
`except KeyError` catches a missing column, `except ValueError` a bad argument
or a corrupt file.
"""

from . import libmarrow

__all__ = [
    "ArrowCorruptError",
    "ArrowException",
    "ArrowIOError",
    "ArrowIndexError",
    "ArrowInternalError",
    "ArrowInvalid",
    "ArrowKeyError",
    "ArrowNotImplementedError",
    "ArrowTypeError",
]


class ArrowException(Exception):
    """Base of every marrow failure; raised as-is for an untagged one."""


class ArrowInvalid(ValueError, ArrowException):
    """A bad argument value or precondition, or a parse error in user text."""


class ArrowTypeError(TypeError, ArrowException):
    """A dtype the operation cannot take, or two that do not match."""


class ArrowKeyError(KeyError, ArrowException):
    """A name that does not resolve: a column, field or parameter."""

    def __str__(self):
        # `KeyError.__str__` is the repr of its argument, which quotes it.
        return ArrowException.__str__(self)


class ArrowIndexError(IndexError, ArrowException):
    """A position out of bounds."""


class ArrowNotImplementedError(NotImplementedError, ArrowException):
    """A legal Arrow or file-format feature marrow does not support."""


class ArrowIOError(OSError, ArrowException):
    """The operating system, an object store or a shared library failed."""


class ArrowCorruptError(ArrowInvalid):
    """Bytes read from a file or buffer that violate the format."""


class ArrowInternalError(RuntimeError, ArrowException):
    """A broken invariant inside marrow: a bug worth reporting."""


# The kinds in `marrow/errors.mojo`, which write their name as the tag.
_BY_TAG = {
    "InvalidError": ArrowInvalid,
    "TypeError": ArrowTypeError,
    "KeyError": ArrowKeyError,
    "IndexError": ArrowIndexError,
    "NotImplementedError": ArrowNotImplementedError,
    "IOError": ArrowIOError,
    "CorruptError": ArrowCorruptError,
    "InternalError": ArrowInternalError,
}


def _translate(exc):
    """The marrow exception `exc` stands for, or None to re-raise it as is.

    Only the binding layer's own flattening is translated: a bare `Exception`,
    or the `ValueError` a failing `__init__` raises. Anything else -- a
    `TypeError` from CPython's argument parsing, a `KeyboardInterrupt` -- is
    already the right exception."""
    if type(exc) not in (Exception, ValueError):
        return None
    message = str(exc)
    tag, sep, rest = message.partition(": ")
    cls = _BY_TAG.get(tag) if sep else None
    if cls is not None:
        return cls(rest)
    elif type(exc) is Exception:
        return ArrowException(message)
    else:
        return None


def _wrap(func, name):
    """`func`, re-raising a flattened marrow error as its class.

    Named by the caller rather than from `func`: reading `__doc__` or
    `__qualname__` off the function inside a Mojo-built `staticmethod`
    segfaults the interpreter."""

    def call(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            translated = _translate(exc)
            if translated is None:
                raise
            raise translated from None

    call.__name__ = call.__qualname__ = name
    return call


def _wraps_method(name, value):
    return type(value).__name__ == "method_descriptor" or (
        name == "__init__" and type(value).__name__ == "wrapper_descriptor"
    )


def install(module):
    """Wrap every function and method of the extension `module` in place.

    Done once, before anything calls into it: the wrapper modules look their
    bindings up through the module (`_ma.table(...)`) at call time, and the
    binding types are mutable heap types, so replacing the attributes covers
    every call site without touching one."""
    for name, value in list(vars(module).items()):
        if name.startswith("_"):
            continue
        elif isinstance(value, type):
            for attr, member in list(vars(value).items()):
                if isinstance(member, staticmethod):
                    setattr(value, attr, staticmethod(_wrap(member.__func__, attr)))
                elif _wraps_method(attr, member):
                    setattr(value, attr, _wrap(member, attr))
        elif callable(value):
            setattr(module, name, _wrap(value, name))


install(libmarrow)
