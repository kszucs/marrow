# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The error taxonomy: every failure marrow raises is an `ArrowError`.

`ArrowError` is the trait; each kind is a concrete error type conforming to it,
named as Python names the exception PyArrow raises for it:

```mojo
raise KeyError(t"drop: column '{name}' not found")
raise CorruptError("parquet: bad magic")
```

A function that fails one way declares that type — `raises CorruptError` —
and a caller `except`s it with the type known. A function that fails several
ways declares `DynError`, the erased box: every `ArrowError` converts into it
implicitly, and `e.isa[KeyError]()` tells them apart, as `DynArray` does for
arrays. Code generic over the error of a callee takes `raises E` with
`E: ArrowError`.

Most of marrow still declares bare `raises`, which keeps only an error's
text. Every kind writes itself with its name first — `"KeyError: drop: ..."` —
so `DynError(e: Error)` recovers the kind from it, and the Python bindings map
it onto the exception class of the same name (`python/marrow/errors.py`). The
names are therefore a contract with Python.

Choosing a kind:

- `InvalidError` — a bad argument value or precondition, the wrong state
  ("writer is closed"), a parse or usage error in user text, arithmetic
  overflow, a lossy conversion, or a value exceeding an offset or id space.
- `TypeError` — a dtype this operation cannot take, or two that do not match.
  A `dispatch_*` refusing a dtype is a `TypeError`, not an `InternalError`: the
  runtime lane reaches it with user data.
- `KeyError` — a name that does not resolve: column, field, parameter.
- `IndexError` — a position the *caller* supplied is out of bounds.
- `NotImplementedError` — a legal Arrow or file-format feature marrow does not
  support, or one this build left out (no GPU, a reader not compiled to decode
  it). An unrecognised enum value read from a file is this too, as in Arrow C++.
- `IOError` — the operating system, an object store or `dlopen` failed.
- `CorruptError` — bytes read from a file or buffer violate the format: bad
  magic, checksum mismatch, a truncated structure, an offset past the end. A
  bounds check is corrupt when the offset came from the bytes, an `IndexError`
  when it came from the caller. A codec failing to *decompress* is corrupt.
- `InternalError` — a broken invariant no input can reach: a marrow bug. A
  codec failing to initialise or compress is internal.
"""


trait ArrowError(Deinitable, ImplicitlyCopyable, Writable):
    """A failure marrow raises: a kind, named by the type, and a message."""

    comptime kind: StaticString
    """The kind's name, and the tag its text starts with."""

    def __init__[M: Writable](out self, message: M):
        """An error whose message is `message`, typically a t-string."""
        ...

    def message(self) -> String:
        """The message, without the kind."""
        ...


struct Failure[name: StaticString](ArrowError):
    """The one implementation behind every kind; see the aliases below."""

    comptime kind = Self.name
    var _message: String

    @inline(.never)
    def __init__[M: Writable](out self, message: M):
        self._message = String(message)

    def message(self) -> String:
        return self._message

    def write_to(self, mut writer: Some[Writer]):
        writer.write(Self.kind, ": ", self._message)


comptime InvalidError = Failure["InvalidError"]
comptime TypeError = Failure["TypeError"]
comptime KeyError = Failure["KeyError"]
comptime IndexError = Failure["IndexError"]
comptime NotImplementedError = Failure["NotImplementedError"]
comptime IOError = Failure["IOError"]
comptime CorruptError = Failure["CorruptError"]
comptime InternalError = Failure["InternalError"]

comptime _KINDS: List[StaticString] = [
    InvalidError.kind,
    TypeError.kind,
    KeyError.kind,
    IndexError.kind,
    NotImplementedError.kind,
    IOError.kind,
    CorruptError.kind,
    InternalError.kind,
]


struct DynError(ImplicitlyCopyable, Writable):
    """Any `ArrowError`, erased: what a function failing several ways raises.

    `kind` is empty for an `Error` that no `ArrowError` wrote — one from the
    stdlib, Python or a C library."""

    var kind: StaticString
    var message: String

    def __init__[M: Writable](out self, kind: StaticString, message: M):
        """An error of the named kind — how a caught error is re-raised with
        context and keeps its kind."""
        self.kind = kind
        self.message = String(message)

    @implicit
    def __init__[E: ArrowError](out self, error: E):
        self.kind = E.kind
        self.message = error.message()

    @implicit
    def __init__(out self, error: Error):
        """Recover the kind a bare `raises` frame kept only as text."""
        var text = String(error)
        self.kind = ""
        self.message = text
        comptime for kind in _KINDS:
            var tag = String(kind, ": ")
            if not self.kind and text.startswith(tag):
                self.kind = kind
                self.message = String(text.removeprefix(tag))

    def isa[E: ArrowError](self) -> Bool:
        """Whether this is an `E`."""
        return self.kind == E.kind

    def write_to(self, mut writer: Some[Writer]):
        if self.kind:
            writer.write(self.kind, ": ")
        writer.write(self.message)
