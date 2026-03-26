"""Arrow columnar arrays — always immutable.

Every typed array (`PrimitiveArray`, `StringArray`, `ListArray`, `StructArray`)
is immutable.  To *build* an array incrementally, use the corresponding builder
from `marrow.builders` and call `finish()`.

`BoolArray` is an alias for `PrimitiveArray[bool_]`.

Array — the trait
-----------------
`Array` is the trait that all typed arrays implement.  It provides the common
read-only interface: `type()`, `null_count()`, `is_valid()`, and `as_any()`.

AnyArray — the type-erased handle
----------------------------------
`AnyArray` is the type-erased, immutable handle used for storage, exchange
(C Data Interface), and visitor dispatch.  The concrete typed array lives on
the heap behind an `ArcPointer`; copies are O(1) ref-count bumps.

ArrayData — generic flat layout
---------------------------------
`ArrayData` is a plain @fieldwise_init struct (same 7 fields as the old
AnyArray) produced on demand by `as_data()`.  It is used for the C Data
Interface, building nested arrays, and other interop where a flat layout is
required.  It is NOT stored inside AnyArray.
"""

from std.memory import memcpy, ArcPointer
from std.sys import size_of
from std.gpu.host import DeviceContext
from std.python import Python, PythonObject
from std.python.conversions import ConvertibleFromPython, ConvertibleToPython
from .buffers import Buffer, BufferBuilder
from .bitmap import Bitmap, BitmapBuilder
from .views import BufferView, BitmapView
from .dtypes import *
from .builders import PrimitiveBuilder, StringBuilder
from .scalars import PrimitiveScalar, StringScalar, ListScalar


trait Array(
    ConvertibleToPython,
    Copyable,
    Equatable,
    ImplicitlyDestructible,
    Movable,
    Sized,
