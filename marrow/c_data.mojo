from std.ffi import external_call, c_char
from std.memory import ArcPointer, memcpy
from std.sys import size_of
from .buffers import (
    Allocation,
    Buffer,
    BufferBuilder,
    DeviceType,
)
from .bitmap import Bitmap

import std.math as math
from std.python import Python, PythonObject
from std.python._cpython import CPython, PyObjectPtr

from .dtypes import *
from .arrays import *

comptime ARROW_FLAG_NULLABLE = 2


# TODO: Fix this
fn empty_release_schema(ptr: UnsafePointer[CArrowSchema, MutAnyOrigin]):
    pass


@fieldwise_init
struct CArrowSchema(Copyable):
    var format: UnsafePointer[c_char, MutAnyOrigin]
    var name: UnsafePointer[c_char, MutAnyOrigin]
    var metadata: UnsafePointer[c_char, MutAnyOrigin]
    var flags: Int64
    var n_children: Int64
    var children: UnsafePointer[
        UnsafePointer[CArrowSchema, MutAnyOrigin], MutAnyOrigin
    ]
    var dictionary: UnsafePointer[CArrowSchema, MutAnyOrigin]
    # TODO(kszucs): release callback must be called otherwise memory gets leaked
    var release: fn(UnsafePointer[CArrowSchema, MutAnyOrigin]) -> None
    var private_data: OpaquePointer[MutAnyOrigin]

    fn __del__(deinit self):
        self.release(UnsafePointer(to=self))

    @staticmethod
    fn from_pyarrow(pyobj: PythonObject) raises -> CArrowSchema:
        var ptr = alloc[CArrowSchema](1)
        pyobj._export_to_c(Int(ptr))
        return ptr.take_pointee()

    fn to_pyarrow(self) raises -> PythonObject:
        var pa = Python.import_module("pyarrow")
        var ptr = UnsafePointer(to=self)
        return pa.Schema._import_from_c(Int(ptr))

    @staticmethod
    fn from_dtype(dtype: DataType) -> CArrowSchema:
        var fmt: String
        var n_children: Int64 = 0
        var children = UnsafePointer[
            UnsafePointer[CArrowSchema, MutAnyOrigin], MutAnyOrigin
        ]()

        if dtype == null:
            fmt = "n"
        elif dtype == bool_:
            fmt = "b"
        elif dtype == int8:
            fmt = "c"
        elif dtype == uint8:
            fmt = "C"
        elif dtype == int16:
            fmt = "s"
        elif dtype == uint16:
            fmt = "S"
        elif dtype == int32:
            fmt = "i"
        elif dtype == uint32:
            fmt = "I"
        elif dtype == int64:
            fmt = "l"
        elif dtype == uint64:
            fmt = "L"
        elif dtype == float16:
            fmt = "e"
        elif dtype == float32:
            fmt = "f"
        elif dtype == float64:
            fmt = "g"
        elif dtype == binary:
            fmt = "z"
        elif dtype.is_string():
            fmt = "u"
        elif dtype.is_fixed_size_list():
            fmt = {"+w:", dtype.size}
            n_children = 1
            children = alloc[UnsafePointer[CArrowSchema, MutAnyOrigin]](1)
            var child = CArrowSchema.from_field(dtype.fields[0])
            children[0].init_pointee_move(child^)
        elif dtype.is_struct():
            fmt = "+s"
            n_children = Int64(len(dtype.fields))
            children = alloc[UnsafePointer[CArrowSchema, MutAnyOrigin]](
                Int(n_children)
            )

            for i in range(n_children):
                var child = CArrowSchema.from_field(dtype.fields[i])
                children[i].init_pointee_move(child^)
        else:
            fmt = ""
            # constrained[False, "Unknown dtype"]()

        return CArrowSchema(
            format=UnsafePointer[c_char, MutAnyOrigin](
                unsafe_from_address=Int(fmt.as_c_string_slice().unsafe_ptr())
            ),
            name=UnsafePointer[c_char, MutAnyOrigin](),
            metadata=UnsafePointer[c_char, MutAnyOrigin](),
            flags=0,
            n_children=n_children,
            children=children,
            dictionary=UnsafePointer[CArrowSchema, MutAnyOrigin](),
            # TODO(kszucs): currently there is no way to pass a mojo callback to C
            release=empty_release_schema,
            private_data=OpaquePointer[MutAnyOrigin](),
        )

    @staticmethod
    fn from_field(field: Field) -> CArrowSchema:
        var flags: Int64 = 0  # TODO: nullable

        var field_name = field.name
        return CArrowSchema(
            format=UnsafePointer[c_char, MutAnyOrigin](),
            name=UnsafePointer[c_char, MutAnyOrigin](
                unsafe_from_address=Int(
                    field_name.as_c_string_slice().unsafe_ptr()
                )
            ),
            metadata=UnsafePointer[c_char, MutAnyOrigin](),
            flags=flags,
            n_children=0,
            children=UnsafePointer[
                UnsafePointer[CArrowSchema, MutAnyOrigin], MutAnyOrigin
            ](),
            dictionary=UnsafePointer[CArrowSchema, MutAnyOrigin](),
            # TODO(kszucs): currently there is no way to pass a mojo callback to C
            release=empty_release_schema,
            private_data=OpaquePointer[MutAnyOrigin](),
        )

    fn to_dtype(self) raises -> DataType:
        var fmt = StringSlice(unsafe_from_utf8_ptr=self.format)
        # TODO(kszucs): not the nicest, but dictionary literals are not supported yet
        if fmt == "n":
            return null
        elif fmt == "b":
            return bool_
        elif fmt == "c":
            return int8
        elif fmt == "C":
            return uint8
        elif fmt == "s":
            return int16
        elif fmt == "S":
            return uint16
        elif fmt == "i":
            return int32
        elif fmt == "I":
            return uint32
        elif fmt == "l":
            return int64
        elif fmt == "L":
            return uint64
        elif fmt == "e":
            return float16
        elif fmt == "f":
            return float32
        elif fmt == "g":
            return float64
        elif fmt == "z":
            return binary
        elif fmt == "u":
            return string
        elif fmt == "+l":
            var field = self.children[0][].to_field()
            return list_(field.dtype.copy())
        elif fmt.startswith("+w:"):
            var size = Int(fmt[3:])
            var field = self.children[0][].to_field()
            return fixed_size_list_(field.dtype.copy(), size)
        elif fmt == "+s":
            var fields = List[Field]()
            for i in range(self.n_children):
                fields.append(self.children[i][].to_field())
            return struct_(fields)
        else:
            raise Error("Unknown format: ", fmt)

    fn to_field(self) raises -> Field:
        var name = StringSlice(unsafe_from_utf8_ptr=self.name)
        var dtype = self.to_dtype()
        var nullable = self.flags & ARROW_FLAG_NULLABLE
        return Field(String(name), dtype^, nullable != 0)


fn _release_c_array(ptr: UnsafePointer[UInt8, MutAnyOrigin]) -> None:
    """Release callback for CArrowArray imported via the C Data Interface.

    Called when the last Buffer (or Bitmap) that references the imported array
    is dropped.  Invokes the C-level release callback so the producer can free
    its resources, then frees the Mojo heap allocation that owns the struct.
    """
    var c_ptr = ptr.bitcast[CArrowArray]()
    c_ptr[].release(c_ptr)
    c_ptr.free()


@fieldwise_init
struct CArrowArray(Movable):
    var length: Int64
    var null_count: Int64
    var offset: Int64
    var n_buffers: Int64
    var n_children: Int64
    var buffers: UnsafePointer[
        UnsafePointer[NoneType, MutAnyOrigin], MutAnyOrigin
    ]
    var children: UnsafePointer[
        UnsafePointer[CArrowArray, MutAnyOrigin], MutAnyOrigin
    ]
    var dictionary: UnsafePointer[CArrowArray, MutAnyOrigin]
    var release: fn(UnsafePointer[CArrowArray, MutAnyOrigin]) -> None
    var private_data: OpaquePointer[MutAnyOrigin]

    fn __del__(deinit self):
        self.release(UnsafePointer(to=self))

    @staticmethod
    fn from_pyarrow(pyobj: PythonObject) raises -> CArrowArray:
        var ptr = alloc[CArrowArray](1)
        pyobj._export_to_c(Int(ptr))
        return ptr.take_pointee()

    fn _to_array(
        self, dtype: DataType, owner: ArcPointer[Allocation]
    ) raises -> Array:
        """Build an Array from this CArrowArray, all buffers sharing one owner.

        All Buffer views hold a copy of `owner` (an ArcPointer, so
        copying just bumps the ref-count).  The C release callback fires
        automatically once the last buffer is dropped.
        """
        # Buffer sizes must cover all elements including the offset, because the
        # raw C buffers start at element 0 regardless of the logical array offset.
        var length = self.length + self.offset

        var bitmap: Optional[Bitmap]
        if self.buffers[0]:
            bitmap = Bitmap(
                Buffer.from_foreign(
                    self.buffers[0],
                    math.ceildiv(Int(length), 8),
                    owner,
                ),
                0,
                Int(length),
            )
        else:
            # null bitmap pointer means all elements are valid
            bitmap = None

        var buffers = List[Buffer]()
        var children = List[Array]()

        if dtype.is_bool():
            if self.n_buffers != 2:
                raise Error(
                    t"bool array must have 2 buffers, got {self.n_buffers}"
                )
            if self.n_children != 0:
                raise Error(
                    t"bool array must have 0 children, got {self.n_children}"
                )
            var values = Buffer.from_foreign(
                self.buffers[1],
                math.ceildiv(Int(length), 8),
                owner,
            )
            buffers.append(values^)
        elif dtype.is_primitive():
            if self.n_buffers != 2:
                raise Error(
                    t"numeric array must have 2 buffers, got {self.n_buffers}"
                )
            if self.n_children != 0:
                raise Error(
                    "numeric array must have 0 children, got",
                    t" {self.n_children}.",
                )
            var values = Buffer.from_foreign(
                self.buffers[1], Int(length) * dtype.byte_width(), owner
            )
            buffers.append(values^)
        elif dtype.is_list():
            if self.n_buffers != 2:
                raise Error(
                    t"list array must have 2 buffers, got {self.n_buffers}"
                )
            if self.n_children != 1:
                raise Error(
                    t"list array must have 1 child, got {self.n_children}"
                )
            # list has only an offsets buffer; child data lives in self.children
            var size = (length + 1) * Int64(size_of[DType.int32]())
            var offsets = Buffer.from_foreign(self.buffers[1], size, owner)
            buffers.append(offsets^)
            # add the single values child array
            var values_field = dtype.fields[0].copy()
            var values_array = self.children[0][]._to_array(
                values_field.dtype, owner
            )
            children.append(values_array^)
        elif dtype.is_string():
            if self.n_buffers != 3:
                raise Error(
                    t"string array must have 3 buffers, got {self.n_buffers}"
                )
            if self.n_children != 0:
                raise Error(
                    t"string array must have 0 children, got {self.n_children}"
                )
            var size = (length + 1) * Int64(size_of[DType.int32]())
            var offsets = Buffer.from_foreign(self.buffers[1], size, owner)
            var data_len = offsets.unsafe_get[DType.int32](Int(length))
            var values = Buffer.from_foreign(self.buffers[2], data_len, owner)
            buffers.append(offsets^)
            buffers.append(values^)
        elif dtype.is_fixed_size_list():
            if self.n_buffers != 1:
                raise Error(
                    "fixed_size_list array must have 1 buffer, got",
                    t" {self.n_buffers}",
                )
            if self.n_children != 1:
                raise Error(
                    "fixed_size_list array must have 1 child, got",
                    t" {self.n_children}",
                )
            var values_field = dtype.fields[0].copy()
            var values_array = self.children[0][]._to_array(
                values_field.dtype, owner
            )
            children.append(values_array^)
        elif dtype.is_struct():
            if self.n_buffers != 1:
                raise Error(
                    t"struct array must have 1 buffer, got {self.n_buffers}"
                )
            if self.n_children != Int64(len(dtype.fields)):
                raise Error(
                    t"struct array must have {len(dtype.fields)} children, got"
                    t" {self.n_children}"
                )
            for i in range(self.n_children):
                var child_field = dtype.fields[i].copy()
                var child_array = self.children[i][]._to_array(
                    child_field.dtype, owner
                )
                children.append(child_array^)
        else:
            raise Error("unsupported dtype for buffer import: ", dtype)

        return Array(
            dtype=dtype.copy(),
            length=Int(self.length),
            nulls=Int(self.null_count),
            bitmap=bitmap,
            buffers=buffers^,
            children=children^,
            offset=Int(self.offset),
        )

    fn _to_device_array(
        self,
        dtype: DataType,
        owner: ArcPointer[Allocation],
        device_type: Int32,
        device_id: Int64,
    ) raises -> Array:
        """Build an Array from device-resident CArrowArray buffers.

        TODO: Zero-copy import of device arrays requires wrapping raw device
        pointers in Mojo's DeviceBuffer.  This is not yet supported because
        DeviceBuffer construction needs an AsyncRT handle that is not provided
        by the C Device Data Interface.  Once Mojo exposes an API to adopt a
        raw device pointer into a DeviceBuffer, this method can be completed.
        """
        raise Error(
            "_to_device_array: zero-copy device array import is not yet"
            " implemented; Mojo does not yet expose a way to wrap raw device"
            " pointers in DeviceBuffer without an AsyncRT handle"
        )

    fn to_array(deinit self, dtype: DataType) raises -> Array:
        """Convert to an Array, taking ownership of the C struct.

        The CArrowArray is moved onto the heap and wrapped in a
        Allocation.  Every Buffer / Bitmap view shares the same
        ArcPointer[Allocation], so the C release callback fires
        automatically when the last buffer referencing this import is dropped.
        """
        var heap_c = alloc[CArrowArray](1)
        heap_c.init_pointee_move(self^)
        var owner = ArcPointer(
            Allocation.foreign(heap_c.bitcast[UInt8](), _release_c_array)
        )
        return heap_c[]._to_array(dtype, owner)


# ---------------------------------------------------------------------------
# Arrow C Device Data Interface
# https://arrow.apache.org/docs/format/CDeviceDataInterface.html
# ---------------------------------------------------------------------------


fn _release_c_device_array(ptr: UnsafePointer[UInt8, MutAnyOrigin]) -> None:
    """Release callback for CArrowDeviceArray imported via the C Device Data Interface.

    Called when the last Buffer that references the imported array is dropped.
    Invokes the C-level release callback on the embedded ArrowArray, then frees
    the Mojo heap allocation.
    """
    var c_ptr = ptr.bitcast[CArrowDeviceArray]()
    c_ptr[].array.release(UnsafePointer(to=c_ptr[].array))
    c_ptr.free()


@fieldwise_init
struct CArrowDeviceArray(Movable):
    """Arrow C Device Data Interface array struct.

    Extends ArrowArray with device-location metadata.  All metadata fields
    (`device_id`, `device_type`, `sync_event`, `reserved`) reside in CPU
    memory; only the buffer *pointers* inside `array.buffers` point to
    device memory.

    Layout matches the C spec:
        struct ArrowDeviceArray {
            struct ArrowArray array;
            int64_t device_id;
            ArrowDeviceType device_type;  // int32_t
            void* sync_event;
            int64_t reserved[3];          // must be zero
        };

    Notes:
        - `reserved0/1/2` must all be zero (spec requirement).
        - `sync_event` should be synchronized via `ctx.synchronize()` before
          accessing buffers if non-null; per-event-type sync is a future enhancement.
        - `from_pyarrow` is not yet implemented — PyArrow's `__arrow_c_device_array__`
          protocol support is still evolving.
    """

    var array: CArrowArray
    var device_id: Int64
    var device_type: Int32
    var _pad: Int32  # explicit padding to align sync_event to 8 bytes (C ABI)
    var sync_event: OpaquePointer[MutAnyOrigin]
    var reserved0: Int64
    var reserved1: Int64
    var reserved2: Int64

    fn to_array(
        deinit self, dtype: DataType, ctx: DeviceContext
    ) raises -> Array:
        """Import a device array into marrow, taking ownership of the C struct.

        The CArrowDeviceArray is moved onto the heap and wrapped in an
        `Allocation`.  All Buffer views share the same `ArcPointer[Allocation]`
        owner so the C release callback fires when the last buffer is dropped.

        If `sync_event` is non-null, `ctx.synchronize()` is called first to
        ensure all preceding device operations are complete before the buffers
        are accessed.

        Args:
            dtype: The Arrow data type describing the array's schema.
            ctx:   The DeviceContext associated with the device buffers.

        Returns:
            An `Array` whose buffers reference the device memory directly
            (zero-copy for device types; CPU for ARROW_DEVICE_CPU).
        """
        if self.sync_event:
            ctx.synchronize()

        var heap_c = alloc[CArrowDeviceArray](1)
        heap_c.init_pointee_move(self^)
        var owner = ArcPointer(
            Allocation.foreign(heap_c.bitcast[UInt8](), _release_c_device_array)
        )

        var device_type = heap_c[].device_type
        var device_id = heap_c[].device_id

        # For CPU device type, delegate to the existing CArrowArray import path.
        if device_type == DeviceType.CPU:
            return heap_c[].array._to_array(dtype, owner)

        # For device memory, wrap each buffer pointer as a DEVICE buffer.
        # The raw pointers in array.buffers point to device-resident memory.
        return heap_c[].array._to_device_array(
            dtype, owner, device_type, device_id
        )


# See: https://arrow.apache.org/docs/format/CStreamInterface.html


@fieldwise_init
struct CArrowArrayStream(Copyable, TrivialRegisterPassable):
    var get_schema: fn(
        UnsafePointer[CArrowArrayStream, MutAnyOrigin],
        UnsafePointer[CArrowSchema, MutAnyOrigin],
    ) -> Int32
    var get_next: fn(
        UnsafePointer[CArrowArrayStream, MutAnyOrigin],
        UnsafePointer[CArrowArray, MutAnyOrigin],
    ) -> Int32
    var get_last_error: fn(
        UnsafePointer[CArrowArrayStream, MutAnyOrigin]
    ) -> UnsafePointer[UInt8, MutAnyOrigin]
    var release: fn(UnsafePointer[CArrowArrayStream, MutAnyOrigin]) -> None
    var private_data: OpaquePointer[MutAnyOrigin]


@fieldwise_init
struct ArrowArrayStream(Copyable):
    """Provide an fiendly interface to the C Arrow Array Stream."""

    var handle: UnsafePointer[CArrowArrayStream, MutAnyOrigin]

    @staticmethod
    fn from_pyarrow(
        pyobj: PythonObject, cpython: CPython
    ) raises -> ArrowArrayStream:
        """Ask a PyArrow table for its arrow array stream interface."""
        var stream = pyobj.__arrow_c_stream__()
        var ptr = cpython.PyCapsule_GetPointer(
            stream.steal_data(), "arrow_array_stream"
        )
        if not ptr:
            raise Error("Failed to get the arrow array stream pointer")

        return ArrowArrayStream(ptr.bitcast[CArrowArrayStream]())

    fn c_schema(self) raises -> CArrowSchema:
        """Return the C variant of the Arrow Schema."""
        var schema = alloc[CArrowSchema](1)
        var err = self.handle[].get_schema(self.handle, schema)
        if err != 0:
            raise Error("Failed to get schema ", err)
        if not schema:
            raise Error("The schema pointer is null")
        return schema.take_pointee()

    fn c_next(self) raises -> CArrowArray:
        """Return the next buffer in the streeam."""
        var arrow_array = alloc[CArrowArray](1)
        var err = self.handle[].get_next(self.handle, arrow_array)
        if err != 0:
            raise Error("Failed to get next arrow array ", err)
        if not arrow_array:
            raise Error("The arrow array pointer is null")
        return arrow_array.take_pointee()
