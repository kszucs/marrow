from std.ffi import external_call, c_char

# from std.io.write import Writable, Writer
from std.memory import ArcPointer, memcpy
from std.python import Python, PythonObject
from std.python._cpython import CPython, PyObjectPtr
from std.sys import size_of

from .dtypes import *
from .arrays import *

comptime ARROW_FLAG_NULLABLE = 2


# TODO: Fix this
fn empty_release_schema(ptr: UnsafePointer[CArrowSchema, MutAnyOrigin]):
    pass


@fieldwise_init
struct CArrowSchema(Copyable, Writable):
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

        if dtype == materialize[null]():
            fmt = "n"
        elif dtype == materialize[bool_]():
            fmt = "b"
        elif dtype == materialize[int8]():
            fmt = "c"
        elif dtype == materialize[uint8]():
            fmt = "C"
        elif dtype == materialize[int16]():
            fmt = "s"
        elif dtype == materialize[uint16]():
            fmt = "S"
        elif dtype == materialize[int32]():
            fmt = "i"
        elif dtype == materialize[uint32]():
            fmt = "I"
        elif dtype == materialize[int64]():
            fmt = "l"
        elif dtype == materialize[uint64]():
            fmt = "L"
        elif dtype == materialize[float16]():
            fmt = "e"
        elif dtype == materialize[float32]():
            fmt = "f"
        elif dtype == materialize[float64]():
            fmt = "g"
        elif dtype == materialize[binary]():
            fmt = "z"
        elif dtype == materialize[string]():
            fmt = "u"
        elif dtype.is_struct():
            print("EEE")

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
                fmt.as_c_string_slice().unsafe_ptr()
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
                field_name.as_c_string_slice().unsafe_ptr()
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
            return materialize[null]()
        elif fmt == "b":
            return materialize[bool_]()
        elif fmt == "c":
            return materialize[int8]()
        elif fmt == "C":
            return materialize[uint8]()
        elif fmt == "s":
            return materialize[int16]()
        elif fmt == "S":
            return materialize[uint16]()
        elif fmt == "i":
            return materialize[int32]()
        elif fmt == "I":
            return materialize[uint32]()
        elif fmt == "l":
            return materialize[int64]()
        elif fmt == "L":
            return materialize[uint64]()
        elif fmt == "e":
            return materialize[float16]()
        elif fmt == "f":
            return materialize[float32]()
        elif fmt == "g":
            return materialize[float64]()
        elif fmt == "z":
            return materialize[binary]()
        elif fmt == "u":
            return materialize[string]()
        elif fmt == "+l":
            var field = self.children[0][].to_field()
            return list_(field.dtype.copy())
        elif fmt == "+s":
            var fields = List[Field]()
            for i in range(self.n_children):
                fields.append(self.children[i][].to_field())
            return struct_(fields)
        else:
            raise Error("Unknown format: " + fmt)

    fn to_field(self) raises -> Field:
        var name = StringSlice(unsafe_from_utf8_ptr=self.name)
        var dtype = self.to_dtype()
        var nullable = self.flags & ARROW_FLAG_NULLABLE
        return Field(String(name), dtype^, nullable != 0)

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this CArrowSchema to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        var metadata = 'metadata="{}", '.format(
            StringSlice(unsafe_from_utf8_ptr=self.metadata)
        ) if self.metadata else ""

        writer.write(
            'CArrowSchema(name="{}", format="{}", {}n_children={})'.format(
                StringSlice(unsafe_from_utf8_ptr=self.name),
                StringSlice(unsafe_from_utf8_ptr=self.format),
                metadata,
                self.n_children,
            )
        )

    fn __str__(self) -> String:
        return String.write(self)

    fn __repr__(self) -> String:
        return String.write(self)


@fieldwise_init
struct CArrowArray(Copyable):
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

    @staticmethod
    fn from_pyarrow(pyobj: PythonObject) raises -> CArrowArray:
        var ptr = alloc[CArrowArray](1)
        pyobj._export_to_c(Int(ptr))
        return ptr.take_pointee()

    fn to_array(self, dtype: DataType) raises -> ArrayData:
        var bitmap: ArcPointer[Bitmap]
        if self.buffers[0]:
            bitmap = ArcPointer(
                Bitmap(Buffer.view(self.buffers[0], self.length, DType.bool))
            )
        else:
            # bitmaps are allowed to be nullptrs by the specification, in this
            # case we allocate a new buffer to hold the null bitmap
            bitmap = ArcPointer(Bitmap.alloc(self.length))
            bitmap[].unsafe_range_set(0, self.length, True)

        var buffers = List[ArcPointer[Buffer]]()
        if dtype.is_numeric() or dtype == materialize[bool_]():
            var buffer = Buffer.view(self.buffers[1], self.length, dtype.native)
            buffers.append(ArcPointer(buffer^))
        elif dtype == materialize[string]():
            var offsets = Buffer.view(
                self.buffers[1], self.length + 1, DType.uint32
            )
            var values_size = Int(offsets.unsafe_get(Int(self.length)))
            var values = Buffer.view(self.buffers[2], values_size, DType.uint8)
            buffers.append(ArcPointer(offsets^))
            buffers.append(ArcPointer(values^))
        elif dtype.is_list():
            var offsets = Buffer.view(
                self.buffers[1], self.length + 1, DType.uint32
            )
            buffers.append(ArcPointer(offsets^))
        elif dtype.is_struct():
            # Since the children buffers are handled below there is nothing to do here.
            pass
        else:
            raise Error("Unknown dtype: " + String(dtype))

        var children = List[ArcPointer[ArrayData]]()
        for i in range(self.n_children):
            var child_field = dtype.fields[i].copy()
            var child_array = self.children[i][].to_array(child_field.dtype)
            children.append(ArcPointer(child_array^))

        return ArrayData(
            dtype=dtype.copy(),
            length=Int(self.length),
            bitmap=bitmap,
            buffers=buffers^,
            children=children^,
            offset=Int(self.offset),
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
    var get_last_error: fn(UnsafePointer[CArrowArrayStream, MutAnyOrigin]) -> UnsafePointer[
        UInt8, MutAnyOrigin
    ]
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
            raise Error("Failed to get schema " + String(err))
        if not schema:
            raise Error("The schema pointer is null")
        return schema.take_pointee()

    fn c_next(self) raises -> CArrowArray:
        """Return the next buffer in the streeam."""
        var arrow_array = alloc[CArrowArray](1)
        var err = self.handle[].get_next(self.handle, arrow_array)
        if err != 0:
            raise Error("Failed to get next arrow array " + String(err))
        if not arrow_array:
            raise Error("The arrow array pointer is null")
        return arrow_array.take_pointee()
