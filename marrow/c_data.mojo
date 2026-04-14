from std.ffi import external_call, c_char
from std.memory import ArcPointer, memcpy
from std.python import Python, PythonObject
from std.python._cpython import CPython, PyObjectPtr
from std.sys import size_of
from .buffers import (
    Allocation,
    Buffer,
    Bitmap,
    DeviceType,
)

import std.math as math

from .dtypes import *
from .arrays import *
from .schema import Schema
from .tabular import RecordBatch, Table

comptime ARROW_FLAG_NULLABLE = 2


def _alloc_c_string(s: String) -> UnsafePointer[c_char, MutAnyOrigin]:
    """Copy a Mojo String into a heap-allocated null-terminated C string.

    The caller owns the returned buffer and must free it when done.

    Note: copies len(s) bytes then writes an explicit null terminator.
    String.unsafe_ptr() is not guaranteed to be null-terminated (SSO inline
    storage leaves bytes past len(s) uninitialized).
    TODO: replace with unsafe_cstr_ptr() once available in this Mojo build.
    """
    var n = s.byte_length()
    var buf = alloc[c_char](n + 1)
    memcpy(dest=buf.bitcast[UInt8](), src=s.unsafe_ptr(), count=n)
    buf.bitcast[UInt8]()[n] = 0
    return UnsafePointer[c_char, MutAnyOrigin](unsafe_from_address=Int(buf))


def _release_schema_capsule(capsule: PyObjectPtr):
    """PyCapsule destructor for "arrow_schema" capsules.

    Called by Python's GC when a schema capsule is collected.
    The capsule holds a raw pointer to a heap-allocated CArrowSchema.
    If its release callback is still set (i.e. not yet consumed by an
    Arrow importer), we call it to free the format/name strings and children,
    then free the struct shell itself.
    If the release callback has been zeroed (capsule consumed via pycapsule
    import), we just free the shell.
    """
    try:
        var py = Python()
        ref cpy = py.cpython()
        var ptr = cpy.PyCapsule_GetPointer(capsule, "arrow_schema")
        if ptr:
            var c_schema = ptr.bitcast[CArrowSchema]()
            # Guard against double-free: an Arrow importer zeroes the release
            # field after taking ownership.
            if UnsafePointer(to=c_schema[].release).bitcast[UInt64]()[0] != 0:
                c_schema[].release(c_schema)
            c_schema.free()
    except:
        pass


def _release_exported_schema(ptr: UnsafePointer[CArrowSchema, MutAnyOrigin]):
    """Arrow release callback for CArrowSchemas exported from Mojo.

    Arrow calls this (via the release function pointer) when it is done with
    an imported schema.  Frees:
    - The heap-allocated child CArrowSchema struct shells (their own release
      callbacks were already invoked by Arrow's recursive import).
    - The children pointer array.
    - The heap-allocated format and name C strings.
    Nulls the release field per the Arrow spec so double-free is detectable.
    """
    for i in range(Int(ptr[].n_children)):
        ptr[].children[i].free()
    if ptr[].n_children > 0:
        ptr[].children.free()
    if ptr[].format:
        ptr[].format.free()
    if ptr[].name:
        ptr[].name.free()
    UnsafePointer(to=ptr[].release).bitcast[UInt64]()[0] = 0


@fieldwise_init
struct CArrowSchema(Copyable, Movable):
    """Arrow C Data Interface schema struct (ArrowSchema).

    Ownership model
    ---------------
    A CArrowSchema value owns its heap resources (format/name C strings,
    children) through the `release` callback.  When the struct is no longer
    needed, `release` must be called exactly once — `__del__` handles this
    automatically, but only if `release` is still non-null.

    Arrow importers take ownership by copying the struct fields then zeroing
    the source's release field (per the Arrow C Data Interface spec).  After a
    transfer the zeroed source is safe to drop.  `__del__` guards against this
    by checking for a null release.

    Lifecycle for Python export:
        1. `from_dtype` / `from_field` / `from_schema` builds the struct value.
        2. `to_pycapsule` moves it onto the heap and wraps it in a PyCapsule.
        3. `_release_schema_capsule` (PyCapsule destructor) calls `release` and
           frees the struct shell when Python GC collects the capsule.
    """

    var format: UnsafePointer[c_char, MutAnyOrigin]
    var name: UnsafePointer[c_char, MutAnyOrigin]
    var metadata: UnsafePointer[c_char, MutAnyOrigin]
    var flags: Int64
    var n_children: Int64
    var children: UnsafePointer[
        UnsafePointer[CArrowSchema, MutAnyOrigin], MutAnyOrigin
    ]
    var dictionary: UnsafePointer[CArrowSchema, MutAnyOrigin]
    var release: def(UnsafePointer[CArrowSchema, MutAnyOrigin]) thin -> None
    var private_data: OpaquePointer[MutAnyOrigin]

    def __del__(deinit self):
        # Guard: release is zeroed by an Arrow importer after it takes ownership,
        # so we only call it when we still own the resources.
        if UnsafePointer(to=self.release).bitcast[UInt64]()[0] != 0:
            self.release(UnsafePointer(to=self))

    @staticmethod
    def from_dtype(
        dtype: AnyDataType,
    ) raises -> CArrowSchema:
        """Build a CArrowSchema value for a DataType.

        The format string is heap-allocated as a raw C string owned by the
        `format` pointer; `_release_exported_schema` frees it.  Child schemas
        are also heap-allocated so the children pointer array survives moves.

        The returned value owns all resources; `__del__` calls
        `_release_exported_schema` when it goes out of scope, unless ownership
        has been transferred via `to_pycapsule`.

        Call `.to_pycapsule()` to wrap the result in a Python capsule.
        """
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
        elif dtype.is_list():
            fmt = "+l"
            n_children = 1
            children = alloc[UnsafePointer[CArrowSchema, MutAnyOrigin]](1)
            # Move child value onto the heap so the pointer stays valid after
            # this stack frame is gone.
            var child0 = CArrowSchema.from_field(
                dtype.as_list_type().value_field().copy()
            )
            var child0_ptr = alloc[CArrowSchema](1)
            child0_ptr.init_pointee_move(child0^)
            children[0] = child0_ptr
        elif dtype.is_fixed_size_list():
            var fsl = dtype.as_fixed_size_list_type()
            fmt = {"+w:", fsl.size}
            n_children = 1
            children = alloc[UnsafePointer[CArrowSchema, MutAnyOrigin]](1)
            var child0 = CArrowSchema.from_field(fsl.value_field().copy())
            var child0_ptr = alloc[CArrowSchema](1)
            child0_ptr.init_pointee_move(child0^)
            children[0] = child0_ptr
        elif dtype.is_struct():
            fmt = "+s"
            var st = dtype.as_struct_type()
            n_children = Int64(len(st.fields))
            children = alloc[UnsafePointer[CArrowSchema, MutAnyOrigin]](
                Int(n_children)
            )
            for i in range(Int(n_children)):
                var child = CArrowSchema.from_field(st.fields[i])
                var child_ptr = alloc[CArrowSchema](1)
                child_ptr.init_pointee_move(child^)
                children[i] = child_ptr
        else:
            raise Error(
                "CArrowSchema.from_dtype: unsupported dtype: {}".format(dtype)
            )

        return CArrowSchema(
            format=_alloc_c_string(fmt),
            name=UnsafePointer[c_char, MutAnyOrigin](),
            metadata=UnsafePointer[c_char, MutAnyOrigin](),
            flags=0,
            n_children=n_children,
            children=children,
            dictionary=UnsafePointer[CArrowSchema, MutAnyOrigin](),
            release=_release_exported_schema,
            private_data=OpaquePointer[MutAnyOrigin](),
        )

    @staticmethod
    def from_field(
        field: Field,
    ) raises -> CArrowSchema:
        """Build a CArrowSchema for a Field.

        Delegates to `from_dtype` and then sets the field name (heap-allocated
        as a raw C string) and nullability flag.
        """
        var c_schema = CArrowSchema.from_dtype(field.dtype)
        c_schema.name = _alloc_c_string(field.name)
        c_schema.flags = Int64(
            ARROW_FLAG_NULLABLE
        ) if field.nullable else Int64(0)
        return c_schema^

    @staticmethod
    def from_schema(
        fields: List[Field],
    ) raises -> CArrowSchema:
        """Build a top-level "+s" CArrowSchema representing a record-batch schema.

        Analogous to `from_dtype` for struct types but without a parent dtype:
        the format is always "+s" and children correspond to the schema fields.
        """
        var n_fields = len(fields)
        var children = UnsafePointer[
            UnsafePointer[CArrowSchema, MutAnyOrigin], MutAnyOrigin
        ]()
        if n_fields > 0:
            children = alloc[UnsafePointer[CArrowSchema, MutAnyOrigin]](
                n_fields
            )
            for i in range(n_fields):
                # Move each child value onto the heap so the pointer is stable.
                var child = CArrowSchema.from_field(fields[i])
                var child_ptr = alloc[CArrowSchema](1)
                child_ptr.init_pointee_move(child^)
                children[i] = child_ptr

        return CArrowSchema(
            format=_alloc_c_string("+s"),
            name=UnsafePointer[c_char, MutAnyOrigin](),
            metadata=UnsafePointer[c_char, MutAnyOrigin](),
            flags=0,
            n_children=Int64(n_fields),
            children=children,
            dictionary=UnsafePointer[CArrowSchema, MutAnyOrigin](),
            release=_release_exported_schema,
            private_data=OpaquePointer[MutAnyOrigin](),
        )

    @staticmethod
    def from_pycapsule(capsule: PythonObject) raises -> CArrowSchema:
        """Take ownership of a CArrowSchema from an "arrow_schema" PyCapsule.

        Copies the struct out of the capsule's raw memory and zeroes the
        source's release field so the capsule destructor does not double-free.
        The returned value owns all resources and will call
        `_release_exported_schema` (or the original producer's callback) when
        it goes out of scope.
        """
        var py = Python()
        ref cpy = py.cpython()
        var src = cpy.PyCapsule_GetPointer(
            capsule._obj_ptr, "arrow_schema"
        ).bitcast[CArrowSchema]()
        var schema = src[].copy()
        UnsafePointer(to=src[].release).bitcast[UInt64]()[0] = 0
        return schema^

    def to_pycapsule(deinit self) raises -> PythonObject:
        """Wrap this schema in a Python "arrow_schema" capsule.

        Moves `self` onto the heap so the capsule can hold a stable pointer.
        Ownership transfers to the capsule: `_release_schema_capsule` is set
        as the PyCapsule destructor and will call `_release_exported_schema`
        when Python GC collects the capsule.

        Typical usage: `CArrowSchema.from_dtype(dtype).to_pycapsule()`
        """
        var py = Python()
        ref cpy = py.cpython()
        # Move self onto the heap; the capsule destructor will free it.
        var ptr = alloc[CArrowSchema](1)
        ptr.init_pointee_move(self^)
        return PythonObject(
            from_owned=cpy.PyCapsule_New(
                ptr.bitcast[NoneType](),
                "arrow_schema",
                _release_schema_capsule,
            )
        )

    def to_dtype(self) raises -> AnyDataType:
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
            var f = self.children[0][].to_field()
            return list_(f.dtype.copy())
        elif fmt.startswith("+w:"):
            var size = Int(String(fmt).removeprefix("+w:"))
            var f = self.children[0][].to_field()
            return fixed_size_list_(f.dtype.copy(), size)
        elif fmt == "+s":
            var fields = List[Field](capacity=Int(self.n_children))
            for i in range(self.n_children):
                fields.append(self.children[i][].to_field())
            return struct_(fields^)
        else:
            raise Error("Unknown format: ", fmt)

    def to_field(self) raises -> Field:
        var name = StringSlice(unsafe_from_utf8_ptr=self.name)
        var dtype = self.to_dtype()
        var nullable = self.flags & ARROW_FLAG_NULLABLE
        return field(String(name), dtype^, nullable != 0)

    def to_schema(self) raises -> Schema:
        """Build a Schema from this top-level struct CArrowSchema."""
        var fields = List[Field]()
        for i in range(self.n_children):
            fields.append(self.children[i][].to_field())
        return Schema(fields=fields^)


def _release_array_capsule(capsule: PyObjectPtr):
    """PyCapsule destructor for "arrow_array" capsules.

    Mirrors `_release_schema_capsule`.  The capsule holds a raw pointer to a
    heap-allocated CArrowArray.  If the release callback is still set we call
    it (which frees buffers, children, and private_data via
    `_release_exported_array`), then free the struct shell.
    """
    try:
        var py = Python()
        ref cpy = py.cpython()
        var ptr = cpy.PyCapsule_GetPointer(capsule, "arrow_array")
        if ptr:
            var c_arr = ptr.bitcast[CArrowArray]()
            # Guard: release is zeroed by _release_exported_array after it runs,
            # or by an Arrow importer after it takes ownership.
            if UnsafePointer(to=c_arr[].release).bitcast[UInt64]()[0] != 0:
                c_arr[].release(c_arr)
            c_arr.free()
    except:
        pass


def _release_imported_array(ptr: UnsafePointer[UInt8, MutAnyOrigin]) -> None:
    """Release callback for CArrowArray imported via the C Data Interface.

    Called when the last Buffer (or Bitmap) that references the imported array
    is dropped.  Invokes the C-level release callback so the producer can free
    its resources, then frees the Mojo heap allocation that owns the struct.
    """
    var c_ptr = ptr.bitcast[CArrowArray]()
    c_ptr[].release(c_ptr)
    c_ptr.free()


def _release_exported_array(ptr: UnsafePointer[CArrowArray, MutAnyOrigin]):
    """Arrow release callback for CArrowArrays exported from Mojo.

    Called (via the release function pointer) when an Arrow consumer is done
    with the array.  Frees:
    - The heap-allocated child CArrowArray struct shells (their own release
      callbacks were already invoked by Arrow's recursive import).
    - The heap-allocated buffers pointer array.
    - The heap-allocated ArrayData in private_data (drops Arc refs so the
      underlying Buffer/Bitmap memory is freed when the last ref goes).
    Nulls the release field per the Arrow spec.
    """
    if ptr[].n_children > 0:
        for i in range(Int(ptr[].n_children)):
            ptr[].children[i].free()
        ptr[].children.free()
    if ptr[].buffers:
        ptr[].buffers.free()
    var data_ptr = ptr[].private_data.bitcast[ArrayData]()
    data_ptr.destroy_pointee()
    data_ptr.free()
    UnsafePointer(to=ptr[].release).bitcast[UInt64]()[0] = 0


@fieldwise_init
struct CArrowArray(Copyable, Movable):
    """Arrow C Data Interface array struct (ArrowArray).

    Ownership model
    ---------------
    Mirrors CArrowSchema.  A CArrowArray value owns its heap resources
    (buffers pointer array, child struct shells, private_data AnyArray copy)
    through the `release` callback.

    Arrow importers take ownership by copying the struct fields then zeroing
    the source's release field (per the Arrow C Data Interface spec).
    `__del__` guards against a null release so dropping a consumed value is safe.

    Lifecycle for Python export (the common path):
        1. `from_array` builds the struct value, heap-allocating an AnyArray copy
           (private_data) and a buffers pointer array.
        2. `to_pycapsule` moves it onto the heap and wraps it in a PyCapsule.
        3. `_release_array_capsule` calls `_release_exported_array` and frees
           the struct shell when Python GC collects the capsule.

    Lifecycle for direct Arrow export (e.g. passing to an Arrow importer):
        1. Build the struct value.
        2. Pass `UnsafePointer(to=c_array)` to the Arrow importer.
        3. The importer copies the struct and zeroes the local release field.
        4. When the local value goes out of scope `__del__` is a no-op.
    """

    var length: Int64
    var null_count: Int64
    var offset: Int64
    var n_buffers: Int64
    var n_children: Int64
    var buffers: UnsafePointer[OpaquePointer[MutAnyOrigin], MutAnyOrigin]
    var children: UnsafePointer[
        UnsafePointer[CArrowArray, MutAnyOrigin], MutAnyOrigin
    ]
    var dictionary: UnsafePointer[CArrowArray, MutAnyOrigin]
    var release: def(UnsafePointer[CArrowArray, MutAnyOrigin]) thin -> None
    var private_data: OpaquePointer[MutAnyOrigin]

    def __del__(deinit self):
        # Guard: release is zeroed by an Arrow importer after it takes ownership.
        if UnsafePointer(to=self.release).bitcast[UInt64]()[0] != 0:
            self.release(UnsafePointer(to=self))

    def to_data(
        self, dtype: AnyDataType, owner: ArcPointer[Allocation]
    ) raises -> ArrayData:
        """Build an ArrayData from this CArrowArray, all buffers sharing one owner.

        All Buffer views hold a copy of `owner` (an ArcPointer, so copying just
        bumps the ref-count).  The C release callback fires automatically once
        the last buffer is dropped.

        Buffer sizes are dtype-dependent (same as Arrow C++ and arrow-rs).
        Typed array construction is delegated to AnyArray.from_data().
        """
        # Buffer sizes must cover all elements including the offset, because the
        # raw C buffers start at element 0 regardless of the logical array offset.
        var length = self.length + self.offset

        var bitmap: Optional[Bitmap[]]
        if self.buffers[0]:
            bitmap = Bitmap(
                Buffer.from_foreign(
                    self.buffers[0],
                    math.ceildiv(Int(length), 8),
                    owner,
                ),
                length=Int(length),
            )
        else:
            bitmap = None

        var buffers = List[Buffer[]](capacity=2)  # worst case for string
        var children = List[ArrayData](capacity=Int(self.n_children))

        if dtype.is_bool():
            buffers.append(
                Buffer.from_foreign(
                    self.buffers[1],
                    math.ceildiv(Int(length), 8),
                    owner,
                )
            )
        elif dtype.is_primitive():
            buffers.append(
                Buffer.from_foreign(
                    self.buffers[1],
                    Int(length) * dtype.byte_width(),
                    owner,
                )
            )
        elif dtype.is_list():
            var size = (length + 1) * Int64(size_of[DType.int32]())
            var offsets = Buffer.from_foreign(self.buffers[1], size, owner)
            buffers.append(offsets^)
            children.append(
                self.children[0][].to_data(
                    dtype.as_list_type().value_type(), owner
                )
            )
        elif dtype.is_string() or dtype.is_binary():
            var size = (length + 1) * Int64(size_of[DType.int32]())
            var offsets = Buffer.from_foreign(self.buffers[1], size, owner)
            var data_len = offsets.unsafe_get[DType.int32](Int(length))
            var values = Buffer.from_foreign(self.buffers[2], data_len, owner)
            buffers.append(offsets^)
            buffers.append(values^)
        elif dtype.is_fixed_size_list():
            children.append(
                self.children[0][].to_data(
                    dtype.as_fixed_size_list_type().value_type(), owner
                )
            )
        elif dtype.is_struct():
            var st = dtype.as_struct_type()
            for i in range(Int(self.n_children)):
                children.append(
                    self.children[i][].to_data(st.fields[i].dtype, owner)
                )
        else:
            raise Error("to_data: unsupported dtype: ", dtype)

        return ArrayData(
            dtype=dtype.copy(),
            length=Int(self.length),
            nulls=Int(self.null_count),
            offset=Int(self.offset),
            bitmap=bitmap,
            buffers=buffers^,
            children=children^,
        )

    def to_array(
        self, dtype: AnyDataType, owner: ArcPointer[Allocation]
    ) raises -> AnyArray:
        """Build an AnyArray from this CArrowArray.  Thin wrapper over to_data.
        """
        return AnyArray.from_data(self.to_data(dtype, owner))

    @staticmethod
    def from_array(array: AnyArray) raises -> CArrowArray:
        """Build a CArrowArray from a Mojo AnyArray.  Thin wrapper over from_data.
        """
        return CArrowArray.from_data(array.to_data())

    @staticmethod
    def from_data(var data: ArrayData) raises -> CArrowArray:
        """Build a CArrowArray value from an ArrayData for export.

        Buffer pointers in the returned struct point directly into the
        ArrayData's ArcPointer-managed memory.  A heap-allocated copy of `data`
        is stored in private_data to keep all ArcPointer ref-counts alive for
        the lifetime of the export; `_release_exported_array` destroys it
        (dropping the Arc refs) when the Arrow consumer is done.

        Call `.to_pycapsule()` to wrap the result in a Python capsule, or pass
        `UnsafePointer(to=c_array)` directly to an Arrow importer.
        """
        # n_buffers and n_children come directly from ArrayData — no dtype branching.
        var n_buffers = Int64(1 + len(data.buffers))  # 1 = validity bitmap slot
        var n_children = Int64(len(data.children))

        # Heap-allocate ArrayData to keep ArcPointer ref-counts alive.
        var data_heap = alloc[ArrayData](1)
        data_heap.init_pointee_move(data^)

        # Heap-allocate the buffers pointer array.
        # buffers[0] = validity bitmap (null pointer means all-valid).
        # buffers[1..n] = data.buffers[0..n-1] in order.
        var buffers = alloc[OpaquePointer[MutAnyOrigin]](Int(n_buffers))
        if data_heap[].bitmap:
            buffers[0] = OpaquePointer[MutAnyOrigin](
                unsafe_from_address=Int(
                    data_heap[].bitmap.value().view().unsafe_ptr()
                )
            )
        else:
            buffers[0] = OpaquePointer[MutAnyOrigin]()
        for i in range(len(data_heap[].buffers)):
            buffers[1 + i] = OpaquePointer[MutAnyOrigin](
                unsafe_from_address=Int(
                    data_heap[].buffers[i].view[DType.uint8]().unsafe_ptr()
                )
            )

        # Recursively build children; each child is moved onto the heap so the
        # pointer in children_ptr remains valid after this stack frame exits.
        var children_ptr = UnsafePointer[
            UnsafePointer[CArrowArray, MutAnyOrigin], MutAnyOrigin
        ]()
        if n_children > 0:
            children_ptr = alloc[UnsafePointer[CArrowArray, MutAnyOrigin]](
                Int(n_children)
            )
            for i in range(Int(n_children)):
                var child = CArrowArray.from_data(
                    data_heap[].children[i].copy()
                )
                var child_ptr = alloc[CArrowArray](1)
                child_ptr.init_pointee_move(child^)
                children_ptr[i] = child_ptr

        return CArrowArray(
            length=Int64(data_heap[].length),
            null_count=Int64(data_heap[].nulls),
            offset=Int64(data_heap[].offset),
            n_buffers=n_buffers,
            n_children=n_children,
            buffers=buffers,
            children=children_ptr,
            dictionary=UnsafePointer[CArrowArray, MutAnyOrigin](),
            release=_release_exported_array,
            # private_data keeps data_heap alive; freed by _release_exported_array.
            private_data=data_heap.bitcast[NoneType](),
        )

    @staticmethod
    def from_pycapsule(capsule: PythonObject) raises -> CArrowArray:
        """Take ownership of a CArrowArray from an "arrow_array" PyCapsule.

        Mirrors `CArrowSchema.from_pycapsule`.
        """
        var py = Python()
        ref cpy = py.cpython()
        var src = cpy.PyCapsule_GetPointer(
            capsule._obj_ptr, "arrow_array"
        ).bitcast[CArrowArray]()
        var array = src[].copy()
        UnsafePointer(to=src[].release).bitcast[UInt64]()[0] = 0
        return array^

    def to_pycapsule(deinit self) raises -> PythonObject:
        """Wrap this array in a Python "arrow_array" capsule.

        Mirrors `CArrowSchema.to_pycapsule`.  Moves `self` onto the heap so
        the capsule can hold a stable pointer.  Ownership transfers to the
        capsule: `_release_array_capsule` calls `_release_exported_array` when
        Python GC collects the capsule.

        Typical usage: `CArrowArray.from_array(arr).to_pycapsule()`
        """
        var py = Python()
        ref cpy = py.cpython()
        # Move self onto the heap; the capsule destructor will free it.
        var ptr = alloc[CArrowArray](1)
        ptr.init_pointee_move(self^)
        return PythonObject(
            from_owned=cpy.PyCapsule_New(
                ptr.bitcast[NoneType](),
                "arrow_array",
                _release_array_capsule,
            )
        )

    def to_array(deinit self, dtype: AnyDataType) raises -> AnyArray:
        """Convert to an AnyArray, taking ownership of the C struct.

        The CArrowArray is moved onto the heap and wrapped in a
        Allocation.  Every Buffer / Bitmap view shares the same
        ArcPointer[Allocation], so the C release callback fires
        automatically when the last buffer referencing this import is dropped.
        """
        var heap_c = alloc[CArrowArray](1)
        heap_c.init_pointee_move(self^)
        var owner = ArcPointer(
            Allocation.foreign(heap_c.bitcast[UInt8](), _release_imported_array)
        )
        return heap_c[].to_array(dtype, owner)


# ---------------------------------------------------------------------------
# Arrow C Device Data Interface
# https://arrow.apache.org/docs/format/CDeviceDataInterface.html
# ---------------------------------------------------------------------------


def _release_c_device_array(ptr: UnsafePointer[UInt8, MutAnyOrigin]) -> None:
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

    def to_array(
        deinit self, dtype: AnyDataType, ctx: DeviceContext
    ) raises -> AnyArray:
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
            An `AnyArray` whose buffers reference the device memory directly
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
        var _ = heap_c[].device_id

        # For CPU device type, delegate to the existing CArrowArray import path.
        if device_type == DeviceType.CPU:
            return heap_c[].array.to_array(dtype, owner)

        # For device memory, wrap each buffer pointer as a DEVICE buffer.
        # TODO: Zero-copy import of device arrays requires wrapping raw device
        # pointers in Mojo's DeviceBuffer.  This is not yet supported because
        # DeviceBuffer construction needs an AsyncRT handle that is not provided
        # by the C Device Data Interface.  Once Mojo exposes an API to adopt a
        # raw device pointer into a DeviceBuffer, this can be completed.
        raise Error(
            "to_array: zero-copy device array import is not yet implemented;"
            " Mojo does not yet expose a way to wrap raw device pointers in"
            " DeviceBuffer without an AsyncRT handle"
        )


# ---------------------------------------------------------------------------
# Arrow C Stream Interface
# https://arrow.apache.org/docs/format/CStreamInterface.html
# ---------------------------------------------------------------------------


struct _StreamPrivateData(Movable):
    """Internal state for an exported CArrowArrayStream.

    Holds the schema fields and record batches that the stream yields.
    `index` tracks the current position in `batches`.
    """

    var fields: List[Field]
    var batches: List[RecordBatch]
    var index: Int

    def __init__(
        out self, var fields: List[Field], var batches: List[RecordBatch]
    ):
        self.fields = fields^
        self.batches = batches^
        self.index = 0


def _stream_get_schema(
    stream_ptr: UnsafePointer[CArrowArrayStream, MutAnyOrigin],
    schema_out: UnsafePointer[CArrowSchema, MutAnyOrigin],
) -> Int32:
    """Stream callback: write the schema into `schema_out`."""
    try:
        var data = stream_ptr[].private_data.bitcast[_StreamPrivateData]()
        schema_out.init_pointee_move(CArrowSchema.from_schema(data[].fields))
        return 0
    except:
        return 1


def _stream_get_next(
    stream_ptr: UnsafePointer[CArrowArrayStream, MutAnyOrigin],
    array_out: UnsafePointer[CArrowArray, MutAnyOrigin],
) -> Int32:
    """Stream callback: write the next batch into `array_out`, or signal end."""
    try:
        var data = stream_ptr[].private_data.bitcast[_StreamPrivateData]()
        if data[].index >= len(data[].batches):
            # Signal end-of-stream: set release to null.
            UnsafePointer(to=array_out[].release).bitcast[UInt64]()[0] = 0
            return 0
        var batch = data[].batches[data[].index].copy()
        data[].index += 1
        var struct_arr: AnyArray = batch.to_struct_array()
        array_out.init_pointee_move(CArrowArray.from_array(struct_arr))
        return 0
    except:
        return 1


def _stream_get_last_error(
    stream_ptr: UnsafePointer[CArrowArrayStream, MutAnyOrigin],
) -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Stream callback: return null (no detailed error tracking)."""
    return UnsafePointer[UInt8, MutAnyOrigin]()


def _stream_release(
    stream_ptr: UnsafePointer[CArrowArrayStream, MutAnyOrigin],
) -> None:
    """Stream callback: free private data and null the release field."""
    var data = stream_ptr[].private_data.bitcast[_StreamPrivateData]()
    data.destroy_pointee()
    data.free()
    UnsafePointer(to=stream_ptr[].release).bitcast[UInt64]()[0] = 0


def _release_stream_capsule(capsule: PyObjectPtr):
    """PyCapsule destructor for "arrow_array_stream" capsules."""
    try:
        var py = Python()
        ref cpy = py.cpython()
        var ptr = cpy.PyCapsule_GetPointer(capsule, "arrow_array_stream")
        if ptr:
            var c_stream = ptr.bitcast[CArrowArrayStream]()
            if UnsafePointer(to=c_stream[].release).bitcast[UInt64]()[0] != 0:
                c_stream[].release(c_stream)
            c_stream.free()
    except:
        pass


@fieldwise_init
struct CArrowArrayStream(Copyable, TrivialRegisterPassable):
    """Arrow C Stream Interface struct (ArrowArrayStream).

    Provides a streaming interface to exchange sequences of record batches.
    Each stream has a fixed schema and yields batches via get_next() until
    end-of-stream (signalled by a null release field on the output array).

    Import:  `from_pycapsule()` → `to_record_batches()`
    Export:  `from_batches()` → `to_pycapsule()`
    """

    var get_schema: def(
        UnsafePointer[CArrowArrayStream, MutAnyOrigin],
        UnsafePointer[CArrowSchema, MutAnyOrigin],
    ) thin -> Int32
    var get_next: def(
        UnsafePointer[CArrowArrayStream, MutAnyOrigin],
        UnsafePointer[CArrowArray, MutAnyOrigin],
    ) thin -> Int32
    var get_last_error: def(
        UnsafePointer[CArrowArrayStream, MutAnyOrigin]
    ) thin -> UnsafePointer[UInt8, MutAnyOrigin]
    var release: def(
        UnsafePointer[CArrowArrayStream, MutAnyOrigin]
    ) thin -> None
    var private_data: OpaquePointer[MutAnyOrigin]

    @staticmethod
    def from_batches(
        var fields: List[Field], var batches: List[RecordBatch]
    ) -> CArrowArrayStream:
        """Build a CArrowArrayStream that yields the given batches.

        The stream takes ownership of the batches; callers should not
        mutate them after this call.
        """
        var data = alloc[_StreamPrivateData](1)
        data.init_pointee_move(_StreamPrivateData(fields^, batches^))
        return CArrowArrayStream(
            get_schema=_stream_get_schema,
            get_next=_stream_get_next,
            get_last_error=_stream_get_last_error,
            release=_stream_release,
            private_data=data.bitcast[NoneType](),
        )

    @staticmethod
    def from_pycapsule(capsule: PythonObject) raises -> CArrowArrayStream:
        """Take ownership of a CArrowArrayStream from an
        "arrow_array_stream" PyCapsule.
        """
        var py = Python()
        ref cpy = py.cpython()
        var src = cpy.PyCapsule_GetPointer(
            capsule._obj_ptr, "arrow_array_stream"
        ).bitcast[CArrowArrayStream]()
        var stream = src[].copy()
        UnsafePointer(to=src[].release).bitcast[UInt64]()[0] = 0
        return stream

    def to_pycapsule(self) raises -> PythonObject:
        """Wrap this stream in a Python "arrow_array_stream" PyCapsule."""
        var py = Python()
        ref cpy = py.cpython()
        var ptr = alloc[CArrowArrayStream](1)
        ptr.init_pointee_copy(self)
        return PythonObject(
            from_owned=cpy.PyCapsule_New(
                ptr.bitcast[NoneType](),
                "arrow_array_stream",
                _release_stream_capsule,
            )
        )

    def to_table(self) raises -> Table:
        """Consume the stream and build a Table.

        Calls get_schema once, then iterates get_next until end-of-stream.
        """
        var heap = alloc[CArrowArrayStream](1)
        heap.init_pointee_copy(self)

        # Get schema.
        var c_schema = alloc[CArrowSchema](1)
        var err = heap[].get_schema(heap, c_schema)
        if err != 0:
            heap[].release(heap)
            heap.free()
            raise Error("CArrowArrayStream: get_schema failed with code ", err)
        var schema = c_schema.take_pointee().to_schema()

        # Iterate batches.
        var batches = List[RecordBatch]()
        while True:
            var c_array = alloc[CArrowArray](1)
            err = heap[].get_next(heap, c_array)
            if err != 0:
                heap[].release(heap)
                heap.free()
                raise Error(
                    "CArrowArrayStream: get_next failed with code ", err
                )
            # End-of-stream: release field is null.
            if UnsafePointer(to=c_array[].release).bitcast[UInt64]()[0] == 0:
                c_array.free()
                break
            var struct_dtype = struct_(schema.fields.copy())
            var arr = c_array.take_pointee().to_array(struct_dtype^)
            var columns = List[AnyArray]()
            for child in arr.as_struct().children:
                columns.append(child.copy())
            batches.append(RecordBatch(schema=schema, columns=columns^))

        # Release the stream.
        heap[].release(heap)
        heap.free()
        return Table.from_batches(schema, batches^)
