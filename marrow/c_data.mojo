from std.ffi import c_char, CStringSlice
from std.memory import ArcPointer, unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.sys import size_of
from .buffers import (
    Allocation,
    Buffer,
    Bitmap,
    DeviceType,
)

import std.math as math

from std.gpu import DeviceContext
from .dtypes import (
    DynType,
    Field,
    FixedSizeBinaryType,
    FixedSizeListType,
    LargeListType,
    ListType,
    MapType,
    binary,
    bool_,
    date32,
    date64,
    day_time_interval,
    decimal128,
    decimal256,
    decimal32,
    decimal64,
    dictionary,
    duration,
    field,
    float16,
    float32,
    float64,
    int16,
    int32,
    int64,
    int8,
    large_binary,
    large_string,
    microsecond,
    millisecond,
    month_day_nano_interval,
    nanosecond,
    null,
    second,
    string,
    struct_,
    time32,
    time64,
    timestamp,
    uint16,
    uint32,
    uint64,
    uint8,
    year_month_interval,
)
from .arrays import (
    DynArray,
    ArrayData,
)
from .schema import Schema
from .tabular import RecordBatch, Table

comptime ARROW_FLAG_NULLABLE = 2
comptime ARROW_FLAG_DICT_ORDERED: Int64 = 1
comptime ARROW_FLAG_MAP_KEYS_SORTED: Int64 = 4


@always_inline
def _null_ptr[T: AnyType]() -> UnsafePointer[T, MutUntrackedOrigin]:
    """Construct an address-zero pointer for C ABI struct fields that may be null.
    """
    # NULL is a valid state for Arrow C ABI pointers. `UnsafePointer` is
    # non-nullable when built from an `IntLiteral`, so route the 0 through the
    # runtime `Int` overload to construct the null pointer the C layout needs.
    return UnsafePointer[T, MutUntrackedOrigin](unsafe_from_address=Int(0))


def _alloc_c_string(s: String) -> UnsafePointer[c_char, MutUntrackedOrigin]:
    """Copy a Mojo String into a heap-allocated null-terminated C string.

    The caller owns the returned buffer and must free it when done.

    Note: copies len(s) bytes then writes an explicit null terminator.
    String.unsafe_ptr() is not guaranteed to be null-terminated (SSO inline
    storage leaves bytes past len(s) uninitialized).
    TODO: replace with unsafe_cstr_ptr() once available in this Mojo build.
    """
    var n = s.byte_length()
    var buf = alloc[c_char](n + 1)
    unsafe_memcpy(dest=buf.bitcast[UInt8](), src=s.unsafe_ptr(), count=n)
    buf.bitcast[UInt8]()[n] = 0
    return UnsafePointer[c_char, MutUntrackedOrigin](
        unsafe_from_address=Int(buf)
    )


def _encode_c_metadata(
    metadata: Dict[String, String],
) raises -> UnsafePointer[c_char, MutUntrackedOrigin]:
    """Encode a Dict into the Arrow C Data Interface metadata blob.

    Format (native byte order, per the spec):
        int32 num_kv_pairs
        for each pair:
            int32 key_length;   bytes key   (no null terminator)
            int32 value_length; bytes value
    Returns a null pointer when the dict is empty.
    """
    if len(metadata) == 0:
        return _null_ptr[c_char]()
    # First pass: compute total byte length.
    var total = 4  # num_kv_pairs
    for entry in metadata.items():
        total += 4 + entry.key.byte_length() + 4 + entry.value.byte_length()
    var buf = alloc[UInt8](total)
    var head = 0

    var n_pairs = Int32(len(metadata))
    unsafe_memcpy(
        dest=buf + head, src=UnsafePointer(to=n_pairs).bitcast[UInt8](), count=4
    )
    head += 4
    for entry in metadata.items():
        var k = entry.key
        var v = entry.value
        var k_len = Int32(k.byte_length())
        unsafe_memcpy(
            dest=buf + head,
            src=UnsafePointer(to=k_len).bitcast[UInt8](),
            count=4,
        )
        head += 4
        unsafe_memcpy(
            dest=buf + head, src=k.unsafe_ptr(), count=k.byte_length()
        )
        head += k.byte_length()
        var v_len = Int32(v.byte_length())
        unsafe_memcpy(
            dest=buf + head,
            src=UnsafePointer(to=v_len).bitcast[UInt8](),
            count=4,
        )
        head += 4
        unsafe_memcpy(
            dest=buf + head, src=v.unsafe_ptr(), count=v.byte_length()
        )
        head += v.byte_length()
    return buf.bitcast[c_char]()


def _decode_c_metadata(
    metadata: UnsafePointer[c_char, MutUntrackedOrigin],
) raises -> Dict[String, String]:
    """Decode an Arrow C Data Interface metadata blob into a Dict."""
    var result = Dict[String, String]()
    if Int(metadata) == 0:
        return result^
    var p = metadata.bitcast[UInt8]()
    var head = 0

    var n_pairs = Int32(0)
    unsafe_memcpy(
        dest=UnsafePointer(to=n_pairs).bitcast[UInt8](), src=p + head, count=4
    )
    head += 4
    for _ in range(Int(n_pairs)):
        var k_len = Int32(0)
        unsafe_memcpy(
            dest=UnsafePointer(to=k_len).bitcast[UInt8](), src=p + head, count=4
        )
        head += 4
        var k = String(
            from_utf8=Span[Byte](unsafe_ptr=p + head, length=Int(k_len))
        )
        head += Int(k_len)
        var v_len = Int32(0)
        unsafe_memcpy(
            dest=UnsafePointer(to=v_len).bitcast[UInt8](), src=p + head, count=4
        )
        head += 4
        var v = String(
            from_utf8=Span[Byte](unsafe_ptr=p + head, length=Int(v_len))
        )
        head += Int(v_len)
        result[k^] = v^
    return result^


def _release_schema_capsule(capsule: PyObjectPtr) abi("C"):
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
        if Int(ptr) != 0:
            var c_schema = ptr.bitcast[CArrowSchema]()
            # Guard against double-free: an Arrow importer zeroes the release
            # field after taking ownership.
            if not c_schema[].is_released():
                c_schema[].release(c_schema)
            c_schema.free()
    except:
        pass


def _release_exported_schema(
    ptr: UnsafePointer[CArrowSchema, MutUntrackedOrigin]
) abi("C"):
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
    if Int(ptr[].format) != 0:
        ptr[].format.free()
    if Int(ptr[].name) != 0:
        ptr[].name.free()
    if Int(ptr[].metadata) != 0:
        ptr[].metadata.free()
    ptr[].mark_released()


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

    var format: UnsafePointer[c_char, MutUntrackedOrigin]
    var name: UnsafePointer[c_char, MutUntrackedOrigin]
    var metadata: UnsafePointer[c_char, MutUntrackedOrigin]
    var flags: Int64
    var n_children: Int64
    var children: UnsafePointer[
        UnsafePointer[CArrowSchema, MutUntrackedOrigin], MutUntrackedOrigin
    ]
    var dictionary: UnsafePointer[CArrowSchema, MutUntrackedOrigin]
    var release: def(UnsafePointer[CArrowSchema, MutUntrackedOrigin]) thin abi(
        "C"
    ) -> None
    var private_data: OpaquePointer[MutUntrackedOrigin]

    def is_released(self) -> Bool:
        """True once ownership has moved on: the C ABI marks a struct consumed
        by nulling its `release` callback, and calling it again is a double
        free."""
        return UnsafePointer(to=self.release).bitcast[UInt64]()[0] == 0

    def mark_released(mut self):
        """Give up ownership: null the callback so nobody can release twice.

        This is what an importer does after taking the resources — the C ABI
        has no other handshake for it."""
        UnsafePointer(to=self.release).bitcast[UInt64]()[0] = 0

    def __del__(deinit self):
        if not self.is_released():
            self.release(
                UnsafePointer(to=self).unsafe_origin_cast[MutUntrackedOrigin]()
            )

    @staticmethod
    def from_dtype(
        dtype: DynType,
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
        var children: UnsafePointer[
            UnsafePointer[CArrowSchema, MutUntrackedOrigin], MutUntrackedOrigin
        ] = _null_ptr[UnsafePointer[CArrowSchema, MutUntrackedOrigin]]()
        var flags: Int64 = 0
        var dictionary_ptr = _null_ptr[CArrowSchema]()

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
        elif dtype.is_large_binary():
            fmt = "Z"
        elif dtype.is_string():
            fmt = "u"
        elif dtype.is_large_string():
            fmt = "U"
        elif dtype.is_list():
            fmt = "+l"
            n_children = 1
            children = alloc[UnsafePointer[CArrowSchema, MutUntrackedOrigin]](1)
            # Move child value onto the heap so the pointer stays valid after
            # this stack frame is gone.
            var child0 = CArrowSchema.from_field(
                dtype.as_list().value_field().copy()
            )
            var child0_ptr = alloc[CArrowSchema](1)
            child0_ptr.unsafe_write(child0^)
            children[0] = child0_ptr
        elif dtype.is_large_list():
            fmt = "+L"
            n_children = 1
            children = alloc[UnsafePointer[CArrowSchema, MutUntrackedOrigin]](1)
            var child0 = CArrowSchema.from_field(
                dtype.as_large_list().value_field().copy()
            )
            var child0_ptr = alloc[CArrowSchema](1)
            child0_ptr.unsafe_write(child0^)
            children[0] = child0_ptr
        elif dtype.is_fixed_size_list():
            ref fsl = dtype.as_fixed_size_list()
            fmt = {"+w:", fsl.size}
            n_children = 1
            children = alloc[UnsafePointer[CArrowSchema, MutUntrackedOrigin]](1)
            var child0 = CArrowSchema.from_field(fsl.value_field().copy())
            var child0_ptr = alloc[CArrowSchema](1)
            child0_ptr.unsafe_write(child0^)
            children[0] = child0_ptr
        elif dtype.is_fixed_size_binary():
            ref fsb = dtype.as_fixed_size_binary()
            fmt = {"w:", fsb.byte_width}
        elif dtype.is_date32():
            fmt = "tdD"
        elif dtype.is_date64():
            fmt = "tdm"
        elif dtype.is_time32():
            var u = dtype.as_time32().unit
            if u == second:
                fmt = "tts"
            else:
                fmt = "ttm"
        elif dtype.is_time64():
            var u = dtype.as_time64().unit
            if u == microsecond:
                fmt = "ttu"
            else:
                fmt = "ttn"
        elif dtype.is_timestamp():
            ref ts = dtype.as_timestamp()
            var uc: String
            if ts.unit == second:
                uc = "s"
            elif ts.unit == millisecond:
                uc = "m"
            elif ts.unit == microsecond:
                uc = "u"
            else:
                uc = "n"
            fmt = {"ts", uc, ":", ts.timezone}
        elif dtype.is_duration():
            var u = dtype.as_duration().unit
            if u == second:
                fmt = "tDs"
            elif u == millisecond:
                fmt = "tDm"
            elif u == microsecond:
                fmt = "tDu"
            else:
                fmt = "tDn"
        elif dtype.is_year_month_interval():
            fmt = "tiM"
        elif dtype.is_day_time_interval():
            fmt = "tiD"
        elif dtype.is_month_day_nano_interval():
            fmt = "tin"
        elif dtype.is_decimal32():
            ref d = dtype.as_decimal32()
            fmt = {"d:", d.precision, ",", d.scale, ",32"}
        elif dtype.is_decimal64():
            ref d = dtype.as_decimal64()
            fmt = {"d:", d.precision, ",", d.scale, ",64"}
        elif dtype.is_decimal128():
            ref d = dtype.as_decimal128()
            fmt = {"d:", d.precision, ",", d.scale}
        elif dtype.is_decimal256():
            ref d = dtype.as_decimal256()
            fmt = {"d:", d.precision, ",", d.scale, ",256"}
        elif dtype.is_struct():
            fmt = "+s"
            ref st = dtype.as_struct()
            n_children = Int64(len(st.fields))
            children = alloc[UnsafePointer[CArrowSchema, MutUntrackedOrigin]](
                Int(n_children)
            )
            for i in range(Int(n_children)):
                var child = CArrowSchema.from_field(st.fields[i])
                var child_ptr = alloc[CArrowSchema](1)
                child_ptr.unsafe_write(child^)
                children[i] = child_ptr
        elif dtype.is_map():
            # "+m" with a single child = the non-nullable "entries" struct of
            # (key, value). keys_sorted rides in the schema flags.
            ref mt = dtype.as_map()
            fmt = "+m"
            n_children = 1
            children = alloc[UnsafePointer[CArrowSchema, MutUntrackedOrigin]](1)
            var entries = CArrowSchema.from_field(mt.entries_field())
            var entries_ptr = alloc[CArrowSchema](1)
            entries_ptr.unsafe_write(entries^)
            children[0] = entries_ptr
            if mt.keys_sorted:
                flags = ARROW_FLAG_MAP_KEYS_SORTED
        elif dtype.is_dictionary():
            ref dt = dtype.as_dictionary()
            ref idx = dt.index_type()
            if idx == int8:
                fmt = "c"
            elif idx == int16:
                fmt = "s"
            elif idx == int32:
                fmt = "i"
            elif idx == int64:
                fmt = "l"
            elif idx == uint8:
                fmt = "C"
            elif idx == uint16:
                fmt = "S"
            elif idx == uint32:
                fmt = "I"
            elif idx == uint64:
                fmt = "L"
            else:
                raise Error(
                    (
                        "CArrowSchema.from_dtype: unsupported dictionary index"
                        " type: "
                    ),
                    idx,
                )
            var dict_schema = CArrowSchema.from_dtype(dt.value_type())
            var dict_schema_ptr = alloc[CArrowSchema](1)
            dict_schema_ptr.unsafe_write(dict_schema^)
            dictionary_ptr = dict_schema_ptr
            if dt.ordered:
                flags = ARROW_FLAG_DICT_ORDERED
        else:
            raise Error(
                "CArrowSchema.from_dtype: unsupported dtype: {}".format(dtype)
            )

        return CArrowSchema(
            format=_alloc_c_string(fmt),
            name=_null_ptr[c_char](),
            metadata=_null_ptr[c_char](),
            flags=flags,
            n_children=n_children,
            children=children,
            dictionary=dictionary_ptr,
            release=_release_exported_schema,
            private_data=_null_ptr[NoneType](),
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
        c_schema.metadata = _encode_c_metadata(field.metadata)
        return c_schema^

    @staticmethod
    def from_schema(schema: Schema) raises -> CArrowSchema:
        """Build a top-level "+s" CArrowSchema representing a record-batch schema.

        Analogous to `from_dtype` for struct types but without a parent dtype:
        the format is always "+s" and children correspond to the schema fields.
        Schema-level `custom_metadata` is encoded into the metadata blob.
        """
        var n_fields = len(schema.fields)
        var children: UnsafePointer[
            UnsafePointer[CArrowSchema, MutUntrackedOrigin], MutUntrackedOrigin
        ] = _null_ptr[UnsafePointer[CArrowSchema, MutUntrackedOrigin]]()
        if n_fields > 0:
            children = alloc[UnsafePointer[CArrowSchema, MutUntrackedOrigin]](
                n_fields
            )
            for i in range(n_fields):
                # Move each child value onto the heap so the pointer is stable.
                var child = CArrowSchema.from_field(schema.fields[i])
                var child_ptr = alloc[CArrowSchema](1)
                child_ptr.unsafe_write(child^)
                children[i] = child_ptr

        return CArrowSchema(
            format=_alloc_c_string("+s"),
            name=_null_ptr[c_char](),
            metadata=_encode_c_metadata(schema.metadata),
            flags=0,
            n_children=Int64(n_fields),
            children=children,
            dictionary=_null_ptr[CArrowSchema](),
            release=_release_exported_schema,
            private_data=_null_ptr[NoneType](),
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
        src[].mark_released()
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
        ptr.unsafe_write(self^)
        return PythonObject(
            from_owned=cpy.PyCapsule_New(
                ptr.bitcast[NoneType](),
                "arrow_schema",
                _release_schema_capsule,
            )
        )

    def to_dtype(self) raises -> DynType:
        var fmt = StringSlice(
            unsafe_from_utf8=CStringSlice(
                unsafe_from_ptr=self.format.as_immutable()
            )
        )
        # Dictionary type: non-null `dictionary` field signals dictionary encoding.
        # The format string is the index type's format (e.g. "i" for int32).
        # Must be checked before the regular format string dispatch.
        if UnsafePointer(to=self.dictionary).bitcast[UInt64]()[0] != 0:
            var index_type: DynType
            if fmt == "c":
                index_type = int8
            elif fmt == "s":
                index_type = int16
            elif fmt == "i":
                index_type = int32
            elif fmt == "l":
                index_type = int64
            elif fmt == "C":
                index_type = uint8
            elif fmt == "S":
                index_type = uint16
            elif fmt == "I":
                index_type = uint32
            elif fmt == "L":
                index_type = uint64
            else:
                raise Error(
                    "CArrowSchema.to_dtype: unknown dictionary index format: ",
                    fmt,
                )
            var value_type = self.dictionary[].to_dtype()
            var ordered = Bool(self.flags & ARROW_FLAG_DICT_ORDERED)
            return dictionary(index_type^, value_type^, ordered).to_dyn()
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
        elif fmt == "Z":
            return large_binary
        elif fmt == "u":
            return string
        elif fmt == "U":
            return large_string
        elif fmt == "+l":
            # Preserve the child Field as-is (its name may not be the default
            # "item" when constructed by other Arrow implementations).
            return ListType(self.children[0][].to_field()).to_dyn()
        elif fmt == "+L":
            return LargeListType(self.children[0][].to_field()).to_dyn()
        elif fmt.startswith("+w:"):
            var size = Int(String(fmt).removeprefix("+w:"))
            return FixedSizeListType(
                self.children[0][].to_field(), size
            ).to_dyn()
        elif fmt.startswith("w:"):
            var width = Int(String(fmt).removeprefix("w:"))
            return FixedSizeBinaryType(width).to_dyn()
        elif fmt == "tdD":
            return date32()
        elif fmt == "tdm":
            return date64()
        elif fmt == "tts":
            return time32(second)
        elif fmt == "ttm":
            return time32(millisecond)
        elif fmt == "ttu":
            return time64(microsecond)
        elif fmt == "ttn":
            return time64(nanosecond)
        elif fmt.startswith("tss:"):
            return timestamp(
                second, String(String(fmt).removeprefix("tss:"))
            ).to_dyn()
        elif fmt.startswith("tsm:"):
            return timestamp(
                millisecond, String(String(fmt).removeprefix("tsm:"))
            ).to_dyn()
        elif fmt.startswith("tsu:"):
            return timestamp(
                microsecond, String(String(fmt).removeprefix("tsu:"))
            ).to_dyn()
        elif fmt.startswith("tsn:"):
            return timestamp(
                nanosecond, String(String(fmt).removeprefix("tsn:"))
            ).to_dyn()
        elif fmt == "tDs":
            return duration(second)
        elif fmt == "tDm":
            return duration(millisecond)
        elif fmt == "tDu":
            return duration(microsecond)
        elif fmt == "tDn":
            return duration(nanosecond)
        elif fmt == "tiM":
            return year_month_interval()
        elif fmt == "tiD":
            return day_time_interval()
        elif fmt == "tin":
            return month_day_nano_interval()
        elif fmt == "+s":
            var fields = List[Field](capacity=Int(self.n_children))
            for i in range(self.n_children):
                fields.append(self.children[i][].to_field())
            return struct_(fields^)
        elif fmt == "+m":
            # "+m" has one child = the entries struct field of (key, value); store
            # it directly, preserving its field names and nullability. keys_sorted
            # rides the schema flags.
            var entries = self.children[0][].to_field()
            var sorted = Bool(self.flags & ARROW_FLAG_MAP_KEYS_SORTED)
            return MapType(entries^, sorted).to_dyn()
        elif fmt.startswith("d:"):
            var rest = String(fmt).removeprefix("d:")
            var parts = rest.split(",")
            var precision = Int(parts[0])
            var scale = Int(parts[1])
            var bit_width = 128  # default for legacy format without bitwidth
            if len(parts) == 3:
                bit_width = Int(parts[2])
            if bit_width == 32:
                return decimal32(precision, scale)
            elif bit_width == 64:
                return decimal64(precision, scale)
            elif bit_width == 256:
                return decimal256(precision, scale)
            else:
                return decimal128(precision, scale)
        else:
            raise Error("Unknown format: ", fmt)

    def to_field(self) raises -> Field:
        var name = StringSlice(
            unsafe_from_utf8=CStringSlice(
                unsafe_from_ptr=self.name.as_immutable()
            )
        )
        var dtype = self.to_dtype()
        var nullable = self.flags & ARROW_FLAG_NULLABLE
        var metadata = _decode_c_metadata(self.metadata)
        return Field(String(name), dtype^, nullable != 0, metadata^)

    def to_schema(self) raises -> Schema:
        """Build a Schema from this top-level struct CArrowSchema."""
        var fields = List[Field]()
        for i in range(self.n_children):
            fields.append(self.children[i][].to_field())
        var metadata = _decode_c_metadata(self.metadata)
        return Schema(fields=fields^, metadata=metadata^)


def _release_array_capsule(capsule: PyObjectPtr) abi("C"):
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
        if Int(ptr) != 0:
            var c_arr = ptr.bitcast[CArrowArray]()
            # Guard: release is zeroed by _release_exported_array after it runs,
            # or by an Arrow importer after it takes ownership.
            if not c_arr[].is_released():
                c_arr[].release(c_arr)
            c_arr.free()
    except:
        pass


def _release_imported_array(
    ptr: UnsafePointer[UInt8, MutUntrackedOrigin]
) -> None:
    """Release callback for CArrowArray imported via the C Data Interface.

    Called when the last Buffer (or Bitmap) that references the imported array
    is dropped.  Invokes the C-level release callback so the producer can free
    its resources, then frees the Mojo heap allocation that owns the struct.
    """
    var c_ptr = ptr.bitcast[CArrowArray]()
    c_ptr[].release(c_ptr)
    c_ptr.free()


def _release_exported_array(
    ptr: UnsafePointer[CArrowArray, MutUntrackedOrigin]
) abi("C"):
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
    if Int(ptr[].buffers) != 0:
        ptr[].buffers.free()
    var data_ptr = ptr[].private_data.bitcast[ArrayData]()
    data_ptr.unsafe_deinit_pointee()
    data_ptr.free()
    ptr[].mark_released()


@fieldwise_init
struct CArrowArray(Copyable, Movable):
    """Arrow C Data Interface array struct (ArrowArray).

    Ownership model
    ---------------
    Mirrors CArrowSchema.  A CArrowArray value owns its heap resources
    (buffers pointer array, child struct shells, private_data DynArray copy)
    through the `release` callback.

    Arrow importers take ownership by copying the struct fields then zeroing
    the source's release field (per the Arrow C Data Interface spec).
    `__del__` guards against a null release so dropping a consumed value is safe.

    Lifecycle for Python export (the common path):
        1. `from_array` builds the struct value, heap-allocating an DynArray copy
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
    var buffers: UnsafePointer[
        OpaquePointer[MutUntrackedOrigin], MutUntrackedOrigin
    ]
    var children: UnsafePointer[
        UnsafePointer[CArrowArray, MutUntrackedOrigin], MutUntrackedOrigin
    ]
    var dictionary: UnsafePointer[CArrowArray, MutUntrackedOrigin]
    var release: def(UnsafePointer[CArrowArray, MutUntrackedOrigin]) thin abi(
        "C"
    ) -> None
    var private_data: OpaquePointer[MutUntrackedOrigin]

    def is_released(self) -> Bool:
        """True once ownership has moved on: the C ABI marks a struct consumed
        by nulling its `release` callback, and calling it again is a double
        free."""
        return UnsafePointer(to=self.release).bitcast[UInt64]()[0] == 0

    def mark_released(mut self):
        """Give up ownership: null the callback so nobody can release twice.

        This is what an importer does after taking the resources — the C ABI
        has no other handshake for it."""
        UnsafePointer(to=self.release).bitcast[UInt64]()[0] = 0

    def __del__(deinit self):
        if not self.is_released():
            self.release(
                UnsafePointer(to=self).unsafe_origin_cast[MutUntrackedOrigin]()
            )

    def to_data(
        self, dtype: DynType, owner: ArcPointer[Allocation]
    ) raises -> ArrayData:
        """Build an ArrayData from this CArrowArray, all buffers sharing one owner.

        All Buffer views hold a copy of `owner` (an ArcPointer, so copying just
        bumps the ref-count).  The C release callback fires automatically once
        the last buffer is dropped.

        Buffer sizes are dtype-dependent (same as Arrow C++ and arrow-rs).
        Typed array construction is delegated to DynArray.from_data().
        """
        # Buffer sizes must cover all elements including the offset, because the
        # raw C buffers start at element 0 regardless of the logical array offset.
        var length = self.length + self.offset

        # Null arrays carry no buffers — `self.buffers` itself may be a null
        # pointer — so skip the validity read for them.
        var bitmap: Optional[Bitmap[]] = None
        if not dtype.is_null() and Int(self.buffers[0]) != 0:
            bitmap = Bitmap(
                Buffer.from_foreign(
                    self.buffers[0],
                    math.ceildiv(Int(length), 8),
                    owner,
                ),
                length=Int(length),
            )

        var buffers = List[Buffer[]](capacity=2)  # worst case for string
        var children = List[ArrayData](capacity=Int(self.n_children))

        if dtype.is_null():
            pass  # no buffers, no children
        elif dtype.is_bool():
            buffers.append(
                Buffer.from_foreign(
                    self.buffers[1], math.ceildiv(Int(length), 8), owner
                )
            )
        elif dtype.is_primitive():
            buffers.append(
                Buffer.from_foreign(
                    self.buffers[1], Int(length) * dtype.byte_width(), owner
                )
            )
        elif dtype.is_string() or dtype.is_binary():
            var offsets = Buffer.from_foreign(
                self.buffers[1],
                (Int(length) + 1) * size_of[DType.int32](),
                owner,
            )
            var n = Int(offsets.unsafe_get[DType.int32](Int(length)))
            buffers.append(offsets^)
            buffers.append(Buffer.from_foreign(self.buffers[2], n, owner))
        elif dtype.is_large_string() or dtype.is_large_binary():
            var offsets = Buffer.from_foreign(
                self.buffers[1],
                (Int(length) + 1) * size_of[DType.int64](),
                owner,
            )
            var n = Int(offsets.unsafe_get[DType.int64](Int(length)))
            buffers.append(offsets^)
            buffers.append(Buffer.from_foreign(self.buffers[2], n, owner))
        elif dtype.is_list():
            buffers.append(
                Buffer.from_foreign(
                    self.buffers[1],
                    (Int(length) + 1) * size_of[DType.int32](),
                    owner,
                )
            )
            children.append(
                self.children[0][].to_data(dtype.as_list().value_type(), owner)
            )
        elif dtype.is_large_list():
            buffers.append(
                Buffer.from_foreign(
                    self.buffers[1],
                    (Int(length) + 1) * size_of[DType.int64](),
                    owner,
                )
            )
            children.append(
                self.children[0][].to_data(
                    dtype.as_large_list().value_type(), owner
                )
            )
        elif dtype.is_fixed_size_binary():
            buffers.append(
                Buffer.from_foreign(
                    self.buffers[1],
                    Int(length) * dtype.as_fixed_size_binary().byte_width,
                    owner,
                )
            )
        elif dtype.is_fixed_size_list():
            children.append(
                self.children[0][].to_data(
                    dtype.as_fixed_size_list().value_type(), owner
                )
            )
        elif dtype.is_struct():
            ref st = dtype.as_struct()
            for i in range(Int(self.n_children)):
                children.append(
                    self.children[i][].to_data(st.fields[i].dtype, owner)
                )
        elif dtype.is_map():
            # Same physical layout as a list: an int32 offsets buffer plus one
            # child = the entries struct (synthesized from the map's key/value).
            buffers.append(
                Buffer.from_foreign(
                    self.buffers[1],
                    (Int(length) + 1) * size_of[DType.int32](),
                    owner,
                )
            )
            var entries_dt = dtype.as_map().entries_field().dtype.copy()
            children.append(self.children[0][].to_data(entries_dt, owner))
        elif dtype.is_dictionary():
            ref dt = dtype.as_dictionary()
            buffers.append(
                Buffer.from_foreign(
                    self.buffers[1],
                    Int(length) * dt.index_type().byte_width(),
                    owner,
                )
            )
            children.append(self.dictionary[].to_data(dt.value_type(), owner))
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
        self, dtype: DynType, owner: ArcPointer[Allocation]
    ) raises -> DynArray:
        """Build an DynArray from this CArrowArray.  Thin wrapper over to_data.
        """
        return DynArray.from_data(self.to_data(dtype, owner))

    @staticmethod
    def from_array(array: DynArray) raises -> CArrowArray:
        """Build a CArrowArray from a Mojo DynArray.  Thin wrapper over from_data.
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
        # Null arrays carry no buffers at all (not even validity).  Every other
        # type has a leading validity slot followed by `data.buffers`.
        var is_null_dtype = data.dtype.is_null()
        var is_dictionary = data.dtype.is_dictionary()
        var n_buffers: Int64
        if is_null_dtype:
            n_buffers = 0
        else:
            n_buffers = Int64(1 + len(data.buffers))  # 1 = validity bitmap slot
        # Dictionary arrays expose values via the `dictionary` field, not children.
        var n_children = Int64(0) if is_dictionary else Int64(
            len(data.children)
        )

        # Heap-allocate ArrayData to keep ArcPointer ref-counts alive.
        var data_heap = alloc[ArrayData](1)
        data_heap.unsafe_write(data^)

        # Heap-allocate the buffers pointer array.
        # buffers[0] = validity bitmap (null pointer means all-valid).
        # buffers[1..n] = data.buffers[0..n-1] in order.
        var buffers: UnsafePointer[
            OpaquePointer[MutUntrackedOrigin], MutUntrackedOrigin
        ] = _null_ptr[OpaquePointer[MutUntrackedOrigin]]()
        if not is_null_dtype:
            buffers = alloc[OpaquePointer[MutUntrackedOrigin]](Int(n_buffers))
            if data_heap[].bitmap:
                buffers[0] = OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=Int(
                        data_heap[].bitmap.value().view().unsafe_ptr()
                    )
                )
            else:
                buffers[0] = _null_ptr[NoneType]()
            for i in range(len(data_heap[].buffers)):
                buffers[1 + i] = OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=Int(
                        data_heap[].buffers[i].view[DType.uint8]().unsafe_ptr()
                    )
                )

        # Recursively build children; each child is moved onto the heap so the
        # pointer in children_ptr remains valid after this stack frame exits.
        var children_ptr: UnsafePointer[
            UnsafePointer[CArrowArray, MutUntrackedOrigin], MutUntrackedOrigin
        ] = _null_ptr[UnsafePointer[CArrowArray, MutUntrackedOrigin]]()
        if n_children > 0:
            children_ptr = alloc[
                UnsafePointer[CArrowArray, MutUntrackedOrigin]
            ](Int(n_children))
            for i in range(Int(n_children)):
                var child = CArrowArray.from_data(
                    data_heap[].children[i].copy()
                )
                var child_ptr = alloc[CArrowArray](1)
                child_ptr.unsafe_write(child^)
                children_ptr[i] = child_ptr

        # For dictionary arrays, children[0] holds the values array; expose it
        # as the C dictionary field.
        var dict_ptr = _null_ptr[CArrowArray]()
        if is_dictionary and len(data_heap[].children) > 0:
            var dict_c = CArrowArray.from_data(data_heap[].children[0].copy())
            var dp = alloc[CArrowArray](1)
            dp.unsafe_write(dict_c^)
            dict_ptr = dp

        return CArrowArray(
            length=Int64(data_heap[].length),
            null_count=Int64(data_heap[].nulls),
            offset=Int64(data_heap[].offset),
            n_buffers=n_buffers,
            n_children=n_children,
            buffers=buffers,
            children=children_ptr,
            dictionary=dict_ptr,
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
        src[].mark_released()
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
        ptr.unsafe_write(self^)
        return PythonObject(
            from_owned=cpy.PyCapsule_New(
                ptr.bitcast[NoneType](),
                "arrow_array",
                _release_array_capsule,
            )
        )

    def to_array(deinit self, dtype: DynType) raises -> DynArray:
        """Convert to an DynArray, taking ownership of the C struct.

        The CArrowArray is moved onto the heap and wrapped in a
        Allocation.  Every Buffer / Bitmap view shares the same
        ArcPointer[Allocation], so the C release callback fires
        automatically when the last buffer referencing this import is dropped.
        """
        var heap_c = alloc[CArrowArray](1)
        heap_c.unsafe_write(self^)
        var owner = ArcPointer(
            Allocation.foreign(heap_c.bitcast[UInt8](), _release_imported_array)
        )
        return heap_c[].to_array(dtype, owner)


# ---------------------------------------------------------------------------
# Arrow C Device Data Interface
# https://arrow.apache.org/docs/format/CDeviceDataInterface.html
# ---------------------------------------------------------------------------


def _release_c_device_array(
    ptr: UnsafePointer[UInt8, MutUntrackedOrigin]
) -> None:
    """Release callback for CArrowDeviceArray imported via the C Device Data Interface.

    Called when the last Buffer that references the imported array is dropped.
    Invokes the C-level release callback on the embedded ArrowArray, then frees
    the Mojo heap allocation.
    """
    var c_ptr = ptr.bitcast[CArrowDeviceArray]()
    c_ptr[].array.release(
        UnsafePointer(to=c_ptr[].array).unsafe_origin_cast[MutUntrackedOrigin]()
    )
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
    var sync_event: OpaquePointer[MutUntrackedOrigin]
    var reserved0: Int64
    var reserved1: Int64
    var reserved2: Int64

    def to_array(
        deinit self, dtype: DynType, ctx: DeviceContext
    ) raises -> DynArray:
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
            An `DynArray` whose buffers reference the device memory directly
            (zero-copy for device types; CPU for ARROW_DEVICE_CPU).
        """
        if Int(self.sync_event) != 0:
            ctx.synchronize()

        var heap_c = alloc[CArrowDeviceArray](1)
        heap_c.unsafe_write(self^)
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

    Holds the schema and record batches that the stream yields.  `index`
    tracks the current position in `batches`.
    """

    var schema: Schema
    var batches: List[RecordBatch]
    var index: Int

    def __init__(out self, var schema: Schema, var batches: List[RecordBatch]):
        self.schema = schema^
        self.batches = batches^
        self.index = 0


def _stream_get_schema(
    stream_ptr: UnsafePointer[CArrowArrayStream, MutUntrackedOrigin],
    schema_out: UnsafePointer[CArrowSchema, MutUntrackedOrigin],
) abi("C") -> Int32:
    """Stream callback: write the schema into `schema_out`."""
    try:
        var data = stream_ptr[].private_data.bitcast[_StreamPrivateData]()
        schema_out.unsafe_write(CArrowSchema.from_schema(data[].schema))
        return 0
    except:
        return 1


def _stream_get_next(
    stream_ptr: UnsafePointer[CArrowArrayStream, MutUntrackedOrigin],
    array_out: UnsafePointer[CArrowArray, MutUntrackedOrigin],
) abi("C") -> Int32:
    """Stream callback: write the next batch into `array_out`, or signal end."""
    try:
        var data = stream_ptr[].private_data.bitcast[_StreamPrivateData]()
        if data[].index >= len(data[].batches):
            # Signal end-of-stream: set release to null.
            array_out[].mark_released()
            return 0
        var batch = data[].batches[data[].index].copy()
        data[].index += 1
        var struct_arr: DynArray = batch.to_struct_array()
        array_out.unsafe_write(CArrowArray.from_array(struct_arr))
        return 0
    except:
        return 1


def _stream_get_last_error(
    stream_ptr: UnsafePointer[CArrowArrayStream, MutUntrackedOrigin],
) abi("C") -> UnsafePointer[UInt8, MutUntrackedOrigin]:
    """Stream callback: return null (no detailed error tracking)."""
    return _null_ptr[UInt8]()


def _stream_release(
    stream_ptr: UnsafePointer[CArrowArrayStream, MutUntrackedOrigin],
) abi("C") -> None:
    """Stream callback: free private data and null the release field."""
    var data = stream_ptr[].private_data.bitcast[_StreamPrivateData]()
    data.unsafe_deinit_pointee()
    data.free()
    stream_ptr[].mark_released()


def _release_stream_capsule(capsule: PyObjectPtr) abi("C"):
    """PyCapsule destructor for "arrow_array_stream" capsules."""
    try:
        var py = Python()
        ref cpy = py.cpython()
        var ptr = cpy.PyCapsule_GetPointer(capsule, "arrow_array_stream")
        if Int(ptr) != 0:
            var c_stream = ptr.bitcast[CArrowArrayStream]()
            if not c_stream[].is_released():
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
        UnsafePointer[CArrowArrayStream, MutUntrackedOrigin],
        UnsafePointer[CArrowSchema, MutUntrackedOrigin],
    ) thin abi("C") -> Int32
    var get_next: def(
        UnsafePointer[CArrowArrayStream, MutUntrackedOrigin],
        UnsafePointer[CArrowArray, MutUntrackedOrigin],
    ) thin abi("C") -> Int32
    var get_last_error: def(
        UnsafePointer[CArrowArrayStream, MutUntrackedOrigin]
    ) thin abi("C") -> UnsafePointer[UInt8, MutUntrackedOrigin]
    var release: def(
        UnsafePointer[CArrowArrayStream, MutUntrackedOrigin]
    ) thin abi("C") -> None
    var private_data: OpaquePointer[MutUntrackedOrigin]

    def is_released(self) -> Bool:
        """True once ownership has moved on — see `CArrowSchema.is_released`."""
        return UnsafePointer(to=self.release).bitcast[UInt64]()[0] == 0

    def mark_released(mut self):
        """Give up ownership: null the callback so nobody can release twice."""
        UnsafePointer(to=self.release).bitcast[UInt64]()[0] = 0

    @staticmethod
    def from_batches(
        var schema: Schema, var batches: List[RecordBatch]
    ) -> CArrowArrayStream:
        """Build a CArrowArrayStream that yields the given batches.

        The stream takes ownership of the batches; callers should not
        mutate them after this call.
        """
        var data = alloc[_StreamPrivateData](1)
        data.unsafe_write(_StreamPrivateData(schema^, batches^))
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
        src[].mark_released()
        return stream

    def to_pycapsule(self) raises -> PythonObject:
        """Wrap this stream in a Python "arrow_array_stream" PyCapsule."""
        var py = Python()
        ref cpy = py.cpython()
        var ptr = alloc[CArrowArrayStream](1)
        ptr.unsafe_write(self)
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
        heap.unsafe_write(self)

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
            if c_array[].is_released():
                c_array.free()
                break
            var struct_dtype = struct_(schema.fields.copy())
            var arr = c_array.take_pointee().to_array(struct_dtype^)
            var columns = List[DynArray]()
            for child in arr.as_struct().children:
                columns.append(child.copy())
            batches.append(RecordBatch(schema=schema, columns=columns^))

        # Release the stream.
        heap[].release(heap)
        heap.free()
        return Table.from_batches(schema, batches^)
