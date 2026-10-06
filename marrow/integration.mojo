# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The Arrow integration-testing JSON format, read into a schema and batches.

This is the format `archery` generates to test Arrow implementations against
each other: a schema, the record batches, and the dictionary batches they
reference, with every buffer spelled out as JSON —

```json
{"schema": {"fields": [...], "metadata": [...]},
 "dictionaries": [{"id": 0, "data": {"count": 3, "columns": [...]}}],
 "batches": [{"count": 3, "columns": [...]}]}
```

A column carries `count`, `VALIDITY` (0/1 per slot) and the buffers its type
needs: `DATA`, `OFFSET`, `VIEWS` with `VARIADIC_DATA_BUFFERS`, and `children`.
The encoding follows Arrow C++'s reader (`arrow/integration/json_internal.cc`):

- 64-bit integers, decimals and large offsets are decimal *strings*, narrower
  integers are numbers; either spelling is accepted for any integer;
- binary values are hexadecimal, string values are text;
- a day-time interval is `{"days", "milliseconds"}`, a month-day-nano interval
  `{"months", "days", "nanoseconds"}`;
- a dictionary-encoded field's `type` and `children` describe its *values*,
  its column holds the indices, and `dictionary.id` names the dictionary batch.

Union, run-end-encoded and list-view columns are not supported.

This is test infrastructure, not a data format: it exists so marrow can take
part in the cross-implementation integration suite with no other Arrow library
in the path.
"""

from emberjson import Document, DocValue, from_json

from .arrays import ArrayData, BoolArray, DictionaryArray, DynArray
from .buffers import Bitmap, Buffer
from .dtypes import (
    DynType,
    Field,
    FixedSizeListType,
    LargeListType,
    ListType,
    MapType,
    PrimitiveType,
    TimeUnit,
    binary,
    binary_view,
    bool_,
    date32,
    date64,
    day_time_interval,
    decimal32,
    decimal64,
    decimal128,
    decimal256,
    dictionary,
    duration,
    fixed_size_binary_,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    large_binary,
    large_string,
    microsecond,
    millisecond,
    month_day_nano_interval,
    nanosecond,
    null,
    second,
    string,
    string_view,
    struct_,
    time32,
    time64,
    timestamp,
    uint8,
    uint16,
    uint32,
    uint64,
    year_month_interval,
)
from .errors import CorruptError, DynError, InvalidError, NotImplementedError
from .io import BufferSource
from .schema import Schema
from .tabular import RecordBatch


@fieldwise_init
struct IntegrationJson(Movable):
    """A parsed integration JSON file: its schema and its record batches."""

    var schema: Schema
    var batches: List[RecordBatch]

    @staticmethod
    def parse(text: StringSlice) raises DynError -> IntegrationJson:
        """Parse integration JSON text."""
        var doc: Document
        try:
            doc = from_json[Document](text)
        except e:
            raise InvalidError(t"integration json: {e}")
        try:
            var root = _Node(doc.root())
            var reader = _Reader(root)
            return reader.read(root)
        except e:
            raise DynError(e)

    @staticmethod
    def read(path: String) raises DynError -> IntegrationJson:
        """Read an integration JSON file."""
        try:
            var source = BufferSource(path)
            return IntegrationJson.parse(
                StringSlice(unsafe_from_utf8=source.read_at(0, source.size()))
            )
        except e:
            raise DynError(e)


# ---------------------------------------------------------------------------
# JSON values
# ---------------------------------------------------------------------------


@fieldwise_init
struct _Node[o: ImmOrigin](TrivialRegisterPassable):
    """One JSON value in the document, with the reads the format needs."""

    var value: DocValue[Self.o]

    def __getitem__(self, key: StringSlice) raises -> Self:
        return Self(self.value[key])

    def has(self, key: StringSlice) raises -> Bool:
        """Whether this object has `key` with a non-null value."""
        var obj = self.value.object()
        return key in obj and not obj[key].is_null()

    def int(self, key: StringSlice) raises -> Int:
        return Int(self.value[key].int())

    def flag(self, key: StringSlice, default: Bool) raises -> Bool:
        """The boolean at `key`, or `default` when it is absent."""
        return self.value[key].bool() if self.has(key) else default

    def string(self, key: StringSlice) raises -> StringSlice[Self.o]:
        return self.value[key].string_slice()

    def items(self, key: StringSlice) raises -> List[Self]:
        """The array at `key`, or none when it is absent."""
        var out = List[Self]()
        if self.has(key):
            for item in self.value[key].array():
                out.append(Self(item))
        return out^

    def metadata(self) raises -> Dict[String, String]:
        """The `[{"key", "value"}]` list at `metadata`, as a dictionary."""
        var out = Dict[String, String]()
        for entry in self.items("metadata"):
            out[String(entry.string("key"))] = String(entry.string("value"))
        return out^

    def bool(self) raises -> Bool:
        return self.value.bool()

    def truthy(self) raises -> Bool:
        """A validity flag: `1`/`0`, or `true`/`false`."""
        if self.value.is_bool():
            return self.value.bool()
        return self.value.int() != 0

    def integer[T: DType](self) raises -> Scalar[T]:
        """An integer spelled as a JSON number or as a decimal string."""
        if self.value.is_string():
            return Self.parse_integer[T](self.value.string_slice())
        elif self.value.is_uint():
            return self.value.uint().cast[T]()
        return self.value.int().cast[T]()

    def member[T: DType](self, key: StringSlice) raises -> Scalar[T]:
        """The integer at `key`, or zero: a null interval may be `{}`."""
        return self[key].integer[T]() if self.has(key) else Scalar[T](0)

    def float(self) raises -> Float64:
        if self.value.is_float():
            return self.value.float()
        return self.value.int().cast[DType.float64]()

    def decoded_size(self, hex: Bool) raises -> Int:
        """How many bytes this string decodes to: hex pairs, or its UTF-8."""
        var text = self.value.string_slice().as_bytes()
        if not hex:
            return len(text)
        if len(text) % 2 != 0:
            raise CorruptError(t"integration json: odd-length hex value")
        return len(text) // 2

    def decode_into(self, buf: Buffer[mut=True], at: Int, hex: Bool) raises:
        """Write this string, decoded, into `buf` from byte `at`."""
        var text = self.value.string_slice().as_bytes()
        if hex:
            for i in range(len(text) // 2):
                buf.unsafe_set[DType.uint8](
                    at + i,
                    Self.hex_digit(text[2 * i]) * 16
                    + Self.hex_digit(text[2 * i + 1]),
                )
        else:
            for i in range(len(text)):
                buf.unsafe_set[DType.uint8](at + i, text[i])

    @staticmethod
    def parse_integer[T: DType](text: StringSlice) raises -> Scalar[T]:
        """A decimal integer, wrapping like the fixed-width type it fills."""
        var bytes = text.as_bytes()
        var i = 0
        var negative = False
        if len(bytes) > 0 and (bytes[0] == UInt8(ord("-"))):
            negative = True
            i = 1
        if i == len(bytes):
            raise CorruptError(t"integration json: '{text}' is not an integer")
        var value = Scalar[T](0)
        while i < len(bytes):
            var digit = bytes[i] - UInt8(ord("0"))
            if digit > 9:
                raise CorruptError(
                    t"integration json: '{text}' is not an integer"
                )
            value = value * 10 + digit.cast[T]()
            i += 1
        return -value if negative else value

    @staticmethod
    def hex_digit(c: UInt8) raises -> UInt8:
        if c >= UInt8(ord("0")) and c <= UInt8(ord("9")):
            return c - UInt8(ord("0"))
        elif c >= UInt8(ord("A")) and c <= UInt8(ord("F")):
            return c - UInt8(ord("A")) + 10
        elif c >= UInt8(ord("a")) and c <= UInt8(ord("f")):
            return c - UInt8(ord("a")) + 10
        raise CorruptError(t"integration json: byte {Int(c)} is not hex")


# ---------------------------------------------------------------------------
# One column's buffers
# ---------------------------------------------------------------------------


@fieldwise_init
struct _Validity(Copyable, Movable):
    var bitmap: Optional[Bitmap[mut=False]]
    var nulls: Int


struct _Column[o: ImmOrigin](TrivialRegisterPassable):
    """One column's JSON: its slot count and the buffers its type needs.

    Every method checks that its buffer has one entry per slot (or one more,
    for offsets), so a mismatch is reported where it is read.
    """

    var node: _Node[Self.o]
    var count: Int

    def __init__(out self, node: _Node[Self.o]) raises:
        self.node = node
        self.count = node.int("count")

    def children(self) raises -> List[_Node[Self.o]]:
        return self.node.items("children")

    def validity(self) raises -> _Validity:
        """The validity bitmap, or none when every slot is valid."""
        if not self.node.has("VALIDITY"):
            return _Validity(None, 0)
        var flags = self.node.items("VALIDITY")
        if len(flags) != self.count:
            raise CorruptError(
                t"integration json: {len(flags)} validity flags for"
                t" {self.count} slots"
            )
        var bits = Bitmap.alloc_zeroed(self.count)
        var nulls = 0
        for i in range(self.count):
            if flags[i].truthy():
                bits.set(i)
            else:
                nulls += 1
        if nulls == 0:
            return _Validity(None, 0)
        return _Validity(bits^.to_immutable(length=self.count), nulls)

    def values(self) raises -> List[_Node[Self.o]]:
        """`DATA`, one value per slot."""
        var data = self.node.items("DATA")
        if len(data) != self.count:
            raise CorruptError(
                t"integration json: {len(data)} values for {self.count} slots"
            )
        return data^

    def offsets(self) raises -> List[Int]:
        """`OFFSET`, one more than there are slots."""
        var offsets = List[Int]()
        for v in self.node.items("OFFSET"):
            offsets.append(Int(v.integer[DType.int64]()))
        if len(offsets) != self.count + 1:
            raise CorruptError(
                t"integration json: {len(offsets)} offsets for"
                t" {self.count} slots"
            )
        return offsets^

    def offset_buffer(
        self, offsets: List[Int], large: Bool
    ) -> Buffer[mut=False]:
        """`offsets` as int64 for the large layouts, int32 otherwise."""
        var buf = Buffer.alloc_zeroed((8 if large else 4) * len(offsets))
        for i in range(len(offsets)):
            if large:
                buf.unsafe_set[DType.int64](i, Int64(offsets[i]))
            else:
                buf.unsafe_set[DType.int32](i, Int32(offsets[i]))
        return buf^.to_immutable()

    def bits(self) raises -> Bitmap[mut=False]:
        """`DATA` of a boolean column, bit-packed."""
        var bits = Bitmap.alloc_zeroed(self.count)
        var data = self.values()
        for i in range(self.count):
            if data[i].bool():
                bits.set(i)
        return bits^.to_immutable(length=self.count)

    def fixed(self, dtype: DynType) raises -> Buffer[mut=False]:
        """`DATA` of a fixed-width column, as `dtype`'s native values."""
        var data = self.values()

        def fill[T: PrimitiveType](d: T) raises {imm} -> Buffer[mut=False]:
            var buf = Buffer.alloc_zeroed[T.native](len(data))
            for i in range(len(data)):
                # A half float is spelled as its uint16 bit pattern, as Arrow
                # C++ writes it.
                comptime if T.native == DType.float16:
                    buf.unsafe_set[DType.uint16](
                        i, data[i].integer[DType.uint16]()
                    )
                elif T.native.is_floating_point():
                    buf.unsafe_set[T.native](
                        i, data[i].float().cast[T.native]()
                    )
                else:
                    buf.unsafe_set[T.native](i, data[i].integer[T.native]())
            return buf^.to_immutable()

        return dtype.dispatch_primitive(fill)

    def binary(self, offsets: List[Int], hex: Bool) raises -> Buffer[mut=False]:
        """`DATA` of a binary or string column, laid out at `offsets`."""
        var buf = Buffer.alloc_zeroed(offsets[self.count])
        var data = self.values()
        for i in range(self.count):
            var size = data[i].decoded_size(hex)
            var span = offsets[i + 1] - offsets[i]
            if size != span:
                raise CorruptError(
                    t"integration json: value {i} is {size} bytes, not {span}"
                )
            data[i].decode_into(buf, offsets[i], hex)
        return buf^.to_immutable()

    def day_time(self) raises -> Buffer[mut=False]:
        """Per slot: int32 days, then int32 milliseconds."""
        var buf = Buffer.alloc_zeroed[DType.int32](2 * self.count)
        var data = self.values()
        for i in range(self.count):
            buf.unsafe_set(2 * i, data[i].member[DType.int32]("days"))
            buf.unsafe_set(
                2 * i + 1, data[i].member[DType.int32]("milliseconds")
            )
        return buf^.to_immutable()

    def month_day_nano(self) raises -> Buffer[mut=False]:
        """Per slot: int32 months, int32 days, int64 nanoseconds."""
        var buf = Buffer.alloc_zeroed(16 * self.count)
        var data = self.values()
        for i in range(self.count):
            buf.unsafe_set(4 * i, data[i].member[DType.int32]("months"))
            buf.unsafe_set(4 * i + 1, data[i].member[DType.int32]("days"))
            buf.unsafe_set(
                2 * i + 1, data[i].member[DType.int64]("nanoseconds")
            )
        return buf^.to_immutable()

    def views(self, hex: Bool) raises -> List[Buffer[mut=False]]:
        """The 16-byte views, then the variadic data buffers they point into."""
        var json_views = self.node.items("VIEWS")
        if len(json_views) != self.count:
            raise CorruptError(
                t"integration json: {len(json_views)} views for"
                t" {self.count} slots"
            )
        var views = Buffer.alloc_zeroed(16 * self.count)
        for i in range(self.count):
            var view = json_views[i]
            var size = view.int("SIZE")
            views.unsafe_set[DType.int32](4 * i, Int32(size))
            if size <= 12:
                var inlined = view["INLINED"]
                if inlined.decoded_size(hex) != size:
                    raise CorruptError(
                        t"integration json: view {i} inlines"
                        t" {inlined.decoded_size(hex)} bytes of {size}"
                    )
                inlined.decode_into(views, 16 * i + 4, hex)
            else:
                var prefix = view["PREFIX_HEX"]
                if prefix.decoded_size(hex=True) != 4:
                    raise CorruptError(
                        t"integration json: view {i} has no 4-byte prefix"
                    )
                prefix.decode_into(views, 16 * i + 4, hex=True)
                views.unsafe_set[DType.int32](
                    4 * i + 2, Int32(view.int("BUFFER_INDEX"))
                )
                views.unsafe_set[DType.int32](
                    4 * i + 3, Int32(view.int("OFFSET"))
                )
        var buffers = List[Buffer[mut=False]]()
        buffers.append(views^.to_immutable())
        for data in self.node.items("VARIADIC_DATA_BUFFERS"):
            var buf = Buffer.alloc_zeroed(data.decoded_size(hex=True))
            data.decode_into(buf, 0, hex=True)
            buffers.append(buf^.to_immutable())
        return buffers^


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


struct _Reader[o: ImmOrigin]:
    """Reads a document: the schema from its fields, then every batch.

    A dictionary batch is decoded the first time a column references it,
    against that column's field, so the order of `dictionaries` in the file
    does not matter, and a dictionary whose values are themselves
    dictionary-encoded resolves the inner one the same way.
    """

    var dictionaries: List[_Node[Self.o]]
    var decoded: Dict[Int, DynArray]

    def __init__(out self, root: _Node[Self.o]) raises:
        self.dictionaries = root.items("dictionaries")
        self.decoded = Dict[Int, DynArray]()

    def read(mut self, root: _Node[Self.o]) raises -> IntegrationJson:
        var nodes = root["schema"].items("fields")
        var fields = List[Field]()
        for node in nodes:
            fields.append(Self.field(node))
        var schema = Schema(fields=fields^, metadata=root["schema"].metadata())

        var batches = List[RecordBatch]()
        for batch in root.items("batches"):
            var columns = batch.items("columns")
            if len(columns) != len(nodes):
                raise CorruptError(
                    t"integration json: a batch has {len(columns)} columns"
                    t" for {len(nodes)} fields"
                )
            var arrays = List[DynArray]()
            for i in range(len(nodes)):
                arrays.append(
                    DynArray.from_data(
                        self.column(
                            schema.fields[i], nodes[i], _Column(columns[i])
                        )
                    )
                )
            batches.append(RecordBatch(schema, arrays^))
        return IntegrationJson(schema^, batches^)

    # --- the schema ---

    @staticmethod
    def field(node: _Node[Self.o]) raises -> Field:
        return Field(
            String(node.string("name")),
            Self.dtype(node),
            node.flag("nullable", True),
            node.metadata(),
        )

    @staticmethod
    def dtype(node: _Node[Self.o]) raises -> DynType:
        """The field's type: its value type, dictionary-encoded if it says so.
        """
        var value_type = Self.value_type(node)
        if not node.has("dictionary"):
            return value_type^
        var encoding = node["dictionary"]
        return dictionary(
            Self.int_type(encoding["indexType"]),
            value_type^,
            encoding.flag("isOrdered", False),
        )

    @staticmethod
    def int_type(spec: _Node[Self.o]) raises -> DynType:
        var width = spec.int("bitWidth")
        var signed = spec["isSigned"].bool()
        if width == 8:
            return int8 if signed else uint8
        elif width == 16:
            return int16 if signed else uint16
        elif width == 32:
            return int32 if signed else uint32
        elif width == 64:
            return int64 if signed else uint64
        raise CorruptError(t"integration json: no {width}-bit integer type")

    @staticmethod
    def time_unit(spec: _Node[Self.o]) raises -> TimeUnit:
        var unit = spec.string("unit")
        if unit == "SECOND":
            return second
        elif unit == "MILLISECOND":
            return millisecond
        elif unit == "MICROSECOND":
            return microsecond
        elif unit == "NANOSECOND":
            return nanosecond
        raise CorruptError(t"integration json: unknown time unit '{unit}'")

    @staticmethod
    def only_child(node: _Node[Self.o]) raises -> Field:
        var children = node.items("children")
        if len(children) != 1:
            var name = node["type"].string("name")
            raise CorruptError(
                t"integration json: a {name} has {len(children)} children,"
                t" not 1"
            )
        return Self.field(children[0])

    @staticmethod
    def value_type(node: _Node[Self.o]) raises -> DynType:
        """The type `node["type"]` and `node["children"]` describe."""
        var spec = node["type"]
        var name = spec.string("name")
        if name == "null":
            return null
        elif name == "bool":
            return bool_
        elif name == "int":
            return Self.int_type(spec)
        elif name == "floatingpoint":
            var precision = spec.string("precision")
            if precision == "HALF":
                return float16
            elif precision == "SINGLE":
                return float32
            elif precision == "DOUBLE":
                return float64
            raise CorruptError(
                t"integration json: unknown precision {precision}"
            )
        elif name == "utf8":
            return string
        elif name == "largeutf8":
            return large_string
        elif name == "binary":
            return binary
        elif name == "largebinary":
            return large_binary
        elif name == "utf8view":
            return string_view
        elif name == "binaryview":
            return binary_view
        elif name == "fixedsizebinary":
            return fixed_size_binary_(spec.int("byteWidth"))
        elif name == "decimal":
            var precision = spec.int("precision")
            var scale = spec.int("scale")
            var width = spec.int("bitWidth") if spec.has("bitWidth") else 128
            if width == 32:
                return decimal32(precision, scale)
            elif width == 64:
                return decimal64(precision, scale)
            elif width == 128:
                return decimal128(precision, scale)
            elif width == 256:
                return decimal256(precision, scale)
            raise CorruptError(t"integration json: no {width}-bit decimal type")
        elif name == "date":
            var unit = spec.string("unit")
            if unit == "DAY":
                return date32()
            elif unit == "MILLISECOND":
                return date64()
            raise CorruptError(t"integration json: unknown date unit '{unit}'")
        elif name == "time":
            var unit = Self.time_unit(spec)
            var wide = unit == microsecond or unit == nanosecond
            if spec.has("bitWidth"):
                var width = spec.int("bitWidth")
                if width != (64 if wide else 32):
                    raise CorruptError(
                        t"integration json: a {width}-bit time in {unit}"
                    )
            if wide:
                return time64(unit)
            return time32(unit)
        elif name == "timestamp":
            var timezone = String()
            if spec.has("timezone"):
                timezone = String(spec.string("timezone"))
            return timestamp(Self.time_unit(spec), timezone)
        elif name == "duration":
            return duration(Self.time_unit(spec))
        elif name == "interval":
            var unit = spec.string("unit")
            if unit == "YEAR_MONTH":
                return year_month_interval()
            elif unit == "DAY_TIME":
                return day_time_interval()
            elif unit == "MONTH_DAY_NANO":
                return month_day_nano_interval()
            raise CorruptError(
                t"integration json: unknown interval unit '{unit}'"
            )
        elif name == "list":
            return ListType(Self.only_child(node))
        elif name == "largelist":
            return LargeListType(Self.only_child(node))
        elif name == "fixedsizelist":
            return FixedSizeListType(
                Self.only_child(node), spec.int("listSize")
            )
        elif name == "map":
            return MapType(
                Self.only_child(node), spec.flag("keysSorted", False)
            )
        elif name == "struct":
            var fields = List[Field]()
            for child in node.items("children"):
                fields.append(Self.field(child))
            return struct_(fields^)
        raise NotImplementedError(
            t"integration json: unsupported type '{name}'"
        )

    # --- the columns ---

    def column(
        mut self, field: Field, node: _Node[Self.o], column: _Column[Self.o]
    ) raises -> ArrayData:
        """The column of `field`, which the JSON field `node` describes."""
        if not field.dtype.is_dictionary():
            return self.values(field.dtype, node, column)
        ref encoding = field.dtype.as_dictionary()
        var indices = self.values(encoding.index_type(), node, column)
        var id = node["dictionary"].int("id")
        return DictionaryArray.from_arrays(
            DynArray.from_data(indices),
            self.dictionary(id, encoding.value_type(), node),
            encoding.ordered,
        ).to_data()

    def dictionary(
        mut self, id: Int, dtype: DynType, node: _Node[Self.o]
    ) raises -> DynArray:
        """Dictionary batch `id`, decoded as `dtype`."""
        if id in self.decoded:
            return self.decoded[id].copy()
        for batch in self.dictionaries:
            if batch.int("id") == id:
                var columns = batch["data"].items("columns")
                if len(columns) != 1:
                    raise CorruptError(
                        t"integration json: dictionary {id} has"
                        t" {len(columns)} columns, not 1"
                    )
                var values = DynArray.from_data(
                    self.values(dtype, node, _Column(columns[0]))
                )
                self.decoded[id] = values.copy()
                return values^
        raise CorruptError(t"integration json: no dictionary with id {id}")

    def values(
        mut self, dtype: DynType, node: _Node[Self.o], column: _Column[Self.o]
    ) raises -> ArrayData:
        """The column, read as `dtype` without dictionary encoding."""
        var count = column.count
        if dtype.is_null():
            return ArrayData(
                dtype=dtype.copy(),
                length=count,
                nulls=count,
                offset=0,
                bitmap=None,
                buffers=[],
                children=[],
            )

        var validity = column.validity()
        var buffers = List[Buffer[mut=False]]()
        var children = List[ArrayData]()

        if dtype.is_bool():
            return BoolArray(
                length=count,
                nulls=validity.nulls,
                offset=0,
                bitmap=validity.bitmap.copy(),
                buffer=column.bits(),
            ).to_data()
        elif dtype.is_binary_like():
            var offsets = column.offsets()
            var large = dtype.is_large_string() or dtype.is_large_binary()
            buffers.append(column.offset_buffer(offsets, large))
            buffers.append(
                column.binary(offsets, hex=not dtype.is_string_like())
            )
        elif dtype.is_fixed_size_binary():
            var width = dtype.as_fixed_size_binary().byte_width
            var offsets = List[Int]()
            for i in range(count + 1):
                offsets.append(i * width)
            buffers.append(column.binary(offsets, hex=True))
        elif dtype.is_string_view() or dtype.is_binary_view():
            buffers = column.views(hex=dtype.is_binary_view())
        elif dtype.is_day_time_interval():
            buffers.append(column.day_time())
        elif dtype.is_month_day_nano_interval():
            buffers.append(column.month_day_nano())
        elif dtype.is_primitive():
            buffers.append(column.fixed(dtype))
        elif (
            dtype.is_list_like()
            or dtype.is_fixed_size_list()
            or dtype.is_struct()
        ):
            if dtype.is_list_like():
                buffers.append(
                    column.offset_buffer(
                        column.offsets(), dtype.is_large_list()
                    )
                )
            children = self.children(dtype.children(), node, column)
        else:
            raise NotImplementedError(
                t"integration json: unsupported type {dtype}"
            )

        return ArrayData(
            dtype=dtype.copy(),
            length=count,
            nulls=validity.nulls,
            offset=0,
            bitmap=validity.bitmap.copy(),
            buffers=buffers^,
            children=children^,
        )

    def children(
        mut self,
        fields: List[Field],
        node: _Node[Self.o],
        column: _Column[Self.o],
    ) raises -> List[ArrayData]:
        """The child columns of `column`, one per field of the nested type."""
        var nodes = node.items("children")
        var columns = column.children()
        if len(nodes) != len(fields) or len(columns) != len(fields):
            var name = node.string("name")
            raise CorruptError(
                t"integration json: column '{name}' has {len(columns)}"
                t" children for {len(fields)} fields"
            )
        var out = List[ArrayData]()
        for i in range(len(fields)):
            out.append(self.column(fields[i], nodes[i], _Column(columns[i])))
        return out^
