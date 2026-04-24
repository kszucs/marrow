"""Arrow integration JSON format reader.

Parses the Apache Arrow integration testing JSON format and converts record
batches to Marrow in-memory arrays.

Format reference: https://arrow.apache.org/docs/format/Integration.html

JSON is parsed via the `json` Mojo package (https://github.com/ehsanmok/json).
The array-building layer is pure Mojo using the standard builders.
"""

from json import load, Value
from .dtypes import *
from .schema import Schema
from .arrays import AnyArray, ArrayData
from .builders import (
    BoolBuilder,
    Int8Builder,
    Int16Builder,
    Int32Builder,
    Int64Builder,
    UInt8Builder,
    UInt16Builder,
    UInt32Builder,
    UInt64Builder,
    Float32Builder,
    Float64Builder,
    StringBuilder,
)
from .buffers import Bitmap, Buffer
from .tabular import RecordBatch


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


struct IntegrationJson(Movable):
    """Holds the schema and record batches parsed from an integration JSON file.
    """

    var schema: Schema
    var batches: List[RecordBatch]

    def __init__(out self, var schema: Schema, var batches: List[RecordBatch]):
        self.schema = schema^
        self.batches = batches^

    def __init__(out self, *, deinit take: Self):
        self.schema = take.schema^
        self.batches = take.batches^


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------


def _dtype_from_json(
    type_obj: Value, children_arr: Value
) raises -> AnyDataType:
    """Convert an Arrow integration JSON type object to a Marrow AnyDataType."""
    var name = type_obj["name"].string_value()
    if name == "int":
        var bit_width = Int(type_obj["bitWidth"].int_value())
        var is_signed = type_obj["isSigned"].bool_value()
        if is_signed:
            if bit_width == 8:
                return int8
            elif bit_width == 16:
                return int16
            elif bit_width == 32:
                return int32
            elif bit_width == 64:
                return int64
        else:
            if bit_width == 8:
                return uint8
            elif bit_width == 16:
                return uint16
            elif bit_width == 32:
                return uint32
            elif bit_width == 64:
                return uint64
        raise Error("unsupported integer bit width: " + String(bit_width))
    elif name == "floatingpoint":
        var precision = type_obj["precision"].string_value()
        if precision == "SINGLE":
            return float32
        elif precision == "DOUBLE":
            return float64
        raise Error("unsupported float precision: " + precision)
    elif name == "bool":
        return bool_
    elif name == "utf8":
        return string
    elif name == "binary":
        return binary
    elif name == "null":
        return null
    elif name == "list":
        var child_field = _field_from_json(children_arr[0])
        return list_(child_field.dtype.copy())
    elif name == "fixedsizelist":
        var child_field = _field_from_json(children_arr[0])
        var list_size = Int(type_obj["listSize"].int_value())
        return fixed_size_list_(child_field.dtype.copy(), list_size)
    elif name == "struct":
        var fields = List[Field]()
        var n = children_arr.array_count()
        for i in range(n):
            fields.append(_field_from_json(children_arr[i]))
        return struct_(fields^)
    raise Error("unsupported type name: " + name)


def _field_from_json(field_obj: Value) raises -> Field:
    """Convert an Arrow integration JSON field object to a Marrow Field."""
    var name = field_obj["name"].string_value()
    var nullable = field_obj["nullable"].bool_value()
    var dtype = _dtype_from_json(field_obj["type"], field_obj["children"])
    return Field(name, dtype^, nullable)


def _schema_from_json(schema_obj: Value) raises -> Schema:
    """Convert an Arrow integration JSON schema object to a Marrow Schema."""
    var schema = Schema()
    var fields_arr = schema_obj["fields"]
    var n = fields_arr.array_count()
    for i in range(n):
        schema.append(_field_from_json(fields_arr[i]))
    return schema^


# ---------------------------------------------------------------------------
# Array parsing helpers
# ---------------------------------------------------------------------------


def _count_nulls(validity_arr: Value, count: Int) raises -> Int:
    """Count null entries in a VALIDITY array."""
    var null_count = 0
    for i in range(count):
        if validity_arr[i].int_value() == 0:
            null_count += 1
    return null_count


def _build_bitmap(
    validity_arr: Value, count: Int, null_count: Int
) raises -> Optional[Bitmap[mut=False]]:
    """Build a validity Bitmap from a JSON VALIDITY array.

    Returns None when there are no nulls (all-valid fast path).
    """
    if null_count == 0:
        return None
    var bm = Bitmap.alloc_zeroed(count)
    for i in range(count):
        if validity_arr[i].int_value() != 0:
            bm.unsafe_set(i)
    return bm.to_immutable()


def _array_from_json(col_obj: Value, dt: AnyDataType) raises -> AnyArray:
    """Convert an Arrow integration JSON column object to a Marrow AnyArray."""
    var count = Int(col_obj["count"].int_value())
    var validity_arr = col_obj["VALIDITY"]

    if dt == bool_:
        var b = BoolBuilder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                b.append(data_arr[i].int_value() != 0)
        return AnyArray(b.finish())

    elif dt == int8:
        var b = Int8Builder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                b.append(Scalar[int8.native](Int(data_arr[i].int_value())))
        return AnyArray(b.finish())

    elif dt == int16:
        var b = Int16Builder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                b.append(Scalar[int16.native](Int(data_arr[i].int_value())))
        return AnyArray(b.finish())

    elif dt == int32:
        var b = Int32Builder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                b.append(Scalar[int32.native](Int(data_arr[i].int_value())))
        return AnyArray(b.finish())

    elif dt == int64:
        var b = Int64Builder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                # INT64 values are stored as strings in integration JSON
                # to avoid floating-point precision loss.
                b.append(Scalar[int64.native](atol(data_arr[i].string_value())))
        return AnyArray(b.finish())

    elif dt == uint8:
        var b = UInt8Builder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                b.append(Scalar[uint8.native](Int(data_arr[i].int_value())))
        return AnyArray(b.finish())

    elif dt == uint16:
        var b = UInt16Builder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                b.append(Scalar[uint16.native](Int(data_arr[i].int_value())))
        return AnyArray(b.finish())

    elif dt == uint32:
        var b = UInt32Builder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                b.append(Scalar[uint32.native](Int(data_arr[i].int_value())))
        return AnyArray(b.finish())

    elif dt == uint64:
        var b = UInt64Builder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                # UINT64 values are stored as strings in integration JSON.
                b.append(
                    Scalar[uint64.native](atol(data_arr[i].string_value()))
                )
        return AnyArray(b.finish())

    elif dt == float32:
        var b = Float32Builder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                b.append(Scalar[float32.native](data_arr[i].float_value()))
        return AnyArray(b.finish())

    elif dt == float64:
        var b = Float64Builder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                b.append(Scalar[float64.native](data_arr[i].float_value()))
        return AnyArray(b.finish())

    elif dt.is_string():
        var b = StringBuilder(count)
        var data_arr = col_obj["DATA"]
        for i in range(count):
            if validity_arr[i].int_value() == 0:
                b.append_null()
            else:
                b.append(data_arr[i].string_value())
        return AnyArray(b.finish())

    elif dt.is_list():
        return _list_array_from_json(col_obj, dt, count, validity_arr)

    elif dt.is_fixed_size_list():
        return _fixed_size_list_array_from_json(
            col_obj, dt, count, validity_arr
        )

    elif dt.is_struct():
        return _struct_array_from_json(col_obj, dt, count, validity_arr)

    raise Error("unsupported dtype in JSON integration reader: " + String(dt))


def _list_array_from_json(
    col_obj: Value,
    dt: AnyDataType,
    count: Int,
    validity_arr: Value,
) raises -> AnyArray:
    """Build a ListArray from integration JSON column data."""
    var lt = dt.as_list_type()
    var child_col = col_obj["children"][0]
    var child_arr = _array_from_json(child_col, lt.value_type())

    # Build int32 offsets buffer (Arrow list offsets are signed int32).
    var offset_json = col_obj["OFFSET"]
    var num_offsets = count + 1
    var offsets_bb = Buffer.alloc_uninit[DType.int32](num_offsets)
    for i in range(num_offsets):
        offsets_bb.unsafe_set[DType.int32](
            i, Int32(Int(offset_json[i].int_value()))
        )
    var offsets_buf = offsets_bb.to_immutable()

    var null_count = _count_nulls(validity_arr, count)
    var bm = _build_bitmap(validity_arr, count, null_count)

    var children = List[ArrayData]()
    children.append(child_arr.to_data())

    return AnyArray.from_data(
        ArrayData(
            dtype=dt.copy(),
            length=count,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            buffers=[offsets_buf^],
            children=children^,
        )
    )


def _fixed_size_list_array_from_json(
    col_obj: Value,
    dt: AnyDataType,
    count: Int,
    validity_arr: Value,
) raises -> AnyArray:
    """Build a FixedSizeListArray from integration JSON column data."""
    var fsl = dt.as_fixed_size_list_type()
    var child_col = col_obj["children"][0]
    var child_arr = _array_from_json(child_col, fsl.value_type())

    var null_count = _count_nulls(validity_arr, count)
    var bm = _build_bitmap(validity_arr, count, null_count)

    var children = List[ArrayData]()
    children.append(child_arr.to_data())

    return AnyArray.from_data(
        ArrayData(
            dtype=dt.copy(),
            length=count,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            buffers=[],
            children=children^,
        )
    )


def _struct_array_from_json(
    col_obj: Value,
    dt: AnyDataType,
    count: Int,
    validity_arr: Value,
) raises -> AnyArray:
    """Build a StructArray from integration JSON column data."""
    var st = dt.as_struct_type()
    var children_json = col_obj["children"]
    var children = List[ArrayData]()
    var num_fields = len(st.fields)
    for i in range(num_fields):
        var child_col = children_json[i]
        var child_arr = _array_from_json(child_col, st.fields[i].dtype.copy())
        children.append(child_arr.to_data())

    var null_count = _count_nulls(validity_arr, count)
    var bm = _build_bitmap(validity_arr, count, null_count)

    return AnyArray.from_data(
        ArrayData(
            dtype=dt.copy(),
            length=count,
            nulls=null_count,
            offset=0,
            bitmap=bm^,
            buffers=[],
            children=children^,
        )
    )


# ---------------------------------------------------------------------------
# Batch and file reading
# ---------------------------------------------------------------------------


def _record_batch_from_json(
    batch_obj: Value,
    schema: Schema,
) raises -> RecordBatch:
    """Convert an Arrow integration JSON batch object to a Marrow RecordBatch.
    """
    var columns_json = batch_obj["columns"]
    var columns = List[AnyArray]()
    var num_fields = len(schema.fields)
    for i in range(num_fields):
        var col_obj = columns_json[i]
        var arr = _array_from_json(col_obj, schema.fields[i].dtype.copy())
        columns.append(arr^)
    return RecordBatch(schema, columns^)


def read_json_file(path: String) raises -> IntegrationJson:
    """Read an Arrow integration JSON file and return the schema and batches.

    The file must follow the Apache Arrow integration testing JSON format
    described at https://arrow.apache.org/docs/format/Integration.html.

    Args:
        path: Filesystem path to the `.json` file.

    Returns:
        An IntegrationJson holding the parsed Schema and List[RecordBatch].
    """
    var root = load(path)
    var schema = _schema_from_json(root["schema"])
    var batches = List[RecordBatch]()
    var batches_json = root["batches"]
    var n = batches_json.array_count()
    for i in range(n):
        batches.append(_record_batch_from_json(batches_json[i], schema))
    return IntegrationJson(schema^, batches^)
