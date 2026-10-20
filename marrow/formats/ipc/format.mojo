# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""The IPC wire format's constants and small records: message header
kinds, type ids and units, the file magic, field nodes, body buffers,
blocks, and what a schema field carries beyond its Arrow type."""

from ... import dtypes as dt


# ---------------------------------------------------------------------------
# Arrow IPC wire protocol constants
# ---------------------------------------------------------------------------

comptime HEADER_SCHEMA: UInt8 = 1
comptime HEADER_DICTIONARY_BATCH: UInt8 = 2
comptime HEADER_RECORD_BATCH: UInt8 = 3
comptime METADATA_VERSION_V5: Int16 = 4
comptime ENDIANNESS_LITTLE: Int16 = 0
# How deeply schema fields may nest.
comptime MAX_NESTING_DEPTH = 64
comptime TYPE_NULL: UInt8 = 1
comptime TYPE_INT: UInt8 = 2
comptime TYPE_FLOATING_POINT: UInt8 = 3
comptime TYPE_BINARY: UInt8 = 4
comptime TYPE_UTF8: UInt8 = 5
comptime TYPE_LARGE_BINARY: UInt8 = 19
comptime TYPE_LARGE_UTF8: UInt8 = 20
comptime TYPE_LARGE_LIST: UInt8 = 21
comptime TYPE_BINARY_VIEW: UInt8 = 23
comptime TYPE_UTF8_VIEW: UInt8 = 24
comptime TYPE_INTERVAL: UInt8 = 11
comptime INTERVAL_UNIT_YEAR_MONTH: UInt16 = 0
comptime INTERVAL_UNIT_DAY_TIME: UInt16 = 1
comptime INTERVAL_UNIT_MONTH_DAY_NANO: UInt16 = 2

comptime TYPE_BOOL: UInt8 = 6
comptime TYPE_DECIMAL: UInt8 = 7
comptime TYPE_DATE: UInt8 = 8
comptime TYPE_TIME: UInt8 = 9
comptime TYPE_TIMESTAMP: UInt8 = 10
comptime TYPE_LIST: UInt8 = 12
comptime TYPE_STRUCT: UInt8 = 13
comptime TYPE_FIXED_SIZE_BINARY: UInt8 = 15
comptime TYPE_FIXED_SIZE_LIST: UInt8 = 16
comptime TYPE_MAP: UInt8 = 17
comptime TYPE_DURATION: UInt8 = 18
comptime PRECISION_HALF: UInt16 = 0
comptime PRECISION_SINGLE: UInt16 = 1
comptime PRECISION_DOUBLE: UInt16 = 2
comptime DATE_UNIT_DAY: UInt16 = 0
comptime DATE_UNIT_MILLISECOND: UInt16 = 1
comptime TIME_UNIT_SECOND: UInt16 = 0
comptime TIME_UNIT_MILLISECOND: UInt16 = 1


def magic() -> List[UInt8]:
    var m = List[UInt8](capacity=8)
    m.append(65)  # 'A'
    m.append(82)  # 'R'
    m.append(82)  # 'R'
    m.append(79)  # 'O'
    m.append(87)  # 'W'
    m.append(49)  # '1'
    m.append(0)
    m.append(0)
    return m^


# ---------------------------------------------------------------------------
# Internal wire format structs
# ---------------------------------------------------------------------------


@fieldwise_init
struct FieldNode(ImplicitlyCopyable, Movable):
    var length: Int64
    var null_count: Int64


@fieldwise_init
struct BodyBuffer(ImplicitlyCopyable, Movable):
    var offset: Int64
    var length: Int64


@fieldwise_init
struct Block(ImplicitlyCopyable, Movable):
    var offset: Int64
    var metadata_length: Int32
    var body_length: Int64


struct _DictLookup(Movable):
    """Result of searching a (_dtype, FieldIpcInfo) tree for a dict_id."""

    var value_type: dt.DynType
    var value_ipc_info: FieldIpcInfo

    def __init__(
        out self,
        var value_type: dt.DynType,
        var value_ipc_info: FieldIpcInfo,
    ):
        self.value_type = value_type^
        self.value_ipc_info = value_ipc_info^


struct FieldIpcInfo(Copyable, Movable):
    """IPC-only metadata shadow for a field: dict_id and value-type children.

    Mirrors the Arrow type tree but carries only IPC serialization metadata
    (dict_ids assigned during schema write/read), decoupled from the logical
    type system.  For dictionary fields, `children` holds the IPC infos of the
    VALUE TYPE's children; for non-dict fields it holds the direct type children
    (list child, struct children).  `dict_id == -1` means no DictionaryEncoding.
    """

    var dict_id: Int
    var children: List[FieldIpcInfo]

    def __init__(out self, dict_id: Int = -1):
        self.dict_id = dict_id
        self.children = List[FieldIpcInfo]()

    def __init__(out self, dict_id: Int, var children: List[FieldIpcInfo]):
        self.dict_id = dict_id
        self.children = children^

    def __init__(out self, *, copy: Self):
        self.dict_id = copy.dict_id
        self.children = copy.children.copy()

    # Explicit (empty) destructor so this self-referential struct
    # (`children: List[FieldIpcInfo]`) is Deinitable; fields are still
    # destroyed automatically after the body runs.
    def __deinit__(deinit self):
        pass

    @staticmethod
    def find(
        dtype: dt.DynType, ipc_info: FieldIpcInfo, target_id: Int
    ) raises -> Optional[_DictLookup]:
        """Search the (dtype, ipc_info) shadow tree for dict_id == target_id.

        Returns a `_DictLookup` whose `value_ipc_info` has `dict_id=-1` with
        the matched node's children, so that nested dicts inside the value type
        can still be resolved during decoding.
        """
        # Every node that is not a dictionary carries -1, which a dictionary
        # batch may name too.
        if ipc_info.dict_id == target_id and dtype.is_dictionary():
            ref d = dtype.as_dictionary()
            var vt_ipc = FieldIpcInfo(-1, ipc_info.children.copy())
            return _DictLookup(d.value_type().copy(), vt_ipc^)
        if dtype.is_dictionary():
            ref d = dtype.as_dictionary()
            var vt_ipc = FieldIpcInfo(-1, ipc_info.children.copy())
            return FieldIpcInfo.find(d.value_type().copy(), vt_ipc, target_id)
        var n = min(dtype.layout().num_children, len(ipc_info.children))
        for i in range(n):
            var found = FieldIpcInfo.find(
                dtype.child_type(i), ipc_info.children[i].copy(), target_id
            )
            if found:
                return found^
        return None

    @staticmethod
    def find_in_schema(
        fields: List[dt.Field],
        ipc_infos: List[FieldIpcInfo],
        target_id: Int,
    ) raises -> Optional[_DictLookup]:
        """Search schema fields and their IPC shadow tree for the given dict_id.
        """
        for i in range(len(fields)):
            if i < len(ipc_infos):
                var found = FieldIpcInfo.find(
                    fields[i].dtype.copy(), ipc_infos[i].copy(), target_id
                )
                if found:
                    return found^
        return None


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def padding_to(pos: Int, alignment: Int) -> Int:
    return (alignment - (pos % alignment)) % alignment


def pad_to(mut buf: List[UInt8], alignment: Int):
    var r = len(buf) % alignment
    if r != 0:
        for _ in range(alignment - r):
            buf.append(UInt8(0))
