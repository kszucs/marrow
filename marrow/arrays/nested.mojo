from std.memory import ArcPointer

from ..buffers import Buffer, Bitmap


struct ListArray(Array):
    var data: ArrayData
    var capacity: Int

    fn __init__(out self, var data: ArrayData) raises:
        if not data.dtype.is_list():
            raise Error(
                "Unexpected dtype {} instead of 'list'".format(data.dtype)
            )
        elif len(data.buffers) != 1:
            raise Error("ListArray requires exactly one buffer")
        elif len(data.children) != 1:
            raise Error("ListArray requires exactly one child array")

        self.capacity = data.length
        self.data = data^

    fn bitmap(self) -> ArcPointer[Bitmap]:
        return self.data.bitmap

    fn offsets(self) -> ArcPointer[Buffer]:
        return self.data.buffers[0]

    fn values(self) -> ArcPointer[ArrayData]:
        return self.data.children[0]

    fn __init__[T: Array](out self, var values: T, capacity: Int = 1):
        """Initialize a list with the given values.

        Default capacity is at least 1 to accomodate the values.

        Args:
            values: Array to use as the first element in the ListArray.
            capacity: The capacity of the ListArray.
        """
        var values_data = values^.take_data()
        var list_dtype = list_(values_data.dtype.copy())

        var bitmap = Bitmap.alloc(capacity)
        bitmap.unsafe_set(0, True)
        var offsets = Buffer.alloc[DType.uint32](capacity + 1)
        offsets.unsafe_set[DType.uint32](0, 0)
        offsets.unsafe_set[DType.uint32](1, UInt32(values_data.length))

        self.capacity = capacity
        self.data = ArrayData(
            dtype=list_dtype^,
            length=1,
            bitmap=ArcPointer(bitmap^),
            buffers=[ArcPointer(offsets^)],
            children=[ArcPointer(values_data^)],
            offset=0,
        )

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data^
        self.capacity = take.capacity

    fn __len__(self) -> Int:
        return self.data.length

    fn as_data(self) -> UnsafePointer[ArrayData, ImmutAnyOrigin]:
        return UnsafePointer(to=self.data)

    fn take_data(deinit self) -> ArrayData:
        return self.data^

    fn is_valid(self, index: Int) -> Bool:
        return self.bitmap()[].unsafe_get(index)

    fn unsafe_append(mut self, is_valid: Bool):
        self.bitmap()[].unsafe_set(self.data.length, is_valid)
        self.offsets()[].unsafe_set[DType.uint32](
            self.data.length + 1, self.values()[].length
        )
        self.data.length += 1

    fn unsafe_get(self, index: Int, out array_data: ArrayData) raises:
        """Access the value at a given index in the list array.

        Use an out argument to allow the caller to re-use memory while iterating over a pyarrow structure.
        """
        var start = Int(
            self.offsets()[].unsafe_get[DType.int32](self.data.offset + index)
        )
        var end = Int(
            self.offsets()[].unsafe_get[DType.int32](
                self.data.offset + index + 1
            )
        )
        ref first_child = self.data.children[0][]
        return ArrayData(
            dtype=first_child.dtype.copy(),
            bitmap=first_child.bitmap,
            buffers=first_child.buffers.copy(),
            offset=start,
            length=end - start,
            children=first_child.children.copy(),
        )

    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this ListArray to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("ListArray(")
        writer.write("length=")
        writer.write(self.data.length)
        writer.write(")")


struct StructArray(Array):
    var data: ArrayData
    var fields: List[Field]
    var capacity: Int

    fn __init__(
        out self,
        var fields: List[Field],
        capacity: Int = 0,
    ):
        var bitmap = Bitmap.alloc(capacity)
        bitmap.unsafe_range_set(0, capacity, True)

        var struct_dtype = struct_(fields)

        self.capacity = capacity
        self.fields = fields^
        self.data = ArrayData(
            dtype=struct_dtype^,
            length=0,
            bitmap=ArcPointer(bitmap^),
            buffers=[],
            children=[],
            offset=0,
        )

    fn __init__(out self, *, var data: ArrayData):
        self.fields = data.dtype.fields.copy()
        self.capacity = data.length
        self.data = data^

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data^
        self.fields = take.fields^
        self.capacity = take.capacity

    fn __len__(self) -> Int:
        return self.data.length

    fn take_data(deinit self) -> ArrayData:
        return self.data^

    fn as_data(self) -> UnsafePointer[ArrayData, ImmutAnyOrigin]:
        return UnsafePointer(to=self.data)

    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this StructArray to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("StructArray(")
        writer.write("length=")
        writer.write(self.data.length)
        writer.write(")")

    fn _index_for_field_name(self, name: StringSlice) raises -> Int:
        for idx, ref field in enumerate(self.data.dtype.fields):
            if field.name == name:
                return idx

        raise Error("Field {} does not exist in this StructArray.".format(name))

    fn unsafe_get(
        self, name: StringSlice
    ) raises -> ref[self.data.children[0]] ArrayData:
        """Access the field with the given name in the struct."""
        return self.data.children[self._index_for_field_name(name)][]
