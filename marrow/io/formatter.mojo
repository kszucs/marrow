from marrow.arrays import *
from marrow.dtypes import *


struct Formatter:
    """Recursively formats and prints ArrayData and typed arrays."""

    # How many elements to print.
    var limit: Int

    fn __init__(out self, limit: Int = 3):
        self.limit = limit

    fn format[
        T: DataType, //, W: Writer
    ](self, mut writer: W, value: PrimitiveArray[T]) raises:
        """Output a PrimitiveArray to the given Writer."""
        writer.write("PrimitiveArray[")
        writer.write(materialize[value.dtype]())
        writer.write("]([")

        for i in range(value.data.length):
            if i > 0:
                writer.write(", ")
            if i >= self.limit:
                writer.write("...")
                break

            if value.is_valid(i):
                writer.write(value.unsafe_get(i))
            else:
                writer.write("NULL")
        writer.write("])")

    fn format[W: Writer](self, mut writer: W, value: ListArray) raises:
        """Output a ListArray to the given Writer."""
        writer.write("ListArray([")
        for i in range(value.data.length):
            if i > 0:
                writer.write(", ")
            if i >= self.limit:
                writer.write("...")
                break

            if value.is_valid(i):
                self.format(writer, value.unsafe_get(i))
            else:
                writer.write("NULL")
        writer.write("])")

    fn format[W: Writer](self, mut writer: W, value: StringArray) raises:
        """Output a StringArray to the given Writer."""
        writer.write("StringArray([")
        for i in range(value.data.length):
            if i > 0:
                writer.write(", ")
            if i >= self.limit:
                writer.write("...")
                break

            if value.is_valid(i):
                writer.write(value.unsafe_get(UInt(i)))
            else:
                writer.write("NULL")
        writer.write("])")

    fn format[W: Writer](self, mut writer: W, array_data: ArrayData) raises:
        """Output a dynamic ArrayData to the given writer."""
        if array_data.dtype.is_numeric():
            comptime for dtype in all_numeric_dtypes:
                if array_data.dtype == materialize[dtype]():
                    self.format(
                        writer,
                        PrimitiveArray[dtype](data=array_data.copy()),
                    )
                    return
        elif array_data.dtype.is_list():
            self.format(writer, ListArray(array_data.copy()))
            return
        elif array_data.dtype.is_struct():
            self.format(writer, StructArray(data=array_data.copy()))
            return
        elif array_data.dtype.is_string():
            self.format(writer, StringArray(data=array_data.copy()))
            return
        raise Error("Unknown dtype {} in format.".format(array_data.dtype))

    fn format[W: Writer](self, mut writer: W, value: StructArray) raises:
        """Output a StructArray to the Writer."""
        writer.write("StructArray({")
        if len(value.data.children) > 0:
            for i in range(len(value.fields)):
                if i > 0:
                    writer.write(", ")
                ref field = value.fields[i]
                writer.write("'")
                writer.write(field.name)
                writer.write("': ")
                ref field_value = value.unsafe_get(field.name)
                self.format(writer, field_value)
        writer.write("})")
