"""Implements a ChunkedArray class for handling pyarrow.Array-like objects."""

from marrow.buffers import Buffer, Bitmap
from marrow.dtypes import DataType, DType


struct ChunkedArray:
    """An array-like composed from a (possibly empty) collection of pyarrow.Arrays.

    [Reference](https://arrow.apache.org/docs/python/generated/pyarrow.ChunkedArray.html#pyarrow-chunkedarray).
    """

    var dtype: DataType
    var length: Int
    var chunks: List[ArrayData]

    fn _compute_length(mut self) -> None:
        """Update the length of the array from the length of its chunks."""
        var total_length = 0
        for chunk in self.chunks:
            total_length += chunk.length
        self.length = total_length

    fn __init__(out self, var dtype: DataType, var chunks: List[ArrayData]):
        self.dtype = dtype^
        self.chunks = chunks^
        self.length = 0
        self._compute_length()

    fn chunk(self, index: Int) -> ref[self.chunks] ArrayData:
        """Returns the chunk at the given index.

        Args:
          index: The desired index.

        Returns:
          A reference to the chunk at the given index.
        """
        return self.chunks[index]

    fn combine_chunks(var self, out combined: ArrayData):
        """Combines all chunks into a single array."""
        var bitmap = ArcPointer(Bitmap.alloc(self.length))
        combined = ArrayData(
            dtype=self.dtype.copy(),
            length=self.length,
            bitmap=bitmap,
            buffers=[],
            children=[],
            offset=0,
        )
        var start = 0
        while self.chunks:
            var chunk = self.chunks.pop(0)
            start += chunk^.append_to_array(combined, start)
        return combined^
