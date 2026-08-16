"""CRC-32, the ISO-3309 / zlib / gzip checksum."""


struct Crc32(Copyable, Movable):
    """Standard CRC-32 (reflected, polynomial `0xEDB88320`) — the ISO-3309 /
    zlib / gzip checksum Parquet uses for its optional per-page checksum.
    Incremental: `update` each byte span in order (Parquet v2 pages checksum the
    levels then the compressed values), then read `value`."""

    var _state: UInt32

    def __init__(out self):
        self._state = UInt32(0xFFFFFFFF)

    def update(mut self, data: Span[UInt8, _]):
        var crc = self._state
        for i in range(len(data)):
            crc ^= UInt32(data[i])
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ UInt32(0xEDB88320)
                else:
                    crc = crc >> 1
        self._state = crc

    def value(self) -> UInt32:
        return self._state ^ UInt32(0xFFFFFFFF)

    @staticmethod
    def compute(data: Span[UInt8, _]) -> UInt32:
        """CRC-32 of a single contiguous span."""
        var c = Self()
        c.update(data)
        return c.value()
