"""Format-agnostic primitives shared across marrow.

Each submodule is a self-contained block that depends on nothing in marrow — no
arrays, no dtypes, no execution context — so it can be read, tested and lifted
out on its own:

| module | what |
|---|---|
| `byteorder` | `LittleEndian` — byte, bit and LEB128-varint reads/writes |
| `datetime` | `CivilDate`, `Epoch`, `floor_div` — proleptic-Gregorian arithmetic |
| `checksum` | `Crc32` — the ISO-3309 / zlib / gzip checksum |
| `hashing` | `Hasher` plus `RapidHash64`, `XxHash64`, `AHash64` |
| `compression` | `CompressionLibs` — `dlopen`ed zstd / snappy / lz4 / zlib / brotli |
| `testing` | `TestSuite` / `BenchSuite` / `Benchmark` — the harness pytest drives |

The names are re-exported here, so `from ..utils import LittleEndian` is the
import everywhere; the submodule split is about where the code *lives*, not
about making callers spell out a path.

**`testing` is the one exception and is deliberately not re-exported.** Every
module in the tree imports `marrow.utils`, and none of them should pull
`std.benchmark` in behind it — test and bench files import
`..utils.testing` explicitly.

This replaced a single 312-line `marrow/utils.mojo` that was four unrelated
things, and a second `marrow/parquet/utils.mojo` that held the codec bindings —
two modules named `utils`, neither describing its contents. Device capability
(`GPU_ENABLED`, `has_accelerator_support`) went the other way, to
`marrow.execution`, where `ExecContext` already owns every question about
whether there is a device.
"""

from .byteorder import LittleEndian
from .checksum import Crc32
from .compression import CompressionLibs
from .datetime import CivilDate, Epoch, floor_div
from .hashing import AHash64, Hasher, RapidHash64, RapidSecret, XxHash64
