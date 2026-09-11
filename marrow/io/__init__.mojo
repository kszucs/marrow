"""Storage: where bytes come from and go, decoupled from what they mean.

`ByteSource` and `ByteSink` are the two seams every format reads and writes
through, so a reader or writer is written once and gains every backend at once.
The backends are a memory map, a local file, an in-memory buffer, and — through
`marrow.io.opendal` — any of the object stores OpenDAL speaks.

Explicit re-exports, never `import *`: a wildcard re-exports whatever the
submodule itself imported, so a name would resolve or not depending on which
file you entered through.
"""

from .opendal import OpenDalStore, OpenDalWriter, OpenDalSource
from .dispatch import DynSink, DynSource
from .core import (
    FOOTER_READ_SIZE,
    BufferedSink,
    ByteSink,
    ByteSource,
    Fetched,
    require_range,
)
from .local import BufferSource, FileSink, MemorySink
from ..utils.uri import StorageOptions, Uri
