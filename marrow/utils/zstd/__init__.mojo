# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Zstandard, in Mojo: what Parquet's ZSTD pages and Arrow IPC's ZSTD buffers
hold (`zstd_compression_format.md`, RFC 8878).

| Module | What it is |
|---|---|
| `bits` | `BitReader`, `BitWriter`: the backward bitstream every entropy coder uses |
| `fse` | `Alphabet`, `Distribution`, `FseEntry`, `FseTable`, `FseEncoder`, `FseState`: Finite State Entropy |
| `huffman` | `HuffmanTable`, `HuffmanEncoder`: the literals' trees and streams |
| `headers` | `FrameHeader`, `BlockHeader`, `LiteralsHeader`, `SequencesHeader`: read and written by one type each |
| `block` | `BlockDecoder`, `RepeatOffsets`: decoding a block |
| `matcher` | `FastMatcher`, `SeqStore`: a block's matches, libzstd's fast strategy |
| `encoder` | `BlockEncoder`, `LiteralsEncoder`, `SequencesEncoder`, `Histogram`: encoding a block |
| `frame` | `Zstd`: frames |

The modules depend on each other in one direction, in the table's order
(`matcher` aside, which only `encoder` uses); every type within them
too.

Dictionaries are refused: neither format uses them. `Zstd.compress` writes
what libzstd 1.5.7's one-shot `ZSTD_compress` writes at level 1, byte for
byte: its fast match finder and window, its block splitting, its rules for
choosing each section's encoding and for repeating a Huffman code, and its
tie-breaking in building one. Parquet's writer takes no level, so level 1 is
the whole of what is needed; `test_zstd.mojo` holds the frames identical.

Against libzstd 1.5.7 on the codec corpora (`benchmarks/codecs/
codec_ab.mojo`), decoding takes 0.94-1.07x its time and compressing
0.93-1.09x, text the slowest both ways. On literal-heavy input two things
carry the decoder: decoding the four literal streams in lockstep, so their
dependency chains overlap (3.7x on the 1 MiB floats), and keeping the bit
reader's bits left-aligned, so a lookup is one shift (a further 1.23x).
"""

from .frame import Zstd
