"""FAT table parsing and chain reconstruction when the FAT is gone.

Reconstruction ladder (most trusted first):
  1. surviving FAT copies (parsing ALL copies, merging on agreement)
  2. directory-entry evidence + contiguity (camera/sequential write patterns)
  3. content anchors (e.g. MP4 sample tables act as a content-level FAT)
Every reconstructed chain carries its evidence in Chain.evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import schema
from .geometry import Geometry


@dataclass
class Chain:
    start: int
    clusters: list[int]
    evidence: str                  # 'fat', 'fat+contig', 'contig', 'content-anchor'
    truncated: bool = False        # did not reach a clean end
    notes: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.clusters)


class FatTable:
    """One FAT copy, parsed per FAT bit width."""

    def __init__(self, gz: Geometry, data: bytes, index: int):
        self.gz = gz
        self.index = index
        self.entries = self._parse(data)

    def _parse(self, data: bytes):
        g = self.gz
        if g.fat_bits == 32:
            import struct
            n = len(data) // 4
            return list(struct.unpack(f'<{n}I', data[:n * 4]))
        import array
        if g.fat_bits == 16:
            a = array.array('H')
            a.frombytes(data[:len(data) // 2 * 2])
            return list(a)
        # FAT12: packed 12-bit entries (schema: 1.5 bytes each)
        out = []
        for i in range(0, len(data) - 1, 3):
            w = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16)
            out.append(w & 0xFFF)
            out.append((w >> 12) & 0xFFF)
        return out

    def mask(self, v: int) -> int:
        return schema.fat_value(v, self.gz.fat_bits)

    def is_eoc(self, v: int) -> bool:
        return schema.fat_is_eoc(self.mask(v), self.gz.fat_bits)

    def is_bad(self, v: int) -> bool:
        return schema.fat_is_bad(self.mask(v), self.gz.fat_bits)

    def get(self, clu: int) -> int | None:
        return self.entries[clu] if clu < len(self.entries) else None

    def walk(self, start: int, limit: int | None = None) -> Chain:
        out, seen = [], set()
        c = start
        lim = limit or self.gz.cluster_count + 2
        truncated = False
        while schema.FAT_RESERVED_FIRST <= c < lim:
            if c in seen:
                truncated = True
                break
            seen.add(c)
            out.append(c)
            v = self.get(c)
            if v is None or self.is_bad(v):
                truncated = True
                break
            if self.is_eoc(v):
                return Chain(start, out, 'fat')
            c = self.mask(v)
        return Chain(start, out, 'fat', truncated=truncated)


def load_fats(f, gz: Geometry) -> list[FatTable]:
    out = []
    for i, off in enumerate(gz.fat_off):
        f.seek(off)
        out.append(FatTable(gz, f.read(gz.fat_sects * gz.bps), i))
    return out


def merge_fats(tables: list[FatTable]) -> dict[int, int]:
    """Consensus FAT: keep entries where copies agree; majority otherwise."""
    votes: dict[int, dict[int, int]] = {}
    for t in tables:
        for i, v in enumerate(t.entries):
            m = t.mask(v)
            if m in (0, 1) or t.is_eoc(v) or t.is_bad(v):
                continue
            votes.setdefault(i, {}).setdefault(m, 0)
            votes[i][m] += 1
    return {i: max(d, key=lambda k: d[k]) for i, d in votes.items() if max(d.values()) >= 2}


def contiguous_chain(gz: Geometry, start: int, size: int,
                     cluster_bytes: int | None = None) -> Chain:
    """The sequential-allocation hypothesis: start..start+n-1."""
    cb = cluster_bytes or gz.cluster_bytes
    n = max(1, (size + cb - 1) // cb)
    end = min(start + n, gz.cluster_count + 2)
    return Chain(start, list(range(start, end)), 'contig',
                 truncated=end < start + n,
                 notes=[f'assumes sequential allocation of {n} clusters'])


def reconstruct(gz: Geometry, fats: list[FatTable], start: int, size: int) -> Chain:
    """Best chain for a file: FAT when it survives and agrees with size,
    otherwise the contiguity hypothesis with its evidence recorded."""
    cb = gz.cluster_bytes
    want = max(1, (size + cb - 1) // cb)
    consensus = merge_fats(fats)
    if start in consensus:
        ch = None
        for t in fats:
            ch = t.walk(start)
            if ch.length >= want and not ch.truncated:
                ch.evidence = 'fat'
                ch.notes.append(f'FAT copy {t.index} yields {ch.length} clusters')
                return ch
    ch = contiguous_chain(gz, start, size)
    if start in consensus:
        ch.evidence = 'fat+contig'
        ch.notes.append(f'FAT disagree/unusable for size={size}; contiguity fallback')
    return ch
