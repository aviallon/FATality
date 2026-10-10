"""Directory mining: slots, LFNs, deleted entries, stale fragments, trees.

A *directory fragment* is any cluster (or contiguous run) holding coherent
32-byte slot structure — including clusters whose owning chain died long ago.
Fragments link into trees via '.'/'..' and directory-entry cluster claims;
when evidence conflicts, fragments stay in SEPARATE trees (conservative split:
never merge unrelated evidence).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ATTR_RO, ATTR_HIDDEN, ATTR_SYSTEM = 0x01, 0x02, 0x04
ATTR_VOLUME, ATTR_DIR, ATTR_ARCHIVE = 0x08, 0x10, 0x20
ATTR_LFN = 0x0F

VALID_ATTRS = {
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
}

NAME_CHARS = set(
    b'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.'
    b"!#$%&'()-@^_`{}~ \x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"
    b"\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"
    b"\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"
    b"\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"
    b"\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"
    b"\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf"
    b"\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef"
    b"\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff\x05"
)


def fat_date(v: int) -> str | None:
    d, m, y = v & 0x1F, (v >> 5) & 0xF, 1980 + (v >> 9)
    if not (1 <= d <= 31 and 1 <= m <= 12 and 2000 <= y <= 2035):
        return None
    return f'{y:04d}-{m:02d}-{d:02d}'


def fat_time(v: int) -> str | None:
    s, m, h = (v & 0x1F) * 2, (v >> 5) & 0x3F, v >> 11
    return f'{h:02d}:{m:02d}:{s:02d}' if h < 24 and m < 59 and s < 59 else None


def lfn_checksum(short11: bytes) -> int:
    s = 0
    for c in short11:
        s = (((s & 1) << 7) + (s >> 1) + c) & 0xFF
    return s


@dataclass
class Slot:
    off: int                      # absolute byte offset in image
    raw: bytes
    kind: str                     # 'entry' | 'lfn' | 'volume' | 'free'
    deleted: bool = False
    name: str = ''                # 8.3 or LFN text
    long_name: str | None = None
    attr: int = 0
    cluster: int = 0
    size: int = 0
    crt_date: str | None = None
    wr_date: str | None = None
    acc_date: str | None = None
    wr_time: str | None = None
    is_dir: bool = False

    @property
    def display(self) -> str:
        return self.long_name or self.name


def parse_slot(b: bytes, off: int) -> Slot | None:
    """Parse one 32-byte directory slot; None = implausible (not slot data)."""
    if len(b) < 32:
        return None
    first = b[0]
    if first == 0x00:
        return None
    attr = b[11]
    if attr not in VALID_ATTRS:
        return None
    deleted = first == 0xE5
    crt_d = fat_date(int.from_bytes(b[16:18], 'little'))
    acc_d = fat_date(int.from_bytes(b[18:20], 'little'))
    wr_d = fat_date(int.from_bytes(b[24:26], 'little'))
    if crt_d is None and acc_d is None and wr_d is None:
        return None

    if attr == ATTR_LFN:
        seq = first & 0x1F
        if not 1 <= seq <= 20:
            return None
        raw_name = b[1:11] + b[14:26] + b[28:32]
        try:
            txt = raw_name.decode('utf-16-le').rstrip('\uffff').rstrip('\x00')
        except UnicodeDecodeError:
            return None
        if not txt or any(ord(c) < 32 for c in txt):
            return None
        return Slot(off=off, raw=b, kind='lfn', deleted=deleted, name=txt,
                    wr_date=wr_d)

    base = bytes([first]) + b[1:8]
    ext = b[8:11]
    if not all(c in NAME_CHARS for c in base + ext):
        return None
    name = base.rstrip(b' ').decode('latin-1')
    e = ext.rstrip(b' ').decode('latin-1')
    full = f'{name}.{e}' if e else name
    clu = b[26] | (b[27] << 8) | (b[20] << 16) | (b[21] << 24)
    size = int.from_bytes(b[28:32], 'little')

    if attr & ATTR_VOLUME and not (attr & (ATTR_DIR | 0x27)):
        return Slot(off=off, raw=b, kind='volume', deleted=deleted, name=full,
                    attr=attr, wr_date=wr_d)
    return Slot(off=off, raw=b, kind='entry', deleted=deleted, name=full,
                attr=attr, cluster=clu, size=size, is_dir=bool(attr & ATTR_DIR),
                crt_date=crt_d, acc_date=acc_d, wr_date=wr_d,
                wr_time=fat_time(int.from_bytes(b[22:24], 'little')))


def merge_lfns(slots: list[Slot]) -> list[Slot]:
    """Attach checksum-validated LFN runs to the short entry that follows."""
    out: list[Slot] = []
    pending: list[Slot] = []
    for s in slots:
        if s.kind == 'lfn':
            pending.append(s)
            continue
        if s.kind == 'entry':
            s.long_name = None
            if pending:
                cs = lfn_checksum(s.raw[:11])
                good = [p for p in pending
                        if p.raw[13] == cs and p.deleted == s.deleted]
                if good:
                    good.sort(key=lambda p: p.raw[0] & 0x1F)
                    s.long_name = ''.join(p.name for p in good)
            pending = []
        out.append(s)
    return out


@dataclass
class DirFragment:
    """A cluster-sized directory snapshot candidate."""
    cluster: int
    off: int
    slots: list[Slot] = field(default_factory=list)

    @property
    def entries(self) -> list[Slot]:
        return [s for s in self.slots if s.kind == 'entry']

    @property
    def self_entry(self) -> Slot | None:
        for s in self.entries:
            if s.name == '.' and not s.deleted:
                return s
        return None

    @property
    def parent_entry(self) -> Slot | None:
        for s in self.entries:
            if s.name == '..' and not s.deleted:
                return s
        return None

    @property
    def names(self) -> list[str]:
        return [s.display for s in self.entries if s.name not in ('.', '..')]


def is_fragment(slots: list[Slot], min_entries: int = 2) -> bool:
    """Coherent directory snapshot.

    The key discriminator against random binary data: real directories have
    runs of ADJACENT valid 32-byte slots (LFN runs, . / .. pairs, entry
    sequences); chance hits in media payloads are scattered singletons.
    """
    entries = [s for s in slots if s.kind == 'entry']
    if len(entries) < min_entries:
        return False
    # require at least one adjacent pair (two valid slots exactly 32B apart)
    by_off = sorted((s.off, s) for s in slots if s.kind in ('entry', 'lfn'))
    adjacent = any(b[0] - a[0] == 32 for a, b in zip(by_off, by_off[1:]))
    if not adjacent:
        return False
    dots = sum(1 for s in entries if s.name in ('.', '..'))
    if dots == 1:                     # half of a . / .. pair: malformed
        return False
    # adjacency is the main discriminator; a couple of plausible refs is
    # enough (stale clusters mix real entries with clobber junk by nature)
    plausible = sum(1 for s in entries if s.cluster == 0 or s.size == 0
                    or s.name in ('.', '..') or s.cluster >= 2)
    has_self = any(s.name == '.' for s in entries)
    return has_self or plausible >= min(2, len(entries))


@dataclass
class Tree:
    """A set of fragments believed to describe one directory hierarchy."""
    root: DirFragment | None
    nodes: dict[int, DirFragment]     # cluster -> fragment
    label: str

    def paths(self) -> dict[str, list[Slot]]:
        """Best-effort path -> entries mapping within this tree."""
        out: dict[str, list[Slot]] = {}
        parent_of: dict[int, int | None] = {}
        for clu, frag in self.nodes.items():
            pe = frag.parent_entry
            parent_of[clu] = pe.cluster if pe and pe.cluster else None
        # also honour entry claims: dir entry pointing at fragment cluster
        claims: dict[int, tuple[int, str]] = {}
        for clu, frag in self.nodes.items():
            for s in frag.entries:
                if s.is_dir and s.cluster in self.nodes and s.name not in ('.', '..'):
                    claims[s.cluster] = (clu, s.display)

        def climb(c: int) -> list[tuple[int, str]]:
            chain = []
            seen = set()
            cur = c
            while cur is not None and cur not in seen:
                seen.add(cur)
                if cur in claims:
                    p, nm = claims[cur]
                    chain.append((p, nm))
                    cur = p
                else:
                    pu = parent_of.get(cur)
                    if pu is not None and pu in self.nodes:
                        chain.append((pu, f'clu{cur}'))
                        cur = pu
                    else:
                        break
            return list(reversed(chain))

        for clu, frag in self.nodes.items():
            sub = '/'.join(n for _, n in climb(clu))
            path = '/' + sub if sub else '/'
            out[path] = frag.entries
        return out
