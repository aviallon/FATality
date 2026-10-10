"""Vectorised whole-image scanning (numpy): directory slots and signatures.

These are the proven hot paths: slot-aligned scans reach several hundred MB/s
by reshaping chunks to (-1, 32) and applying boolean masks over lookup tables.
"""
from __future__ import annotations

import numpy as np

from .dirtree import ATTR_LFN, VALID_ATTRS, NAME_CHARS, parse_slot, Slot
from .geometry import Geometry


def _char_table() -> np.ndarray:
    t = np.zeros(256, dtype=bool)
    for c in NAME_CHARS:
        t[c] = True
    return t


_CHAR = _char_table()
_ATTR = np.zeros(256, dtype=bool)
for a in VALID_ATTRS:
    _ATTR[a] = True


def scan_slots(f, start: int, end: int, chunk: int = 64 << 20,
               min_plausible: int = 1) -> list[Slot]:
    """All plausible 32-byte directory slots in [start, end), in order."""
    out: list[Slot] = []
    f.seek(start)
    consumed = 0
    total = end - start
    while consumed < total:
        buf = f.read(min(chunk, total - consumed))
        if not buf:
            break
        base = start + consumed
        n = len(buf) // 32 * 32
        arr = np.frombuffer(buf[:n], dtype=np.uint8).reshape(-1, 32)
        mask = (_CHAR[arr[:, :11]].all(axis=1)
                & _ATTR[arr[:, 11]]
                & ~np.all(arr == 0, axis=1))
        # name charset check for LFN rows is different: permit any utf16-ish
        lfn_rows = arr[:, 11] == ATTR_LFN
        mask |= (lfn_rows & _ATTR[arr[:, 11]])
        for idx in np.nonzero(mask)[0]:
            off = base + int(idx) * 32
            s = parse_slot(bytes(arr[idx]), off)
            if s is not None:
                out.append(s)
        consumed += len(buf)
    return out


def scan_signatures(f, start: int, end: int, sigs: dict[str, bytes],
                    limit_per_sig: int = 1_000_000, chunk: int = 64 << 20) -> dict[str, list[int]]:
    """Byte-exact signature index with chunk-overlap handling."""
    hits: dict[str, list[int]] = {k: [] for k in sigs}
    maxlen = max(len(s) for s in sigs.values())
    f.seek(start)
    consumed = 0
    carry = b''
    total = end - start
    while consumed < total:
        buf = f.read(min(chunk, total - consumed))
        if not buf:
            break
        data = carry + buf
        base = start + consumed - len(carry)
        for name, sig in sigs.items():
            if len(hits[name]) >= limit_per_sig:
                continue
            pos = 0
            while True:
                i = data.find(sig, pos)
                if i < 0:
                    break
                hits[name].append(base + i)
                pos = i + 1
                if len(hits[name]) >= limit_per_sig:
                    break
        consumed += len(buf)
        carry = data[-(maxlen - 1):] if maxlen > 1 else b''
    return hits


def scan_fragments(f, gz: Geometry, slots: list[Slot], min_entries: int = 2):
    """Group slot hits per cluster; return directory-fragment candidates."""
    from .dirtree import DirFragment, is_fragment, merge_lfns
    by_cluster: dict[int, list[Slot]] = {}
    for s in slots:
        if not gz.in_data(s.off):
            continue
        clu = gz.cluster_of(s.off)
        by_cluster.setdefault(clu, []).append(s)
    frags = []
    for clu, ss in sorted(by_cluster.items()):
        ss.sort(key=lambda s: s.off)
        # adjacency test on RAW slots (merge_lfns consumes LFN rows, which
        # would leave false gaps between the entries they annotate)
        if is_fragment(ss, min_entries):
            frags.append(DirFragment(cluster=clu, off=gz.cluster_off(clu),
                                     slots=merge_lfns(ss)))
    return frags
