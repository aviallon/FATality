"""Container-independent H.264 media salvage.

An MP4's ftyp/moov are single points of failure: one damaged byte in a box
size field and the container walk rejects the whole file.  The media itself
is not: mdat payloads are length-prefixed NAL chains (AVCC), and a chain
validates itself step by step — a damage point breaks the chain there and
nowhere else.  GoPro cameras additionally interleave 16-byte per-sample
'GP' descriptors (see carve.mp4.gp_descriptors) with every sample, so the
walk steps over those too.

  * :func:`media_runs` — validated NAL runs anywhere in an image (pure
    codec structure, no geometry, no container);
  * :func:`annexb`    — convert a run to Annex-B (start codes) so frames
    are playable without any timing metadata.

Runs are honest partials: pixels without time.  A run stops at the first
break — never guessed past it.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

__all__ = ['MediaRun', 'media_runs', 'annexb']

MAX_NAL = 4 << 20          # video slices can be large; anything beyond is noise
MIN_RUN_BYTES = 64 << 10   # a real media run, not a stray pair of NALs
MIN_RUN_NALS = 4


@dataclass(frozen=True)
class MediaRun:
    off: int
    end: int
    nals: int
    vcl: int                # video-coded NALs (types 1..5) seen

    @property
    def size(self) -> int:
        return self.end - self.off


def _nal_header_ok(h: int) -> bool:
    # AUD NALs are 1-2 bytes: never impose a payload minimum here
    return not (h & 0x80) and 1 <= (h & 0x1F) <= 24


def _gp_desc_size(region: bytes, p: int, n: int) -> int | None:
    """GoPro per-sample descriptor at p -> sample size, else None.

    Layout (carve.mp4.gp_descriptors): b'GP' + type + flag(0..3) +
    size(BE u32 @+4) + delta(BE u32 @+8) + ... + 0 (@+15), followed by
    `size` bytes of AVCC sample.
    """
    if p + 16 > n or region[p:p + 2] != b'GP':
        return None
    if region[p + 3] not in (0, 1, 2, 3) or region[p + 15] != 0:
        return None
    size = struct.unpack('>I', region[p + 4:p + 8])[0]
    if not (0 < size < (64 << 20)) or p + 16 + size > n:
        return None
    return size


def _walk_nals(region: bytes, p: int, end: int) -> list[tuple[int, int]]:
    """NAL (offset, payload-len) list for sample region [p, end); exact."""
    out: list[tuple[int, int]] = []
    while p + 4 <= end:
        ln = struct.unpack('>I', region[p:p + 4])[0]
        if not (1 <= ln <= MAX_NAL) or p + 4 + ln > end:
            return []
        if not _nal_header_ok(region[p + 4]):
            return []
        out.append((p, ln))
        p += 4 + ln
    return out if p == end else []


def _stats(nals: list[tuple[int, int]], region: bytes) -> tuple[int, int]:
    vcl = sum(1 for q, _ in nals if (region[q + 4] & 0x1F) <= 5)
    return len(nals), vcl


def _gp_chain_runs(region: bytes, n: int, consumed: bytearray) -> list[MediaRun]:
    """Runs over GoPro descriptor chains: consecutive NAL-valid samples merge."""
    runs: list[MediaRun] = []
    j = region.find(b'GP')
    while 0 <= j < n:
        if consumed[j]:
            j = region.find(b'GP', j + 1)
            continue
        g = _gp_desc_size(region, j, n)
        if g is None:
            j = region.find(b'GP', j + 1)
            continue
        # walk the whole chain from here, buffering consecutive good samples
        p = j
        run_start: int | None = None
        run_end = 0
        rnals = rvcl = 0
        while True:
            g = _gp_desc_size(region, p, n)
            if g is None:
                break
            nals = _walk_nals(region, p + 16, p + 16 + g)
            if nals:
                cnt, vcl = _stats(nals, region)
                if run_start is None:
                    run_start = p
                rnals += cnt
                rvcl += vcl
                run_end = p + 16 + g
            else:
                if (run_start is not None and rnals >= MIN_RUN_NALS
                        and run_end - run_start >= MIN_RUN_BYTES and rvcl):
                    runs.append(MediaRun(run_start, run_end, rnals, rvcl))
                run_start, rnals, rvcl = None, 0, 0
            p += 16 + g
        if (run_start is not None and rnals >= MIN_RUN_NALS
                and run_end - run_start >= MIN_RUN_BYTES and rvcl):
            runs.append(MediaRun(run_start, run_end, rnals, rvcl))
        for q in range(j, min(p, n)):
            consumed[q] = 1
        j = region.find(b'GP', max(p, j + 1))
    return runs


def _plain_nal_runs(region: bytes, n: int, consumed: bytearray) -> list[MediaRun]:
    """Runs over plain AVCC chains (non-GoPro mdat).

    Seeds: vectorised two-level check (this NAL header and the next both
    validate) — sparse enough to walk each candidate once.
    """
    runs: list[MediaRun] = []
    if n < 16:
        return []
    rb = np.frombuffer(region, dtype=np.uint8)
    candidates: set[int] = set()
    for phase in range(4):
        start = (4 - phase) % 4
        count = (n - start - 4) // 4
        if count <= 0:
            continue
        lens = np.frombuffer(region, dtype='>u4', count=count,
                             offset=start).astype(np.int64)
        offs = start + 4 * np.nonzero((lens >= 1) & (lens <= MAX_NAL))[0]
        if not len(offs):
            continue
        ls = lens[(lens >= 1) & (lens <= MAX_NAL)]
        h1 = rb[offs + 4]
        ok1 = (h1 & 0x80) == 0
        t1 = h1 & 0x1F
        ok1 &= (t1 >= 1) & (t1 <= 24)
        nxt = offs + 4 + ls
        ok2 = (nxt + 4 < n) & (ls >= 1)
        if not ok2.any():
            continue
        safe = np.where(ok2, nxt, 0)
        h2 = rb[safe + 4]
        ok2 &= (h2 & 0x80) == 0
        t2 = h2 & 0x1F
        ok2 &= (t2 >= 1) & (t2 <= 24)
        for o in offs[ok1 & ok2]:
            candidates.add(int(o))
    for off in sorted(candidates):
        if off >= n or consumed[off]:
            continue
        p, nals, vcl, start = off, 0, 0, off
        while p + 4 <= n:
            ln = struct.unpack('>I', region[p:p + 4])[0]
            if not (1 <= ln <= MAX_NAL) or p + 4 + ln > n:
                break
            if not _nal_header_ok(region[p + 4]):
                break
            if (region[p + 4] & 0x1F) <= 5:
                vcl += 1
            nals += 1
            p += 4 + ln
        for q in range(start, min(p, n)):
            consumed[q] = 1
        if nals >= MIN_RUN_NALS and p - start >= MIN_RUN_BYTES and vcl:
            runs.append(MediaRun(off=off, end=p, nals=nals, vcl=vcl))
    return runs


def media_runs(data: bytes, lo: int = 0, hi: int | None = None) -> list[MediaRun]:
    """All validated NAL runs in ``data[lo:hi]`` (GoPro-aware, linear)."""
    hi = len(data) if hi is None else min(hi, len(data))
    region = bytes(data[lo:hi])
    n = len(region)
    if n < 16:
        return []
    consumed = bytearray(n)
    runs = _gp_chain_runs(region, n, consumed) + _plain_nal_runs(region, n, consumed)
    runs.sort(key=lambda r: r.off)
    return [MediaRun(off=r.off + lo, end=r.end + lo, nals=r.nals, vcl=r.vcl)
            for r in runs]


def annexb(data: bytes, run: MediaRun) -> bytes:
    """AVCC run -> Annex-B (4-byte start codes): playable without a moov.

    GoPro sample descriptors are stepped over (metadata, not NALs).
    """
    out = bytearray()
    p, n = run.off, run.end
    while p + 4 <= n:
        ln = struct.unpack('>I', data[p:p + 4])[0]
        if 1 <= ln <= MAX_NAL and p + 4 + ln <= n and _nal_header_ok(data[p + 4]):
            out += b'\x00\x00\x00\x01'
            out += data[p + 4:p + 4 + ln]
            p += 4 + ln
            continue
        g = _gp_desc_size(data, p, n)
        if g is not None:
            nals = _walk_nals(data, p + 16, p + 16 + g)
            if nals:
                for q, ln2 in nals:
                    out += b'\x00\x00\x00\x01'
                    out += data[q + 4:q + 4 + ln2]
                p += 16 + g
                continue
        break
    return bytes(out)
