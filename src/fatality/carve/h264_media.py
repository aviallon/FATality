"""Container-independent media salvage (H.264/AVC and H.265/HEVC).

An MP4's ftyp/moov are single points of failure: one damaged byte in a box
size field and the container walk rejects the whole file.  The media itself
is not: mdat payloads are length-prefixed NAL chains (AVCC), and a chain
validates itself step by step — a damage point breaks the chain there and
nowhere else.

Two lanes, deliberately both kept:

  * GoPro lane (specialised): 16-byte per-sample 'GP' descriptors mark every
    sample — a content-level FAT that works even in a destroyed container.
  * Plain lane (general): AVCC/HEVC chains anywhere, e.g. smartphone and
    camera mdat.

Real-world mdat interleaves video with non-NAL chunks (AAC audio, telemetry).
A chain break is therefore ambiguous: it is either damage (stop) or bounded
interleave (continue after it).  The walker resolves this by bounded
resync: it re-validates a chain start within GAP_MAX and records every
bridged span in :attr:`MediaRun.gaps`.  It never emits bytes from a gap and
never walks unvalidated structure — damage with no valid chain ahead ends
the run.

  * :func:`media_runs` — validated NAL runs anywhere in an image;
  * :func:`annexb`    — run -> Annex-B (start codes), gaps skipped, so
    frames are playable without any container or timing metadata.
"""
from __future__ import annotations

import struct
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np

__all__ = ['MediaRun', 'media_runs', 'annexb']

MAX_NAL = 4 << 20          # video slices can be large; anything beyond is noise
MIN_RUN_BYTES = 64 << 10   # a real media run, not a stray pair of NALs
MIN_RUN_NALS = 4
GAP_MAX = 32 << 10         # bounded resync window for interleaved non-NAL
                           # chunks (AAC/telemetry are small); damage beyond
                           # it ends the run


@dataclass(frozen=True)
class MediaRun:
    off: int
    end: int
    nals: int
    vcl: int                # video-coded NALs seen
    gaps: tuple[tuple[int, int], ...] = ()   # bridged non-NAL spans (evidence)

    @property
    def size(self) -> int:
        return self.end - self.off

    @property
    def media_bytes(self) -> int:
        return self.size - sum(e - s for s, e in self.gaps)


def _nal_header_ok(h: int) -> bool:
    # AUD NALs are 1-2 bytes: never impose a payload minimum here
    return not (h & 0x80) and 1 <= (h & 0x1F) <= 24


def _hevc_header_ok(b0: int, b1: int) -> bool:
    """H.265 NAL header (2 bytes): forbidden_zero | type(6) | layer(6) | tid."""
    ntype = (b0 >> 1) & 0x3F
    return not (b0 & 0x80) and ntype <= 40 and not (b1 & 0x80)


def _step_ok(region: bytes, p: int, end: int) -> int | None:
    """Length-prefixed NAL at p validates (AVC or HEVC header) -> payload len."""
    if p + 5 >= end:
        return None
    ln = struct.unpack('>I', region[p:p + 4])[0]
    if not (1 <= ln <= MAX_NAL) or p + 4 + ln > end:
        return None
    if not (_nal_header_ok(region[p + 4])
            or _hevc_header_ok(region[p + 4], region[p + 5])):
        return None
    return ln


def _is_vcl(region: bytes, q: int) -> bool:
    h = region[q + 4]
    if _nal_header_ok(h):
        return (h & 0x1F) <= 5
    return ((h >> 1) & 0x3F) <= 31          # HEVC VCL types 0..31


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
    """Exact NAL (offset, payload-len) walk over one sample region."""
    out: list[tuple[int, int]] = []
    while p + 4 <= end:
        ln = _step_ok(region, p, end)
        if ln is None:
            return []
        out.append((p, ln))
        p += 4 + ln
    return out if p == end else []


def _candidates(region: bytes, n: int) -> list[int]:
    """Offsets whose NAL step validates AND whose successor does too.

    Two-level seeding: sparse enough that walking each candidate once is
    linear in practice, while a random length+header coincidence almost
    never chains twice.
    """
    cands: set[int] = set()
    rb = np.frombuffer(region, dtype=np.uint8)
    for phase in range(4):
        start = (4 - phase) % 4
        count = (n - start - 4) // 4
        if count <= 0:
            continue
        lens = np.frombuffer(region, dtype='>u4', count=count,
                             offset=start).astype(np.int64)
        mask = (lens >= 1) & (lens <= MAX_NAL)
        idx = np.nonzero(mask)[0]
        if not len(idx):
            continue
        offs = start + 4 * idx
        ls = lens[idx]
        h1 = rb[offs + 4]
        h2b = rb[np.minimum(offs + 5, n - 1)]
        ok1 = ((h1 & 0x80) == 0) & (((h1 & 0x1F) >= 1) & ((h1 & 0x1F) <= 24)
                                    | ((((h1 >> 1) & 0x3F) <= 40)
                                       & ((h2b & 0x80) == 0)))
        nxt = offs + 4 + ls
        ok2 = (nxt + 5 < n)
        safe = np.where(ok2, nxt, 0)
        nh1 = rb[safe + 4]
        nh2 = rb[np.minimum(safe + 5, n - 1)]
        ok2 &= ((nh1 & 0x80) == 0) & (((nh1 & 0x1F) >= 1) & ((nh1 & 0x1F) <= 24)
                                      | ((((nh1 >> 1) & 0x3F) <= 40)
                                         & ((nh2 & 0x80) == 0)))
        for o in offs[ok1 & ok2]:
            cands.add(int(o))
    return sorted(cands)


def _walk_bridged(region: bytes, start: int, n: int,
                  cands: list[int]) -> tuple[int, int, int, list[tuple[int, int]]]:
    """Walk NAL steps from `start`; on a break, bounded resync via cands.

    Returns (end, nals, vcl, gaps).  Bytes inside gaps are never emitted —
    they are recorded so the evidence shows exactly what was skipped.
    """
    p = start
    nals = vcl = 0
    gaps: list[tuple[int, int]] = []
    while p + 4 <= n:
        ln = _step_ok(region, p, n)
        if ln is not None:
            if _is_vcl(region, p):
                vcl += 1
            nals += 1
            p += 4 + ln
            continue
        i = bisect_left(cands, p + 1)
        q = cands[i] if i < len(cands) else n
        if q >= n or q - p > GAP_MAX:
            break
        gaps.append((p, q))
        p = q
    return p, nals, vcl, gaps


def _emit(runs: list[MediaRun], run_start: int | None, run_end: int,
          rnals: int, rvcl: int, gaps: list[tuple[int, int]],
          lo: int = 0) -> None:
    if run_start is None:
        return
    span = run_end - run_start
    media = span - sum(e - s for s, e in gaps)
    if rnals >= MIN_RUN_NALS and media >= MIN_RUN_BYTES and rvcl:
        runs.append(MediaRun(off=run_start + lo, end=run_end + lo,
                             nals=rnals, vcl=rvcl,
                             gaps=tuple((s + lo, e + lo) for s, e in gaps)))


def _gp_chain_runs(region: bytes, n: int, consumed: bytearray) -> list[MediaRun]:
    """GoPro lane: runs over descriptor chains.

    A non-NAL sample (AAC audio, telemetry) is a bounded gap — bridged when
    the next descriptor's sample validates within GAP_MAX; a damaged sample
    (large, no valid chain ahead) ends the run.
    """
    runs: list[MediaRun] = []
    j = region.find(b'GP')
    while 0 <= j < n:
        if consumed[j] or _gp_desc_size(region, j, n) is None:
            j = region.find(b'GP', j + 1)
            continue
        p = j
        run_start: int | None = None
        rnals = rvcl = 0
        gaps: list[tuple[int, int]] = []
        gap_start: int | None = None
        while True:
            g = _gp_desc_size(region, p, n)
            if g is None:
                break
            nals = _walk_nals(region, p + 16, p + 16 + g)
            if nals:
                if gap_start is not None:
                    gaps.append((gap_start, p))
                    gap_start = None
                if run_start is None:
                    run_start = p
                cnt = len(nals)
                rvcl += sum(1 for q, _ in nals if _is_vcl(region, q))
                rnals += cnt
            else:
                if run_start is not None:
                    if gap_start is None:
                        gap_start = p
                    if p + 16 + g - gap_start > GAP_MAX:
                        _emit(runs, run_start, gap_start, rnals, rvcl, gaps)
                        run_start, rnals, rvcl = None, 0, 0
                        gaps = []
                        gap_start = None
            p += 16 + g
        if gap_start is not None and (p - gap_start > GAP_MAX or run_start is None):
            if run_start is not None:
                _emit(runs, run_start, gap_start, rnals, rvcl, gaps)
            run_start, rnals, rvcl = None, 0, 0
            gaps = []
        _emit(runs, run_start, p, rnals, rvcl, gaps)
        for q in range(j, min(p, n)):
            consumed[q] = 1
        j = region.find(b'GP', max(p, j + 1))
    return runs


def _plain_nal_runs(region: bytes, n: int, consumed: bytearray) -> list[MediaRun]:
    """General lane: AVCC/HEVC runs with bounded gap resync (audio interleave)."""
    runs: list[MediaRun] = []
    if n < 16:
        return []
    cands = _candidates(region, n)
    for off in cands:
        if off >= n or consumed[off]:
            continue
        end, nals, vcl, gaps = _walk_bridged(region, off, n, cands)
        for q in range(off, min(end, n)):
            consumed[q] = 1
        _emit(runs, off, end, nals, vcl, gaps)
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
    return [MediaRun(off=r.off + lo, end=r.end + lo, nals=r.nals, vcl=r.vcl,
                     gaps=tuple((s + lo, e + lo) for s, e in r.gaps))
            for r in runs]


def annexb(data: bytes, run: MediaRun) -> bytes:
    """Run -> Annex-B (4-byte start codes); gap bytes are never emitted."""
    gapset = run.gaps
    out = bytearray()
    p, n = run.off, run.end
    gi = 0
    while p + 4 <= n:
        while gi < len(gapset) and gapset[gi][1] <= p:
            gi += 1
        if gi < len(gapset) and gapset[gi][0] <= p < gapset[gi][1]:
            p = gapset[gi][1]        # skip recorded non-NAL span
            continue
        ln = _step_ok(data, p, n)
        if ln is None:
            break
        out += b'\x00\x00\x00\x01'
        out += data[p + 4:p + 4 + ln]
        p += 4 + ln
    return bytes(out)
