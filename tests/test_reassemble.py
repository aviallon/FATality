"""Tests for fragmented-MP4 reassembly.

Two layers:
  * synthetic images (fast, fully controlled): prove the engine rebuilds a
    fragmented GoPro-style MP4 BYTE-EXACTLY, including the extent-switch
    tiers (structural validation, flush lattice) and the moov-less path;
  * the gold test on the real FAT-zeroed SD card image: GOPR3655.MP4 must
    reassemble with zero missing bytes, every sample contract clean, the
    documented duration, and a fully clean decoder run (the content claim
    that structural checks alone cannot make).  Skipped when the image is
    absent.
"""
from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from fatality.carve.mp4_reassemble import (
    FileModel, _cluster_grid, nal_walk_ok, devc_walk_ok, parse_moov,
    reconcile_maps, reassemble, walk_atoms,
)
from fatality.geometry import Geometry

# --------------------------------------------------------------- helpers --

def _atom(kind: bytes, payload: bytes) -> bytes:
    return struct.pack('>I', 8 + len(payload)) + kind + payload


def _hdlr(handler: bytes) -> bytes:
    return _atom(b'hdlr', b'\x00' * 8 + handler + b'\x00' * 13)


def _stbl(codec: bytes, chunk_offs: list[int], sizes: list[int],
          delta: int = 1001, entry_extra: int = 20) -> bytes:
    entry = struct.pack('>I', 8 + entry_extra) + codec + b'\x00' * entry_extra
    stsd = _atom(b'stsd', b'\x00' * 4 + struct.pack('>I', 1) + entry)
    stts = _atom(b'stts', b'\x00' * 4 + struct.pack('>I', 1)
                 + struct.pack('>II', len(sizes), delta))
    stsc = _atom(b'stsc', b'\x00' * 4 + struct.pack('>I', 1)
                 + struct.pack('>III', 1, 1, 1))
    stsz = _atom(b'stsz', b'\x00' * 4 + struct.pack('>II', 0, len(sizes))
                 + b''.join(struct.pack('>I', s) for s in sizes))
    stco = _atom(b'stco', b'\x00' * 4 + struct.pack('>I', len(chunk_offs))
                 + b''.join(struct.pack('>I', c) for c in chunk_offs))
    return _atom(b'stbl', stsd + stts + stsc + stsz + stco)


def _trak(track_id: int, handler: bytes, codec: bytes,
          chunk_offs: list[int], sizes: list[int], delta: int = 1001,
          entry_extra: int = 20) -> bytes:
    tkhd = _atom(b'tkhd', b'\x00\x00\x00\x07' + b'\x00' * 8
                 + struct.pack('>III', track_id, 0, 0)
                 + b'\x00' * 8 + b'\x00' * 8 + b'\x00' * 36
                 + struct.pack('>II', 1920 << 16, 1440 << 16))
    mdhd = _atom(b'mdhd', b'\x00' * 4 + b'\x00' * 8
                 + struct.pack('>II', 60000, 60000) + b'\x00' * 4)
    mdia = _atom(b'mdia', mdhd + _hdlr(handler)
                 + _atom(b'minf', _stbl(codec, chunk_offs, sizes, delta, entry_extra)))
    return _atom(b'trak', tkhd + mdia)


def _moov(samples: list[tuple[int, int]], desc_offs: list[int],
          creation: int = 3_800_000_000) -> bytes:
    """moov with a meta/fdsc track (the descriptors) + one vide/avc1 track."""
    mvhd = _atom(b'mvhd', b'\x00' * 4
                 + struct.pack('>IIII', creation, creation, 60000, 2 * 60000)
                 + struct.pack('>IH', 0x00010000, 0x0100)
                 + b'\x00' * 70 + struct.pack('>I', 3))
    fdsc = _trak(2, b'meta', b'fdsc', desc_offs, [16] * len(desc_offs), delta=1)
    vide = _trak(1, b'vide', b'avc1', [o for o, _ in samples],
                 [s for _, s in samples], delta=1001, entry_extra=78)
    return _atom(b'moov', mvhd + fdsc + vide)


def _fake_sample(size: int, nals: list[int], seed: int) -> bytes:
    """AVCC sample: `nals` NAL units of the given payload lengths."""
    assert sum(4 + n for n in nals) == size
    out = b''
    k = 0
    for n in nals:
        payload = bytes(((seed * 37 + k + j * 17) ^ (j >> 8)) & 0xFF
                        for j in range(n))
        out += struct.pack('>I', n) + payload
        k += 1
    return out


def _descriptor(size: int, delta: int = 1001) -> bytes:
    return b'GP' + b'\x00\x03' + struct.pack('>II', size, delta) + b'\x00' * 4


def _build_file(sample_specs: list[tuple[list[int], int]]) -> tuple[bytes, list, list]:
    """ftyp + mdat(descriptor+samples) + moov.  sample_specs: (nal sizes, seed)."""
    sizes = [sum(4 + n for n in nals) for nals, _ in sample_specs]
    desc_offs = []
    off = 20 + 8
    body = b''
    for i, (nals, seed) in enumerate(sample_specs):
        desc_offs.append(off)
        body += _descriptor(sizes[i]) + _fake_sample(sizes[i], nals, seed)
        off += 16 + sizes[i]
    mdat = _atom(b'mdat', body)
    samples = [(d + 16, s) for d, s in zip(desc_offs, sizes)]
    ftyp = _atom(b'ftyp', b'mp41' + b'\x00' * 4 + b'mp41')
    moov_off = 20 + len(mdat)
    moov = _moov(samples, desc_offs)
    return ftyp + mdat + moov, samples, desc_offs


def _synthetic_geometry(size: int) -> Geometry:
    return Geometry(source='synthetic', part_off=0, part_size=size,
                    bps=512, spc=1, reserved=1, num_fats=1, fat_sects=0)


def _scatter(file_bytes: bytes, extents: list[tuple[int, int, int]], image_size: int,
             noise_seed: int = 99) -> bytes:
    """Place file ranges (file_start, file_end, image_off) into a noisy image."""
    img = bytearray(bytes(((i * 1103515245 + noise_seed) >> 16) & 0xFF
                          for i in range(image_size)))
    for a, b, at in extents:
        img[at:at + (b - a)] = file_bytes[a:b]
    return bytes(img)


# ------------------------------------------------------------ unit tests --

def test_nal_walk():
    good = struct.pack('>I', 3) + b'abc' + struct.pack('>I', 2) + b'de'
    assert nal_walk_ok(good, len(good))
    assert not nal_walk_ok(good[:-1], len(good))          # chain must close
    bad = struct.pack('>I', 3) + b'abc' + struct.pack('>I', 9) + b'de'
    assert not nal_walk_ok(bad, len(bad))
    assert not nal_walk_ok(b'', 1)


def test_devc_walk():
    s = b'DEVC' + struct.pack('>I', 4) + b'abcd' + b'DEVC' + struct.pack('>I', 1) + b'x\x00\x00\x00'
    assert devc_walk_ok(s, len(s))
    assert not devc_walk_ok(s[:-1], len(s) - 1)
    assert not devc_walk_ok(b'NOPE' + s[4:], len(s))


def test_parse_moov_roundtrip():
    specs = [([500, 300, 200], 1), ([700, 100], 2)]
    file_bytes, samples, desc_offs = _build_file(specs)
    # locate the moov atom within the file bytes
    for a in walk_atoms(file_bytes):
        if a.kind == b'moov':
            info = parse_moov(file_bytes[a.off:a.off + a.size], phys=777)
            break
    else:
        pytest.fail('no moov in synthetic file')
    assert info.physical_offset == 777
    assert info.timescale == 60000 and info.duration == 120000
    by = info.by_tag()
    vide = by[('vide', 'avc1')]
    fdsc = by[('meta', 'fdsc')]
    assert vide.sample_count == 2 and vide.sample_sizes == [1012, 808]
    assert vide.chunk_offsets == [o for o, _ in samples]
    assert fdsc.sample_count == 2 and fdsc.chunk_offsets == desc_offs
    assert vide.width == 1920 and vide.height == 1440
    model = FileModel('syn', info, 0, 0, 0, 100)
    assert len(model.descs) == 2
    assert model.descs[0].expected[:2] == b'GP'
    assert model.descs[0].expected == _descriptor(1012)


def test_cluster_grid():
    gz = _synthetic_geometry(1 << 20)
    assert _cluster_grid(gz, 0, 2048) == [512, 1024, 1536, 2048]
    assert _cluster_grid(gz, 1024, 1536) == [1536]
    assert _cluster_grid(None, 0, 70000) == [32768, 65536]


def test_reconcile_maps():
    """Two files whose provisional boundaries overlap; the solver abuts them."""
    a = dict(name='a', runs=[[0, 4096, 100_000], [4096, 8192, 200_000]],
             switches=[dict(lo=3000, hi=5000, cands=[4096, 4352], chosen=4352)])
    b = dict(name='b', runs=[[0, 2048, 104_096], [2048, 6144, 210_000]],
             switches=[dict(lo=1500, hi=2500, cands=[2048, 2304], chosen=2304)])
    report = reconcile_maps([a, b])
    # a's run0 must end at or before b's run0 start: 100000 + B_a <= 104096
    assert a['runs'][0][1] == 4096
    assert report['final_overlap'] == 0
    assert a['switches'][0]['chosen'] == 4096


# ------------------------------------------------- synthetic reassemblies --

SYN_SAMPLES_CONTIG = [
    ([400, 300, 200], 1), ([600, 400], 2), ([500, 500, 300], 3),
    ([700, 100], 4), ([450, 450, 400], 5),
]


def test_contiguous_reassembly(tmp_path):
    """Level 1: one extent, everything validates, byte-exact output."""
    file_bytes, samples, _ = _build_file(SYN_SAMPLES_CONTIG)
    off = 4096
    img = _scatter(file_bytes, [(0, len(file_bytes), off)], 1 << 17)
    (tmp_path / 'img.bin').write_bytes(img)
    out = tmp_path / 'out.mp4'
    with open(tmp_path / 'img.bin', 'rb') as f:
        res = reassemble(f, _synthetic_geometry(len(img)), off, out)
    assert res.status == 'complete'
    assert res.missing_ranges == []
    assert res.bytes_written == len(file_bytes)
    assert out.read_bytes() == file_bytes
    assert res.sample_stats.nal_bad == 0 and res.sample_stats.descriptors_bad == 0
    assert res.sample_stats.descriptors_ok == 5
    assert res.moov.width == 1920 and res.moov.duration_s == 2.0
    assert len(res.runs) == 1
    assert res.runs[0] == (0, len(file_bytes), off)


def _dense_straddle(sample_off: int, n_pre: int = 6, cluster: int = 512,
                    nal: int = 200) -> tuple[list[int], int]:
    """NAL list for a straddling sample + the boundary the test targets.

    Short NALs (204-byte pitch < cluster grid) put a length prefix within
    every candidate window, so only the true boundary keeps the chain
    closing: splits before it misread a later prefix, splits after it
    misread the prefix sitting exactly on it.
    """
    step = nal + 4
    m = next(m for m in range(0, cluster) if (sample_off + m * step) % cluster == 0)
    return [nal] * (m + 5), sample_off + m * step


def test_fragmented_structural_switches(tmp_path):
    """Level 3, structural tier: the straddling sample's NAL walk pins the
    boundary content-wise (strict score maximum, no tie to break)."""
    f0, f1 = ([390, 390], 1), ([558, 558], 2)              # 788 / 1124 bytes
    f2, f3, f4 = ([602, 602], 4), ([700, 100], 5), ([450, 450, 400], 7)
    off = 44 + (16 + 788) + (16 + 1124)                    # sample #2 start
    nals1, b1 = _dense_straddle(off)
    off += sum(4 + n for n in nals1) + 16 + (16 + 1212)    # past sample #3
    nals2, b2 = _dense_straddle(off)
    specs = [f0, f1, (nals1, 3), f2, (nals2, 6), f3, f4]
    file_bytes, samples, desc_offs = _build_file(specs)
    assert samples[2][0] == 1988 and samples[4][0] == off
    assert b1 % 512 == 0 and b2 % 512 == 0
    assert samples[2][0] < b1 < samples[2][0] + samples[2][1]
    assert samples[4][0] < b2 < samples[4][0] + samples[4][1]
    x0, x1, x2 = 4096, 300_000, 600_000                    # ragged gaps
    extents = [(0, b1, x0), (b1, b2, x1), (b2, len(file_bytes), x2)]
    img = _scatter(file_bytes, extents, 1 << 21)
    (tmp_path / 'img.bin').write_bytes(img)
    out = tmp_path / 'out.mp4'
    with open(tmp_path / 'img.bin', 'rb') as f:
        res = reassemble(f, _synthetic_geometry(len(img)), x0, out)
    assert res.status == 'complete'
    assert res.missing_ranges == []
    assert out.read_bytes() == file_bytes                  # byte-exact rebuild
    assert res.sample_stats.nal_bad == 0 and res.sample_stats.descriptors_bad == 0
    assert len(res.switches) == 2
    assert [e.chosen for e in res.switches] == [b1, b2]
    assert all(e.method == 'structural' for e in res.switches)


def test_fragmented_flush_lattice(tmp_path):
    """Level 3, lattice tier: the straddling sample is one giant NAL whose
    payload swallows every candidate, so the NAL walk ties everywhere and
    the flush lattice picks the truth (gaps of one quantum G, boundaries on
    the lattice)."""
    f0, f1 = ([390, 390], 1), ([558, 558], 2)
    f2, f3, f4 = ([602, 602], 4), ([700, 100], 5), ([450, 450, 400], 7)
    giant = [250, 3496]                                    # 3754 bytes total
    specs = [f0, f1, (giant, 3), f2, (giant, 6), f3, f4]
    file_bytes, samples, desc_offs = _build_file(specs)
    b1, b2 = 4096, 8192                                    # the lattice points
    assert samples[2][0] + 254 < b1 < samples[2][0] + samples[2][1]
    assert samples[4][0] + 254 < b2 < samples[4][0] + samples[4][1]
    for i, b in ((2, b1), (4, b2)):                         # one lattice point
        lo, hi = samples[i][0] - 16, samples[i][0] + samples[i][1]
        assert [B for B in range(lo + 1, hi + 1) if B % 4096 == 0] == [b]
    g = 4096
    x0 = 8192
    x1 = x0 + b1 + g                                       # gap = one quantum
    x2 = x1 + (b2 - b1) + g
    extents = [(0, b1, x0), (b1, b2, x1), (b2, len(file_bytes), x2)]
    img = _scatter(file_bytes, extents, 1 << 21)
    (tmp_path / 'img.bin').write_bytes(img)
    out = tmp_path / 'out.mp4'
    with open(tmp_path / 'img.bin', 'rb') as f:
        res = reassemble(f, _synthetic_geometry(len(img)), x0, out)
    assert res.status == 'complete'
    assert out.read_bytes() == file_bytes                  # byte-exact rebuild
    assert [e.chosen for e in res.switches] == [b1, b2]
    assert all(e.method == 'flush-lattice' for e in res.switches)
    assert res.sample_stats.nal_bad == 0


def test_moovless_metadata_only(tmp_path):
    """Level 4: no moov anywhere -> 'metadata-only', no output written."""
    ftyp = _atom(b'ftyp', b'mp41' + b'\x00' * 4 + b'mp41')
    mdat = _atom(b'mdat', b'\x11' * 4096)
    file_bytes = ftyp + mdat
    off = 8192
    img = _scatter(file_bytes, [(0, len(file_bytes), off)], 1 << 16, noise_seed=7)
    assert b'moov' not in img
    (tmp_path / 'img.bin').write_bytes(img)
    out = tmp_path / 'out.mp4'
    with open(tmp_path / 'img.bin', 'rb') as f:
        res = reassemble(f, _synthetic_geometry(len(img)), off, out)
    assert res.status == 'metadata-only'
    assert not out.exists()
    assert any('mdat' in n for n in res.notes)


# --------------------------------------------------------------- gold -----

IMAGE = Path('/home/aviallon/Documents/Backups/Manon/SD_Card_Manon-v2.img')
GOPR3655_OFF = 17235968            # cluster 272 = 8388608 + 270*32768
GOPR3655_SIZE = 1256672901
GOPR3655_DURATION = 222.72

pytestmark_gold = pytest.mark.skipif(
    not IMAGE.exists(), reason='fixture image SD_Card_Manon-v2.img not present')


def _gold_geometry() -> Geometry:
    # FAT layout reconstructed from ground truth (the FAT itself is zeroed):
    # cluster 272 == absolute 17235968 -> 32 KiB clusters, data at 8 MiB.
    return Geometry(source='fixture', part_off=0, part_size=8053063680,
                    bps=512, spc=64, reserved=32, num_fats=2, fat_sects=8176)


@pytest.fixture(scope='module')
def gold(tmp_path_factory):
    if not IMAGE.exists():
        pytest.skip('fixture image SD_Card_Manon-v2.img not present')
    out_dir = tmp_path_factory.mktemp('gold')
    out = out_dir / 'GOPR3655.MP4'
    with open(IMAGE, 'rb') as f:
        res = reassemble(f, _gold_geometry(), GOPR3655_OFF, out, verify=True)
    return res, out


@pytestmark_gold
def test_gold_gopr3655_zero_missing(gold):
    res, out = gold
    assert res.status == 'complete'
    assert res.bytes_written == GOPR3655_SIZE
    assert out.stat().st_size == GOPR3655_SIZE
    assert res.missing_ranges == []                      # zero missing bytes
    assert res.file_size == GOPR3655_SIZE
    assert res.mapped_bytes == GOPR3655_SIZE


@pytestmark_gold
def test_gold_gopr3655_sample_stats(gold):
    res, _ = gold
    s = res.sample_stats
    # exact counts: the moov contract of GOPR3655.MP4 (source verify.json)
    assert (s.descriptors_ok, s.descriptors_bad) == (24012, 0)
    assert (s.nal_ok, s.nal_bad) == (13349, 0)
    assert (s.gpmd_ok, s.gpmd_bad) == (222, 0)
    assert s.all_ok


@pytestmark_gold
def test_gold_gopr3655_extent_walk(gold):
    res, _ = gold
    assert len(res.runs) == 142                          # 141 extent switches
    assert len(res.switches) == 141
    assert all(e.cands and e.chosen in e.cands for e in res.switches)
    assert res.moov is not None
    assert abs(res.moov.duration_s - GOPR3655_DURATION) < 0.01
    assert (res.moov.width, res.moov.height) == (1920.0, 1440.0)


@pytestmark_gold
def test_gold_gopr3655_ffprobe_duration(gold):
    _, out = gold
    if shutil.which('ffprobe') is None:
        pytest.skip('ffprobe not on PATH')
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                        'format=duration', '-of', 'csv=p=0', str(out)],
                       capture_output=True, text=True)
    assert p.returncode == 0
    assert abs(float(p.stdout.strip()) - GOPR3655_DURATION) < 0.01


@pytestmark_gold
def test_gold_gopr3655_decodes_clean(gold):
    """Content claim: the whole 222.72 s video decodes without one
    corruption signature — what distinguishes correct extent boundaries
    from structurally-valid-but-wrong ones."""
    _, out = gold
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg not on PATH')
    p = subprocess.run(['ffmpeg', '-v', 'warning', '-i', str(out), '-f', 'null', '-'],
                       capture_output=True, text=True)
    sig = ('concealing', 'error while decoding', 'corrupt',
           'failed to parse', 'sps_id')
    bad = [l for l in p.stderr.splitlines() if any(s in l.lower() for s in sig)]
    assert bad == []


def test_flush_lattice_gcd_interleaved_quanta(tmp_path):
    """Regression (the GOPR3666 case): interleaved writers make the extent
    gaps G and 2G, so the delta steps are NOT constant.  The lattice must
    still apply, with G inferred as the GCD of the steps; the old
    constant-delta rule silently disabled it here and the fallback
    misattributed boundaries."""
    f0, f1 = ([390, 390], 1), ([558, 558], 2)
    f2, f3, f4 = ([602, 602], 4), ([700, 100], 5), ([450, 450, 400], 7)
    giant = [250, 3496]                                    # ties at any split
    specs = [f0, f1, (giant, 3), f2, (giant, 6), f3, f4]
    file_bytes, samples, desc_offs = _build_file(specs)
    b1, b2 = 4096, 8192
    for i, b in ((2, b1), (4, b2)):
        lo, hi = samples[i][0] - 16, samples[i][0] + samples[i][1]
        assert [B for B in range(lo + 1, hi + 1) if B % 4096 == 0] == [b]
    g = 4096
    x0 = 8192
    x1 = x0 + b1 + g            # gap G  -> step  b1+g
    x2 = x1 + (b2 - b1) + 2 * g  # gap 2G -> step (b2-b1)+2g  (NON-constant)
    assert x1 - x0 != x2 - x1, 'steps must differ: this is the regression'
    extents = [(0, b1, x0), (b1, b2, x1), (b2, len(file_bytes), x2)]
    img = _scatter(file_bytes, extents, 1 << 21)
    (tmp_path / 'img.bin').write_bytes(img)
    out = tmp_path / 'out.mp4'
    with open(tmp_path / 'img.bin', 'rb') as f:
        res = reassemble(f, _synthetic_geometry(len(img)), x0, out)
    assert res.status == 'complete'
    assert out.read_bytes() == file_bytes
    assert [e.chosen for e in res.switches] == [b1, b2]
    assert all(e.method == 'flush-lattice' for e in res.switches)
    assert res.sample_stats.nal_bad == 0
