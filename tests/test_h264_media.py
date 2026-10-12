"""Container-independent media salvage + header-tolerant MP4 bounds."""
from __future__ import annotations

import struct

from fatality.carve.h264_media import annexb, media_runs
from fatality.carve.mp4 import Mp4Handler, _recover_bounds, walk_atoms


def _nal(payload: bytes, ntype: int = 1, ref_idc: int = 2) -> bytes:
    return struct.pack('>I', len(payload) + 1) + bytes([(ref_idc << 5) | ntype]) + payload


def _avcc_run(nals: int, size: int = 2000) -> bytes:
    body = bytearray()
    for i in range(nals):
        ntype = 5 if i == 0 else 1          # IDR then slices
        body += _nal(bytes([i % 251]) * size, ntype)
    return bytes(body)


def test_media_runs_finds_validated_run_in_noise():
    import random
    noise = random.Random(42).randbytes
    run = _avcc_run(20, 4000)               # 20 NALs, 80KB
    data = noise(100_000) + run + noise(100_000)
    runs = media_runs(data)
    assert len(runs) == 1
    r = runs[0]
    assert r.nals == 20 and r.vcl == 20
    assert data[r.off:r.end] == run


def test_media_run_bridges_interleave_but_never_emits_gaps():
    import random
    run = _avcc_run(20, 8000)
    damaged = bytearray(run)
    # break NAL #10's length field (as an interleaved non-NAL chunk would)
    pos = 0
    for _ in range(10):
        pos += 4 + struct.unpack('>I', damaged[pos:pos + 4])[0]
    damaged[pos:pos + 4] = b'\xff\xff\xff\xff'
    data = bytes(damaged) + random.Random(7).randbytes(64_000)
    runs = media_runs(data)
    assert runs and runs[0].nals == 19          # both valid stretches, minus the broken one
    assert len(runs[0].gaps) == 1               # the skipped span is RECORDED
    g0, g1 = runs[0].gaps[0]
    assert g0 == pos
    out = annexb(data, runs[0])
    assert out.count(b'\x00\x00\x00\x01') == 19
    assert bytes([10]) * 100 not in out         # gap bytes are never emitted


def test_media_run_stops_when_no_valid_chain_ahead():
    run = _avcc_run(10, 16000)      # 5 good NALs = 80KB: over the run threshold
    damaged = bytearray(run)
    pos = 0
    for _ in range(5):
        pos += 4 + struct.unpack('>I', damaged[pos:pos + 4])[0]
    damaged[pos:pos + 4] = b'\xff\xff\xff\xff'
    # nothing valid after the break: zeroed damage (what a lost cluster
    # looks like), longer than GAP_MAX
    data = bytes(damaged[:pos]) + bytes(64_000)
    runs = media_runs(data)
    assert runs and runs[0].end == pos          # never guessed past the break
    assert runs[0].nals == 5


def test_hevc_chain_found_and_annexb():
    import random
    noise = random.Random(42).randbytes

    def hevc_nal(payload: bytes, ntype: int) -> bytes:
        b0 = ntype << 1                          # forbidden=0, layer=0, tid=1
        return struct.pack('>I', len(payload) + 2) + bytes([b0, 0x01]) + payload
    body = b''
    for i in range(60):
        ntype = 19 if i == 0 else 32 if i == 1 else 33 if i == 2 else 1
        body += hevc_nal(bytes([i % 251]) * 3000, ntype)
    data = noise(2000) + body + noise(2000)
    runs = media_runs(data)
    assert runs
    r = runs[0]
    # the run must CONTAIN the synthetic chain (noise may bridge in extras)
    assert r.off <= 2000 and r.end >= 2000 + len(body)
    assert r.nals >= 60
    out = annexb(data, r)
    assert out.startswith(b'\x00\x00\x00\x01')
    assert out.count(b'\x00\x00\x00\x01') >= 60


def test_media_runs_rejects_random_data():
    import random
    assert media_runs(random.Random(42).randbytes(2_000_000)) == []


def test_annexb_has_start_codes():
    import random
    noise = random.Random(42).randbytes
    run_data = _avcc_run(80, 1000)           # 80KB: over the run threshold
    data = noise(1000) + run_data + noise(1000)
    r = media_runs(data)[0]
    out = annexb(data, r)
    assert out.startswith(b'\x00\x00\x00\x01')
    assert out.count(b'\x00\x00\x00\x01') == 80
    # every NAL payload preserved
    assert out[4:1001] in run_data


def test_recover_bounds_survives_clobbered_ftyp_size():
    # ftyp box with garbage size field, then mdat with a sane size
    ftyp = struct.pack('>I', 0xdeadbeef) + b'ftypisom' + b'\x00' * 28   # 40B box
    mdat_payload = b'\x00' * 500
    mdat = struct.pack('>I', 8 + len(mdat_payload)) + b'mdat' + mdat_payload
    data = ftyp + mdat
    assert walk_atoms(data) == [] or walk_atoms(data)[0].size == 0xdeadbeef
    end = _recover_bounds(data)
    assert end == len(data)


def test_mp4_extract_metadata_only_when_bounds_exceed_window():
    h = Mp4Handler()
    # a moov+mdat whose size fields walk past the provided buffer
    bad = (struct.pack('>I', 8) + b'ftypisom'
           + struct.pack('>I', 1 << 30) + b'mdat' + b'\x00' * 64
           + struct.pack('>I', 1 << 30) + b'moov' + b'\x00' * 64)
    import io
    res = h.extract(io.BytesIO(bad), 0)
    assert res.status == 'partial'
    assert 'bytes' not in res.extras          # never claim unbounded output
    assert any('exceed readable window' in n for n in res.notes)


def test_gap_heavy_run_is_not_emitted():
    """A run whose span is mostly bridged gap has no media to deliver."""
    import random
    noise = random.Random(42).randbytes
    run = _avcc_run(8, 2000)                   # 16KB of real media
    data = noise(2000) + run + noise(200_000)  # huge non-NAL surroundings
    runs = media_runs(data)
    for r in runs:
        assert r.media_bytes >= 64 * 1024     # emission threshold is on MEDIA
