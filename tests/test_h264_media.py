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


def test_media_runs_stops_at_damage():
    import random
    run = _avcc_run(20, 8000)
    damaged = bytearray(run)
    # break NAL #10's length field
    pos = 0
    for _ in range(10):
        pos += 4 + struct.unpack('>I', damaged[pos:pos + 4])[0]
    damaged[pos:pos + 4] = b'\xff\xff\xff\xff'
    data = bytes(damaged) + random.Random(7).randbytes(64_000)
    runs = media_runs(data)
    assert runs and runs[0].nals == 10       # exactly the valid prefix
    assert runs[0].end == pos                # never guessed past the break


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
