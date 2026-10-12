"""Real-world MP4 shapes: smartphone HEVC, fragmented MP4, camera moov-at-end.

Fixtures are committed under tests/fixtures/mp4 (ffmpeg-generated, small).
These pin the generality the tool must keep while the GoPro lane stays
specialised.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from fatality.carve.h264_media import annexb, media_runs
from fatality.carve.mp4 import Mp4Handler, nal_walk_ok, walk_atoms

FIX = Path(__file__).parent / 'fixtures' / 'mp4'


@pytest.mark.parametrize('name', ['phone_hevc_moovend.mp4',
                                  'screen_fmp4.mp4',
                                  'cam_h264_moovend.mp4'])
def test_extract_complete_on_realworld_shapes(name):
    p = FIX / name
    if not p.exists():
        pytest.skip('fixture missing')
    h = Mp4Handler()
    res = h.extract(io.BytesIO(p.read_bytes()), 0)
    assert res.status == 'complete', (res.status, res.notes)
    assert 'bytes' in res.extras
    assert len(res.extras['bytes']) == p.stat().st_size


@pytest.mark.parametrize('name', ['phone_hevc_moovend.mp4',
                                  'screen_fmp4.mp4',
                                  'cam_h264_moovend.mp4'])
def test_media_runs_cover_video_payload(name):
    p = FIX / name
    if not p.exists():
        pytest.skip('fixture missing')
    data = p.read_bytes()
    runs = media_runs(data)
    assert runs, 'no media runs found'
    covered = sum(r.media_bytes for r in runs)
    total = p.stat().st_size
    assert covered >= 0.5 * total, f'covered only {covered}/{total}'


def test_annexb_output_is_start_coded_stream():
    p = FIX / 'phone_hevc_moovend.mp4'
    if not p.exists():
        pytest.skip('fixture missing')
    data = p.read_bytes()
    runs = media_runs(data)
    out = b''.join(annexb(data, r) for r in runs)
    assert out.startswith(b'\x00\x00\x00\x01')
    assert out.count(b'\x00\x00\x00\x01') > 50


def test_hevc_samples_pass_sample_gate():
    # nal_walk_ok must accept HEVC headers (2-byte: type in bits 1..6)
    hevc_sample = (b'\x00\x00\x00\x08' + bytes([32 << 1, 0x01]) + b'\x00' * 6
                   + b'\x00\x00\x00\x05' + bytes([19 << 1, 0x01]) + b'\x00' * 3)
    assert nal_walk_ok(hevc_sample)


def test_fmp4_has_moof_boxes():
    p = FIX / 'screen_fmp4.mp4'
    if not p.exists():
        pytest.skip('fixture missing')
    kinds = [a.kind for a in walk_atoms(p.read_bytes())]
    assert b'moof' in kinds
    assert b'mdat' in kinds
