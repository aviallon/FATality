"""Recovered files must look like the original files — timestamps included."""
from __future__ import annotations

import os

from fatality.cli import _apply_datetime


def test_apply_datetime_sets_exif_format(tmp_path):
    p = tmp_path / 'photo.jpg'
    p.write_bytes(b'\xff\xd8\xff\xd9')
    _apply_datetime(p, '2024:08:02 15:19:05')
    st = os.stat(p)
    import datetime
    got = datetime.datetime.fromtimestamp(st.st_mtime)
    assert (got.year, got.month, got.day, got.hour, got.minute) == (2024, 8, 2, 15, 19)


def test_apply_datetime_iso_and_none(tmp_path):
    p = tmp_path / 'a.bin'
    p.write_bytes(b'x')
    _apply_datetime(p, '2023-01-02T03:04:05')
    import datetime
    assert datetime.datetime.fromtimestamp(os.stat(p).st_mtime).year == 2023
    # None or junk must not crash or touch the file
    _apply_datetime(p, None)
    _apply_datetime(p, 'not a date')


def test_usable_name_rejects_mojibake():
    from fatality.cli import _usable_name
    assert _usable_name('G0013656.JPG', 'x.bin') == 'G0013656.JPG'
    assert _usable_name('café.jpg', 'x.bin') == 'café.jpg'
    # broken LFN: control characters -> anonymous fallback
    bad = '\x8b ñÖ\x8dR³ö.\x8b\x8d«'
    assert _usable_name(bad, 'clu0000271.bin') == 'clu0000271.bin'


def test_mp4_naming_gaps_and_twins():
    from fatality.cli import _assign_mp4_names
    members = [
        # named LRV + its full-res twin (same creation)
        dict(off=1, name='GOPR3655.LRV', creation='2024-08-02T10:00:00', size=74_570_824),
        dict(off=2, name=None, creation='2024-08-02T10:00:00', size=1_256_672_901),
        # unnamed pairs filling counter gaps 3666/3667 (photos cover 3656-3665/3669+)
        dict(off=3, name=None, creation='2024-08-02T11:00:00', size=727_563_375),
        dict(off=4, name=None, creation='2024-08-02T11:00:00', size=43_154_794),
        dict(off=5, name=None, creation='2024-08-02T12:00:00', size=886_248_100),
        dict(off=6, name=None, creation='2024-08-02T12:00:00', size=52_601_460),
        # lone orphan twin (primary destroyed): honest fallback, never a guess
        dict(off=7, name=None, creation='2024-08-02T13:00:00', size=514_450),
    ]
    photos = (['G0013656.JPG'] + [f'G001{n}.JPG' for n in range(3657, 3666)]
              + [f'G002{n}.JPG' for n in range(3669, 3690)])
    out = _assign_mp4_names(members, photos)
    assert out[2] == 'GOPR3655.MP4'          # twin of the named LRV
    assert out[3] == 'GOPR3666.MP4'          # gap fill, creation order
    assert out[4] == 'GOPR3666.LRV'
    assert out[5] == 'GOPR3667.MP4'
    assert out[6] == 'GOPR3667.LRV'
    # a lone orphan twin is NEVER given a guessed counter
    assert 'GOPR' not in out[7] and '.partial.' in out[7]
