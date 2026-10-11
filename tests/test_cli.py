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
