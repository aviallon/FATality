"""Shared fixtures: build tiny FAT filesystem images in memory/tmp_path.

Tests must not need real evidence images: a deterministic FAT builder lets us
assert exact byte-level behaviour, including the quick-format scenario (FATs
zeroed, data and directory contents intact) that FATality exists for.
"""
from __future__ import annotations

import struct

import pytest

BPS = 512
SPC = 1                     # 512-byte clusters for tiny fixtures
RESERVED = 32
NUM_FATS = 2
FAT_SECTS = 1               # 512B FAT = 128 FAT32 entries
ROOT_CLU = 2
DATA_OFF = (RESERVED + NUM_FATS * FAT_SECTS) * BPS


def fat_date(y: int, m: int, d: int) -> int:
    return ((y - 1980) << 9) | (m << 5) | d


def fat_time(h: int, mi: int, s: int) -> int:
    return (h << 11) | (mi << 5) | (s // 2)


def short_entry(name: bytes, ext: bytes, cluster: int, size: int,
                attr: int = 0x20, deleted: bool = False,
                y=2024, m=8, d=2) -> bytes:
    e = bytearray(32)
    e[0:8] = name.ljust(8)
    e[8:11] = ext.ljust(3)
    if deleted:
        e[0] = 0xE5
    e[11] = attr
    dt = fat_date(y, m, d)
    e[14:16] = struct.pack('<H', fat_time(15, 19, 5))
    e[16:18] = struct.pack('<H', dt)
    e[18:20] = struct.pack('<H', dt)
    e[22:24] = struct.pack('<H', fat_time(15, 19, 5))
    e[24:26] = struct.pack('<H', dt)
    e[26:28] = struct.pack('<H', cluster & 0xFFFF)
    e[20:22] = struct.pack('<H', cluster >> 16)
    e[28:32] = struct.pack('<I', size)
    return bytes(e)


def lfn_entries(long_name: str, checksum: int) -> list[bytes]:
    """LFN slots for a name, in on-disk order (last segment first)."""
    units = [long_name[i:i + 13] for i in range(0, len(long_name), 13)]
    out = []
    for segno, seg in enumerate(units, start=1):
        e = bytearray(32)
        e[0] = segno | (0x40 if segno == len(units) else 0)
        chars = seg + ('\x00' if len(seg) < 13 else '')
        raw = (chars + '\uffff' * 13)[:13].encode('utf-16-le')
        e[1:11] = raw[0:10]
        e[11] = 0x0F
        e[12] = 0
        e[13] = checksum
        e[14:26] = raw[10:22]
        e[26:28] = b'\x00\x00'
        e[28:32] = raw[22:26]
        out.append(bytes(e))
    return list(reversed(out))


def lfn_checksum(short11: bytes) -> int:
    s = 0
    for c in short11:
        s = (((s & 1) << 7) + (s >> 1) + c) & 0xFF
    return s


def build_fat32_image() -> bytearray:
    """A ~40KB FAT32 image: root dir (cluster 2) with an orphaned-style tree.

    Layout (cluster size 512B):
      clu2: root   -> DCIM (dir, clu3), HELLO.TXT (file, clu5, 11 bytes)
      clu3: DCIM   -> '.', '..', PHOTO.JPG (file, clu4, deleted)
      clu4: 'fake-jpeg-payload' bytes
      clu5: 'hello fatality' payload (11 bytes)
    """
    img = bytearray(DATA_OFF + 16 * SPC * BPS)
    # boot sector (FAT32 BPB)
    bs = bytearray(512)
    bs[0:3] = b'\xeb\x58\x90'
    bs[3:11] = b'mkfs.fat'
    bs[11:13] = struct.pack('<H', BPS)
    bs[13] = SPC
    bs[14:16] = struct.pack('<H', RESERVED)
    bs[16] = NUM_FATS
    bs[17:19] = struct.pack('<H', 0)          # FAT32: no root entries
    bs[19:21] = struct.pack('<H', 0)
    bs[21] = 0xF8
    bs[22:24] = struct.pack('<H', 0)          # FATSz16 = 0 -> FAT32
    bs[24:26] = struct.pack('<H', 63)
    bs[26:28] = struct.pack('<H', 255)
    bs[28:32] = struct.pack('<I', RESERVED)   # hidden
    bs[32:36] = struct.pack('<I', len(img) // BPS)
    bs[36:40] = struct.pack('<I', FAT_SECTS)
    bs[44:48] = struct.pack('<I', ROOT_CLU)
    bs[66] = 0x29
    bs[67:71] = b'\x4f\x1f\x63\x18'
    bs[71:82] = b'NO NAME    '
    bs[82:90] = b'FAT32   '
    bs[510:512] = b'\x55\xaa'
    img[0:512] = bs

    # FATs: chain root(2)->EOF, DCIM(3)->EOF, photo(4)->EOF, hello(5)->EOF
    for fat_i in range(NUM_FATS):
        fat = bytearray(FAT_SECTS * BPS)

        def put(c, v, fat=fat):     # bind: closure must capture THIS copy
            struct.pack_into('<I', fat, c * 4, v)
        put(0, 0x0FFFFFF8)
        put(1, 0x0FFFFFFF)
        put(2, 0x0FFFFFFF)
        put(3, 0x0FFFFFFF)
        put(4, 0x0FFFFFFF)
        put(5, 0x0FFFFFFF)
        off = (RESERVED + fat_i * FAT_SECTS) * BPS
        img[off:off + len(fat)] = fat

    # root dir (cluster 2)
    dcim = short_entry(b'DCIM', b'   ', 3, 0, attr=0x10)
    hello_lfn = lfn_entries('hello fatality file.txt', lfn_checksum(b'HELLO   TXT'))
    hello = short_entry(b'HELLO', b'TXT', 5, 11)
    root = b''.join([dcim] + hello_lfn + [hello])
    img[DATA_OFF:DATA_OFF + len(root)] = root

    # DCIM dir (cluster 3): deleted PHOTO.JPG entry (orphan evidence)
    dots = short_entry(b'.', b'   ', 3, 0, attr=0x10) + \
        short_entry(b'..', b'   ', 0, 0, attr=0x10)
    photo_lfn = lfn_entries('GOPR0001.JPG', lfn_checksum(b'GOPR0001JPG'))
    # deleted: LFN seq bytes marked 0xE5 as well
    photo_lfn = [bytes([0xE5]) + p[1:] for p in photo_lfn]
    photo = short_entry(b'GOPR0001', b'JPG', 4, 16, deleted=True)
    d = dots + b''.join(photo_lfn) + photo
    img[DATA_OFF + 1 * SPC * BPS:DATA_OFF + 1 * SPC * BPS + len(d)] = d

    # payloads (cluster 4 = photo data, cluster 5 = hello)
    img[DATA_OFF + 2 * SPC * BPS:DATA_OFF + 2 * SPC * BPS + 16] = b'fake-jpeg-data!'
    img[DATA_OFF + 3 * SPC * BPS:DATA_OFF + 3 * SPC * BPS + 11] = b'hello fatality'
    return img


@pytest.fixture
def fat32_image(tmp_path):
    img = build_fat32_image()
    p = tmp_path / 'fixture.img'
    p.write_bytes(img)
    return p


@pytest.fixture
def quickformatted_image(tmp_path):
    """Same image after a quick format: FATs zeroed, everything else intact."""
    img = build_fat32_image()
    for fat_i in range(NUM_FATS):
        off = (RESERVED + fat_i * FAT_SECTS) * BPS
        img[off:off + FAT_SECTS * BPS] = bytes(FAT_SECTS * BPS)
    p = tmp_path / 'quickfmt.img'
    p.write_bytes(img)
    return p
