"""JPEG handler: extraction, embedded naming, phantom classification."""
import io

from PIL import Image

from fatality.carve.jpeg import JpegHandler


def make_jpeg(size=(256, 256), seed=42) -> bytes:
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, 'JPEG', quality=92)
    return buf.getvalue()


def test_complete_jpeg_extracts_and_validates(tmp_path):
    h = JpegHandler()
    jpg = make_jpeg()
    p = tmp_path / 'in.bin'
    p.write_bytes(jpg)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    assert res.status == 'complete'
    assert res.size == len(jpg)
    out = tmp_path / 'out.jpg'
    out.write_bytes(res.extras['bytes'])
    v = h.validate(str(out))
    assert v.ok, v.detail


def test_name_record_recovered_from_content(tmp_path):
    h = JpegHandler()
    jpg = make_jpeg()
    rec = b'DCIM\\100GOPRO\\G0013656.JPG\x00' + b'2024:08:02 15:19:05'
    # inject record into an APP segment area (before SOS) via padding comment
    padded = jpg[:2] + b'\xff\xfe' + (len(rec) + 2).to_bytes(2, 'big') + rec + jpg[2:]
    p = tmp_path / 'in.bin'
    p.write_bytes(padded)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    assert res.name == 'G0013656.JPG'
    assert res.datetime == '2024:08:02 15:19:05'


def test_soi_in_foreign_data_is_phantom(tmp_path):
    h = JpegHandler()
    # SOI + SOS-looking garbage with no SOF: not a file
    junk = b'\xff\xd8\xff\xda\x00\x08' + bytes(range(64))
    p = tmp_path / 'in.bin'
    p.write_bytes(junk)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    assert res.status == 'phantom'


def test_truncated_jpeg_salvages_prefix(tmp_path):
    h = JpegHandler()
    jpg = make_jpeg()                      # entropy-dominated: headers are tiny
    assert len(jpg) > 20000
    truncated = jpg[:len(jpg) // 2]        # cut mid-entropy, past SOS
    p = tmp_path / 'in.bin'
    p.write_bytes(truncated)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    assert res.status == 'partial'
    assert res.size > 1000
    assert res.notes


def test_broken_jpeg_still_attaches_embedded_preview(tmp_path):
    """Regression: a truncated photo must still carry its embedded preview
    (the best surviving view), and the FIRST preview must not be dropped."""
    h = JpegHandler()
    thumb = b'\xff\xd8\xff\xdb\x00\x04\x00\x00\xff\xd9'
    app1 = b'\xff\xe1' + (2 + len(thumb)).to_bytes(2, 'big') + thumb
    jpg = make_jpeg()
    parent = jpg[:2] + app1 + jpg[2:]
    broken = parent[:len(parent) // 2]     # cut mid-entropy: no EOI survives
    p = tmp_path / 'in.bin'
    p.write_bytes(broken)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    assert res.status == 'partial'
    previews = res.extras.get('previews') or []
    assert previews, 'embedded preview missing from broken-file salvage'
    poff, plen = previews[0]
    assert poff == 6 and plen == len(thumb)   # after SOI(2) + APP1 hdr(4)
    assert res.extras['bytes'][poff:poff + plen] == thumb


def test_tiny_eoi_junk_is_not_a_jpeg(tmp_path):
    """A 4-byte SOI..EOI in entropy data is a false positive, not salvage."""
    h = JpegHandler()
    junk = b'\xff\xd8\xff\xd9' + bytes(range(256)) * 8
    p = tmp_path / 'in.bin'
    p.write_bytes(junk)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    assert res.status == 'phantom'


# ---------------------------------------------------------- MPF (DC-007) ----

def _build_mpf(primary: bytes, secondaries: list[bytes], endian: str = 'MM') -> bytes:
    """A multi-picture JPEG: primary with an MPF APP2 index + appended images."""
    import struct
    n = 1 + len(secondaries)
    ent_off = 8 + 2 + 24                 # TIFF header + IFD(2 entries)
    e = '<' if endian == 'II' else '>'
    ifd = (struct.pack(e + 'H', 2)
           + struct.pack(e + 'HHII', 0xB001, 4, 1, n)
           + struct.pack(e + 'HHII', 0xB002, 7, 16 * n, ent_off))
    tiff = (b'II\x2a\x00' if endian == 'II' else b'MM\x00\x2a') \
        + struct.pack(e + 'I', 8) + ifd
    entries = bytearray(16 * n)
    payload = b'MPF\x00' + tiff + bytes(entries)
    app2 = b'\xff\xe2' + (2 + len(payload)).to_bytes(2, 'big') + payload
    prim = primary[:2] + app2 + primary[2:]
    s1 = len(prim)
    bo = 'little' if endian == 'II' else 'big'
    def entry(attrs, size, off):
        # MP Entry = 16 bytes: attr(4) + size(4) + offset(4) + dep(4);
        # fields follow the MP Endian byte order (CIPA DC-007 §5.2.2.1)
        return (attrs.to_bytes(4, bo) + size.to_bytes(4, bo)
                + off.to_bytes(4, bo) + b'\x00' * 4)
    entries[0:16] = entry(0xA0030000, s1, 0)
    pos = s1
    for j, sec in enumerate(secondaries):
        entries[16 * (j + 1):16 * (j + 2)] = entry(0x00010000, len(sec), pos)
        pos += len(sec)
    payload = b'MPF\x00' + tiff + bytes(entries)
    app2 = b'\xff\xe2' + (2 + len(payload)).to_bytes(2, 'big') + payload
    prim = primary[:2] + app2 + primary[2:]
    return prim + b''.join(secondaries)


def test_mpf_complete_file_includes_appended_images(tmp_path):
    h = JpegHandler()
    prim, sec1, sec2 = make_jpeg(), make_jpeg((160, 120), 1), make_jpeg((960, 720), 2)
    full = _build_mpf(prim, [sec1, sec2])
    p = tmp_path / 'in.bin'
    p.write_bytes(full)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    assert res.status == 'complete'
    assert len(res.extras['bytes']) == len(full)      # the COMPLETE MPF file
    assert res.size == len(full)
    mpf = res.extras['mpf']
    assert len(mpf) == 3
    assert res.extras['previews'] == [(o, s) for o, s, _a in mpf[1:]]
    assert any('MPF' in n for n in res.notes)


def test_mpf_index_previews_are_exact(tmp_path):
    h = JpegHandler()
    sec = make_jpeg((160, 120), 3)
    full = _build_mpf(make_jpeg(), [sec])
    p = tmp_path / 'in.bin'
    p.write_bytes(full)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    off, size = res.extras['previews'][0]
    assert res.extras['bytes'][off:off + size] == sec


def test_unindexed_trailing_stream_never_merged(tmp_path):
    """Regression: absorbing unindexed trailing streams MERGED neighbouring
    files on a contiguous card (8 scenario regressions).  Only an index
    proves membership — unindexed streams stay separate carves."""
    h = JpegHandler()
    prim, sec = make_jpeg(), make_jpeg((160, 120), 4)
    full = prim + sec
    p = tmp_path / 'in.bin'
    p.write_bytes(full)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    assert res.size == len(prim)       # the second file stays separate


def test_garbage_after_eoi_never_absorbed(tmp_path):
    import random
    h = JpegHandler()
    junk = random.Random(42).randbytes(50_000)
    full = make_jpeg() + junk
    p = tmp_path / 'in.bin'
    p.write_bytes(full)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    assert res.size == len(make_jpeg())     # only the standard-bounded part


def test_mpf_little_endian_index(tmp_path):
    """MP Entry fields follow the MP Endian byte order — an II file must
    parse exactly like an MM one (spec correction, not 'always big-endian')."""
    h = JpegHandler()
    sec = make_jpeg((160, 120), 7)
    full = _build_mpf(make_jpeg(), [sec], endian='II')
    p = tmp_path / 'in.bin'
    p.write_bytes(full)
    with open(p, 'rb') as f:
        res = h.extract(f, 0)
    assert len(res.extras.get('mpf', [])) == 2
    off, size = res.extras['previews'][0]
    assert res.extras['bytes'][off:off + size] == sec
