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
