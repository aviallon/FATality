"""Schema layer: field tables are the single source of on-disk offsets."""
import json

from conftest import build_fat32_image, lfn_checksum, lfn_entries, short_entry

from fatality import schema
from fatality.geometry import scan_geometries


def test_dirent_schema_roundtrip():
    raw = short_entry(b'GOPR3655', b'MP4', 272, 1256672901)
    e = schema.DIRENT.parse(raw)
    assert e['name8'] == 'GOPR3655'
    assert e['ext'] == 'MP4'
    assert e['attr'] == 0x20
    assert schema.dirent_cluster(e) == 272
    assert e['size'] == 1256672901
    assert schema.dirent_short_name(e) == 'GOPR3655.MP4'


def test_dirent_high_cluster_word():
    raw = short_entry(b'X', b'   ', (0x1234 << 16) | 5, 0, attr=0x10)
    e = schema.DIRENT.parse(raw)
    assert e['cluster_hi'] == 0x1234 and e['cluster_lo'] == 5
    assert schema.dirent_cluster(e) == (0x1234 << 16) | 5


def test_lfnent_schema_assembles_text():
    slots = lfn_entries('very long file name.txt', lfn_checksum(b'HELLO   TXT'))
    e = schema.LFNENT.parse(slots[0])
    assert e['attr'] == schema.ATTR_LFN
    parsed = [schema.lfn_text(schema.LFNENT.parse(s)) for s in slots]
    # on-disk order is last-segment-first
    assert ''.join(reversed(parsed)) == 'very long file name.txt'
    assert parsed[-1].startswith('very long fi')


def test_bpb_schema_matches_geometry():
    img = build_fat32_image()
    g = scan_geometries(__import__('io').BytesIO(bytes(img)), len(img))[0]
    b = schema.BPB.parse(bytes(img[:512]))
    assert b['bps'] == g.bps == 512
    assert b['spc'] == g.spc == 1
    assert b['reserved'] == g.reserved == 32
    assert b['num_fats'] == g.num_fats == 2
    assert b['fat_sects32'] == g.fat_sects == 1
    assert b['root_cluster'] == g.root_cluster == 2
    assert schema.variants(b) == 'fat32'
    assert b['signature'] == b'\x55\xaa'


def test_fat_specs_semantics():
    assert schema.fat_is_eoc(0x0FFFFFFF, 32)
    assert schema.fat_is_eoc(0x0FFFFFF8, 32)
    assert not schema.fat_is_eoc(0x0FFFFFF7, 32)
    assert schema.fat_is_bad(0x0FFFFFF7, 32)
    assert schema.fat_is_eoc(0xFFFF, 16) and schema.fat_is_bad(0xFFF7, 16)
    assert schema.fat_is_eoc(0x0FFF, 12) and schema.fat_is_bad(0x0FF7, 12)
    assert schema.fat_value(0xF1234567, 16) == 0x4567


def test_export_is_json_serialisable():
    out = json.loads(schema.dumps())
    assert set(out['structs']) >= {'BootSector', 'DirEntry', 'LFNEntry'}
    assert out['structs']['DirEntry']['size'] == 32
    assert out['fat_specs']['32']['eoc_min'] == 0x0FFFFFF8


def test_struct_coverage_is_documented():
    """Every schema declares what it covers; DIRENT spans its full 32 bytes."""
    cov = schema.DIRENT.coverage()
    assert cov[0][0] == 0 and cov[-1][1] == 32
    for s in schema.ALL_STRUCTS:
        for lo, hi in s.coverage():
            assert 0 <= lo < hi <= s.size, s.name
