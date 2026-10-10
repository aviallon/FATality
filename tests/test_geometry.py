"""Geometry parsing + quick-format survival tests."""
from fatality.geometry import scan_geometries
from fatality.scan import scan_fragments, scan_slots


def _geo(path):
    with open(path, 'rb') as f:
        geos = scan_geometries(f, path.stat().st_size)
    return geos


def test_geometry_parsed(fat32_image):
    geos = _geo(fat32_image)
    assert geos, 'no geometry found'
    g = geos[0]
    assert g.bps == 512
    assert g.spc == 1
    assert g.fat_bits == 32
    assert g.data_off == (32 + 2 * 1) * 512
    assert g.cluster_count > 0


def test_quickformat_zero_fats_detected(quickformatted_image):
    from fatality.fat import load_fats, merge_fats
    geos = _geo(quickformatted_image)
    g = geos[0]
    with open(quickformatted_image, 'rb') as f:
        fats = load_fats(f, g)
        assert len(fats) == 2
        assert merge_fats(fats) == {}, 'zeroed FAT must have no consensus chains'


def test_names_survive_quickformat(quickformatted_image):
    """The core promise: names and data survive a FAT-zeroing quick format."""
    geos = _geo(quickformatted_image)
    g = geos[0]
    with open(quickformatted_image, 'rb') as f:
        end = g.part_off + (g.part_size or (quickformatted_image.stat().st_size - g.part_off))
        slots = scan_slots(f, g.data_off, end)
        frags = scan_fragments(f, g, slots)
    names = {n for fr in frags for n in fr.names}
    assert 'DCIM' in names
    assert any(n.lower().startswith('hello fatality') for n in names)
    # deleted photo with LFN evidence still names itself
    all_displays = {s.display for fr in frags for s in fr.entries}
    assert any('GOPR0001' in d for d in all_displays)
