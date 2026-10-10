"""FAT parsing + chain reconstruction evidence labels."""
from fatality.fat import contiguous_chain, load_fats, merge_fats, reconstruct
from fatality.geometry import scan_geometries


def _geo(path):
    with open(path, 'rb') as f:
        return scan_geometries(f, path.stat().st_size)[0]


def test_live_fat_walks_chain(fat32_image):
    g = _geo(fat32_image)
    with open(fat32_image, 'rb') as f:
        fats = load_fats(f, g)
        ch = fats[0].walk(2)
    assert ch.clusters == [2]
    assert ch.evidence == 'fat'
    assert not ch.truncated


def test_zeroed_fat_falls_back_to_contiguity(quickformatted_image):
    g = _geo(quickformatted_image)
    with open(quickformatted_image, 'rb') as f:
        fats = load_fats(f, g)
        ch = reconstruct(g, fats, 4, 16)
    assert ch.evidence == 'contig'
    assert ch.clusters == [4]
    assert ch.notes


def test_multi_cluster_contiguity_claim():
    from fatality.geometry import Geometry
    g = Geometry(source='t', part_off=0, part_size=1 << 20, bps=512, spc=1,
                 reserved=32, num_fats=2, fat_sects=1, root_entries=0)
    ch = contiguous_chain(g, 10, 1200)      # 1200 bytes = 3 clusters of 512
    assert ch.clusters == [10, 11, 12]
    assert ch.evidence == 'contig'


def test_consensus_needs_agreement(fat32_image):
    g = _geo(fat32_image)
    with open(fat32_image, 'rb') as f:
        fats = load_fats(f, g)
    # identical copies agree on the EOC markers we wrote
    assert 2 not in merge_fats(fats)      # EOC entries are not chain links
