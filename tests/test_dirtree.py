"""Directory slot parsing, LFN reassembly, fragment linking."""
from conftest import lfn_entries, short_entry

from fatality.dirtree import is_fragment, lfn_checksum, merge_lfns, parse_slot
from fatality.trees import link_fragments, tree_label


def test_short_entry_roundtrip():
    e = short_entry(b'GOPR3655', b'MP4', 272, 1256672901)
    s = parse_slot(e, 0)
    assert s is not None
    assert s.name == 'GOPR3655.MP4'
    assert s.cluster == 272
    assert s.size == 1256672901
    assert s.wr_date == '2024-08-02'


def test_deleted_entry_parsed_with_lfn_name():
    lfn = lfn_entries('GOPR0001.JPG', lfn_checksum(b'GOPR0001JPG'))
    lfn = [bytes([0xE5]) + p[1:] for p in lfn]
    e = short_entry(b'GOPR0001', b'JPG', 4, 16, deleted=True)
    slots = [parse_slot(x, i * 32) for i, x in enumerate(lfn + [e])]
    merged = merge_lfns([s for s in slots if s])
    assert merged[-1].deleted
    assert merged[-1].display == 'GOPR0001.JPG'


def test_lfn_checksum_validation_rejects_mismatch():
    lfn = lfn_entries('HELLO.TXT', checksum=0x42)   # wrong checksum
    e = short_entry(b'HELLO', b'TXT', 5, 11)
    slots = [s for s in (parse_slot(x, i * 32) for i, x in enumerate(lfn + [e])) if s]
    merged = merge_lfns(slots)
    assert merged[-1].long_name is None


def test_adjacency_discriminates_fragments():
    a = parse_slot(short_entry(b'FILE', b'TXT', 5, 11), 0)
    b = parse_slot(short_entry(b'FILE2', b'TXT', 6, 11), 32)
    far = parse_slot(short_entry(b'FILE3', b'TXT', 7, 11), 4096)
    assert is_fragment([a, b])               # adjacent pair
    assert not is_fragment([a, far])         # scattered singletons = media data


def test_fragment_linking_by_entry_claim():
    from fatality.dirtree import DirFragment
    root_s = parse_slot(short_entry(b'MISC', b'   ', 4, 0, attr=0x10), 0)
    root_s.is_dir = True
    frag2 = DirFragment(cluster=2, off=0, slots=[root_s])
    sub = DirFragment(cluster=4, off=0, slots=[
        parse_slot(short_entry(b'.', b'   ', 4, 0, attr=0x10), 0),
        parse_slot(short_entry(b'..', b'   ', 0, 0, attr=0x10), 32),
    ])
    comps = link_fragments([frag2, sub], max_cluster=100)
    multi = [c for c in comps.values() if len(c) > 1]
    assert len(multi) == 1
    assert tree_label(multi[0]) == 'root_clu2'
