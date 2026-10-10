"""Scenario harness: scoring classes, baseline deltas, regression gate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))

from scenario import (
    RunResult,
    Score,
    build_manifest,
    compare,
    load_manifest,
    render_report,
    save_manifest,
    score_extraction,
)


def _tree(tmp_path):
    gt = tmp_path / 'groundtruth'
    (gt / 'DCIM').mkdir(parents=True)
    (gt / 'DCIM' / 'A.JPG').write_bytes(b'JPEGDATA-A')
    (gt / 'B.TXT').write_bytes(b'hello')
    (gt / 'C.BIN').write_bytes(b'will be lost')
    pf = tmp_path / 'postformat'
    pf.mkdir()
    (pf / 'NEW.TXT').write_bytes(b'written after format')
    return gt


def test_build_manifest(tmp_path):
    _tree(tmp_path)
    ts = build_manifest(tmp_path)
    files = {t.path: t for t in ts if t.kind == 'file'}
    assert set(files) == {'groundtruth/DCIM/A.JPG', 'groundtruth/B.TXT',
                          'groundtruth/C.BIN', 'postformat/NEW.TXT'}
    assert files['groundtruth/B.TXT'].size == 5
    assert files['postformat/NEW.TXT'].after_format
    save_manifest(tmp_path, ts)
    assert load_manifest(tmp_path)[0].path == ts[0].path


def test_scoring_classes(tmp_path):
    _tree(tmp_path)
    ts = build_manifest(tmp_path)
    ex = tmp_path / 'extracted'
    (ex / 'x').mkdir(parents=True)
    (ex / 'x' / 'A.JPG').write_bytes(b'JPEGDATA-A')      # intact (hash match)
    (ex / 'x' / 'B.TXT').write_bytes(b'hel')             # partial (name, short)
    (ex / 'x' / 'EXTRA.JPG').write_bytes(b'junk')        # false positive
    res = score_extraction(ts, ex, {})
    by = {s.path: s for s in res.scores}
    assert by['groundtruth/DCIM/A.JPG'].cls == 'intact' and by['groundtruth/DCIM/A.JPG'].sha_match
    assert by['groundtruth/B.TXT'].cls == 'partial'
    assert by['groundtruth/B.TXT'].bytes_recovered == 3
    assert by['groundtruth/C.BIN'].cls == 'lost'
    assert res.false_positives == 1
    assert res.counts()['intact'] == 1


def test_name_only_via_named_index(tmp_path):
    _tree(tmp_path)
    ts = build_manifest(tmp_path)
    ex = tmp_path / 'extracted'
    ex.mkdir()
    res = score_extraction(ts, ex, {'c.bin': Path('/nowhere/C.BIN')})
    by = {s.path: s for s in res.scores}
    assert by['groundtruth/C.BIN'].cls == 'name-only'
    assert by['groundtruth/C.BIN'].name_found


def test_compare_regression_fails():
    base = dict(counts={'intact': 2, 'partial': 0, 'name-only': 0, 'lost': 0},
                false_positives=0,
                scores=[dict(path='A', cls='intact', bytes_recovered=10),
                        dict(path='B', cls='intact', bytes_recovered=10)])
    cur = RunResult(scenario='s', image='i', false_positives=1, scores=[
        Score(path='A', cls='intact', bytes_recovered=10, size=10),
        Score(path='B', cls='partial', bytes_recovered=4, size=10),
    ])
    comp = compare(base, cur)
    assert comp['verdict'] == 'FAIL'
    assert comp['regressions'] == 1
    assert comp['false_positives'] == 1
    rep = render_report('s', cur, comp)
    assert 'REGRESSED' in rep and 'FAIL' in rep


def test_compare_improvement_passes():
    base = dict(counts={'intact': 1, 'partial': 1, 'name-only': 0, 'lost': 0},
                false_positives=2,
                scores=[dict(path='A', cls='partial', bytes_recovered=4),
                        dict(path='B', cls='intact', bytes_recovered=10)])
    cur = RunResult(scenario='s', image='i', false_positives=0, scores=[
        Score(path='A', cls='intact', bytes_recovered=10, size=10),
        Score(path='B', cls='intact', bytes_recovered=10, size=10),
        Score(path='C', cls='partial', bytes_recovered=3, size=9),
    ])
    comp = compare(base, cur)
    assert comp['verdict'] == 'PASS'
    assert comp['improvements'] == 1
    assert comp['class_delta']['intact'] == 1
    kinds = {d['change'] for d in comp['deltas']}
    assert 'IMPROVED' in kinds and 'NEW' in kinds


def test_results_json_roundtrip(tmp_path):
    """Regression: cmd_report deserializes scores as Score objects, not dicts."""
    import json as _json
    from dataclasses import asdict

    from scenario import cmd_report
    cur = RunResult(scenario='s', image='i', scores=[
        Score(path='A', cls='intact', bytes_recovered=10, size=10)])
    results = tmp_path / 'results.json'
    results.write_text(_json.dumps(asdict(cur)))
    sc = tmp_path / 'scenario'
    sc.mkdir()
    args = type('Args', (), dict(scenario=str(sc), results=str(results),
                                 out=None, tolerance=0.0,
                                 baseline=None, theoretical=None))
    assert cmd_report(args) == 0          # no baseline -> report-only


def test_composite_heals_data_but_not_metadata(tmp_path):
    """The user's layering idea, with the essential guard: zero-holes in
    DATA clusters are healed from the pre-format snapshot, but FATs and the
    root cluster keep their post-format zeros (their zeros ARE the format's
    semantics — filling them would un-format the image)."""
    from conftest import DATA_OFF, build_fat32_image
    from scenario import composite
    pre = bytearray(build_fat32_image())          # GT state (dirs + files)
    fmt = bytearray(pre)                          # fake 'formatted+sanitize'
    fmt[3 * 512:4 * 512] = bytes(512)             # zero FAT2 tail / keep shape
    # sanitize: wipe the ENTIRE data area incl. root, then write one new
    # cluster (postformat usage)
    fmt[DATA_OFF:] = bytes(len(fmt) - DATA_OFF)
    fmt[DATA_OFF + 2 * 512:DATA_OFF + 3 * 512] = b'NEW-DATA' + bytes(504)
    p_pre, p_fmt = tmp_path / 'pre.img', tmp_path / 'fmt.img'
    p_pre.write_bytes(pre)
    p_fmt.write_bytes(fmt)
    stats = composite(p_pre, p_fmt, tmp_path / 'out.img')
    out = (tmp_path / 'out.img').read_bytes()
    assert stats['healed_clusters'] > 0
    # data healed: GT payload is back (cluster 5 = 'hello fatality')
    # and so is a subdirectory listing (clusters >= 3 are fair game)
    assert b'hello fatality' in out[DATA_OFF:]
    assert b'OPR0001' in out[DATA_OFF:]      # clu3: deleted-entry name remnant
    # postformat cluster untouched
    assert out[DATA_OFF + 2 * 512:DATA_OFF + 2 * 512 + 8] == b'NEW-DATA'
    # root cluster (2) keeps its formatted zeros: NOT healed
    assert out[DATA_OFF:DATA_OFF + 32] == bytes(32)


def test_best_candidate_attribution(tmp_path):
    """Regression: an empty dir-entry stub with the right NAME must not
    shadow a content-matching carve with a generated name — the scorer
    takes the BEST candidate across name and fingerprint channels."""
    gt = tmp_path / 'groundtruth'
    gt.mkdir()
    truth = b'\xff\xd8\xff' + b'A' * 5000 + b'\xff\xd9'
    (gt / 'PHOTO.JPG').write_bytes(truth)
    ts = build_manifest(tmp_path)
    ex = tmp_path / 'extracted'
    (ex / 'tree').mkdir(parents=True)
    (ex / 'tree' / 'PHOTO.JPG').write_bytes(b'')          # empty stub, right name
    (ex / 'carved').mkdir()
    (ex / 'carved' / 'jpeg_off999').write_bytes(truth[:4000])  # partial carve
    res = score_extraction(ts, ex, {},
                           {'groundtruth/PHOTO.JPG': truth})
    s = res.scores[0]
    assert s.cls == 'partial'
    assert s.bytes_recovered == 4000, s.evidence
