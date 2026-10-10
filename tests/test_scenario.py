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
