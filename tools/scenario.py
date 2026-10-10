#!/usr/bin/env python3
"""Scenario harness: score recovery runs against ground truth, diff baselines.

A scenario is a directory::

    scenarios/<name>/
        groundtruth/     files copied into the image BEFORE the quick format
        postformat/      files copied AFTER it (the 'someone used the card'
                         step that clobbers part of the old data)
        manifest.json    exact ground truth: path, size, sha256, class
        recipe.ps1       Windows image build (the only sanctioned builder)
        expected.json    blessed baseline of recovery scores (what 'we usually
                         recover'); absent = report-only mode

Commands:
    manifest SCENARIO            regenerate manifest.json from groundtruth/
    run SCENARIO IMAGE           run fatality extraction, score vs manifest
    report SCENARIO RESULTS...   delta report vs expected.json (exit 1 on
                                 negative deltas beyond tolerance)

Scoring per ground-truth file (the recovery classes that matter):
    intact      all bytes recovered, content hash matches
    partial     some bytes recovered (fraction reported)
    name-only   name survived somewhere, no usable content
    lost        nothing
plus a false-positive count (recovered items matching no ground truth).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

CLASSES = ('intact', 'partial', 'name-only', 'lost')


@dataclass
class Truth:
    path: str
    size: int
    sha256: str
    kind: str = 'file'              # file | dir
    after_format: bool = False      # lives in postformat/ (not expected lost)
    note: str = ''


@dataclass
class Score:
    path: str
    cls: str                        # one of CLASSES
    bytes_recovered: int = 0
    size: int = 0
    sha_match: bool = False
    name_found: bool = False
    evidence: str = ''


@dataclass
class RunResult:
    scenario: str
    image: str
    scores: list[Score] = field(default_factory=list)
    false_positives: int = 0
    extra_named: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        c = {k: 0 for k in CLASSES}
        for s in self.scores:
            c[s.cls] = c.get(s.cls, 0) + 1
        return c


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_manifest(scenario: Path) -> list[Truth]:
    """Manifest from the groundtruth/ and postformat/ trees."""
    out: list[Truth] = []
    for sub, after in (('groundtruth', False), ('postformat', True)):
        root = scenario / sub
        if not root.exists():
            continue
        for p in sorted(root.rglob('*')):
            rel = p.relative_to(root).as_posix()
            if p.is_dir():
                out.append(Truth(path=f'{sub}/{rel}', size=0, sha256='',
                                 kind='dir', after_format=after))
            else:
                data = p.read_bytes()
                out.append(Truth(path=f'{sub}/{rel}', size=len(data),
                                 sha256=sha256_bytes(data), kind='file',
                                 after_format=after))
    return out


def load_manifest(scenario: Path) -> list[Truth]:
    raw = json.loads((scenario / 'manifest.json').read_text())
    return [Truth(**t) for t in raw['files']]


def save_manifest(scenario: Path, truths: list[Truth]) -> None:
    (scenario / 'manifest.json').write_text(json.dumps(
        dict(files=[asdict(t) for t in truths]), indent=1))


def score_extraction(manifest: list[Truth], extracted: Path,
                     named_index: dict[str, Path]) -> RunResult:
    """Score one extraction output tree against the manifest.

    `named_index` maps recovered file names (case-insensitive) to paths,
    taken from the fatality manifest.json (dir entries + content-embedded
    names); the extraction tree itself is scanned for content matches.
    """
    # hash every recovered file once
    rec_hashes: dict[str, list[Path]] = {}
    rec_by_name: dict[str, list[Path]] = {}
    if extracted.exists():
        for p in extracted.rglob('*'):
            if not p.is_file() or p.suffix in ('.json', '.md'):
                continue
            h = sha256_bytes(p.read_bytes())
            rec_hashes.setdefault(h, []).append(p)
            rec_by_name.setdefault(p.name.lower(), []).append(p)

    res = RunResult(scenario='', image=str(extracted))
    for t in manifest:
        if t.kind == 'dir':
            continue
        s = Score(path=t.path, cls='lost', size=t.size)
        hits = rec_hashes.get(t.sha256, [])
        base = Path(t.path).name.lower()
        s.name_found = bool(named_index.get(base) or rec_by_name.get(base))
        if hits:
            s.cls = 'intact'
            s.sha_match = True
            s.bytes_recovered = t.size
            s.evidence = f'content match: {hits[0]}'
        else:
            # partial: same name recovered but content differs
            cands = rec_by_name.get(base, [])
            if cands:
                best = max(cands, key=lambda p: p.stat().st_size)
                got = best.stat().st_size
                if 0 < got < t.size:
                    s.cls = 'partial'
                    s.bytes_recovered = got
                    s.evidence = f'name match, {got}/{t.size} bytes: {best}'
                elif got >= t.size:
                    s.cls = 'partial'
                    s.bytes_recovered = t.size
                    s.evidence = f'name match, content differs: {best}'
                else:
                    s.cls = 'name-only'
                    s.evidence = f'name match, empty: {best}'
            elif s.name_found:
                s.cls = 'name-only'
                s.evidence = 'name recovered (entry/record), no content'
        res.scores.append(s)

    {id(s) for s in res.scores if s.cls != 'lost'}
    named_paths = {p for v in rec_by_name.values() for p in v}
    truth_names = {Path(t.path).name.lower() for t in manifest if t.kind == 'file'}
    fp = [p for p in named_paths if p.name.lower() not in truth_names]
    res.false_positives = len(fp)
    res.extra_named = sorted(str(p) for p in fp)[:50]
    return res


def compare(baseline: dict, current: RunResult, tolerance: float = 0.0) -> dict:
    """Per-file and per-class deltas vs the blessed baseline."""
    base_scores = {s['path']: s for s in baseline.get('scores', [])}
    deltas = []
    regressions = 0
    improvements = 0
    for s in current.scores:
        b = base_scores.get(s.path)
        bcls = b['cls'] if b else 'lost'
        if b is None:
            deltas.append(dict(path=s.path, change='NEW', base=None, cur=s.cls,
                               bytes=f"{s.bytes_recovered}/{s.size}"))
            continue
        if s.cls == bcls and s.bytes_recovered == b.get('bytes_recovered', 0):
            continue
        rank = {c: i for i, c in enumerate(reversed(CLASSES))}   # lost=0..intact=3
        direction = 'IMPROVED' if rank[s.cls] > rank[bcls] else 'REGRESSED'
        if direction == 'REGRESSED':
            regressions += 1
        else:
            improvements += 1
        deltas.append(dict(path=s.path, change=direction, base=bcls, cur=s.cls,
                           bytes=f"{b.get('bytes_recovered', 0)}->{s.bytes_recovered}/{s.size}"))
    cls_delta = {}
    bcounts = baseline.get('counts', {})
    for k in CLASSES:
        cls_delta[k] = current.counts().get(k, 0) - bcounts.get(k, 0)
    return dict(deltas=deltas, regressions=regressions, improvements=improvements,
                class_delta=cls_delta,
                false_positives=(current.false_positives
                                 - baseline.get('false_positives', 0)),
                verdict='FAIL' if regressions > tolerance else 'PASS')


def render_report(scenario: str, current: RunResult, comp: dict) -> str:
    """Markdown report for GitHub's step summary."""
    lines = [f'# Recovery regression report — {scenario}', '']
    c = current.counts()
    lines.append('## Current scores')
    lines.append('| class | count |')
    lines.append('|---|---|')
    for k in CLASSES:
        d = comp['class_delta'].get(k, 0)
        arrow = f' ({d:+d})' if d else ''
        lines.append(f'| {k} | {c.get(k, 0)}{arrow} |')
    lines.append(f"| false positives | {current.false_positives} "
                 f"({comp['false_positives']:+d}) |")
    lines.append('')
    lines.append(f"## Verdict: **{comp['verdict']}** "
                 f"(regressions: {comp['regressions']}, "
                 f"improvements: {comp['improvements']})")
    lines.append('')
    if comp['deltas']:
        lines.append('## Deltas vs baseline')
        lines.append('| direction | path | baseline | now | bytes |')
        lines.append('|---|---|---|---|---|')
        for d in comp['deltas']:
            lines.append(f"| {d['change']} | `{d['path']}` | {d['base']} | "
                         f"{d['cur']} | {d['bytes']} |")
    else:
        lines.append('No per-file differences vs baseline.')
    return '\n'.join(lines) + '\n'


def cmd_manifest(args) -> int:
    sc = Path(args.scenario)
    ts = build_manifest(sc)
    save_manifest(sc, ts)
    print(f'manifest.json: {len(ts)} entries')
    return 0


def cmd_run(args) -> int:
    sc, img = Path(args.scenario), Path(args.image)
    out = Path(args.out or (sc / 'last-run'))
    out.mkdir(parents=True, exist_ok=True)
    if img.suffix == '.gz':
        import gzip
        raw = out / 'image.raw'
        raw.write_bytes(gzip.decompress(img.read_bytes()))
        img = raw
    # 1) extraction through the real tool
    subprocess.run([sys.executable, '-m', 'fatality.cli', 'extract', str(img),
                    '-o', str(out / 'extracted')], check=True)
    # 2) name index from fatality's own manifest
    fat_manifest = json.loads((out / 'extracted' / 'manifest.json').read_text())
    named = {}
    for it in fat_manifest['items']:
        if it.get('name'):
            named[it['name'].lower()] = Path(it['path'])
    # 3) score
    res = score_extraction(load_manifest(sc), out / 'extracted', named)
    res.scenario = sc.name
    res.image = str(img)
    (out / 'results.json').write_text(json.dumps(asdict(res), indent=1))
    print(json.dumps(res.counts(), indent=1))
    print(f'false positives: {res.false_positives}')
    print(f'written {out / "results.json"}')
    return 0


def cmd_report(args) -> int:
    sc = Path(args.scenario)
    res = RunResult(**json.loads(Path(args.results).read_text()))
    exp = sc / 'expected.json'
    if not exp.exists():
        rep = (f'# Recovery report — {sc.name}\n\n**No blessed baseline yet.**\n\n'
               + render_report(sc.name, res, compare({}, res)))
        Path(args.out).write_text(rep) if args.out else print(rep)
        print('baseline missing: report-only mode (no FAIL possible)',
              file=sys.stderr)
        return 0
    comp = compare(json.loads(exp.read_text()), res, tolerance=args.tolerance)
    rep = render_report(sc.name, res, comp)
    if args.out:
        Path(args.out).write_text(rep)
    print(rep)
    return 0 if comp['verdict'] == 'PASS' else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='scenario')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('manifest')
    p.add_argument('scenario')
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser('run')
    p.add_argument('scenario')
    p.add_argument('image')
    p.add_argument('-o', '--out')
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser('report')
    p.add_argument('scenario')
    p.add_argument('results')
    p.add_argument('-o', '--out')
    p.add_argument('--tolerance', type=float, default=0.0)
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
