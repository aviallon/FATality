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


def classify_offset(g, off: int) -> str:
    """Which filesystem structure a byte offset belongs to."""
    if off < g.part_off:
        return 'mbr'
    rel = off - g.part_off
    sector = rel // g.bps
    if sector < g.reserved:
        if sector == 0:
            return 'boot-sector'
        if sector == 1:
            return 'fsinfo'
        return f'reserved-sector-{sector}'
    fat_end = g.reserved + g.num_fats * g.fat_sects
    if sector < fat_end:
        return 'fat'
    if off < g.data_off:
        return 'fat12/16-root'
    clu = g.cluster_of(off)
    return f'cluster-{clu}' + ('-ROOT' if clu == 2 else '')


def damage(pre: Path, out: Path, seed: int, density: float = 0.002) -> dict:
    """Seeded random corruption of IMPORTANT structures (reproducible).

    Targets: boot sector, backup boot, FSInfo, FAT copies, the root cluster,
    plus a seeded scatter of data clusters.  Every damaged span is recorded
    with its structure label so the theoretical ceiling can be proven from
    the damage map alone.
    """
    import shutil

    import numpy as _np

    from fatality.geometry import scan_geometries

    rng = _np.random.default_rng(seed)
    with open(pre, 'rb') as f:
        g = scan_geometries(f, pre.stat().st_size)[0]
    shutil.copyfile(pre, out)
    events = []

    def hit(off: int, length: int, why: str):
        data = rng.integers(0, 256, length, dtype=_np.uint8).tobytes()
        with open(out, 'r+b') as fo:
            fo.seek(off)
            fo.write(data)
        events.append(dict(offset=off, length=length,
                           structure=classify_offset(g, off), reason=why))

    # targeted structure damage
    hit(g.part_off, 512, 'boot sector')
    hit(g.part_off + 6 * g.bps, 512, 'backup boot sector')
    hit(g.part_off + 1 * g.bps, 512, 'fsinfo')
    for i, fo in enumerate(g.fat_off):
        for _ in range(3):
            at = fo + int(rng.integers(0, g.fat_sects * g.bps - 4096))
            hit(at, 4096, f'fat copy {i} random span')
    hit(g.cluster_off(2), 512, 'root directory')

    # seeded scatter over data clusters (their bytes matter as much as
    # metadata: corrupting a file's clusters lowers its theoretical ceiling)
    n_scatter = max(1, int(g.cluster_count * density))
    for clu in rng.choice(_np.arange(3, g.cluster_count + 2), size=n_scatter,
                          replace=False):
        hit(g.cluster_off(int(clu)), 512, 'random data cluster')

    return dict(seed=seed, density=density, events=events,
                rule='targeted metadata (boot/fsinfo/fat/root) + seeded scatter')


def theoretical(scenario: Path, pre: Path, damaged: Path) -> dict:
    """The OMNISCIENT ceiling: what could be salvaged from the damaged image.

    Oracle = the pristine pre-damage image: its directory entries give each
    file's chain, its FAT gives the exact clusters, and comparing cluster
    bytes against the damaged image tells exactly which bytes still exist.
    No recovery technique can exceed this; the gap is the tool's fault, the
    ceiling is the damage's fault.
    """
    from fatality.fat import load_fats, reconstruct
    from fatality.geometry import scan_geometries
    from fatality.scan import scan_fragments, scan_slots

    with open(pre, 'rb') as f:
        g = scan_geometries(f, pre.stat().st_size)[0]
        fats = load_fats(f, g)
        end = g.part_off + (g.part_size or (pre.stat().st_size - g.part_off))
        frags = scan_fragments(f, g, scan_slots(f, g.data_off, end))
        pre_entries = {}
        for fr in frags:
            for s in fr.entries:
                if s.name in ('.', '..') or s.is_dir or s.cluster < 2:
                    continue
                pre_entries[s.display.lower()] = (s.cluster, s.size, fr.cluster, s.off)

        ceilings = []
        with open(damaged, 'rb') as fd:
            for t in load_manifest(scenario):
                if t.kind != 'file':
                    continue
                base = Path(t.path).name.lower()
                ent = pre_entries.get(base)
                if ent is None:
                    ceilings.append(dict(path=t.path, theoretical_bytes=0,
                                         size=t.size, cls='lost',
                                         note='no pristine directory entry'))
                    continue
                clu, size, dir_clu, slot_off = ent
                ch = reconstruct(g, fats, clu, size)
                # byte-precise ceiling: only bytes within [0, size) count
                intact_bytes = 0
                remaining = size
                for c in ch.clusters:
                    off = g.cluster_off(c)
                    f.seek(off)
                    a = f.read(g.cluster_bytes)
                    fd.seek(off)
                    b = fd.read(g.cluster_bytes)
                    take = min(remaining, len(a))
                    intact_bytes += sum(1 for i in range(take) if a[i] == b[i])
                    remaining -= take
                    if remaining <= 0:
                        break
                # name survival: the directory slot's cluster intact?
                f.seek(g.cluster_off(dir_clu))
                a = f.read(g.cluster_bytes)
                fd.seek(g.cluster_off(dir_clu))
                b = fd.read(g.cluster_bytes)
                name_ok = a == b
                cls = 'intact' if intact_bytes == size else (
                    'partial' if intact_bytes else 'lost')
                ceilings.append(dict(path=t.path, theoretical_bytes=intact_bytes,
                                     size=size, cls=cls,
                                     name_recoverable=bool(name_ok),
                                     chain_clusters=len(ch.clusters)))
    return dict(rule='oracle = pristine image chains; bytes compared per cluster',
                files=ceilings,
                totals=dict(theoretical=sum(c['theoretical_bytes'] for c in ceilings),
                            size=sum(c['size'] for c in ceilings)))


def composite(pre: Path, fmt: Path, out: Path) -> dict:
    """Composite a quick-formatted image over its pre-format snapshot.

    Windows' format may sanitize (UNMAP -> zero) the whole volume even in
    quick mode; real SD cards leave the data area untouched.  Healing rule:
    for every data cluster >= 3, where the formatted image is ALL ZERO and
    the pre-format snapshot holds data, fill from the snapshot.

    Metadata (reserved + FATs) and the root cluster are NEVER filled: their
    post-format zeros are the format's own semantics (cleared FATs/root),
    and resurrecting them would un-format the image.
    """
    import shutil

    from fatality.geometry import scan_geometries

    with open(fmt, 'rb') as f:
        g = scan_geometries(f, fmt.stat().st_size)[0]
    cb = g.cluster_bytes
    total_clusters = g.cluster_count
    shutil.copyfile(fmt, out)
    healed = 0
    healed_bytes = 0
    with open(pre, 'rb') as fp, open(out, 'r+b') as fo:
        for c in range(3, total_clusters + 2):
            off = g.cluster_off(c)
            fo.seek(off)
            cur = fo.read(cb)
            if len(cur) < cb or any(cur):
                continue                     # formatted image has real data
            fp.seek(off)
            old = fp.read(cb)
            if len(old) < cb or not any(old):
                continue                     # nothing to heal with
            fo.seek(off)
            fo.write(old)
            healed += 1
            healed_bytes += cb
    return dict(cluster_bytes=cb, total_clusters=total_clusters,
                healed_clusters=healed, healed_bytes=healed_bytes,
                rule='data clusters >= 3 only; metadata and root untouched')


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


def prefix_match(a: bytes, b: bytes) -> int:
    """Count of leading identical bytes — how much of a partial file is real."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def overlap_match(fragment: bytes, truth: bytes) -> int:
    """Bytes of `fragment` occurring verbatim at ONE position inside truth
    (a salvaged NAL run starts mid-file; prefix-match cannot see it)."""
    if len(fragment) < 64:
        return 0
    pos = truth.find(fragment[:256])
    if pos < 0:
        return 0
    n = 0
    while pos + n < len(truth) and n < len(fragment) and truth[pos + n] == fragment[n]:
        n += 1
    return n


def attr_bytes(rec: bytes, tb: bytes) -> int:
    return max(prefix_match(rec, tb), aligned_count(rec, tb))


def aligned_count(rec: bytes, tb: bytes) -> int:
    """Total bytes correct at their TRUE position (anchored by first 256B).

    The honest salvage metric: a file 99% correct with damage holes has far
    more real bytes than its longest matching prefix suggests.  Never counts
    across misalignment — the anchor must match first.
    """
    if len(rec) < 64:
        return 0
    pos = tb.find(rec[:256])
    if pos < 0:
        return 0
    seg = tb[pos:pos + len(rec)]
    return sum(1 for a, b in zip(seg, rec, strict=False) if a == b)


def score_extraction(manifest: list[Truth], extracted: Path,
                     named_index: dict[str, Path],
                     truth_bytes: dict[str, bytes] | None = None) -> RunResult:
    """Score one extraction output tree against the manifest.

    `named_index` maps recovered file names (case-insensitive) to paths,
    taken from the fatality manifest.json (dir entries + content-embedded
    names); the extraction tree itself is scanned for content matches.
    """
    # hash every recovered file once
    rec_hashes: dict[str, list[Path]] = {}
    rec_by_name: dict[str, list[Path]] = {}
    rec_files: list[tuple[Path, int, bytes]] = []     # (path, size, head 4K)
    if extracted.exists():
        for p in extracted.rglob('*'):
            if not p.is_file() or p.suffix in ('.json', '.md'):
                continue
            with open(p, 'rb') as fh:
                head = fh.read(4096)
                fh.seek(0)
                data = fh.read()
            rec_hashes.setdefault(sha256_bytes(data), []).append(p)
            rec_by_name.setdefault(p.name.lower(), []).append(p)
            rec_files.append((p, len(data), head))

    res = RunResult(scenario='', image=str(extracted))
    matched_files: set[Path] = set()
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
            matched_files.update(hits)
        else:
            tb = truth_bytes.get(t.path) if truth_bytes else None
            # candidate channels: name matches (incl. empty stubs from
            # directory extraction) AND content-fingerprint matches (carves
            # with generated names).  Take the BEST, never the first.
            cands = list(rec_by_name.get(base, []))
            if tb is not None:
                head = tb[:4096]
                cands += [p for p, sz, hd in rec_files
                          if p not in cands and len(hd) >= 64
                          and hd == head[:len(hd)] and 0 < sz <= 4 * t.size]
            if tb is not None:
                # mid-file fragment channel: a salvaged NAL run starts INSIDE
                # the file — attribute by in-file overlap, not by head match
                cands += [p for p, sz, hd in rec_files
                          if p not in cands and len(hd) >= 64
                          and tb.find(hd[:256]) >= 0 and sz >= 1024]
            if cands:
                matched_files.update(cands)
                if tb is not None:
                    got, best = max(((attr_bytes(p.read_bytes(), tb), p)
                                     for p in cands), key=lambda x: x[0])
                else:
                    best = max(cands, key=lambda p: p.stat().st_size)
                    got = best.stat().st_size
                if 0 < got < t.size:
                    s.cls = 'partial'
                    s.bytes_recovered = got
                    s.evidence = (f'best match {got}/{t.size} bytes '
                                  f'(position-correct): {best}')
                elif got >= t.size:
                    s.cls = 'partial'
                    s.bytes_recovered = t.size
                    s.evidence = f'content match beyond size: {best}'
                else:
                    s.cls = 'name-only'
                    s.evidence = f'name/fingerprint match, no usable bytes: {best}'
            elif s.name_found:
                s.cls = 'name-only'
                s.evidence = 'name recovered (entry/record), no content'
        res.scores.append(s)

    {id(s) for s in res.scores if s.cls != 'lost'}
    named_paths = {p for v in rec_by_name.values() for p in v}
    truth_names = {Path(t.path).name.lower() for t in manifest if t.kind == 'file'}
    # a file matched to ground truth (by hash or by name) is NOT a false
    # positive, even if it was recovered under a generated name
    fp = [p for p in named_paths
          if p not in matched_files and p.name.lower() not in truth_names]
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


def render_report(scenario: str, current: RunResult, comp: dict,
                  theoretical: dict | None = None) -> str:
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
    if theoretical:
        lines.append('')
        lines.append('## Retrieved vs theoretical ceiling')
        lines.append('| path | ceiling bytes | retrieved | efficiency |')
        lines.append('|---|---|---|---|')
        got_by = {s.path: s.bytes_recovered for s in current.scores}
        tot_ceil = tot_got = 0
        for f in theoretical['files']:
            ceil = f['theoretical_bytes']
            got = got_by.get(f['path'], 0)
            tot_ceil += ceil
            tot_got += got
            eff = (100.0 * got / ceil) if ceil else 0.0
            lines.append(f"| `{f['path']}` | {ceil} | {got} | {eff:.1f}% |")
        eff_all = (100.0 * tot_got / tot_ceil) if tot_ceil else 0.0
        lines.append('')
        lines.append(f'**Overall salvage efficiency: {tot_got}/{tot_ceil} bytes '
                     f'= {eff_all:.1f}% of the theoretical maximum.**')
        lines.append(f"(ceiling damaged away: "
                     f"{sum(f['size'] for f in theoretical['files']) - tot_ceil} bytes)")
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
    # ALWAYS a fresh extraction directory: stale outputs from earlier runs
    # silently match hashes/names and inflate scores (a blessed baseline was
    # once scored against files that predate the image being tested).
    import shutil
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
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
    # 3) score (with ground-truth bytes for precise partial measurement)
    manifest = load_manifest(sc)
    truth_bytes = {t.path: (sc / t.path).read_bytes()
                   for t in manifest
                   if t.kind == 'file' and (sc / t.path).exists()}
    res = score_extraction(manifest, out / 'extracted', named, truth_bytes)
    res.scenario = sc.name
    res.image = str(img)
    (out / 'results.json').write_text(json.dumps(asdict(res), indent=1))
    print(json.dumps(res.counts(), indent=1))
    print(f'false positives: {res.false_positives}')
    print(f'written {out / "results.json"}')
    return 0


def cmd_report(args) -> int:
    sc = Path(args.scenario)
    results_path = Path(args.results)
    if not results_path.exists():
        print(f'results not found: {results_path} (did the run step fail?)',
              file=sys.stderr)
        return 2
    res_raw = json.loads(results_path.read_text())
    res_raw['scores'] = [Score(**s) for s in res_raw['scores']]
    res = RunResult(**res_raw)
    th = None
    th_path = Path(args.theoretical) if args.theoretical else (
        results_path.parent / 'theoretical.json')
    if th_path.exists():
        th = json.loads(th_path.read_text())
    exp = sc / 'expected.json'
    if args.baseline:
        exp = Path(args.baseline)
    if not exp.exists():
        rep = (f'# Recovery report — {sc.name}\n\n**No blessed baseline yet.**\n\n'
               + render_report(sc.name, res, compare({}, res), th))
        Path(args.out).write_text(rep) if args.out else print(rep)
        print('baseline missing: report-only mode (no FAIL possible)',
              file=sys.stderr)
        return 0
    comp = compare(json.loads(exp.read_text()), res, tolerance=args.tolerance)
    rep = render_report(sc.name, res, comp, th)
    if args.out:
        Path(args.out).write_text(rep)
    print(rep)
    return 0 if comp['verdict'] == 'PASS' else 1


def cmd_damage(args) -> int:
    sc = Path(args.scenario)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stats = damage(Path(args.pre), outdir / 'image-damaged.vfat', args.seed,
                   density=args.density)
    (outdir / 'damage.json').write_text(json.dumps(stats, indent=1))
    th = theoretical(sc, Path(args.pre), outdir / 'image-damaged.vfat')
    (outdir / 'theoretical.json').write_text(json.dumps(th, indent=1))
    print(json.dumps(dict(seed=stats['seed'], events=len(stats['events']),
                          ceiling=th['totals']), indent=1))
    print('written', outdir / 'damage.json', 'and', outdir / 'theoretical.json')
    return 0


def cmd_theoretical(args) -> int:
    th = theoretical(Path(args.scenario), Path(args.pre), Path(args.damaged))
    if args.out:
        Path(args.out).write_text(json.dumps(th, indent=1))
    print(json.dumps(th['totals'], indent=1))
    return 0


def cmd_compose(args) -> int:
    stats = composite(Path(args.pre), Path(args.formatted), Path(args.out))
    print(json.dumps(stats, indent=1))
    return 0


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

    p = sub.add_parser('damage')
    p.add_argument('scenario')
    p.add_argument('pre', help='pristine pre-damage image (image-prefmt.vfat)')
    p.add_argument('-o', '--out', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--density', type=float, default=0.002)
    p.set_defaults(fn=cmd_damage)

    p = sub.add_parser('theoretical')
    p.add_argument('scenario')
    p.add_argument('pre')
    p.add_argument('damaged')
    p.add_argument('-o', '--out')
    p.set_defaults(fn=cmd_theoretical)

    p = sub.add_parser('composite')
    p.add_argument('pre', help='pre-format snapshot (image-prefmt.vfat)')
    p.add_argument('formatted', help='quick-formatted image (image.vfat)')
    p.add_argument('out', help='healed output image')
    p.set_defaults(fn=cmd_compose)

    p = sub.add_parser('report')
    p.add_argument('scenario')
    p.add_argument('results')
    p.add_argument('-o', '--out')
    p.add_argument('--tolerance', type=float, default=0.0)
    p.add_argument('--baseline', help='alternate expected.json (variants)')
    p.add_argument('--theoretical', help='theoretical.json (ceiling report)')
    p.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
