"""FATality command line.

  fatality scan    IMAGE -o OUT     index geometry/FAT/dirs/signatures
  fatality extract IMAGE -o OUT     recover into generation-split hierarchies
  fatality verify  OUT              decode-level validation of an extraction
  fatality report  OUT              (re)render manifest + graveyard

Output layout of `extract` (conservative split policy):
  OUT/<tree-label>/...        one hierarchy per linked fragment component
  OUT/orphans/clu<N>/...      unlinked fragments, kept separate
  OUT/carved/<handler>/...    content carves without tree evidence
  OUT/manifest.json           provenance, status, damage, validation
  OUT/graveyard.md            names whose data is gone
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import carve
from .fat import load_fats, reconstruct
from .geometry import scan_geometries
from .report import Item, Manifest
from .scan import scan_fragments, scan_signatures, scan_slots
from .trees import link_fragments, tree_label

SIGS = {
    'jpeg': b'\xff\xd8\xff',
    'mp4_ftyp': b'ftyp',
    'moov': b'moov',
    'gopro_record': b'DCIM\\',
}

EXT_HANDLER = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.thm': 'jpeg',
               '.mp4': 'mp4', '.lrv': 'mp4', '.mov': 'mp4'}


def cmd_scan(args) -> int:
    img = Path(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = img.stat().st_size
    with open(img, 'rb') as f:
        geos = scan_geometries(f, size)
        print(f'geometries: {len(geos)}')
        report = dict(image=str(img), size=size,
                      geometries=[g.summary() for g in geos])
        g = geos[0]
        print(f'primary: {g.summary()}')

        fats = load_fats(f, g)
        from .fat import merge_fats
        consensus = merge_fats(fats)
        print(f'FAT copies: {len(fats)}, live chain entries (consensus): {len(consensus)}')
        report['fat_entries'] = len(consensus)

        end = g.part_off + (g.part_size or (size - g.part_off))
        slots = scan_slots(f, g.data_off, end)
        frags = scan_fragments(f, g, slots)
        print(f'directory slots (plausible): {len(slots)}')
        print(f'directory fragments: {len(frags)}')
        report['slots'] = len(slots)
        report['fragments'] = [dict(cluster=fr.cluster, entries=len(fr.entries),
                                    names=fr.names[:20]) for fr in frags]

        hits = scan_signatures(f, g.data_off, end, SIGS)
        report['signatures'] = {k: len(v) for k, v in hits.items()}
        print(f'signatures: {report["signatures"]}')
        (out / 'scan.json').write_text(json.dumps(report, indent=1))
        print(f'written {out / "scan.json"}')
    return 0


def _safe(name: str) -> str:
    return name.replace('/', '_').replace('\\', '_').replace('\x00', '') or 'unnamed'


def cmd_extract(args) -> int:
    img = Path(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = img.stat().st_size
    man = Manifest(image=str(img))
    counts = dict(phantom=0, degraded=0)

    with open(img, 'rb') as f:
        geos = scan_geometries(f, size)
        man.geometries = [g.summary() for g in geos]
        g = geos[0]
        end = g.part_off + (g.part_size or (size - g.part_off))
        fats = load_fats(f, g)

        # ---- 1) directory evidence, organised into linked trees ----
        slots = scan_slots(f, g.data_off, end)
        frags = scan_fragments(f, g, slots)
        comps = link_fragments(frags, g.cluster_count)
        for comp in comps.values():
            label = tree_label(comp)
            tree_root = out / label if len(comp) > 1 else (out / 'orphans' / label)
            for fr in comp:
                sub = tree_root / f'clu{fr.cluster}'
                for s in fr.entries:
                    if s.name in ('.', '..') or s.is_dir or s.kind != 'entry':
                        continue
                    ev = [f'dir-fragment clu{fr.cluster}', f'tree:{label}']
                    if s.cluster >= 2 and s.size > 0:
                        ch = reconstruct(g, fats, s.cluster, s.size)
                        ev.append(f'chain:{ch.evidence}')
                        f.seek(g.cluster_off(s.cluster))
                        data = f.read(min(s.size, ch.length * g.cluster_bytes))
                        path = sub / _safe(s.display)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                        status = 'complete' if len(data) == s.size else 'partial'
                        validated = 'unvalidated'
                        hname = EXT_HANDLER.get(Path(s.display).suffix.lower())
                        if hname:
                            v = carve.by_name(hname).validate(str(path))
                            validated = f'{"ok" if v.ok else "fail"}: {v.detail}'
                            if status == 'complete' and not v.ok:
                                status = 'partial'
                                counts['degraded'] += 1
                                ev.append('decode-validation failed: downgraded')
                        man.items.append(Item(path=str(path), status=status,
                                              source='dir-entry', size=len(data),
                                              name=s.display, datetime=s.wr_date,
                                              evidence=ev, generation=label,
                                              validated=validated))
                    else:
                        man.items.append(Item(path='', status='destroyed',
                                              source='dir-entry', name=s.display,
                                              datetime=s.wr_date,
                                              evidence=ev + ['no data pointer/size'],
                                              generation=label))

        # ---- 2) handler carving on signature hits (decode-gated) ----
        hits = scan_signatures(f, g.data_off, end, SIGS)
        for h in carve.handlers():
            key = {'jpeg': 'jpeg', 'mp4': 'mp4_ftyp'}.get(h.name)
            if key is None:
                continue
            for off in hits.get(key, [])[:args.max_carve]:
                res = h.extract(f, off)
                if res.status == 'phantom':
                    counts['phantom'] += 1
                    continue
                if res.status not in ('complete', 'partial') or 'bytes' not in res.extras:
                    if res.name:
                        man.ghosts.append(dict(name=res.name, kind=h.name,
                                               datetime=res.datetime,
                                               survived='identity only'))
                    continue
                name = res.name or f'{h.name}_off{off:013d}'
                path = out / 'carved' / h.name / _safe(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(res.extras['bytes'])
                v = h.validate(str(path))
                status = res.status
                if status == 'complete' and not v.ok:
                    status = 'partial'
                    counts['degraded'] += 1
                    res.notes.append('decode-validation failed: downgraded')
                man.items.append(Item(path=str(path), status=status,
                                      source=f'carve:{h.name}', offset=off,
                                      size=res.size, name=res.name,
                                      datetime=res.datetime,
                                      evidence=res.notes, damage=res.damage,
                                      generation='carved',
                                      validated=f'{"ok" if v.ok else "fail"}: {v.detail}'))

        # ---- 3) GoPro media-counter inference -> named ghosts ----
        from .carve.gopro import counter_inference
        names = [i.name for i in man.items if i.name] + [gh['name'] for gh in man.ghosts]
        ci = counter_inference([n for n in names if n])
        for num in ci['gaps']:
            man.ghosts.append(dict(name=f'counter-gap-{num:04d}',
                                   kind='gopro-media-counter', datetime=None,
                                   survived='numbering gap'))

        man.save(out)
        (out / 'graveyard.md').write_text(man.graveyard_md())
        summary = man.summary()
        summary.update(counts)
        print(json.dumps(summary, indent=1))
        print(f'written {out}/manifest.json and graveyard.md')
    return 0


def cmd_verify(args) -> int:
    out = Path(args.out)
    man = json.loads((out / 'manifest.json').read_text())
    fails = 0
    for it in man['items']:
        if it['status'] not in ('complete', 'partial') or not it['path']:
            continue
        hname = it['source'].split(':', 1)[1] if it['source'].startswith('carve:') else None
        if not hname:
            hname = EXT_HANDLER.get(Path(it['path']).suffix.lower())
        if not hname:
            continue
        v = carve.by_name(hname).validate(it['path'])
        print(f'{"OK " if v.ok else "FAIL"} {it["path"]}: {v.detail}')
        fails += 0 if v.ok else 1
    print(f'failures: {fails}')
    return 1 if fails else 0


def cmd_report(args) -> int:
    out = Path(args.out)
    man = json.loads((out / 'manifest.json').read_text())
    (out / 'graveyard.md').write_text(
        Manifest(image=man['image'], items=[], ghosts=man['ghosts']).graveyard_md())
    print('graveyard.md re-rendered')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='fatality')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('scan'); p.add_argument('image'); p.add_argument('-o', '--out', default='fatality-out')
    p.set_defaults(fn=cmd_scan)
    p = sub.add_parser('extract'); p.add_argument('image'); p.add_argument('-o', '--out', default='fatality-out')
    p.add_argument('--max-carve', type=int, default=10_000)
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser('verify'); p.add_argument('out'); p.set_defaults(fn=cmd_verify)
    p = sub.add_parser('report'); p.add_argument('out'); p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
