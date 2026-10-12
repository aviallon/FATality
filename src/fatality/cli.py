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
from .carve.h264_media import MediaRun, media_runs
from .carve.h264_media import annexb as nal_annexb
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


def _apply_datetime(path: Path, dt: str | None) -> None:
    """Set the file's mtime from recovered metadata (EXIF / FAT entry).

    Content-identical is not enough: a recovered photo that shows today's
    date instead of the shooting date is visibly not the original file.
    """
    if not dt:
        return
    import datetime as _dt
    import os
    for fmt in ('%Y:%m:%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            t = _dt.datetime.strptime(dt[:19], fmt)
        except ValueError:
            continue
        ts = t.timestamp()
        os.utime(path, (ts, ts))
        return


def _primary_geometry(f, size: int):
    """Best geometry: real BPB if any survives, else MBR-inferred; None =
    nothing to work with (caller exits cleanly rather than crashing)."""
    from .geometry import geometry_from_mbr_only
    geos = scan_geometries(f, size)
    if geos:
        return geos[0]
    return geometry_from_mbr_only(f, size)


def cmd_scan(args) -> int:
    img = Path(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    size = img.stat().st_size
    with open(img, 'rb') as f:
        g = _primary_geometry(f, size)
        if g is None:
            print('no usable geometry: MBR and all boot sectors unreadable',
                  file=sys.stderr)
            return 2
        report = dict(image=str(img), size=size, geometries=[g.summary()])
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
    mp4_targets: list[tuple[int, str, str | None]] = []
    counts = dict(phantom=0, degraded=0)

    with open(img, 'rb') as f:
        g = _primary_geometry(f, size)
        if g is None:
            print('no usable geometry: MBR and all boot sectors unreadable',
                  file=sys.stderr)
            return 2
        man.geometries = [g.summary()]
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
                        if not data:
                            # name evidence only: an empty file is garbage
                            man.items.append(Item(path='', status='name-only',
                                                  source='dir-entry', name=s.display,
                                                  datetime=s.wr_date,
                                                  evidence=ev + ['no data bytes'],
                                                  generation=label))
                            continue
                        path = sub / _safe(s.display)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                        _apply_datetime(path, s.wr_date)
                        if s.display.upper().endswith(('.MP4', '.LRV')):
                            mp4_targets.append((g.cluster_off(s.cluster),
                                                s.display, s.wr_date))
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
        covered: list[tuple[int, int]] = []
        pending = []
        for h in carve.handlers():
            key = {'jpeg': 'jpeg', 'mp4': 'mp4_ftyp'}.get(h.name)
            if key is None:
                continue
            for hit in hits.get(key, [])[:args.max_carve]:
                # signature hits point at the magic itself; mp4's magic
                # ('ftyp') sits 4 bytes INTO the box — handlers get the box
                # start (this off-by-four made every mp4 carve phantom)
                off = hit - 4 if key == 'mp4_ftyp' else hit
                res = h.extract(f, off)
                if res.status == 'phantom':
                    counts['phantom'] += 1
                    continue
                if res.status not in ('complete', 'partial') or 'bytes' not in res.extras:
                    if res.name:
                        man.ghosts.append(dict(name=res.name, kind=h.name,
                                               datetime=res.datetime,
                                               survived='identity only'))
                    if h.name == 'mp4' and res.status != 'phantom':
                        # no carveable bytes, but a real ftyp anchor: the
                        # reassembly engine maps fragments on its own — this
                        # is how videos without a surviving dir entry get
                        # attempted at all
                        mp4_targets.append((off, res.name or f'mp4_off{off:013d}',
                                            res.datetime))
                    continue
                name = res.name or f'{h.name}_off{off:013d}'
                pending.append((off, h, res, name))
                if h.name == 'mp4':
                    mp4_targets.append((off, name, res.datetime))

        # write BIG first: a carve whose bytes live inside a complete file
        # is redundant — recorded as 'subsumed', never written (this was the
        # single biggest source of garbage in the output)
        complete_blobs: list[tuple[str, bytes]] = []
        for off, h, res, name in sorted(pending,
                                        key=lambda t: -len(t[2].extras['bytes'])):
            blob = res.extras['bytes']
            subsumed_by = next((nm for nm, b in complete_blobs if blob in b), None)
            if subsumed_by:
                man.items.append(Item(path='', status='subsumed',
                                      source=f'carve:{h.name}', offset=off,
                                      size=res.size, name=res.name,
                                      evidence=[f'bytes contained in {subsumed_by}: '
                                                'not written']))
                continue
            path = out / 'carved' / h.name / _safe(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            _apply_datetime(path, res.datetime)
            v = h.validate(str(path))
            status = res.status
            if status == 'complete' and not v.ok:
                status = 'partial'
                counts['degraded'] += 1
                res.notes.append('decode-validation failed: downgraded')
            if status == 'complete':
                complete_blobs.append((path.name, blob))
                if h.name == 'mp4':
                    covered.append((off, off + len(blob)))
            # embedded previews as companions ONLY for broken files — a
            # complete file already carries its preview inside its own bytes
            if status != 'complete':
                for pi, (poff, plen) in enumerate(res.extras.get('previews') or []):
                    if plen <= 0:
                        continue
                    pv = blob[poff:poff + plen]
                    ppath = path.with_name(f'{path.stem}.PREVIEW{pi:02d}.jpg')
                    ppath.write_bytes(pv)
                    _apply_datetime(ppath, res.datetime)
                    pv_res = h.validate(str(ppath))
                    man.items.append(Item(
                        path=str(ppath),
                        status='complete' if pv_res.ok else 'partial',
                        source='carve:jpeg:preview', offset=off + poff,
                        size=len(pv), name=ppath.name,
                        evidence=[f'embedded preview of {path.name}'],
                        generation='carved',
                        validated=f'{"ok" if pv_res.ok else "fail"}: {pv_res.detail}'))
            man.items.append(Item(path=str(path), status=status,
                                  source=f'carve:{h.name}', offset=off,
                                  size=res.size, name=res.name,
                                  datetime=res.datetime,
                                  evidence=res.notes, damage=res.damage,
                                  generation='carved',
                                  validated=f'{"ok" if v.ok else "fail"}: {v.detail}'))

        # ---- 3) MP4 reassembly: the deliverable is the whole MOVIE ----
        # the contiguous-span carve / chain dump is an INTERMEDIATE: by
        # default it is removed once a validated reassembly supersedes it
        done_stems: set[str] = set()
        seen_offs: set[int] = set()
        for off, name, dt in mp4_targets:
            # strip ONLY a real .mp4 extension: Path('x.LRV').stem would eat
            # '.LRV' and mislabel a low-res reassembly as the full-res video
            stem = name[:-4] if name.upper().endswith('.MP4') else name
            if stem in done_stems or off in seen_offs:
                continue
            done_stems.add(stem)
            seen_offs.add(off)
            out_mp4 = out / 'carved' / 'mp4' / f'{stem}.reassembled.mp4'
            out_mp4.parent.mkdir(parents=True, exist_ok=True)
            try:
                r = carve.by_name('mp4').reassemble(f, g, off, out_mp4,
                                                    verify=True)
            except Exception as e:                       # noqa: BLE001
                man.items.append(Item(path='', status='failed',
                                      source='reassemble:mp4', offset=off,
                                      name=name,
                                      evidence=[f'reassembly raised {e}: '
                                                'intermediates kept']))
                continue
            # engine scratch (scan cache etc.): intermediate, removed by default
            if r.work_dir and not args.keep_intermediates:
                import shutil
                shutil.rmtree(r.work_dir, ignore_errors=True)
            st = r.sample_stats
            ok = r.status == 'complete' and st.all_ok
            if out_mp4.exists():
                _apply_datetime(out_mp4, dt)
            man.items.append(Item(path=str(out_mp4) if out_mp4.exists() else '',
                                  status='complete' if ok else r.status,
                                  source='reassemble:mp4', offset=off,
                                  size=r.bytes_written, name=out_mp4.name,
                                  datetime=dt,
                                  evidence=[f'samples: nal_ok={st.nal_ok} '
                                            f'nal_bad={st.nal_bad} '
                                            f'desc_ok={st.descriptors_ok} '
                                            f'gpmd_ok={st.gpmd_ok}',
                                            f'mapped={r.mapped_bytes}/'
                                            f'{r.file_size}',
                                            *r.notes],
                                  generation='carved',
                                  validated=('ok' if st.all_ok else 'fail')))
            if ok:
                covered.append((off, off + r.bytes_written))
                if not args.keep_intermediates:
                    for it in man.items:
                        nm = (it.name or '').upper()
                        if (it.path and it.name
                                and nm.endswith(('.MP4', '.LRV'))
                                and (it.name[:-4] if it.name.upper().endswith('.MP4')
                                     else it.name) == stem
                                and it.source in ('carve:mp4', 'dir-entry')):
                            old = Path(it.path)
                            if old.exists() and old != out_mp4:
                                old.unlink()
                            it.path = ''
                            it.status = 'superseded'
                            it.evidence = list(it.evidence or []) + [
                                f'intermediate removed: superseded by '
                                f'{out_mp4.name}']

        # ---- 4) container-independent H.264 media salvage ----
        # No ftyp/moov needed: validated NAL runs are frames without time.
        # Runs already inside a decode-complete MP4 are skipped (no double
        # output); window-duplicates and overlapping runs are merged away.
        win, overlap = 64 << 20, 16 << 20
        found: dict[tuple[int, int], tuple[int, int]] = {}   # (off, end) -> (nals, vcl)
        pos = g.data_off
        while pos < end:
            f.seek(pos)
            blob = f.read(win + overlap)
            if not blob:
                break
            for run in media_runs(blob):
                abs_off, abs_end = pos + run.off, pos + run.end
                if any(a <= abs_off < b for a, b in covered):
                    continue
                found.setdefault((abs_off, abs_end), (run.nals, run.vcl))
            pos += win
        # drop runs overlapping an earlier one (window duplicates of a
        # run that straddles a window boundary start at the same offset;
        # partial overlaps are resolved in favour of the earlier run)
        kept: list[tuple[int, int, int, int]] = []
        prev_end = -1
        for (a, b), (nals, vcl) in sorted(found.items()):
            if a < prev_end:
                continue
            kept.append((a, b, nals, vcl))
            prev_end = b
        kept.sort(key=lambda r: r[1] - r[0], reverse=True)
        omitted = max(0, len(kept) - args.max_carve)
        for a, b, nals, vcl in kept[:args.max_carve]:
            f.seek(a)
            chunk = f.read(b - a)
            frames = nal_annexb(chunk, MediaRun(0, len(chunk), nals, vcl))
            # never write empty/negligible runs, and never duplicates of
            # what a complete file already contains
            if len(frames) < 1024:
                continue
            if any(frames in blob for _, blob in complete_blobs):
                continue
            name = f'h264_run_{a:013d}.h264'
            path = out / 'carved' / 'h264' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(frames)
            man.items.append(Item(path=str(path), status='partial',
                                  source='media:h264', offset=a,
                                  size=b - a,
                                  evidence=[f'NAL run: {nals} NALs, '
                                            f'{vcl} video-coded, '
                                            f'{b - a} bytes, no container'
                                            ' (frames without time)'],
                                  generation='carved',
                                  validated='unvalidated'))
        if omitted:
            man.items.append(Item(path='', status='metadata-only',
                                  source='media:h264', size=0,
                                  evidence=[f'{omitted} smaller NAL runs '
                                            'not written (cap '
                                            f'{args.max_carve})']))

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


def cmd_schema(args) -> int:
    from . import schema
    text = schema.dumps()
    if args.out:
        Path(args.out).write_text(text)
        print(f'written {args.out}')
    else:
        print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='fatality')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('scan')
    p.add_argument('image')
    p.add_argument('-o', '--out', default='fatality-out')
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser('extract')
    p.add_argument('image')
    p.add_argument('-o', '--out', default='fatality-out')
    p.add_argument('--max-carve', type=int, default=10_000)
    p.add_argument('--keep-intermediates', action='store_true',
                   help='keep raw spans/chain dumps that reassembly supersedes')
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser('verify')
    p.add_argument('out')
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser('report')
    p.add_argument('out')
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser('schema')
    p.add_argument('-o', '--out')
    p.set_defaults(fn=cmd_schema)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
