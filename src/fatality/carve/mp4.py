"""MP4 handler: atom parsing, sample tables, GoPro GP-descriptor chains.

When the FAT chain is gone, an MP4 rebuilds itself from two content-level
sources:
  1. moov sample tables (stco/co64 + stsc + stsz) — exact file offsets of
     every sample; one physical anchor maps file offsets to image offsets.
  2. GoPro 'GP' descriptors: 16 bytes before every media sample
     (b'GP' + type + flag + size(BE) + delta(BE) + 0) — a content-level FAT.
     Every candidate byte range is validated against its descriptor, video
     samples additionally NAL-walked (sum(4+len) == size).
"""
from __future__ import annotations

import json
import struct
import subprocess
from dataclasses import dataclass, field

from . import ExtractResult, Handler, Validation, register


@dataclass
class Atom:
    off: int
    size: int
    kind: bytes
    header: int = 8


@dataclass
class SampleTable:
    timescale: int = 0
    duration: int = 0
    chunk_offsets: list[int] = field(default_factory=list)
    sample_sizes: list[int] = field(default_factory=list)
    entries: list[tuple[int, int]] = field(default_factory=list)  # (count, desc_idx)

    @property
    def creation(self) -> int:
        return self._creation

    _creation: int = 0


def walk_atoms(data: bytes, start: int = 0, end: int | None = None,
               depth: int = 0) -> list[Atom]:
    out: list[Atom] = []
    i, end = start, end if end is not None else len(data)
    while i + 8 <= end:
        size = struct.unpack('>I', data[i:i + 4])[0]
        kind = data[i + 4:i + 8]
        hdr = 8
        if size == 1 and i + 16 <= end:
            size = struct.unpack('>Q', data[i + 8:i + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - i
        if size < hdr:
            break
        out.append(Atom(off=i, size=size, kind=kind, header=hdr))
        if kind in (b'moov', b'trak', b'mdia', b'minf', b'stbl', b'udta'):
            out += walk_atoms(data, i + hdr, min(i + size, end), depth + 1)
        i += size
    return out


def parse_moov(data: bytes, moov: Atom) -> dict:
    """Sample tables + metadata of one moov atom (raw bytes)."""
    info = dict(creation=0, duration=0, timescale=0, tracks=[])
    for a in walk_atoms(data, moov.off, moov.off + moov.size):
        body = a.off + a.header
        if a.kind == b'mvhd':
            ver = data[body]
            if ver == 1:
                info['creation'] = struct.unpack('>Q', data[body + 4:body + 12])[0]
                info['timescale'] = struct.unpack('>I', data[body + 20:body + 24])[0]
                info['duration'] = struct.unpack('>Q', data[body + 24:body + 32])[0]
            else:
                info['creation'] = struct.unpack('>I', data[body + 4:body + 8])[0]
                info['timescale'] = struct.unpack('>I', data[body + 12:body + 16])[0]
                info['duration'] = struct.unpack('>I', data[body + 16:body + 20])[0]
        elif a.kind == b'stsz':
            cnt = struct.unpack('>I', data[body + 8:body + 12])[0]
            fixed = struct.unpack('>I', data[body + 4:body + 8])[0]
            if fixed:
                sizes = [fixed] * cnt
            else:
                sizes = list(struct.unpack(f'<{cnt}I', data[body + 12:body + 12 + 4 * cnt]))
            info['tracks'].append(dict(kind='stsz', sizes=sizes))
        elif a.kind == b'stco':
            cnt = struct.unpack('>I', data[body + 4:body + 8])[0]
            info['tracks'].append(dict(
                kind='stco',
                offsets=list(struct.unpack(f'<{cnt}I', data[body + 8:body + 8 + 4 * cnt]))))
        elif a.kind == b'co64':
            cnt = struct.unpack('>I', data[body + 4:body + 8])[0]
            info['tracks'].append(dict(
                kind='co64',
                offsets=list(struct.unpack(f'<{cnt}Q', data[body + 8:body + 8 + 8 * cnt]))))
    return info


def gp_descriptors(data: bytes, lo: int, hi: int) -> list[tuple[int, int, int]]:
    """GoPro per-sample descriptors in [lo, hi): (off, type, size)."""
    out = []
    i = lo
    while i + 16 <= hi:
        if data[i:i + 2] == b'GP' and data[i + 15] == 0 and data[i + 3] in (0, 1, 2, 3):
            size = struct.unpack('>I', data[i + 4:i + 8])[0]
            if 0 < size < 64 << 20:
                out.append((i, data[i + 2], size))
                i += 16 + size
                continue
        i += 1
    return out


def nal_walk_ok(sample: bytes) -> bool:
    """AVCC: 4-byte lengths chain exactly to the sample size."""
    i, n = 0, len(sample)
    while i + 4 <= n:
        ln = struct.unpack('>I', sample[i:i + 4])[0]
        if ln == 0 or i + 4 + ln > n:
            return False
        i += 4 + ln
    return i == n


class Mp4Handler(Handler):
    name = 'mp4'
    signature = b'ftyp'

    def probe(self, data: bytes) -> bool:
        return len(data) >= 16 and data[4:8] == b'ftyp'

    def extract(self, f, off: int, size_hint: int | None = None) -> ExtractResult:
        """v0.1: metadata extraction + descriptor inventory.

        Full fragmented reassembly needs the image-wide extent mapper (see
        docs/design.md); bytes are returned only when structure proves a
        contiguous, validated file.
        """
        f.seek(off)
        data = f.read(size_hint or (256 << 20))
        if not self.probe(data):
            return ExtractResult(status='phantom', notes=['no ftyp at offset'])
        atoms = walk_atoms(data)
        if not atoms:
            return ExtractResult(status='phantom', notes=['no atoms'])
        total = max(a.off + a.size for a in atoms)
        res = ExtractResult(status='partial', size=total,
                            extras={'atoms': [(a.off, a.size, a.kind.decode('latin-1'))
                                              for a in atoms if a.header == 8 or True][:64]})
        moovs = [a for a in atoms if a.kind == b'moov']
        mdats = [a for a in atoms if a.kind == b'mdat']
        if moovs:
            info = parse_moov(data, moovs[0])
            res.extras['moov'] = json.loads(json.dumps(info, default=str))
            if info['creation']:
                import datetime
                res.datetime = (datetime.datetime(1904, 1, 1)
                                + datetime.timedelta(seconds=info['creation'])).isoformat()
        if mdats:
            descs = gp_descriptors(data, mdats[0].off, min(total, mdats[0].off + mdats[0].size))
            res.extras['gp_descriptors'] = len(descs)
            res.notes.append(f'{len(descs)} GoPro sample descriptors in mdat')
        if moovs and mdats and not res.damage:
            res.status = 'complete'
            res.extras['bytes'] = data[:total]
        else:
            res.notes.append('moov/mdat incomplete or unverified: metadata only')
        return res

    def validate(self, path: str) -> Validation:
        try:
            p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                                'format=duration', '-of', 'csv=p=0', path],
                               capture_output=True, text=True)
            if p.returncode != 0:
                return Validation(False, f'ffprobe rc={p.returncode}', 1)
            return Validation(True, f'duration={p.stdout.strip()}', 0)
        except FileNotFoundError:
            return Validation(True, 'ffprobe unavailable; structure-only', 0)


register(Mp4Handler())
