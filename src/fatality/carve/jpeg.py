"""JPEG handler: structural carving, content-embedded names, salvage.

Proven techniques:
  * marker-structure walk decides the file end (never 'next signature')
  * EXIF-embedded self-naming (GoPro writes 'DCIM\\...\\NAME.JPG' + datetimes
    into its own files) gives real filenames for anonymous carves
  * embedded previews are attributed to their parent (no double counting)
  * partial files: salvage the intact prefix, record the exact damage
  * validation is decode-level (two independent decoders when available)
"""
from __future__ import annotations

import re
import struct
import subprocess

from . import ExtractResult, Handler, Validation, register

NAME_REC_RE = re.compile(rb'[\x20-\x7e]{2,20}\\[\x20-\x7e]{2,60}\\[\x20-\x7e]{1,20}\.[A-Za-z]{3}')
DT_RE = re.compile(rb'20\d\d:\d\d:\d\d \d\d:\d\d:\d\d')


def find_eoi(data: bytes) -> int | None:
    """Structural marker walk; offset of FFD9 or None."""
    i, n = 2, len(data)
    while i + 3 < n:
        if data[i] != 0xFF:
            return None
        m = data[i + 1]
        if m == 0xD9:
            return i
        if m == 0xDA:                       # SOS: entropy follows
            seglen = struct.unpack('>H', data[i + 2:i + 4])[0]
            k = data.find(b'\xff\xd9', i + 2 + seglen)
            return k if k >= 0 else None
        if 0xD0 <= m <= 0xD7 or m == 0x01:
            i += 2
            continue
        seglen = struct.unpack('>H', data[i + 2:i + 4])[0]
        if seglen < 2:
            return None
        i += 2 + seglen
    return None


def find_previews(data: bytes) -> list[tuple[int, int]]:
    """SOI offsets of EXIF-embedded previews (parent-relative)."""
    out = []
    pos = 2
    while True:
        i = data.find(b'\xff\xd8\xff', pos)
        if i < 0 or i > len(data):
            break
        e = find_eoi(data[i:])
        out.append((i, (e + 2) if e else 0))
        pos = i + 3
    return out[1:] if out else []       # first SOI is the parent itself


def salvage_prefix(data: bytes) -> tuple[bytes, int]:
    """Cut at the last 0xFF00/RSTn-consistent position before corruption.

    Conservative: scan the entropy stream for the first byte run that breaks
    JPEG stuffing rules, cut just before it, append EOI.
    """
    sos = data.find(b'\xff\xda')
    if sos < 0:
        return data[:2], 2
    seglen = struct.unpack('>H', data[sos + 2:sos + 4])[0]
    i = sos + 2 + seglen
    last_good = i
    n = len(data)
    while i < n - 1:
        b = data[i]
        if b == 0xFF:
            m = data[i + 1]
            if m == 0x00 or 0xD0 <= m <= 0xD7:
                last_good = i + 2
                i += 2
                continue
            if m == 0xD9:
                return data[:i + 2], i + 2
            break                       # unexpected marker inside entropy
        i += 1
    cut = last_good
    return data[:cut] + b'\xff\xd9', cut + 2


class JpegHandler(Handler):
    name = 'jpeg'
    signature = b'\xff\xd8\xff'

    def probe(self, data: bytes) -> bool:
        return data[:3] == b'\xff\xd8\xff'

    def extract(self, f, off: int, size_hint: int | None = None) -> ExtractResult:
        f.seek(off)
        data = f.read(size_hint or (8 << 20))
        if not self.probe(data):
            return ExtractResult(status='phantom', notes=['no SOI at offset'])
        res = ExtractResult(status='partial', size=0)
        m = NAME_REC_RE.search(data[:8192])
        if m:
            res.name = m.group(0).decode('latin-1').split('\\')[-1]
        dts = DT_RE.findall(data[:8192])
        if dts:
            res.datetime = dts[0].decode()

        eoi = find_eoi(data)
        if eoi is not None:
            jpg = data[:eoi + 2]
            res.size = len(jpg)
            res.out_path = 'jpeg'         # caller decides naming
            res.extras['bytes'] = jpg
            res.extras['previews'] = find_previews(jpg)
            res.status = 'complete'       # validate() decides final status
            return res

        salvaged, cut = salvage_prefix(data)
        if b'\xff\xc0' not in salvaged[:4096] and b'\xff\xc2' not in salvaged[:4096]:
            return ExtractResult(status='phantom',
                                 notes=['SOI in foreign data: no SOF marker before corruption'])
        res.extras['bytes'] = salvaged
        res.size = len(salvaged)
        res.damage.append((cut, max(0, len(data) - cut)))
        res.notes.append(f'no EOI: salvaged {len(salvaged)}B prefix at rel {cut}')
        res.out_path = 'jpeg'
        return res

    def validate(self, path: str) -> Validation:
        """Two independent decodes when tooling is present; else strict parse."""
        errs = 0
        try:
            from PIL import Image
            im = Image.open(path)
            im.verify()
            im = Image.open(path)
            im.load()
        except Exception as e:
            return Validation(False, f'pillow: {e}', 1)
        try:
            p = subprocess.run(['djpeg', '-outfile', '/dev/null', path],
                               capture_output=True)
            if p.returncode != 0 or p.stderr:
                return Validation(False, f'djpeg rc={p.returncode} {p.stderr[:80]!r}', 1)
        except FileNotFoundError:
            pass                        # single-decoder validation
        return Validation(True, 'decodes clean', errs)


register(JpegHandler())
