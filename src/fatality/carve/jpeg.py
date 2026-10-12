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


def salvage_rows(data: bytes) -> int:
    """How many rows of REAL content decode (truncated-tolerant).

    The value gate for anonymous fragments: measured on a real card, the
    fragments are bimodal — either real image rows or nothing at all.
    Returns -1 when undecidable (no Pillow): the caller keeps the fragment.
    """
    try:
        import io

        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(io.BytesIO(data))
        img.load()
        px = img.convert('L')
        w, h = px.size
        d = list(px.getdata())
        rows = 0
        for y in range(h):
            if len(set(d[y * w:(y + 1) * w])) > 8:
                rows += 1
        return rows
    except Exception:                              # noqa: BLE001
        return -1


def parse_mpf(data: bytes) -> list[tuple[int, int, int]]:
    """MPF (CIPA DC-007) multi-picture index -> (offset, size, attrs) per image.

    The MPF APP2 segment indexes ALL images of the file (primary + appended
    previews/alternates) with exact byte ranges — proven provenance, unlike
    content heuristics.  Per CIPA DC-007: the offset base is the "Start of
    Offset" = the MP Endian field address (right after 'MPF\0'); entry 1 is
    special-cased at file start.  MP Entry fields follow the MP Endian byte
    order (NOT always big-endian).  Entries pointing to anything that is not
    a complete JPEG stream are REFUSED — never guess boundaries.
    """
    i = 2
    app2 = -1
    while i + 4 <= len(data) and data[i] == 0xFF:
        m = data[i + 1]
        if m == 0xDA:                    # SOS: header region over
            break
        ln = int.from_bytes(data[i + 2:i + 4], 'big')
        if m == 0xE2 and data[i + 4:i + 8] == b'MPF\x00':
            app2 = i
            break
        i += 2 + ln
    if app2 < 0:
        return []
    import struct as _s
    tiff = data[app2 + 4 + 4:]           # after 'MPF\0' comes the TIFF header
    if len(tiff) < 8 or tiff[2:4] not in (b'\x00\x2a', b'\x2a\x00'):
        return []
    le = tiff[0:2] == b'II'

    def u32(o: int) -> int:
        return _s.unpack_from('<I' if le else '>I', tiff, o)[0]

    def u16(o: int) -> int:
        return _s.unpack_from('<H' if le else '>H', tiff, o)[0]
    ifd = u32(4)
    if not 0 <= ifd < len(tiff) - 2:
        return []
    n = u16(ifd)
    num_images = 0
    ent_blob: tuple[int, int] | None = None
    for k in range(n):
        e = ifd + 2 + 12 * k
        if e + 12 > len(tiff):
            return []
        tag, typ, cnt = u16(e), u16(e + 2), u32(e + 4)
        if tag == 0xB001 and typ == 4:
            num_images = u32(e + 8)
        elif tag == 0xB002:
            ent_blob = (u32(e + 8), cnt)
    if not ent_blob or num_images == 0:
        return []
    off, cnt = ent_blob
    # the ARRAY length is authoritative (ExifTool/ExifReader do the same);
    # NumberOfImages is advisory
    n_ent = min(cnt // 16, 256)
    if n_ent == 0 or off + 16 * n_ent > len(tiff):
        return []
    raw: list[tuple[int, int, int]] = []
    for j in range(n_ent):
        ent = tiff[off + 16 * j: off + 16 * j + 16]
        # MP Entry fields follow the MP Endian byte order (CIPA DC-007 §5.2.2.1)
        fmt = '<III' if le else '>III'
        attrs, size, imoff = _s.unpack_from(fmt, ent, 0)
        raw.append((imoff, size, attrs))
    def valid_at(base: int, o: int, s: int) -> bool:
        p = base + o
        # declared size may exclude the EOI (observed on GoPro: -2): accept
        # when a complete marker-walked stream starts at p within s + slack
        if not (p >= 0 and p + 3 <= len(data) and s > 0
                and data[p:p + 3] == b'\xff\xd8\xff'):
            return False
        e = find_eoi(data[p:p + s + 4])
        return e is not None and e + 2 <= s + 4

    out: list[tuple[int, int, int]] = []
    for j, (o, s, a) in enumerate(raw):
        if s <= 0:
            continue
        bases: tuple[int, ...]
        if j == 0:
            bases = (0,)                      # entry 1: file start (special)
        else:
            # "Start of Offset" = MP Endian field address (app2 + 8);
            # file-absolute kept as fallback for non-conforming writers
            bases = (app2 + 8, app2, app2 + 4, 0)
        for base in bases:
            if valid_at(base, o, s):
                p = base + o
                ee = find_eoi(data[p:p + s + 4])
                out.append((p, (ee + 2) if ee is not None else s, a))
                break
    return out


def absorb_trailing_streams(data: bytes, start: int, limit: int = 8) -> int:
    """Generic boundary-driven inclusion (extension-agnostic).

    After the known end, absorb complete JPEG streams while their boundaries
    are STANDARD: a full marker walk must reach its own EOI and the stream
    must contain an SOF.  Bytes whose boundaries cannot be derived are never
    absorbed, whatever they contain.
    """
    end = start
    absorbed = 0
    while absorbed < limit and end + 3 <= len(data) \
            and data[end:end + 3] == b'\xff\xd8\xff':
        e = find_eoi(data[end:])
        if e is None or e < 1024:
            break
        probe = data[end:end + e + 2]
        if b'\xff\xc0' not in probe and b'\xff\xc2' not in probe:
            break
        end += e + 2
        absorbed += 1
    return end


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
    return out                # parent SOI is never in the list (scan starts at 2)


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
            # a real image has an SOF and a body: a 4-byte 'carve' is a
            # false positive on entropy data, not salvage
            if len(jpg) < 256 or not (b'\xff\xc0' in jpg or b'\xff\xc2' in jpg):
                return ExtractResult(status='phantom',
                                     notes=['too small / no SOF: not a real image'])
            # MPF (CIPA DC-007): the index knows ALL images of the file —
            # appended previews included.  The recovered file is the COMPLETE
            # multi-picture file, not just the primary stream.
            mpf = parse_mpf(data)
            if mpf:
                end_mpf = max(o + s for o, s, _a in mpf)
                if end_mpf > len(jpg):
                    jpg = data[:end_mpf]
                res.extras['mpf'] = mpf
                res.extras['previews'] = [(o, s) for o, s, _a in mpf[1:]]
                res.notes.append(
                    f'MPF index: {len(mpf)} image(s), file is the complete '
                    'multi-picture container')
            else:
                res.extras['previews'] = find_previews(jpg)
            # NOT absorbed: unindexed trailing streams have standard
            # boundaries AS STREAMS but their MEMBERSHIP in this file is not
            # derivable — measured failure: on a contiguous card it MERGED
            # neighbouring files into one.  Only an index (MPF) proves
            # membership; unindexed streams stay separate carves.
            res.size = len(jpg)
            res.out_path = 'jpeg'         # caller decides naming
            res.extras['bytes'] = jpg
            res.status = 'complete'       # validate() decides final status
            return res

        salvaged, cut = salvage_prefix(data)
        if b'\xff\xc0' not in salvaged[:4096] and b'\xff\xc2' not in salvaged[:4096]:
            return ExtractResult(status='phantom',
                                 notes=['SOI in foreign data: no SOF marker before corruption'])
        res.extras['bytes'] = salvaged
        res.size = len(salvaged)
        # a broken file still carries its embedded previews — attach them:
        # for a truncated photo the preview is the best surviving view
        res.extras['previews'] = find_previews(salvaged)
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
