"""GoPro family support: name records, THM/LRV/GPR, media-counter inference.

GoPro files carry their own identity:
  * JPEG/THM: EXIF name record ('...\\DIR\\NAME.EXT' + make/model/datetime)
  * MP4/LRV:  moov metadata may embed the path; GP descriptors identify layout
  * the media counter is SHARED between photos and videos — numbering gaps
    prove how many files existed even when their metadata is gone
  (named-ghost inference).
"""
from __future__ import annotations

import re

from . import ExtractResult, Handler, Validation, register

PATH_REC_RE = re.compile(
    rb'[A-Z]:\\[\x20-\x7e]{2,60}'
    rb'|[\x20-\x7e]{2,20}\\[\x20-\x7e]{2,60}\\[\x20-\x7e]{1,20}\.[A-Za-z]{3}')
GOPRO_NAME_RE = re.compile(r'(GOPR|GH0\d|GL0\d|GX0\d|GPFR|G)(\d{4})\.([A-Z]{3})')
DT_RE = re.compile(rb'20\d\d:\d\d:\d\d \d\d:\d\d:\d\d')


def name_records(data: bytes) -> list[dict]:
    """All self-identifying path records inside a payload."""
    out = []
    for m in PATH_REC_RE.finditer(data[:1 << 20]):
        txt = m.group(0).decode('latin-1')
        dts = DT_RE.findall(data[max(0, m.start() - 64):m.end() + 512])
        out.append(dict(path=txt, off=m.start(),
                        datetimes=[d.decode() for d in dts[:3]]))
    return out


def counter_inference(names: list[str]) -> dict:
    """Gap analysis over GoPro media counters -> named ghosts."""
    nums = []
    for n in names:
        m = GOPRO_NAME_RE.match(n.upper())
        if m:
            nums.append(int(m.group(2)))
    ghosts = []
    if nums:
        lo, hi = min(nums), max(nums)
        present = set(nums)
        ghosts = [x for x in range(lo, hi + 1) if x not in present]
    return dict(min=min(nums) if nums else None,
                max=max(nums) if nums else None,
                gaps=ghosts,
                note='gaps are destroyed-or-never-taken; correlate with timestamps')


class GoproHandler(Handler):
    """Metadata handler: never claims file bytes, only identity."""
    name = 'gopro'
    signature = None

    def probe(self, data: bytes) -> bool:
        return bool(PATH_REC_RE.search(data[:4096]))

    def extract(self, f, off: int, size_hint: int | None = None) -> ExtractResult:
        f.seek(off)
        data = f.read(size_hint or (1 << 20))
        recs = name_records(data)
        if not recs:
            return ExtractResult(status='phantom', notes=['no name record'])
        r = recs[0]
        return ExtractResult(status='metadata-only', name=r['path'].split('\\')[-1],
                             datetime=r['datetimes'][0] if r['datetimes'] else None,
                             extras={'records': recs},
                             notes=['identity recovered from embedded record'])

    def validate(self, path: str) -> Validation:
        return Validation(True, 'metadata handler', 0)


register(GoproHandler())
