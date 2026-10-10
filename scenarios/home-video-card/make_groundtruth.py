#!/usr/bin/env python3
"""Generate the ground truth for the 'home-video-card' scenario.

Run once; the produced files AND manifest.json are blessed (committed).
Deterministic inputs (fixed seeds) so regeneration is reproducible in
practice, but the committed bytes are the contract.

Content is deliberately quasi-real, mirroring the Manon case:
  * DCIM/100GOPRO: GoPro-style photos + THM + a small MP4
  * MISC/: VERSION.TXT / SD_INFO.TXT text files
  * Android-ish dirs (MUSIC, ALARMS) and LFN names
  * postformat/: copied after the quick format (clobbers old data)
  * doomed/: deleted BEFORE the format (name ghosts)
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

SCEN = Path(__file__).resolve().parent


def jpeg(seed: int, size=(1920, 1440), quality=85) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, 'JPEG', quality=quality)
    return buf.getvalue()


def mp4(path: Path, seconds: int = 8) -> None:
    subprocess.run([
        'ffmpeg', '-y', '-v', 'error', '-f', 'lavfi',
        '-i', f'testsrc2=size=1280x720:rate=30:duration={seconds}',
        '-pix_fmt', 'yuv420p', '-c:v', 'libx264', '-preset', 'ultrafast',
        str(path)], check=True)


def main() -> None:
    gt = SCEN / 'groundtruth'
    pf = SCEN / 'postformat'
    gopro = gt / 'DCIM' / '100GOPRO'
    misc = gt / 'MISC'
    for d in (gopro, misc, gt / 'MUSIC', gt / 'Alarms', pf / 'Android' / 'data',
              pf / 'LOST.DIR'):
        d.mkdir(parents=True, exist_ok=True)

    # GoPro-style media — REALISTIC sizes (2-3MB photos): the ground truth
    # must dwarf the post-format usage, or every survivor gets clobbered and
    # the scenario measures destruction instead of recovery.
    (gopro / 'GOPR3655.JPG').write_bytes(jpeg(1))
    (gopro / 'G0013656.JPG').write_bytes(jpeg(2))
    (gopro / 'G0013657.JPG').write_bytes(jpeg(3))
    (gopro / 'G0013658.JPG').write_bytes(jpeg(5))
    (gopro / 'G0013659.JPG').write_bytes(jpeg(6))
    (gopro / 'G0013660.JPG').write_bytes(jpeg(7))
    (gopro / 'G0013661.JPG').write_bytes(jpeg(8))
    (gopro / 'GOPR3655.THM').write_bytes(jpeg(4, size=(160, 120), quality=70))
    mp4(gopro / 'GOPR3655.MP4')
    mp4(gopro / 'GOPR3655.LRV.MP4', seconds=3)

    # text metadata (LFN + short names)
    (misc / 'VERSION.TXT').write_bytes(b'{"info version":"2.0","camera type":"SYNTHETIC"}\n')
    (misc / 'SD_INFO.TXT').write_bytes(b'Card Slot ID: 0\nProduct Name: TESTCARD\n')
    (gt / 'a file with spaces and a very long name indeed.txt').write_bytes(
        b'LFN torture test ' * 32)
    (gt / 'MUSIC' / 'track.mp3').write_bytes(b'\xff\xfb\x90\x00' + bytes(2048))
    (gt / 'Alarms' / 'alarm.mp3').write_bytes(b'\xff\xfb\x90\x00' + bytes(1024))

    # post-format usage (clobbers the old data area like real usage does) —
    # kept SMALL on purpose: it must eat only the first cluster or two
    (pf / 'Android' / 'data' / 'nomedia').write_bytes(b'')
    (pf / 'LOST.DIR' / 'placeholder').write_bytes(b'android wrote me after the format')
    (pf / 'DCIM' / 'NEWPHOTO.JPG').parent.mkdir(parents=True, exist_ok=True)
    (pf / 'DCIM' / 'NEWPHOTO.JPG').write_bytes(jpeg(9, size=(640, 480), quality=80))

    print('ground truth written under', SCEN)
    print('now: python tools/scenario.py manifest scenarios/home-video-card')


if __name__ == '__main__':
    main()
