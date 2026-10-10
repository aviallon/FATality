"""Filesystem geometry: partitions, BPB parsing, stale boot sectors.

Everything here is evidence-level: a *geometry* describes one way the image
may have been organised. Quick-formats and re-formatings leave multiple
geometries behind (the old data area still speaks the old layout), so this
module can return SEVERAL candidate geometries, newest first.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from . import schema


@dataclass
class Geometry:
    """A parsed FAT filesystem layout (one 'generation' candidate)."""
    source: str                      # 'mbr-partition', 'superfloppy', 'stale-boot@0x...'
    part_off: int = 0                # byte offset of the volume
    part_size: int = 0               # byte size of the volume (0 = to EOF)
    bps: int = 0                     # bytes per sector
    spc: int = 0                     # sectors per cluster
    reserved: int = 0
    num_fats: int = 2
    fat_sects: int = 0
    root_entries: int = 0            # FAT12/16 root dir slots
    root_cluster: int = 2            # FAT32
    fat_bits: int = 32
    oem: str = ""
    label: str = ""
    serial: int = 0

    @property
    def cluster_bytes(self) -> int:
        return self.bps * self.spc

    @property
    def fat_off(self) -> list[int]:
        return [self.part_off + (self.reserved + i * self.fat_sects) * self.bps
                for i in range(self.num_fats)]

    @property
    def root_off(self) -> int:
        """Byte offset of the root directory (FAT12/16 fixed area)."""
        return self.part_off + (self.reserved + self.num_fats * self.fat_sects) * self.bps

    @property
    def data_off(self) -> int:
        extra = self.root_entries * 32 / self.bps if self.fat_bits < 32 else 0
        import math
        return self.root_off + math.ceil(extra) * self.bps

    @property
    def cluster_count(self) -> int:
        data_sects = (self.part_size // self.bps) - self.reserved - self.num_fats * self.fat_sects
        if self.fat_bits < 32:
            import math
            data_sects -= math.ceil(self.root_entries * 32 / self.bps)
        return data_sects // self.spc

    def cluster_off(self, n: int) -> int:
        return self.data_off + (n - 2) * self.cluster_bytes

    def cluster_of(self, off: int) -> int:
        return 2 + (off - self.data_off) // self.cluster_bytes

    def in_data(self, off: int) -> bool:
        return off >= self.data_off

    def summary(self) -> dict:
        return dict(source=self.source, part_off=self.part_off, part_size=self.part_size,
                    bps=self.bps, spc=self.spc, cluster_bytes=self.cluster_bytes,
                    reserved=self.reserved, num_fats=self.num_fats, fat_sects=self.fat_sects,
                    fat_bits=self.fat_bits, root_entries=self.root_entries,
                    root_cluster=self.root_cluster, data_off=self.data_off,
                    cluster_count=self.cluster_count, label=self.label)


def _parse_bpb(src: str, vol_off: int, vol_size: int, bs: bytes) -> Geometry | None:
    if len(bs) < 512:
        return None
    try:
        b = schema.BPB.parse(bs)
    except ValueError:
        return None
    if b['signature'] != b'\x55\xaa':
        return None
    bps, spc = b['bps'], b['spc']
    reserved, num_fats, root_entries = b['reserved'], b['num_fats'], b['root_entries']
    if not (bps in (512, 1024, 2048, 4096) and 1 <= spc <= 128 and reserved >= 1
            and num_fats in (1, 2) and spc & (spc - 1) == 0):
        return None
    fat16_sects = b['fat_sects16']
    fat_sects = fat16_sects or b['fat_sects32']
    if fat_sects == 0:
        return None
    total = b['total16'] or b['total32']
    if vol_size == 0:
        vol_size = total * bps
    # FAT32 iff the FAT size lives in the 32-bit field (FATSz16 == 0)
    fat_bits = 16 if fat16_sects else 32
    root_cluster = b['root_cluster'] if fat_bits == 32 else 0
    # refine FAT bits by cluster count (FAT spec rule)
    g = Geometry(source=src, part_off=vol_off, part_size=vol_size, bps=bps, spc=spc,
                 reserved=reserved, num_fats=num_fats, fat_sects=fat_sects,
                 root_entries=root_entries, root_cluster=max(root_cluster, 2),
                 fat_bits=fat_bits,
                 oem=b['oem'],
                 serial=b['serial'] if fat_bits == 32 else 0,
                 label=(b['label'] if fat_bits == 32 else bs[43:54].decode('latin-1', 'replace')))
    if g.fat_bits == 32:
        return g
    g.fat_bits = 12 if schema.refine_fat12_16(g.cluster_count) == 'fat12' else 16
    return g


def scan_geometries(f, size: int) -> list[Geometry]:
    """All plausible geometries: partitioned and stale/superfloppy boot sectors."""
    geos: list[Geometry] = []
    mbr = f.read(512)
    if len(mbr) == 512 and mbr[510:512] == b'\x55\xaa':
        for i in range(4):
            ent = mbr[0x1BE + 16 * i:0x1CE + 16 * i]
            lba, sects = struct.unpack('<II', ent[8:16])
            if lba and sects:
                off = lba * 512
                f.seek(off)
                g = _parse_bpb(f'mbr-partition[{i}]', off, sects * 512, f.read(512))
                if g:
                    geos.append(g)
    if not geos:
        f.seek(0)
        g = _parse_bpb('superfloppy', 0, size, f.read(512))
        if g:
            geos.append(g)
    # stale boot sectors elsewhere (former formats / misaligned volumes)
    candidates: list[int] = []
    f.seek(0)
    chunk_size = 64 << 20
    consumed = 0
    while consumed < size:
        buf = f.read(chunk_size)
        if not buf:
            break
        pos = 0
        while True:
            j = buf.find(b'\x55\xaa', pos)
            if j < 0 or j + 1 >= len(buf):
                break
            off = consumed + j - 1
            if off % 512 == 0 and off >= 512:
                candidates.append(off)
            pos = j + 2
        consumed += len(buf)
    for off in candidates:
        if any(x.part_off == off for x in geos):
            continue
        f.seek(off)
        g = _parse_bpb(f'stale-boot@{off}', off, 0, f.read(512))
        if g:
            geos.append(g)
    return geos
