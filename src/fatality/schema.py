"""Declarative on-disk structure schemas for the FAT family.

Every byte offset/width the parsers know lives HERE, as data, so parsing is
schema-driven: a structure is a table of :class:`Field` records and a generic
reader interprets buffers against it.  Semantic rules (variant-dependent
fields, EOC/bad marker ranges, name assembly) sit in small pure functions
next to the tables they describe.

Conventions: multi-byte integers are little-endian (FAT on-disk); 'bytes' is
raw, 'utf16' is a UTF-16LE chunk with NUL/0xFFFF padding stripped.

Schemas are introspectable/exportable: ``export()`` renders the whole model
as JSON-able dict — useful for docs and for other tools.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Field:
    name: str
    offset: int
    size: int
    kind: str = 'bytes'        # u8 | u16le | u32le | u64le | bytes | utf16 | ascii
    doc: str = ''

    def read(self, buf: bytes, base: int = 0):
        b = buf[base + self.offset:base + self.offset + self.size]
        if len(b) < self.size:
            raise ValueError(f'{self.name}: buffer too short')
        if self.kind == 'u8':
            return b[0]
        if self.kind == 'u16le':
            return struct.unpack('<H', b)[0]
        if self.kind == 'u32le':
            return struct.unpack('<I', b)[0]
        if self.kind == 'u64le':
            return struct.unpack('<Q', b)[0]
        if self.kind == 'utf16':
            return b.decode('utf-16-le').rstrip('\uffff').rstrip('\x00')
        if self.kind == 'ascii':
            return b.decode('latin-1')
        return bytes(b)


@dataclass(frozen=True)
class Struct:
    name: str
    size: int
    fields: tuple[Field, ...]
    doc: str = ''

    def parse(self, buf: bytes, base: int = 0) -> dict:
        return {f.name: f.read(buf, base) for f in self.fields}

    def coverage(self) -> list[tuple[int, int]]:
        """Sorted (start, end) byte spans covered by fields."""
        return sorted((f.offset, f.offset + f.size) for f in self.fields)


# --------------------------------------------------------------- MBR ------

PARTITION_ENTRY = Struct('PartitionEntry', 16, (
    Field('status', 0, 1, 'u8', '0x80 = active'),
    Field('chs_first', 1, 3, 'bytes'),
    Field('type', 4, 1, 'u8', 'partition type id (0x0b/0x0c FAT32)'),
    Field('chs_last', 5, 3, 'bytes'),
    Field('lba_start', 8, 4, 'u32le'),
    Field('sectors', 12, 4, 'u32le'),
), 'MBR partition table entry (4 at 0x1BE).')


# -------------------------------------------------------------- BPB -------

BPB = Struct('BootSector', 512, (
    Field('jump', 0, 3, 'bytes'),
    Field('oem', 3, 8, 'ascii'),
    Field('bps', 11, 2, 'u16le', 'bytes per sector (512/1024/2048/4096)'),
    Field('spc', 13, 1, 'u8', 'sectors per cluster (power of two)'),
    Field('reserved', 14, 2, 'u16le', 'reserved sectors incl. boot + FSInfo'),
    Field('num_fats', 16, 1, 'u8'),
    Field('root_entries', 17, 2, 'u16le', 'FAT12/16 root dir slots; 0 on FAT32'),
    Field('total16', 19, 2, 'u16le', 'total sectors if < 0x10000 else 0'),
    Field('media', 21, 1, 'u8'),
    Field('fat_sects16', 22, 2, 'u16le', 'FAT size in sectors (FAT12/16)'),
    Field('spt', 24, 2, 'u16le', 'sectors per track (legacy)'),
    Field('heads', 26, 2, 'u16le', 'legacy'),
    Field('hidden', 28, 4, 'u32le', 'sectors before the volume'),
    Field('total32', 32, 4, 'u32le'),
    Field('fat_sects32', 36, 4, 'u32le', 'FAT size in sectors (FAT32)'),
    Field('ext_flags', 40, 2, 'u16le', 'FAT32 active-FAT mirroring flags'),
    Field('fs_version', 42, 2, 'u16le'),
    Field('root_cluster', 44, 4, 'u32le', 'FAT32 root directory cluster'),
    Field('fsinfo_sector', 48, 2, 'u16le'),
    Field('backup_boot', 50, 2, 'u16le'),
    Field('drive_number', 64, 1, 'u8'),
    Field('boot_sig', 66, 1, 'u8', '0x29 = serial/label present'),
    Field('serial', 67, 4, 'u32le'),
    Field('label', 71, 11, 'ascii'),
    Field('fstype', 82, 8, 'ascii', 'cosmetic, e.g. "FAT32   "'),
    Field('boot_code', 90, 420, 'bytes'),
    Field('signature', 510, 2, 'bytes', '0x55AA'),
), 'FAT boot sector / BIOS Parameter Block. FAT12/16 vs 32 variants '
   'differ in which size/root fields are authoritative (see variants()).')


def variants(bpb: dict) -> str:
    """'fat32' | 'fat16' | 'fat12' — by which fields the image uses."""
    if bpb['fat_sects16'] == 0 and bpb['root_entries'] == 0:
        return 'fat32'
    return 'fat16'                     # width refined later by cluster count


def refine_fat12_16(cluster_count: int) -> str:
    return 'fat12' if cluster_count < 4085 else 'fat16'


# ------------------------------------------------------------- FSInfo -----

FSINFO = Struct('FSInfo', 512, (
    Field('lead_sig', 0, 4, 'bytes', 'RRaA'),
    Field('free_count', 484, 4, 'u32le', '0xFFFFFFFF = unknown'),
    Field('next_free', 488, 4, 'u32le'),
    Field('trail_sig', 508, 4, 'bytes', '0xAA550000'),
), 'FAT32 FSInfo sector (usually sector 1).')


# --------------------------------------------------------- directory ------

DIRENT = Struct('DirEntry', 32, (
    Field('name8', 0, 8, 'ascii'),
    Field('ext', 8, 3, 'ascii'),
    Field('attr', 11, 1, 'u8'),
    Field('ntres', 12, 1, 'u8', 'NT reserved: 0x18 = lowercase both'),
    Field('crt_tenth', 13, 1, 'u8', 'creation time, 1/100 s'),
    Field('crt_time', 14, 2, 'u16le'),
    Field('crt_date', 16, 2, 'u16le'),
    Field('acc_date', 18, 2, 'u16le'),
    Field('cluster_hi', 20, 2, 'u16le'),
    Field('wr_time', 22, 2, 'u16le'),
    Field('wr_date', 24, 2, 'u16le'),
    Field('cluster_lo', 26, 2, 'u16le'),
    Field('size', 28, 4, 'u32le'),
), 'FAT 8.3 directory entry (32 bytes). LFN entries share the shape '
   'with attr == 0x0F and a different meaning (see LFNENT).')

LFNENT = Struct('LFNEntry', 32, (
    Field('seq', 0, 1, 'u8', '1..20; 0x40 marks the last slot'),
    Field('name1', 1, 10, 'utf16', 'UTF-16 chars 1-5'),
    Field('attr', 11, 1, 'u8', 'always 0x0F'),
    Field('type', 12, 1, 'u8', '0'),
    Field('checksum', 13, 1, 'u8', 'checksum of the short name'),
    Field('name2', 14, 12, 'utf16', 'UTF-16 chars 6-11'),
    Field('cluster_lo', 26, 2, 'u16le', '0'),
    Field('name3', 28, 4, 'utf16', 'UTF-16 chars 12-13'),
), 'Long File Name slot: name chars live in three non-contiguous chunks — '
   'the date-like bytes of a short entry are name payload here.')

ATTR_RO, ATTR_HIDDEN, ATTR_SYSTEM = 0x01, 0x02, 0x04
ATTR_VOLUME, ATTR_DIR, ATTR_ARCHIVE = 0x08, 0x10, 0x20
ATTR_LFN = 0x0F
ATTR_MASK = 0x3F


def lfn_text(slots: dict) -> str:
    """Assemble the long name from one LFN slot's parsed chunks."""
    return slots['name1'] + slots['name2'] + slots['name3']


def lfn_checksum(short11: bytes) -> int:
    s = 0
    for c in short11:
        s = (((s & 1) << 7) + (s >> 1) + c) & 0xFF
    return s


def dirent_cluster(e: dict) -> int:
    return e['cluster_lo'] | (e['cluster_hi'] << 16)


def dirent_short_name(e: dict) -> str:
    base = e['name8'].rstrip(' ')
    ext = e['ext'].rstrip(' ')
    return f'{base}.{ext}' if ext else base


# ----------------------------------------------------------- FAT table ----

@dataclass(frozen=True)
class FatSpec:
    width: int
    entry_bytes: float              # 1.5 for FAT12
    eoc_min: int
    bad: int
    mask: int


FAT_SPECS = {
    12: FatSpec(12, 1.5, 0x0FF8, 0x0FF7, 0x0FFF),
    16: FatSpec(16, 2.0, 0xFFF8, 0xFFF7, 0xFFFF),
    32: FatSpec(32, 4.0, 0x0FFFFFF8, 0x0FFFFFF7, 0x0FFFFFFF),
}

FAT_RESERVED_FIRST = 2              # clusters 0-1 are reserved markers


def fat_value(raw: int, width: int) -> int:
    """Masked cluster/EOF/bad value of one FAT entry."""
    return raw & FAT_SPECS[width].mask


def fat_is_eoc(v: int, width: int) -> bool:
    return v >= FAT_SPECS[width].eoc_min


def fat_is_bad(v: int, width: int) -> bool:
    return v == FAT_SPECS[width].bad


# ---------------------------------------------------------- exporting -----

ALL_STRUCTS = (PARTITION_ENTRY, BPB, FSINFO, DIRENT, LFNENT)


def export() -> dict:
    """The whole schema as JSON-able data (docs, other languages, diffs)."""
    out = {'structs': {}}
    for s in ALL_STRUCTS:
        out['structs'][s.name] = dict(
            size=s.size, doc=s.doc,
            fields=[asdict(f) for f in s.fields])
    out['fat_specs'] = {str(k): asdict(v) for k, v in FAT_SPECS.items()}
    return out


def dumps(indent: int = 2) -> str:
    return json.dumps(export(), indent=indent)
