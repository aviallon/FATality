# FATality

Scavenge everything physically recoverable from an arbitrarily broken FAT
filesystem image: quick-formatted cards, zeroed/corrupt FATs, lost roots,
orphaned directory entries, stale directory clusters from former formats.

Born from the recovery of a quick-formatted GoPro SD card where both FAT
copies were entirely zeroed and every byte of media was still recovered:
31 named photos, 5 videos reassembled from content-level chains, and the
graveyard of what was truly gone.

## Principles

1. **The image is evidence.** Opened read-only, always. Never write in place.
2. **Everything is a hypothesis with a confidence.** Directory fragments,
   chain reconstructions and generations are *evidence-backed guesses*, each
   recorded in a manifest with what backed it.
3. **Conservative split.** When origins are ambiguous, recovered content goes
   into *separate* hierarchies (per filesystem generation / recovered root /
   epoch) — never merged — with orphan trees beside each candidate.
4. **Specialised extraction beats carving.** File-format handlers reconstruct
   what signature carving cannot: fragmented containers, names embedded in
   content, partially clobbered payloads (salvage + honest damage maps).
5. **A recovered file is not 'complete' because its size looks right.**
   Validation is decode-level; anything with missing ranges is `partial`.

## What it does

```
fatality scan    IMAGE            # index: geometry, FATs, dir slots, signatures
fatality extract IMAGE -o DIR     # recover into generation-split hierarchies
fatality verify  DIR              # decode-level verification of the output
fatality report  DIR              # manifest + graveyard (names without data)
```

Pipeline stages:

1. **Geometry** — MBR/GPT partitions, BPB parsing (FAT12/16/32), backup
   boot sectors, FSInfo; scans for *stale* boot sectors of former formats
   (each defines a candidate older geometry).
2. **FAT scavenging** — all FAT copies parsed; when zeroed/corrupt, chains
   are reconstructed from directory evidence + contiguity + content anchors
   (e.g. MP4 sample tables act as a content-level FAT).
3. **Directory mining** — every 32-byte slot of every cluster scanned
   (live, deleted 0xE5, LFN reassembly); clusters with entry density become
   *directory fragments*, including ones whose chains are long gone.
4. **Generation clustering** — fragments that agree (shared `.`, `..`,
   consistent timestamps, matching cluster claims) group into trees;
   conflicting snapshots become separate generations.
5. **Specialised extraction** — pluggable handlers (`probe/extract/validate`):
   - `jpeg`: structural marker walk, EXIF self-naming (GoPro-style embedded
     path records), embedded-preview attribution, MCU-boundary salvage of
     partially clobbered files, double-decode validation.
   - `mp4`: atom parsing, moov sample tables (stco/co64/stsc/stsz), GoPro
     `GP` per-sample descriptor chains, fragmented reassembly, damage maps.
   - `gopro`: THM/LRV/GPR family, name records, media-counter gap inference
     (destroyed files reconstructed as named ghosts).
6. **Honest reporting** — manifest.json with per-file provenance and status
   (`complete` / `partial` / `metadata-only`), and graveyard.md for names
   whose data is gone.

## Install / develop

```sh
nix develop            # python, uv, verification tools
uv venv .venv && uv pip install -e '.[dev]'
git config core.hooksPath hooks   # install the committed hooks
pytest                 # synthetic FAT-image fixtures
```

Quality gates (all green, all enforced):

- **ruff** + **mypy** — `pre-commit` hook runs both on every commit;
  `pre-push` re-runs them plus the full test suite.
- **CI** (GitHub Actions) runs the identical checks on every push/PR —
  hooks can be bypassed locally, CI cannot.
- `fatality schema` dumps the on-disk structure model (see `src/fatality/schema.py`)
  as JSON — parsing is schema-driven, field offsets live in one place.

## Status

v0.1 — core implementation. See `docs/design.md` for the evidence model.
