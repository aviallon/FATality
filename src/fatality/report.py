"""Manifest + graveyard reporting.

The manifest records, per recovered item: provenance, status, damage and
what backed every hypothesis. The graveyard records names whose data is
gone — a destroyed file with a surviving name must never vanish silently.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Item:
    path: str
    status: str                      # complete | partial | metadata-only | destroyed
    source: str                      # 'dir-entry', 'carve:<handler>', 'content-name'
    offset: int | None = None
    size: int = 0
    name: str | None = None
    datetime: str | None = None
    evidence: list[str] = field(default_factory=list)
    damage: list[tuple[int, int]] = field(default_factory=list)
    generation: str = 'unknown'
    validated: str = 'unvalidated'


@dataclass
class Manifest:
    image: str
    geometries: list[dict] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    ghosts: list[dict] = field(default_factory=list)

    def save(self, out_dir: Path):
        (out_dir / 'manifest.json').write_text(json.dumps(
            dict(image=self.image, geometries=self.geometries,
                 items=[asdict(i) for i in self.items], ghosts=self.ghosts),
            indent=1, default=str))

    def graveyard_md(self) -> str:
        lines = ['# Graveyard — names without recoverable data', '']
        lines.append('| name | kind | datetime | what survived |')
        lines.append('|---|---|---|---|')
        for g in self.ghosts:
            lines.append(f"| {g.get('name', '?')} | {g.get('kind', '?')} "
                         f"| {g.get('datetime') or '-'} | {g.get('survived', 'metadata only')} |")
        destroyed = [i for i in self.items if i.status == 'destroyed']
        if destroyed:
            lines += ['', '## Destroyed files with surviving directory entries', '']
            for i in destroyed:
                lines.append(f'- {i.name} (cluster evidence: {", ".join(i.evidence) or "n/a"})')
        return '\n'.join(lines) + '\n'

    def summary(self) -> dict:
        by: dict[str, int] = {}
        for i in self.items:
            by[i.status] = by.get(i.status, 0) + 1
        return dict(items=len(self.items), by_status=by, ghosts=len(self.ghosts))
