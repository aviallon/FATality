"""Tree linking: group directory fragments into candidate hierarchies.

Conservative split policy: fragments join the same tree only on concrete
evidence ('.' / '..' parent refs, directory-entry cluster claims, contiguous
continuations of the same directory). Everything unlinked stays in its own
orphan group — unrelated evidence is never merged.

Union-find over fragment clusters with typed edges, then each component
becomes one output hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .dirtree import DirFragment


@dataclass
class UnionFind:
    parent: dict[int, int] = field(default_factory=dict)
    edge: dict[tuple[int, int], str] = field(default_factory=dict)

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int, why: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
            self.edge[(min(a, b), max(a, b))] = why


def link_fragments(frags: list[DirFragment], max_cluster: int) -> dict[int, list[DirFragment]]:
    """Components of fragment clusters -> fragments, with edge evidence."""
    by_clu = {f.cluster: f for f in frags}
    uf = UnionFind()
    for f in frags:
        uf.find(f.cluster)

    for f in frags:
        pe = f.parent_entry
        if pe and pe.cluster in by_clu and pe.cluster != f.cluster:
            uf.union(f.cluster, pe.cluster, 'parent-ref')
        for s in f.entries:
            if (s.is_dir and s.name not in ('.', '..')
                    and s.cluster in by_clu and s.cluster != f.cluster):
                uf.union(f.cluster, s.cluster, f'entry-claim:{s.display}')
        nxt = f.cluster + 1
        if nxt in by_clu:
            nf = by_clu[nxt]
            # contiguity continuation: next fragment has no '.' self-entry
            # (directory continuation clusters) or its '..' points where ours does
            if nf.self_entry is None or (pe and nf.parent_entry
                                        and nf.parent_entry.cluster == pe.cluster):
                uf.union(f.cluster, nxt, 'contig-continuation')

    comps: dict[int, list[DirFragment]] = {}
    for f in frags:
        comps.setdefault(uf.find(f.cluster), []).append(f)
    return comps


def tree_label(comp: list[DirFragment]) -> str:
    """Stable, descriptive label for a component.

    Root choice: the fragment whose '..' points at the volume root (cluster 0
    or absent) and which claims the most child fragments; fallback to the
    lowest cluster with a '.' self-entry.
    """
    def parent_clu(f: DirFragment):
        pe = f.parent_entry
        return pe.cluster if pe else None

    roots = [f for f in comp if parent_clu(f) in (0, None)]
    if not roots:
        selfs = [f for f in comp if f.self_entry]
        roots = selfs or comp
    # tie-break: most child-dir claims wins, then lowest cluster
    def claims(f: DirFragment) -> int:
        return sum(1 for s in f.entries if s.is_dir and s.name not in ('.', '..'))
    root = min(roots, key=lambda f: (-claims(f), f.cluster))
    nm = root.self_entry.name if root.self_entry else ''
    return f'root_clu{root.cluster}_{nm}' if nm not in ('', '.') else f'root_clu{root.cluster}'
