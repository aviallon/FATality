"""Extraction handler registry.

A handler turns raw bytes at an offset into a validated file plus provenance.
Handlers must:
  * never trust sizes blindly (structure decides)
  * return 'partial' + damage map whenever any byte range is missing/clobbered
  * extract identity from CONTENT when names exist inside payloads
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractResult:
    status: str                       # 'complete' | 'partial' | 'metadata-only' | 'phantom'
    out_path: str | None = None
    size: int = 0
    name: str | None = None           # name recovered from content/records
    datetime: str | None = None
    damage: list[tuple[int, int]] = field(default_factory=list)  # (rel_off, length)
    extras: dict = field(default_factory=dict)                   # handler-specific
    notes: list[str] = field(default_factory=list)


@dataclass
class Validation:
    ok: bool
    detail: str
    errors: int = 0


class Handler:
    name = 'base'
    signature: bytes | None = None    # cheap prefilter, None = no prefilter

    def probe(self, data: bytes) -> bool:
        raise NotImplementedError

    def extract(self, f, off: int, size_hint: int | None = None) -> ExtractResult:
        raise NotImplementedError

    def validate(self, path: str) -> Validation:
        raise NotImplementedError


_REGISTRY: list[Handler] = []


def register(h: Handler) -> Handler:
    _REGISTRY.append(h)
    return h


def handlers() -> list[Handler]:
    return list(_REGISTRY)


def by_name(n: str) -> Handler:
    for h in _REGISTRY:
        if h.name == n:
            return h
    raise KeyError(n)


# import for registration side effects
from . import jpeg, mp4, gopro  # noqa: E402,F401
