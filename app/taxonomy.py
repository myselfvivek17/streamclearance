"""Content taxonomy: the shared language between video moments and law provisions.

Two levels: `<parent>.<leaf>` (e.g. `gambling.promotion`). Stage A tags video
moments with leaves; provisions target leaves (or a whole parent, e.g.
`gambling`). Each leaf maps to one legacy `Category` so the old UI and the
legacy stage B keep working.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.schemas import Category

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "data" / "taxonomy.json"


@dataclass(frozen=True)
class Leaf:
    id: str
    label: str
    description: str
    category: Category
    policy_only: bool = False  # set by deterministic code (app/synthetic.py), never by a model

    @property
    def parent(self) -> str:
        return self.id.split(".", 1)[0]


@lru_cache(maxsize=1)
def leaves() -> dict[str, Leaf]:
    raw = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))["leaves"]
    out = {}
    for item in raw:
        leaf = Leaf(item["id"], item["label"], item["description"], Category(item["category"]),
                    bool(item.get("policy_only", False)))
        if leaf.id.count(".") != 1:
            raise ValueError(f"taxonomy leaf {leaf.id!r} must be <parent>.<leaf>")
        out[leaf.id] = leaf
    return out


def parents() -> set[str]:
    return {leaf.parent for leaf in leaves().values()}


def is_leaf(value: str) -> bool:
    return value in leaves()


def is_target(value: str) -> bool:
    """Provisions may target a leaf or a whole parent (e.g. "gambling")."""
    return value in leaves() or value in parents()


def parent_of(value: str) -> str:
    return value.split(".", 1)[0]


def children(parent: str) -> list[str]:
    return [lid for lid, leaf in leaves().items() if leaf.parent == parent]


def covers(target: str, leaf: str) -> bool:
    """True if a provision targeting `target` covers a moment tagged `leaf`."""
    return target == leaf or target == parent_of(leaf)


def category_of(value: str) -> Category:
    """Legacy category of a leaf (or of a parent's first leaf)."""
    if value in leaves():
        return leaves()[value].category
    kids = children(value)
    return leaves()[kids[0]].category if kids else Category.other


def categories_for(values: list[str]) -> list[Category]:
    out: list[Category] = []
    for v in values:
        cat = category_of(v)
        if cat not in out:
            out.append(cat)
    return out


def clean(values: list[str]) -> list[str]:
    """Keep known leaves only, de-duplicated, order preserved."""
    out: list[str] = []
    for v in values:
        v = str(v).strip().lower()
        if is_leaf(v) and v not in out:
            out.append(v)
    return out


def label(value: str) -> str:
    if value in leaves():
        return leaves()[value].label
    return value.replace("_", " ")


def prompt_list() -> str:
    """One line per leaf, for prompts (policy-only leaves are set in code, so not listed)."""
    return "\n".join(f"- {leaf.id}: {leaf.description}" for leaf in leaves().values()
                     if not leaf.policy_only)
