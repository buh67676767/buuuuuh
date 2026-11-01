"""Local persistence for LLM-generated (or hand-written) strategy specs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from .spec_strategy import StrategySpec

STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "strategies"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "strategy"


def save_spec(spec: StrategySpec) -> Path:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(spec.name)
    path = STORE_DIR / f"{slug}.json"
    counter = 2
    while path.exists():
        path = STORE_DIR / f"{slug}_{counter}.json"
        counter += 1
    path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return path


def delete_spec(slug: str) -> None:
    path = STORE_DIR / f"{slug}.json"
    if path.exists():
        path.unlink()


def load_specs() -> List[tuple[str, StrategySpec]]:
    """Returns a list of (slug, StrategySpec) for every saved strategy."""
    if not STORE_DIR.exists():
        return []
    results = []
    for path in sorted(STORE_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            spec = StrategySpec.model_validate(data)
            results.append((path.stem, spec))
        except Exception:
            continue  # skip corrupt/incompatible files rather than crash the app
    return results
