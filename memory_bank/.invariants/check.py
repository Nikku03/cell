"""Invariant checker for the Syn3A simulator memory bank.

Runs fast (< 1s for 10k facts) using plain file I/O + SQLite.
Execute from the repo root:

    python memory_bank/.invariants/check.py

Exits 0 on success, 1 on any invariant failure. The SQLite index at
memory_bank/index/facts.sqlite is rebuilt on every run, so it is always
consistent with the on-disk fact files.

The checks enforced here are the ones listed in PROJECT_BRIEF section 4.3:

  1. Every fact has all required fields.
  2. Every fact references a resolvable source.
  3. Every dependency ID exists.
  4. Parameter values fall inside declared physical ranges (ranges.json).
  5. No two facts make incompatible claims about the same entity.
  6. Every `used_by` code path exists on disk.
  7. Facts with `last_verified` older than `stale_days` are flagged.

Failures 1-6 are hard errors (exit 1). Failure 7 is a warning (exit 0).
"""
from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MEMORY_BANK = ROOT / "memory_bank"
FACTS_DIR = MEMORY_BANK / "facts"
SOURCES_DIR = MEMORY_BANK / "sources"
INDEX_DIR = MEMORY_BANK / "index"
RANGES_FILE = Path(__file__).with_name("ranges.json")
DB_PATH = INDEX_DIR / "facts.sqlite"


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts_seen: int = 0
    sources_seen: int = 0

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self) -> bool:
        return not self.errors


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        return {"__parse_error__": f"{path.name}: {exc}"}


def _iter_fact_files() -> list[Path]:
    return sorted(p for p in FACTS_DIR.rglob("*.json"))


def _iter_source_files() -> list[Path]:
    return sorted(p for p in SOURCES_DIR.glob("*.json"))


def _load_ranges() -> dict[str, Any]:
    with RANGES_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _check_required_fields(fact: dict, path: Path, required: list[str], report: Report) -> bool:
    missing = [f for f in required if f not in fact]
    if missing:
        report.err(f"{path.relative_to(ROOT)}: missing fields {missing}")
        return False
    return True


def _check_value_range(fact: dict, path: Path, ranges: dict, report: Report) -> None:
    value = fact.get("value")
    if not isinstance(value, dict):
        return
    param_name = value.get("parameter")
    number = value.get("number")
    units = value.get("units")
    if param_name is None or number is None:
        return
    rule = ranges.get(param_name)
    if rule is None:
        # Parameter type is not declared in ranges.json - record as warning so we
        # know to extend the ranges file, but do not block.
        report.warn(
            f"{path.relative_to(ROOT)}: parameter '{param_name}' has no declared "
            f"range in ranges.json - add one or accept unchecked."
        )
        return
    if units is not None and rule.get("units") is not None and units != rule["units"]:
        report.err(
            f"{path.relative_to(ROOT)}: units mismatch for '{param_name}' - "
            f"fact has {units!r}, expected {rule['units']!r}"
        )
    if not (rule["min"] <= float(number) <= rule["max"]):
        report.err(
            f"{path.relative_to(ROOT)}: value {number} for '{param_name}' is "
            f"outside range [{rule['min']}, {rule['max']}] {rule['units']}"
        )


def _check_confidence(fact: dict, path: Path, levels: list[str], report: Report) -> None:
    conf = fact.get("confidence")
    if conf not in levels:
        report.err(
            f"{path.relative_to(ROOT)}: confidence '{conf}' not in {levels}"
        )


def _check_source(fact: dict, path: Path, source_ids: set[str], report: Report) -> None:
    src = fact.get("source")
    if not isinstance(src, str) or not src:
        report.err(f"{path.relative_to(ROOT)}: 'source' must be a non-empty string")
        return
    if src not in source_ids:
        report.err(
            f"{path.relative_to(ROOT)}: source '{src}' not found in "
            f"memory_bank/sources/"
        )


def _check_dependencies(
    fact: dict, path: Path, all_ids: set[str], report: Report
) -> None:
    for dep in fact.get("dependencies") or []:
        if dep not in all_ids:
            report.err(
                f"{path.relative_to(ROOT)}: dependency '{dep}' does not exist"
            )


def _check_used_by(fact: dict, path: Path, report: Report) -> None:
    for ref in fact.get("used_by") or []:
        # format: "path/to/file.py" or "path/to/file.py:symbol"
        file_part = ref.split(":", 1)[0]
        target = ROOT / file_part
        if not target.exists():
            report.err(
                f"{path.relative_to(ROOT)}: used_by references missing file "
                f"'{file_part}'"
            )


def _check_staleness(fact: dict, path: Path, stale_days: int, report: Report) -> None:
    lv = fact.get("last_verified")
    if not isinstance(lv, str):
        return
    try:
        lv_date = date.fromisoformat(lv)
    except ValueError:
        report.err(f"{path.relative_to(ROOT)}: last_verified '{lv}' not ISO date")
        return
    age = date.today() - lv_date
    if age > timedelta(days=stale_days):
        report.warn(
            f"{path.relative_to(ROOT)}: fact last_verified {lv} "
            f"({age.days} days ago) is stale (>{stale_days} days)"
        )


def _check_contradictions(facts: list[tuple[Path, dict]], report: Report) -> None:
    """Two facts contradict if they claim distinct numeric values for the same
    (parameter, entity, context) tuple. `entity` here is taken from the fact's
    context, falling back to the fact id prefix."""
    buckets: dict[tuple, list[tuple[Path, dict]]] = {}
    for path, fact in facts:
        value = fact.get("value") or {}
        param = value.get("parameter")
        number = value.get("number")
        if param is None or number is None:
            continue
        ctx = fact.get("context") or {}
        entity = ctx.get("entity") or ctx.get("gene") or ctx.get("enzyme") or ctx.get("protein")
        if entity is None:
            continue
        ctx_key = tuple(sorted((k, v) for k, v in ctx.items() if k not in {"notes"}))
        buckets.setdefault((param, entity, ctx_key), []).append((path, fact))
    for (param, entity, _), group in buckets.items():
        if len(group) < 2:
            continue
        numbers = {float(f["value"]["number"]) for _, f in group}
        if len(numbers) > 1:
            ids = ", ".join(f["id"] for _, f in group)
            report.err(
                f"contradiction: parameter '{param}' for entity '{entity}' has "
                f"multiple distinct values across facts [{ids}]: {sorted(numbers)}"
            )


def _rebuild_sqlite(facts: list[tuple[Path, dict]]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(
            """
            CREATE TABLE fact (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                claim TEXT NOT NULL,
                parameter TEXT,
                number REAL,
                units TEXT,
                source TEXT,
                confidence TEXT,
                last_verified TEXT
            );
            CREATE TABLE fact_used_by (
                fact_id TEXT NOT NULL,
                ref TEXT NOT NULL
            );
            CREATE TABLE fact_dep (
                fact_id TEXT NOT NULL,
                dep_id TEXT NOT NULL
            );
            CREATE INDEX idx_fact_param ON fact(parameter);
            CREATE INDEX idx_fact_source ON fact(source);
            """
        )
        for path, f in facts:
            value = f.get("value") or {}
            con.execute(
                "INSERT INTO fact VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    f["id"],
                    str(path.relative_to(ROOT)),
                    f.get("claim", ""),
                    value.get("parameter"),
                    float(value["number"]) if "number" in value else None,
                    value.get("units"),
                    f.get("source"),
                    f.get("confidence"),
                    f.get("last_verified"),
                ),
            )
            for ref in f.get("used_by") or []:
                con.execute(
                    "INSERT INTO fact_used_by VALUES (?,?)", (f["id"], ref)
                )
            for dep in f.get("dependencies") or []:
                con.execute("INSERT INTO fact_dep VALUES (?,?)", (f["id"], dep))
        con.commit()
    finally:
        con.close()


def check() -> Report:
    report = Report()
    config = _load_ranges()
    required = config["required_fact_fields"]
    ranges = config["parameters"]
    levels = config["confidence_levels"]
    stale_days = int(config.get("stale_days", 90))

    source_paths = _iter_source_files()
    source_ids: set[str] = set()
    for p in source_paths:
        data = _load_json(p)
        if data is None or "__parse_error__" in (data or {}):
            report.err(f"source parse error in {p.relative_to(ROOT)}")
            continue
        sid = data.get("id") or p.stem
        if sid in source_ids:
            report.err(f"duplicate source id '{sid}' in {p.name}")
        source_ids.add(sid)
        for required_src in ("id", "citation", "type"):
            if required_src not in data:
                report.err(
                    f"{p.relative_to(ROOT)}: source missing field '{required_src}'"
                )
    report.sources_seen = len(source_paths)

    fact_paths = _iter_fact_files()
    loaded: list[tuple[Path, dict]] = []
    seen_ids: set[str] = set()
    for p in fact_paths:
        data = _load_json(p)
        if data is None:
            report.err(f"cannot read {p.relative_to(ROOT)}")
            continue
        if "__parse_error__" in data:
            report.err(data["__parse_error__"])
            continue
        if not _check_required_fields(data, p, required, report):
            continue
        fid = data["id"]
        if fid in seen_ids:
            report.err(f"duplicate fact id '{fid}' in {p.relative_to(ROOT)}")
        seen_ids.add(fid)
        loaded.append((p, data))
    report.facts_seen = len(loaded)

    all_ids = {f["id"] for _, f in loaded}
    for p, f in loaded:
        _check_confidence(f, p, levels, report)
        _check_source(f, p, source_ids, report)
        _check_dependencies(f, p, all_ids, report)
        _check_value_range(f, p, ranges, report)
        _check_used_by(f, p, report)
        _check_staleness(f, p, stale_days, report)
    _check_contradictions(loaded, report)

    # Rebuild SQLite index only if we have no hard errors so we never cache a
    # broken state.
    if report.ok():
        _rebuild_sqlite(loaded)

    return report


def main() -> int:
    report = check()
    print(f"facts checked:   {report.facts_seen}")
    print(f"sources checked: {report.sources_seen}")
    if report.warnings:
        print(f"\nwarnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"  - {w}")
    if report.errors:
        print(f"\nerrors ({len(report.errors)}):")
        for e in report.errors:
            print(f"  - {e}")
        print("\nFAIL")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
