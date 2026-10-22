"""
Explicit artefact selection shared by the evaluation scripts.

output/ and evaluation/output/ hold runs over more than one corpus (the 55-file
primary swamp, the 6-file TAB smoke tests, the 1268-document TAB corpus) and several
runs per corpus (July, October, the variance passes). Choosing a file because it is the
newest silently mixed corpora once, so these helpers never do that:

  * an explicitly passed path is used as given, after the corpus check;
  * otherwise a file is auto-selected only if it is the ONLY candidate that passes the
    corpus check;
  * if several candidates pass, the caller exits non-zero listing them, and the user
    must pin one on the command line.
"""

import sys
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


class SelectionError(SystemExit):
    def __init__(self, msg: str):
        print(f"ERROR: {msg}", file=sys.stderr)
        super().__init__(2)


def select_artefact(directory: Path, pattern: str, *, explicit: str | Path | None = None,
                    accept: Callable[[Path], str | None] | None = None,
                    what: str = "", option: str = "") -> Path | None:
    """Return one path or None (no candidate fits). `accept(path)` returns None when
    the file belongs to the expected corpus, else a short reason it does not."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SelectionError(f"{what}: {p} does not exist")
        reason = accept(p) if accept else None
        if reason:
            raise SelectionError(f"{what}: {p.name} rejected - {reason}")
        return p
    candidates = sorted(directory.glob(pattern))
    fitting = [c for c in candidates if not (accept and accept(c))]
    if len(fitting) <= 1:
        return fitting[0] if fitting else None
    listing = "\n  ".join(c.name for c in fitting)
    raise SelectionError(
        f"{what}: {len(fitting)} candidate files fit, refusing to guess which one is meant:\n"
        f"  {listing}\nPin one with {option}{'' if option.endswith('=') else ' '}<path>.")


def file_set_check(expected: Iterable[str], column: str = "file_name",
                   require_complete: bool = False) -> Callable[[Path], str | None]:
    """accept() for metadata CSVs: every file in the CSV must belong to `expected`."""
    expected = {str(f).strip() for f in expected}

    def check(path: Path) -> str | None:
        try:
            got = {str(f).strip() for f in pd.read_csv(path, usecols=[column])[column]}
        except Exception as e:
            return f"unreadable ({type(e).__name__})"
        foreign = got - expected
        if not got or foreign:
            return (f"{len(got & expected)} of {len(got)} files belong to the expected corpus "
                    f"({len(foreign)} foreign, e.g. {sorted(foreign)[:2]})")
        if require_complete and got != expected:
            return f"partial run ({len(got)} of {len(expected)} files)"
        return None
    return check


def row_count_check(n: int) -> Callable[[Path], str | None]:
    """accept() for metadata CSVs over a corpus identified only by its size (TAB, 1268)."""
    def check(path: Path) -> str | None:
        try:
            rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        except Exception as e:
            return f"unreadable ({type(e).__name__})"
        return None if rows == n else f"{rows} rows, expected {n}"
    return check


def parse_pins(values: list[str] | None) -> dict[str, str]:
    """--csv key=path (repeatable) -> {key: path}."""
    pins = {}
    for v in values or []:
        key, sep, path = v.partition("=")
        if not sep or not path:
            raise SelectionError(f"--csv expects key=path, got {v!r}")
        pins[key.strip()] = path.strip()
    return pins


def prefix_key(prefix: str) -> str:
    """metadata_llm_phi3_mini_ -> llm_phi3_mini (the key used with --csv)."""
    return prefix.removeprefix("metadata_").rstrip("_")
