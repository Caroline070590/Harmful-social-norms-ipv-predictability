#!/usr/bin/env python3
"""Compile the historical exports without importing or executing their stages.

This smoke check detects syntax failures only. It does not validate scientific
results, resolve input paths, install dependencies, or download datasets.
"""

from pathlib import Path
import sys


def main() -> int:
    failures = []
    paths = sorted(Path(__file__).resolve().parent.rglob("*.py"))
    for path in paths:
        try:
            # Compile in memory: no imports, model fits, downloads or pyc files.
            compile(path.read_bytes(), str(path), "exec")
        except (SyntaxError, OSError) as error:
            failures.append((path, error))
    if failures:
        for path, error in failures:
            print(f"FAIL {path.name}: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {len(paths)} Python files compile; no analysis was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
