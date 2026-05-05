"""
Count functional lines of code across the project.

TOGGLES (edit the CONFIG block below):
    INCLUDE_COMMENTS : True  → count lines starting with #
                       False → exclude comment-only lines
    INCLUDE_SCRIPTS  : True  → include files inside a 'scripts/' folder
                       False → skip them
    INCLUDE_TESTS    : True  → include files inside a 'tests/' folder
                       False → skip them
"""

from pathlib import Path
import sys

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these toggles
# ══════════════════════════════════════════════════════════════════════════════
INCLUDE_COMMENTS = False   # True → count '#' comment lines | False → exclude
INCLUDE_SCRIPTS  = False    # True → include scripts/ folder  | False → skip
INCLUDE_TESTS    = False    # True → include tests/ folder    | False → skip
# ══════════════════════════════════════════════════════════════════════════════


def is_excluded(filepath: Path, root: Path) -> bool:
    parts = filepath.relative_to(root).parts
    if not INCLUDE_SCRIPTS and "scripts" in parts:
        return True
    if not INCLUDE_TESTS and "tests" in parts:
        return True
    return False


def count_lines(filepath: Path) -> int:
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return 0
    return sum(
        1 for line in lines
        if line.strip()
        and (INCLUDE_COMMENTS or not line.strip().startswith("#"))
    )


root     = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
py_files = [f for f in sorted(root.rglob("*.py")) if not is_excluded(f, root)]
results  = {f: count_lines(f) for f in py_files}

total_files = len(results)
total_lines = sum(results.values())
top2        = sorted(results.items(), key=lambda x: x[1], reverse=True)[:2]

print(f"\nSettings      : comments={'on' if INCLUDE_COMMENTS else 'off'}"
      f"  |  scripts={'on' if INCLUDE_SCRIPTS else 'off'}"
      f"  |  tests={'on' if INCLUDE_TESTS else 'off'}")
print(f"Files scanned : {total_files}")
print(f"Functional LOC: {total_lines:,}")
print(f"\nTop {len(top2)} files by functional LOC:")
for path, count in top2:
    print(f"  {path.relative_to(root)}: {count:,} lines")