#!/usr/bin/env python3
"""Check every shell command in the notebook against the scripts' real argparse.

The notebook documents ~28 scripts it cannot execute here (no model access), so
its commands are the one part that silent-rots. This catches a renamed flag or a
moved script before a reader does. Run it after editing tools/build_notebook.py.
"""
import ast, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOS = [ROOT / "semantic-compass", ROOT / "arithmetic-circuit-discovery"]
CMD = re.compile(r"python (experiments/[\w/]+\.py|investigate_helix_usage_validated\.py)"
                 r"((?:[^\n#']|\\\n)*)")


def declared_flags(path: Path):
    """Flags the script passes to add_argument(). None if it has no parser."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return None
    flags = {
        a.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
        for a in n.args
        if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-")
    }
    return flags or None


def main() -> int:
    nb = json.loads((ROOT / "experiments_end_to_end.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"])

    problems = []
    scripts = set()
    for script, tail in set(CMD.findall(src)):
        scripts.add(script)
        repo = next((r for r in REPOS if (r / script).exists()), None)
        if repo is None:
            problems.append(f"MISSING SCRIPT   {script}")
            continue
        flags = declared_flags(repo / script)
        if flags is None:
            continue  # no argparse: nothing to check
        unknown = sorted(set(re.findall(r"(--[\w-]+)", tail)) - flags)
        if unknown:
            problems.append(
                f"UNKNOWN FLAG(S)  {repo.name}/{script}: {unknown}\n"
                f"                 script declares: {sorted(flags)}"
            )

    print(f"notebook references {len(scripts)} scripts")
    if problems:
        print(f"\n{len(problems)} problem(s):\n")
        for p in problems:
            print("  " + p)
        return 1
    print("all scripts exist and all flags validate against argparse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
