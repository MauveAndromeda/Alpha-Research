#!/usr/bin/env python3
"""
Pre-commit / CI check: fail if any tracked file contains a hardcoded API key.

Patterns detected:
  - "sk-" followed by 10+ alphanumeric/dash/underscore chars (DeepSeek / OpenAI keys)

Usage:
    python scripts/check_no_secrets.py          # exits 0 if clean, 1 if secrets found
"""

import re
import subprocess
import sys

PATTERN = re.compile(r'"sk-[a-zA-Z0-9_\-]{10,}"')


def main():
    # Get list of tracked files
    result = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        print("ERROR: git ls-files failed", file=sys.stderr)
        sys.exit(2)

    files = result.stdout.strip().split("\n")
    violations = []

    for fpath in files:
        if not fpath or fpath.endswith((".parquet", ".png", ".jpg", ".ico", ".woff", ".woff2")):
            continue
        try:
            with open(fpath, "r", errors="ignore") as f:
                for lineno, line in enumerate(f, 1):
                    if PATTERN.search(line):
                        violations.append((fpath, lineno, line.strip()[:120]))
        except (OSError, UnicodeDecodeError):
            continue

    if violations:
        print(f"FAIL: Found {len(violations)} hardcoded API key(s):", file=sys.stderr)
        for fpath, lineno, snippet in violations:
            print(f"  {fpath}:{lineno}  {snippet}", file=sys.stderr)
        sys.exit(1)

    print("OK: No hardcoded API keys found in tracked files.")
    sys.exit(0)


if __name__ == "__main__":
    main()
