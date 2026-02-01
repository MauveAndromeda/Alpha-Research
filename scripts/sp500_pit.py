#!/usr/bin/env python3
"""
One-click S&P 500 point-in-time constituents fetcher for Codespaces/CI.

Data source: https://github.com/fja05680/sp500
- Auto-picks latest "S&P 500 Historical Components & Changes(MM-DD-YYYY).csv" in repo root via GitHub API.
- Downloads to cache, then provides PIT membership snapshots by date.

Usage:
  python scripts/sp500_pit.py fetch --cache data/sp500 --refresh
  python scripts/sp500_pit.py members 2018-06-01
  python scripts/sp500_pit.py diff 2018-06-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

import requests

try:
    import pandas as pd
except Exception as e:
    raise SystemExit("Missing dependency: pandas. Install with `pip install pandas`") from e


GITHUB_OWNER = "fja05680"
GITHUB_REPO = "sp500"
GITHUB_API_CONTENTS = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents"

DATED_FILE_RE = re.compile(
    r"^S&P 500 Historical Components & Changes\((\d{2})-(\d{2})-(\d{4})\)\.csv$"
)
FALLBACK_FILE = "S&P 500 Historical Components & Changes.csv"


def _github_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "sp500-pit-fetcher",
    }
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_mm_dd_yyyy(name: str) -> Optional[dt.date]:
    m = DATED_FILE_RE.match(name)
    if not m:
        return None
    mm, dd, yyyy = map(int, m.groups())
    return dt.date(yyyy, mm, dd)


def _pick_latest_dataset_entry(entries: list) -> dict:
    """
    Prefer the latest dated file. Fallback to base CSV if no dated file exists.
    """
    dated: List[Tuple[dt.date, dict]] = []
    fallback: Optional[dict] = None

    for e in entries:
        name = e.get("name", "")
        if name == FALLBACK_FILE:
            fallback = e
        d = _parse_mm_dd_yyyy(name)
        if d:
            dated.append((d, e))

    if dated:
        dated.sort(key=lambda x: x[0])
        return dated[-1][1]

    if fallback:
        return fallback

    raise RuntimeError("No suitable CSV found in repo root.")


def _download(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def _detect_columns(df: "pd.DataFrame") -> Tuple[str, str]:
    cols = [c.strip() for c in df.columns.astype(str).tolist()]
    lower = [c.lower() for c in cols]

    date_candidates = ["date", "dt", "asof", "day"]
    date_col = None
    for cand in date_candidates:
        if cand in lower:
            date_col = cols[lower.index(cand)]
            break
    if date_col is None:
        date_col = cols[0]

    tick_candidates = ["tickers", "ticker", "symbols", "symbol",
                       "components", "constituents"]
    tick_col = None
    for cand in tick_candidates:
        if cand in lower:
            tick_col = cols[lower.index(cand)]
            break
    if tick_col is None:
        tick_col = cols[1] if len(cols) > 1 else cols[0]

    return date_col, tick_col


def _split_tickers(s: str) -> Set[str]:
    if s is None:
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return {t.strip() for t in s.split(",") if t.strip()}


@dataclass
class SP500PIT:
    cache_dir: Path = field(default_factory=lambda: Path("data/sp500"))
    refresh: bool = False

    _csv_path: Optional[Path] = field(default=None, init=False, repr=False)
    _df: Optional["pd.DataFrame"] = field(default=None, init=False, repr=False)
    _date_col: Optional[str] = field(default=None, init=False, repr=False)
    _tick_col: Optional[str] = field(default=None, init=False, repr=False)

    def fetch(self) -> Path:
        _safe_mkdir(self.cache_dir)

        meta_path = self.cache_dir / "sp500_pit_meta.json"
        csv_path = self.cache_dir / "sp500_pit_raw.csv"

        if (not self.refresh) and meta_path.exists() and csv_path.exists():
            self._csv_path = csv_path
            return csv_path

        r = requests.get(GITHUB_API_CONTENTS, headers=_github_headers(), timeout=30)
        r.raise_for_status()
        entries = r.json()
        if not isinstance(entries, list):
            raise RuntimeError(f"Unexpected GitHub API response: {type(entries)}")

        chosen = _pick_latest_dataset_entry(entries)
        name = chosen.get("name")
        download_url = chosen.get("download_url")
        if not download_url:
            raise RuntimeError(f"Missing download_url for entry: {name}")

        _download(download_url, csv_path)

        meta = {
            "source_repo": f"{GITHUB_OWNER}/{GITHUB_REPO}",
            "chosen_name": name,
            "download_url": download_url,
            "fetched_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self._csv_path = csv_path
        return csv_path

    def load(self) -> "pd.DataFrame":
        if self._df is not None:
            return self._df

        if self._csv_path is None or not self._csv_path.exists():
            self.fetch()

        assert self._csv_path is not None
        df = pd.read_csv(self._csv_path)

        date_col, tick_col = _detect_columns(df)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        self._df = df
        self._date_col = date_col
        self._tick_col = tick_col
        return df

    def members(self, asof: str | dt.date | dt.datetime) -> List[str]:
        """Return S&P 500 members as of the given date."""
        df = self.load()
        assert self._date_col and self._tick_col

        asof_ts = pd.to_datetime(asof)
        sub = df[df[self._date_col] <= asof_ts]
        if sub.empty:
            row = df.iloc[0]
        else:
            row = sub.iloc[-1]

        tickers = sorted(_split_tickers(row[self._tick_col]))
        return tickers

    def latest_date(self) -> dt.date:
        df = self.load()
        assert self._date_col
        last = df[self._date_col].iloc[-1]
        return last.date()

    def added_removed_since(self, asof: str | dt.date | dt.datetime) -> Tuple[List[str], List[str]]:
        """Compare members(asof) vs members(latest). Return (added, removed)."""
        now_members = set(self.members(self.latest_date()))
        past_members = set(self.members(asof))
        added = sorted(now_members - past_members)
        removed = sorted(past_members - now_members)
        return added, removed

    def export_long(self, out_path: Path) -> Path:
        """Export normalized long-format membership: date,ticker"""
        df = self.load()
        assert self._date_col and self._tick_col

        rows = []
        for _, r in df.iterrows():
            d = r[self._date_col]
            tickers = _split_tickers(r[self._tick_col])
            for t in tickers:
                rows.append((d.date().isoformat(), t))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(rows, columns=["date", "ticker"]).sort_values(
            ["date", "ticker"])
        if out_path.name.endswith(".gz"):
            out_df.to_csv(out_path, index=False, compression="gzip")
        else:
            out_df.to_csv(out_path, index=False)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Download latest PIT constituents CSV")
    p_fetch.add_argument("--cache", default="data/sp500")
    p_fetch.add_argument("--refresh", action="store_true")

    p_members = sub.add_parser("members", help="Print tickers as of a date")
    p_members.add_argument("date", help="YYYY-MM-DD")
    p_members.add_argument("--cache", default="data/sp500")

    p_diff = sub.add_parser("diff", help="Print added/removed since date")
    p_diff.add_argument("date", help="YYYY-MM-DD")
    p_diff.add_argument("--cache", default="data/sp500")

    p_export = sub.add_parser("export-long", help="Export long-format membership")
    p_export.add_argument("--out", default="data/sp500/sp500_members_long.csv.gz")
    p_export.add_argument("--cache", default="data/sp500")
    p_export.add_argument("--refresh", action="store_true")

    args = parser.parse_args()
    pit = SP500PIT(
        cache_dir=Path(getattr(args, "cache", "data/sp500")),
        refresh=getattr(args, "refresh", False))

    if args.cmd == "fetch":
        path = pit.fetch()
        print(f"[OK] cached CSV: {path}")

    elif args.cmd == "members":
        tickers = pit.members(args.date)
        print(f"{len(tickers)} members as of {args.date}")
        print("\n".join(tickers))

    elif args.cmd == "diff":
        added, removed = pit.added_removed_since(args.date)
        print(f"=== ADDED ({len(added)}) ===")
        print("\n".join(added))
        print(f"=== REMOVED ({len(removed)}) ===")
        print("\n".join(removed))

    elif args.cmd == "export-long":
        pit.refresh = getattr(args, "refresh", False)
        pit.fetch()
        out = pit.export_long(Path(args.out))
        print(f"[OK] wrote: {out}")


if __name__ == "__main__":
    main()
