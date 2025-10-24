#!/usr/bin/env python3
"""Fetch recent GitHub commits for course sites and emit a JSON changelog."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    courses = data.get("courses", {})
    if not isinstance(courses, dict):
        raise ValueError("Invalid course configuration: 'courses' must be a mapping")
    return courses


def fmt_display_date(iso_timestamp: str) -> str:
    try:
        dt_obj = dt.datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return dt_obj.strftime("%b %d, %Y")
    except Exception:
        return iso_timestamp


def fetch_commits(repo: str, limit: int, token: str | None) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/commits"
    params = {"per_page": max(1, min(limit, 50))}
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    entries: List[Dict[str, str]] = []
    for commit in payload:
        commit_data = commit.get("commit", {})
        message = (commit_data.get("message") or "").splitlines()[0]
        author = commit_data.get("author") or {}
        html_url = commit.get("html_url") or ""
        timestamp = author.get("date") or commit_data.get("committer", {}).get("date", "")
        entries.append(
            {
                "message": message,
                "author": author.get("name") or commit_data.get("committer", {}).get("name") or "",
                "date_iso": timestamp,
                "date_display": fmt_display_date(timestamp) if timestamp else "",
                "url": html_url,
            }
        )
    return entries


def build_changelog(courses: Dict[str, Any], limit: int, token: str | None) -> Dict[str, Any]:
    changelog: Dict[str, Any] = {}
    for slug, info in courses.items():
        repo = info.get("repo", "").strip()
        if not repo:
            continue
        try:
            changelog[slug] = {
                "repo": repo,
                "display_name": info.get("display_name", slug),
                "entries": fetch_commits(repo, limit, token),
            }
        except requests.HTTPError as exc:
            changelog[slug] = {
                "repo": repo,
                "error": f"HTTP {exc.response.status_code}",
                "entries": [],
            }
        except Exception as exc:  # pragma: no cover - defensive
            changelog[slug] = {
                "repo": repo,
                "error": str(exc),
                "entries": [],
            }
    return changelog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch course changelog from GitHub")
    parser.add_argument("--config", type=Path, required=True, help="YAML file describing course repos")
    parser.add_argument("--out", type=Path, required=True, help="Path to write JSON changelog")
    parser.add_argument("--limit", type=int, default=5, help="Number of commits per course")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise SystemExit(f"Config file not found: {args.config}")

    courses = load_config(args.config)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    changelog = build_changelog(courses, args.limit, token)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(changelog, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
