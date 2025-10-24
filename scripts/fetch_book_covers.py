#!/usr/bin/env python3
"""Fetch book covers for library items listed in ``resources.csv``."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

import requests


SEARCH_URL = "https://openlibrary.org/search.json"
COVER_FMT = "https://covers.openlibrary.org/b/{key}/{value}-{size}.jpg?default=false"

ISBN_KEYS = (
    "isbn",
    "isbn_13",
    "isbn13",
    "isbn_10",
    "isbn10",
    "isbn-10",
    "isbn-13",
)
OLID_KEYS = (
    "olid",
    "openlibrary_id",
    "openlibrary_work",
    "openlibrary_edition",
)
COVER_ID_KEYS = (
    "openlibrary_cover_id",
    "cover_id",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RESOURCES = REPO_ROOT / "static" / "csv" / "resources.csv"
DEFAULT_DEST = REPO_ROOT / "static" / "img" / "book_cover_thumbnails"
DEFAULT_SIZE = "L"
DEFAULT_TIMEOUT = 20

ISBN_CLEAN_RE = re.compile(r"[^0-9Xx]")
OLID_CLEAN_RE = re.compile(r"[^A-Za-z0-9]")
TOKEN_SPLIT_RE = re.compile(r"[\s,;/|]+")


def normalize_key(value: str) -> str:
    return (value or "").strip().lower().replace(" ", "_")


def build_header_map(fieldnames: Sequence[str]) -> dict[str, str]:
    return {normalize_key(name): name for name in fieldnames if name}


def row_value(row: dict[str, str], header_map: dict[str, str], *keys: str) -> str:
    for key in keys:
        actual = header_map.get(normalize_key(key))
        if actual is None:
            continue
        return (row.get(actual) or "").strip()
    return ""


def split_tokens(value: str) -> Iterator[str]:
    for token in TOKEN_SPLIT_RE.split(value or ""):
        trimmed = token.strip()
        if trimmed:
            yield trimmed


def unique(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


@dataclass
class LibraryItem:
    index: int
    book_id: str
    title: str
    author: str
    isbns: List[str]
    olids: List[str]
    cover_ids: List[str]


def collect_isbns(row: dict[str, str], header_map: dict[str, str]) -> List[str]:
    values: List[str] = []
    for key in ISBN_KEYS:
        raw = row_value(row, header_map, key)
        if not raw:
            continue
        for token in split_tokens(raw):
            cleaned = ISBN_CLEAN_RE.sub("", token).upper()
            if cleaned:
                values.append(cleaned)
    return unique(values)


def collect_olids(row: dict[str, str], header_map: dict[str, str]) -> List[str]:
    values: List[str] = []
    for key in OLID_KEYS:
        raw = row_value(row, header_map, key)
        if not raw:
            continue
        for token in split_tokens(raw):
            cleaned = OLID_CLEAN_RE.sub("", token).upper()
            if cleaned:
                values.append(cleaned)
    return unique(values)


def collect_cover_ids(row: dict[str, str], header_map: dict[str, str]) -> List[str]:
    values: List[str] = []
    for key in COVER_ID_KEYS:
        raw = row_value(row, header_map, key)
        if not raw:
            continue
        for token in split_tokens(raw):
            cleaned = re.sub(r"[^0-9]", "", token)
            if cleaned:
                values.append(cleaned)
    return unique(values)


def looks_like_library(row: dict[str, str], header_map: dict[str, str]) -> bool:
    export_id = row_value(row, header_map, "export_id", "group", "group_id", "section_id")
    obj_id = row_value(row, header_map, "id")
    entry_type = row_value(row, header_map, "type")

    def matches(token: str) -> bool:
        lower = token.lower()
        return lower.startswith("lib-") or "library" in lower

    for token in (export_id, obj_id):
        if token and matches(token):
            return True

    if entry_type and entry_type.strip().lower() == "book":
        return True

    return False


def extract_library_items(rows: List[dict[str, str]], header_map: dict[str, str]) -> List[LibraryItem]:
    items: List[LibraryItem] = []
    for idx, row in enumerate(rows):
        if not looks_like_library(row, header_map):
            continue
        book_id = row_value(row, header_map, "id") or row_value(row, header_map, "export_id")
        if not book_id:
            continue
        title = row_value(row, header_map, "name", "title")
        author = row_value(row, header_map, "author")
        items.append(
            LibraryItem(
                index=idx,
                book_id=book_id.strip(),
                title=title.strip(),
                author=author.strip(),
                isbns=collect_isbns(row, header_map),
                olids=collect_olids(row, header_map),
                cover_ids=collect_cover_ids(row, header_map),
            )
        )
    return items


def load_resources(path: pathlib.Path) -> Tuple[List[dict[str, str]], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV at {path} has no header")
        rows = []
        for raw in reader:
            row = {field: (raw.get(field) or "").strip() for field in reader.fieldnames}
            rows.append(row)
    header_map = build_header_map(reader.fieldnames)
    return rows, header_map


def score_doc(doc: dict, want_title: str, want_author: str) -> int:
    score = 0
    if "eng" in (doc.get("language") or []):
        score += 2
    title = (doc.get("title") or "").lower()
    author_blob = " ".join((doc.get("author_name") or [])).lower()
    if want_title and title == want_title.lower():
        score += 2
    if want_author and want_author.lower() in author_blob:
        score += 1
    return score


def search_open_library(title: str, author: str, timeout: int) -> List[Tuple[str, str]]:
    params = {"title": title, "limit": 5}
    if author:
        params["author"] = author
    try:
        response = requests.get(SEARCH_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[warn] search failed for '{title}': {exc}", file=sys.stderr)
        return []

    docs = response.json().get("docs", [])
    if not docs:
        return []

    docs.sort(key=lambda d: score_doc(d, title, author), reverse=True)
    top = docs[0]

    candidates: List[Tuple[str, str]] = []
    if top.get("cover_i"):
        candidates.append(("id", str(top["cover_i"])) )
    for edition in (top.get("edition_key") or [])[:5]:
        candidates.append(("olid", edition))
    for isbn in (top.get("isbn") or [])[:5]:
        candidates.append(("isbn", isbn))
    return candidates


def download_candidate(key: str, value: str, size: str, timeout: int) -> Tuple[str, bytes] | Tuple[None, None]:
    url = COVER_FMT.format(key=key, value=value, size=size)
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        print(f"[warn] cover fetch error for {key}:{value}: {exc}", file=sys.stderr)
        return (None, None)

    content_type = response.headers.get("content-type", "")
    if response.status_code == 200 and content_type.startswith("image/"):
        return url, response.content
    return (None, None)


def gather_direct_candidates(item: LibraryItem) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    for cid in item.cover_ids:
        candidates.append(("id", cid))
    for olid in item.olids:
        candidates.append(("olid", olid))
    for isbn in item.isbns:
        candidates.append(("isbn", isbn))
    return candidates


def fetch_cover_bytes(item: LibraryItem, size: str, timeout: int, pause: float) -> Tuple[str, bytes] | Tuple[None, None]:
    attempted: set[Tuple[str, str]] = set()

    for candidate in gather_direct_candidates(item):
        if candidate in attempted:
            continue
        attempted.add(candidate)
        url, content = download_candidate(*candidate, size=size, timeout=timeout)
        if content:
            return url, content
        if pause:
            time.sleep(pause)

    if not item.title:
        return (None, None)

    for candidate in search_open_library(item.title, item.author, timeout):
        if candidate in attempted:
            continue
        attempted.add(candidate)
        url, content = download_candidate(*candidate, size=size, timeout=timeout)
        if content:
            return url, content
        if pause:
            time.sleep(pause)

    return (None, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Open Library covers for local library entries.")
    parser.add_argument("--resources", type=pathlib.Path, default=DEFAULT_RESOURCES, help="Path to resources CSV")
    parser.add_argument("--dest", type=pathlib.Path, default=DEFAULT_DEST, help="Directory for downloaded covers")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="Open Library cover size (S, M, L)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Request timeout in seconds")
    parser.add_argument("--pause", type=float, default=0.15, help="Pause between API calls in seconds")
    parser.add_argument("--limit", type=int, help="Maximum number of downloads in this run")
    parser.add_argument("--force", action="store_true", help="Re-download even if a thumbnail already exists")
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    resources_path = args.resources if args.resources.is_absolute() else (pathlib.Path.cwd() / args.resources)
    dest_path = args.dest if args.dest.is_absolute() else (pathlib.Path.cwd() / args.dest)
    args.resources = resources_path.resolve()
    args.dest = dest_path.resolve()

    if not args.resources.exists():
        print(f"Resources CSV not found at {args.resources}", file=sys.stderr)
        sys.exit(1)

    try:
        rows, header_map = load_resources(args.resources)
    except Exception as exc:
        print(f"Failed to load resources CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    library_items = extract_library_items(rows, header_map)
    if not library_items:
        print("No library entries detected.")
        return

    args.dest.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped_existing = 0
    missing = 0

    for item in library_items:
        if args.limit is not None and downloaded >= args.limit:
            break

        if not item.book_id:
            missing += 1
            continue

        filename = f"{item.book_id}.jpg"
        out_path = args.dest / filename

        if not args.force and out_path.exists() and out_path.stat().st_size > 0:
            skipped_existing += 1
            continue

        url, content = fetch_cover_bytes(item, args.size, args.timeout, args.pause)
        if not content:
            missing += 1
            print(f"[info] no cover found for '{item.title or item.book_id}'")
            continue

        if args.dry_run:
            print(f"[dry-run] Would save cover for {item.book_id} from {url}")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)
            relative: pathlib.Path | str = out_path
            if out_path.is_absolute():
                for base in (REPO_ROOT, pathlib.Path.cwd()):
                    try:
                        relative = out_path.relative_to(base)
                        break
                    except ValueError:
                        continue
            if isinstance(relative, pathlib.Path):
                relative = relative.as_posix()
            print(f"Saved {relative} ({url})")
        downloaded += 1

    print(
        f"Done. downloaded={downloaded} skipped_existing={skipped_existing} missing={missing}"
    )


if __name__ == "__main__":
    main()
