#!/usr/bin/env python3
"""Fetch YouTube thumbnails for video items listed in ``resources.csv``."""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import requests


# YouTube thumbnail URL patterns
# mqdefault.jpg is 320x180, hqdefault.jpg is 480x360, maxresdefault.jpg is 1280x720
THUMBNAIL_URL = "https://img.youtube.com/vi/{video_id}/mqdefault.jpg"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_RESOURCES = REPO_ROOT / "static" / "csv" / "resources.csv"
DEFAULT_DEST = REPO_ROOT / "static" / "img" / "youtube_thumbnails"
DEFAULT_TIMEOUT = 20

# Regex patterns to extract YouTube video IDs
YOUTUBE_VIDEO_RE = re.compile(
    r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})'
)
YOUTUBE_PLAYLIST_RE = re.compile(
    r'youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)'
)


def normalize_key(value: str) -> str:
    return (value or "").strip().lower().replace(" ", "_")


def build_header_map(fieldnames: list[str]) -> dict[str, str]:
    return {normalize_key(name): name for name in fieldnames if name}


def row_value(row: dict[str, str], header_map: dict[str, str], *keys: str) -> str:
    for key in keys:
        actual = header_map.get(normalize_key(key))
        if actual is None:
            continue
        return (row.get(actual) or "").strip()
    return ""


def slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


@dataclass
class VideoItem:
    index: int
    name: str
    slug: str
    link: str
    video_id: str | None
    playlist_id: str | None
    explicit_video_id: str | None  # From CSV video_id column


def extract_youtube_id(url: str) -> Tuple[str | None, str | None]:
    """Extract video ID or playlist ID from a YouTube URL."""
    video_match = YOUTUBE_VIDEO_RE.search(url)
    if video_match:
        return video_match.group(1), None

    playlist_match = YOUTUBE_PLAYLIST_RE.search(url)
    if playlist_match:
        return None, playlist_match.group(1)

    return None, None


def looks_like_video(row: dict[str, str], header_map: dict[str, str]) -> bool:
    """Check if this row is a video entry."""
    entry_type = row_value(row, header_map, "type")
    link = row_value(row, header_map, "link", "url")

    if entry_type.lower() == "video":
        return True
    if link and ("youtube.com" in link or "youtu.be" in link):
        return True
    return False


def extract_video_items(rows: List[dict[str, str]], header_map: dict[str, str]) -> List[VideoItem]:
    """Extract video items from CSV rows."""
    items: List[VideoItem] = []
    for idx, row in enumerate(rows):
        if not looks_like_video(row, header_map):
            continue

        name = row_value(row, header_map, "name", "title")
        link = row_value(row, header_map, "link", "url")
        explicit_vid = row_value(row, header_map, "video_id", "youtube_id", "yt_id")

        if not name or not link:
            continue

        video_id, playlist_id = extract_youtube_id(link)

        # Skip if we can't extract any YouTube ID and no explicit ID provided
        if not video_id and not playlist_id and not explicit_vid:
            continue

        items.append(
            VideoItem(
                index=idx,
                name=name.strip(),
                slug=slugify(name),
                link=link.strip(),
                video_id=video_id,
                playlist_id=playlist_id,
                explicit_video_id=explicit_vid if explicit_vid else None,
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


def get_playlist_first_video_id(playlist_id: str, timeout: int) -> str | None:
    """
    Try to get the first video ID from a playlist.
    This uses a simple approach - trying common first video patterns.
    For a more robust solution, you'd need the YouTube Data API.
    """
    # For playlists, we can try to fetch the playlist page and extract a video ID
    # But this is unreliable without API access. Instead, we'll return None
    # and let the caller handle it (e.g., by using a placeholder or skipping)
    return None


def fetch_thumbnail(video_id: str, timeout: int) -> bytes | None:
    """Fetch thumbnail for a YouTube video."""
    url = THUMBNAIL_URL.format(video_id=video_id)
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            # Check if it's actually an image (not a placeholder)
            content_type = response.headers.get("content-type", "")
            if content_type.startswith("image/"):
                # YouTube returns a gray placeholder for invalid video IDs
                # Check if the image is large enough to be real content
                if len(response.content) > 1000:  # Real thumbnails are > 1KB
                    return response.content
    except requests.RequestException as exc:
        print(f"[warn] thumbnail fetch error for {video_id}: {exc}", file=sys.stderr)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch YouTube thumbnails for video entries.")
    parser.add_argument("--resources", type=pathlib.Path, default=DEFAULT_RESOURCES, help="Path to resources CSV")
    parser.add_argument("--dest", type=pathlib.Path, default=DEFAULT_DEST, help="Directory for downloaded thumbnails")
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

    video_items = extract_video_items(rows, header_map)
    if not video_items:
        print("No video entries detected.")
        return

    args.dest.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped_existing = 0
    skipped_playlist = 0
    missing = 0

    for item in video_items:
        if args.limit is not None and downloaded >= args.limit:
            break

        # Check for existing thumbnail
        candidate_paths = [
            args.dest / f"{item.slug}{ext}"
            for ext in (".jpg", ".jpeg", ".png", ".webp")
        ]
        existing_path = next((p for p in candidate_paths if p.exists()), None)

        if existing_path and not args.force and existing_path.stat().st_size > 0:
            skipped_existing += 1
            continue

        out_path = existing_path if existing_path else candidate_paths[0]

        # Determine video ID to fetch (explicit > extracted > playlist fallback)
        video_id = item.explicit_video_id or item.video_id
        if not video_id and item.playlist_id:
            # For playlists without explicit video_id, we can't easily get the first video
            print(f"[info] skipping playlist '{item.name}' - add video_id column to CSV for playlist thumbnails")
            skipped_playlist += 1
            continue

        if not video_id:
            missing += 1
            print(f"[info] no video ID found for '{item.name}'")
            continue

        content = fetch_thumbnail(video_id, args.timeout)
        if not content:
            missing += 1
            print(f"[info] no thumbnail found for '{item.name}' (video_id: {video_id})")
            continue

        if args.dry_run:
            print(f"[dry-run] Would save thumbnail for '{item.name}' to {out_path.name}")
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
            print(f"Saved {relative}")
        downloaded += 1

        if args.pause:
            time.sleep(args.pause)

    print(
        f"Done. downloaded={downloaded} skipped_existing={skipped_existing} "
        f"skipped_playlist={skipped_playlist} missing={missing}"
    )


if __name__ == "__main__":
    main()
