#!/usr/bin/env python3
"""
Process Bookmarks Queue - Claims items from Google Sheet queue and archives them.

This script:
1. Claims a pending item from the Google Sheet queue
2. Calls archive_tweet.py to fetch and archive the tweet
3. Marks the item as complete (or releases it for retry on failure)

Requires:
    - SHEETS_WEBHOOK_URL environment variable (Google Apps Script web app URL)
    - TWITTER_BEARER_TOKEN environment variable (for archive_tweet.py)
"""
import csv
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARCHIVER = ROOT / "scripts" / "archive_tweet.py"
RESOURCES_CSV = ROOT / "static" / "csv" / "resources.csv"

WEBHOOK_URL = os.environ.get("SHEETS_WEBHOOK_URL")
SHARED_SECRET = os.environ.get("QUEUE_SHARED_SECRET", "")

MAX_PER_RUN = 1  # Process one item per run to stay within API limits

def post_json(url: str, payload: dict) -> dict:
    """POST JSON to webhook and return response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        return json.loads(text)
    except urllib.error.HTTPError as e:
        print(f"[error] HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return {"error": str(e)}
    except json.JSONDecodeError:
        return {"_raw": text}
    except Exception as e:
        print(f"[error] Request failed: {e}", file=sys.stderr)
        return {"error": str(e)}

def claim_one() -> dict | None:
    payload = {"action": "claim"}
    if SHARED_SECRET:
        payload["secret"] = SHARED_SECRET
    res = post_json(WEBHOOK_URL, payload)
    return res.get("item")

def complete_item(item_id: int | str, meta: dict) -> None:
    payload = {"action": "complete", "id": item_id, "meta": meta}
    if SHARED_SECRET:
        payload["secret"] = SHARED_SECRET
    _ = post_json(WEBHOOK_URL, payload)

def release_item(item_id: int | str) -> None:
    payload = {"action": "release", "id": item_id}
    if SHARED_SECRET:
        payload["secret"] = SHARED_SECRET
    _ = post_json(WEBHOOK_URL, payload)

def to_sheet_meta(arch_meta: dict) -> dict:
    return {
        "name":   arch_meta.get("name") or arch_meta.get("title") or "",
        "author": arch_meta.get("author") or "",
        "type":   ("thread" if "Thread" in (arch_meta.get("title") or "") else "tweet"),
        "link":   arch_meta.get("url") or "",
    }

def append_to_resources_csv(meta: dict) -> None:
    """Append archived tweet to resources.csv for display in learning section."""
    # Parse the tweet date to get year for export_id
    tweet_date = meta.get("date", "")[:10]  # YYYY-MM-DD
    year = tweet_date[:4] if tweet_date else "2025"

    # Build media paths as JSON for the notes field
    images = meta.get("images", [])
    videos = meta.get("videos", [])
    media_json = json.dumps({"images": images, "videos": videos}) if (images or videos) else ""

    # Escape tweet text for CSV (replace newlines, handle quotes)
    tweet_text = (meta.get("text") or "").replace("\n", " ").replace("\r", "")

    row = {
        "export_id": f"learning-{year}",
        "name": meta.get("name") or "",
        "type": "thread" if "Thread" in (meta.get("title") or "") else "tweet",
        "link": meta.get("url") or "",
        "author": meta.get("author") or "",
        "date_added": tweet_date,
        "cover": "",
        "checked_out": "",
        "rating": "",
        "notes": tweet_text,
        "media": media_json,
    }

    # Check if CSV exists and has headers
    file_exists = RESOURCES_CSV.exists()

    # Read existing rows to check for duplicates
    if file_exists:
        with open(RESOURCES_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for existing in reader:
                if existing.get("link") == row["link"]:
                    print(f"[info] Tweet already in resources.csv, skipping: {row['link']}", file=sys.stderr)
                    return

    # Get fieldnames from existing file or use defaults
    fieldnames = ["export_id", "name", "type", "link", "author", "date_added", "cover", "checked_out", "rating", "notes", "media"]
    if file_exists:
        with open(RESOURCES_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                # Add 'media' field if not present
                fieldnames = list(reader.fieldnames)
                if "media" not in fieldnames:
                    fieldnames.append("media")

    # Append row
    with open(RESOURCES_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[info] Added to resources.csv: {row['name'][:50]}...", file=sys.stderr)

def archive_url(url: str) -> dict | None:
    proc = subprocess.run(
        ["python", str(ARCHIVER), url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"[error] archiver failed ({proc.returncode})\nSTDERR:\n{proc.stderr}", file=sys.stderr)
        return None

    try:
        last_line = proc.stdout.strip().splitlines()[-1]
        obj = json.loads(last_line)
        return obj.get("archived")
    except Exception as e:
        print(f"[error] could not parse archiver output: {e}\nSTDOUT:\n{proc.stdout}", file=sys.stderr)
        return None

def main():
    if not WEBHOOK_URL:
        print("[info] No SHEETS_WEBHOOK_URL set; exiting.", file=sys.stderr)
        sys.exit(0)

    # Verify Twitter token is set before trying to process
    if not os.environ.get("TWITTER_BEARER_TOKEN"):
        print("[error] TWITTER_BEARER_TOKEN not set; exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Using webhook: {WEBHOOK_URL[:50]}...", file=sys.stderr)
    processed_count = 0

    for _ in range(MAX_PER_RUN):
        item = claim_one()
        if not item:
            print("[info] No queued items.", file=sys.stderr)
            break

        item_id = item.get("id")
        url = (item.get("url") or "").strip()
        if not url:
            print(f"[warn] Claimed item {item_id} missing URL; releasing.", file=sys.stderr)
            release_item(item_id)
            continue

        print(f"[info] Archiving: {url} (queue id {item_id})", file=sys.stderr)

        meta = archive_url(url)
        if not meta:
            print(f"[warn] Archiving failed for {url}. Releasing queue item {item_id}.", file=sys.stderr)
            release_item(item_id)
            continue

        try:
            # Add to resources.csv for display in learning section
            append_to_resources_csv(meta)

            sheet_meta = to_sheet_meta(meta)
            complete_item(item_id, sheet_meta)
            processed_count += 1
            print(f"[info] Completed: {url}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] Could not complete queue item {item_id}: {e}", file=sys.stderr)
            release_item(item_id)

    print(f"[info] Run finished. Processed {processed_count} item(s).", file=sys.stderr)

if __name__ == "__main__":
    main()