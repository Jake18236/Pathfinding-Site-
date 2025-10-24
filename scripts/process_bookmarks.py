#!/usr/bin/env python3
"""
Sheet-queue controller for the tweet archiver.

Flow:
1) POST {action:"claim"} to Apps Script → returns one NEW item {id, url} and marks it PROCESSING (checkbox TRUE + note).
2) Run archive_tweet.py on that URL → produces markdown + media, returns meta.
3) POST {action:"complete", id, meta} to Apps Script → marks DONE and appends row to Learning tab.

Environment:
  SHEETS_WEBHOOK_URL   - required (Apps Script Web App URL)
  QUEUE_SHARED_SECRET  - optional (must match Apps Script SHARED_SECRET if set)

Repo layout assumptions:
  scripts/archive_tweet.py
  static/md/tweets, static/img/tweets, static/vid/tweets exist (archiver writes there)
"""

import json
import os
import pathlib
import subprocess
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARCHIVER = ROOT / "scripts" / "archive_tweet.py"

WEBHOOK_URL = os.environ.get("SHEETS_WEBHOOK_URL")
SHARED_SECRET = os.environ.get("QUEUE_SHARED_SECRET", "")

# Process only ONE item per run (protects free API & keeps workflows short)
MAX_PER_RUN = 1


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    try:
        return json.loads(text)
    except Exception:
        return {"_raw": text}


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


def to_sheet_meta(arch_meta: dict) -> dict:
    """Map archive_tweet.py output to the Learning sheet schema inputs."""
    return {
        "name":   arch_meta.get("name") or arch_meta.get("title") or "",
        "author": arch_meta.get("author") or "",
        "type":   ("thread" if "Thread" in (arch_meta.get("title") or "") else "tweet"),
        "link":   arch_meta.get("url") or "",
    }


def archive_url(url: str) -> dict | None:
    """Run archive_tweet.py and return its 'archived' meta dict."""
    proc = subprocess.run(
        ["python", str(ARCHIVER), url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"[error] archiver failed ({proc.returncode})\nSTDERR:\n{proc.stderr}", file=sys.stderr)
        return None

    # Last line of stdout is a JSON object like: {"archived": {...}, "md": "path"}
    try:
        last_line = proc.stdout.strip().splitlines()[-1]
        obj = json.loads(last_line)
        return obj.get("archived")
    except Exception as e:
        print(f"[error] could not parse archiver output: {e}\nSTDOUT:\n{proc.stdout}", file=sys.stderr)
        return None


def main():
    if not WEBHOOK_URL:
        print("No SHEETS_WEBHOOK_URL set; exiting.")
        sys.exit(0)

    processed_count = 0

    for _ in range(MAX_PER_RUN):
        item = claim_one()
        if not item:
            print("No queued items.")
            break

        item_id = item.get("id")
        url = (item.get("url") or "").strip()
        if not url:
            print(f"[warn] Claimed item {item_id} missing URL; skipping.")
            continue

        print(f"Archiving: {url} (queue id {item_id})")

        meta = archive_url(url)
        if not meta:
            print(f"[warn] Archiving failed for {url}. Leave row as PROCESSING; you may uncheck to requeue.")
            continue

        # Send minimal fields back for Learning
        try:
            sheet_meta = to_sheet_meta(meta)
            complete_item(item_id, sheet_meta)
            processed_count += 1
            print(f"Completed {url} → appended to Learning, marked DONE.")
        except Exception as e:
            print(f"[warn] Could not complete queue item {item_id}: {e}")

    print(f"Run finished. Processed {processed_count} item(s).")


if __name__ == "__main__":
    main()