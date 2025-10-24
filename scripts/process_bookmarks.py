#!/usr/bin/env python3
import pathlib, re, subprocess, json, os, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ====== YOUR TREE ======
BOOKMARKS           = ROOT / "static/txt/bookmarks.txt"
BOOKMARKS_PROCESSED = ROOT / "static/txt/processed_bookmarks.txt"
ARCHIVER            = ROOT / "scripts" / "archive_tweet.py"
# =======================

WEBHOOK_URL = os.environ.get("SHEETS_WEBHOOK_URL")

def read_lines(p: pathlib.Path):
    if not p.exists(): return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]

def write_lines(p: pathlib.Path, lines):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def is_tweet_url(url: str) -> bool:
    return bool(re.search(r"(https?://)?(x\.com|twitter\.com)/.+/status/\d+", url))

def post_rows(rows):
    if not WEBHOOK_URL or not rows: return
    data = json.dumps({"rows": rows}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_URL, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        _ = resp.read()

def to_sheet_row(meta: dict) -> dict:
    return {
        "name":   meta.get("name") or meta.get("title") or "",
        "author": meta.get("author") or "",
        "type":   ("thread" if "Thread" in (meta.get("title") or "") else "tweet"),
        "link":   meta.get("url") or "",
        "date":   (meta.get("date") or "")[:10],
    }

def main():
    processed = set(ln for ln in read_lines(BOOKMARKS_PROCESSED) if ln and not ln.startswith("#"))
    queue = [ln for ln in read_lines(BOOKMARKS) if ln and not ln.startswith("#")]
    to_process = [u for u in queue if is_tweet_url(u) and u not in processed]

    if not to_process:
        print("No new bookmarks to archive.")
        return

    # optional safety cap for API usage
    MAX_PER_RUN = 10
    to_process = to_process[:MAX_PER_RUN]

    print(f"Archiving {len(to_process)} new URL(s)…")
    rows_for_sheet = []
    for url in to_process:
        try:
            r = subprocess.run(
                ["python", str(ARCHIVER), url],
                capture_output=True, text=True, check=True
            )
            out = r.stdout.strip().splitlines()[-1]
            meta = json.loads(out).get("archived", {})
            if meta:
                rows_for_sheet.append(to_sheet_row(meta))
        except subprocess.CalledProcessError as e:
            print(f"[error] Failed to archive {url}: {e}\n{e.stderr}")

    if rows_for_sheet:
        try:
            post_rows(rows_for_sheet)
            print(f"Logged {len(rows_for_sheet)} row(s) to Google Sheets.")
        except Exception as e:
            print(f"[warn] Failed to log to Google Sheets: {e}")

    write_lines(BOOKMARKS_PROCESSED, sorted(set(list(processed) + to_process)))

if __name__ == "__main__":
    main()