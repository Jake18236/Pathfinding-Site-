#!/usr/bin/env python3
"""
Tweet Archiver - Fetches a single tweet using twarc2 and saves as Markdown.

Usage:
    python scripts/archive_tweet.py <tweet_url>

Requires:
    - TWITTER_BEARER_TOKEN environment variable
    - twarc2 installed (pip install twarc)
    - python-slugify installed (pip install python-slugify)
"""
import json, subprocess, tempfile, shutil, pathlib, re, datetime, sys, os
from slugify import slugify

# ====== OUTPUT DIRECTORIES ======
MD_DIR   = pathlib.Path("static/md/tweets")         # Markdown output
IMG_DIR  = pathlib.Path("static/img/tweets")        # images (jpg/png/gif)
VID_DIR  = pathlib.Path("static/vid/tweets")        # videos (mp4)
for d in (MD_DIR, IMG_DIR, VID_DIR):
    d.mkdir(parents=True, exist_ok=True)
# ================================

def get_bearer_token():
    """Get Twitter bearer token from environment."""
    token = os.environ.get("TWITTER_BEARER_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TWITTER_BEARER_TOKEN environment variable not set")
    return token

def run_to_file(cmd, outpath: pathlib.Path):
    """Run a command that writes to stdout; save stdout to a file."""
    print(f"[debug] Running: {' '.join(cmd)}", file=sys.stderr)
    with open(outpath, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, text=True, env=os.environ, stdout=f, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr}")

def run_optional(cmd):
    """Run a command that may fail (for optional operations like media download)."""
    print(f"[debug] Running (optional): {' '.join(cmd)}", file=sys.stderr)
    r = subprocess.run(cmd, text=True, env=os.environ, capture_output=True)
    if r.returncode != 0:
        print(f"[warn] Optional command failed: {' '.join(cmd)}\n{r.stderr}", file=sys.stderr)
        return False
    return True

def iso_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def tweet_id_from_url(url: str):
    """Extract tweet ID from a Twitter/X URL."""
    m = re.search(r"/status/(\d+)", url)
    return m.group(1) if m else ""

def extract_author(obj):
    """Extract author username from tweet object."""
    for key in ("author", "user"):
        if key in obj and isinstance(obj[key], dict):
            u = obj[key].get("username") or obj[key].get("screen_name")
            if u:
                return f"@{u}"
    return f"@{obj.get('author_id', 'unknown')}"

def fetch_tweet(url: str, workdir: pathlib.Path):
    """
    Fetch a single tweet using twarc2.

    Note: On Twitter's free API tier, we can only fetch individual tweets,
    not full conversations/threads. This is a limitation of the free tier.
    """
    raw_jsonl  = workdir / "tweet.jsonl"
    flat_jsonl = workdir / "tweet_flat.jsonl"

    bearer_token = get_bearer_token()

    tid = tweet_id_from_url(url) or url.strip()
    if not re.fullmatch(r"\d+", tid):
        raise RuntimeError(f"Could not extract numeric tweet ID from URL: {url}")

    print(f"[info] Fetching tweet {tid}...", file=sys.stderr)

    # Fetch single tweet (works on free tier)
    run_to_file(["twarc2", "--bearer-token", bearer_token, "tweet", tid], raw_jsonl)

    # Flatten the JSON for easier processing
    run_to_file(["twarc2", "--bearer-token", bearer_token, "flatten", str(raw_jsonl)], flat_jsonl)

    # Try to download media (non-fatal if it fails)
    tmp_media = workdir / "media"
    tmp_media.mkdir(exist_ok=True)
    media_success = run_optional([
        "twarc2", "--bearer-token", bearer_token,
        "media", str(raw_jsonl), "--download-dir", str(tmp_media)
    ])

    if not media_success:
        print("[info] Media download skipped or failed (tweet may have no media)", file=sys.stderr)

    # Move downloaded media to permanent locations
    moved = []
    for f in tmp_media.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            dest_dir = IMG_DIR
        elif ext in (".mp4", ".mov", ".m4v"):
            dest_dir = VID_DIR
        else:
            dest_dir = IMG_DIR
        target = dest_dir / f.name
        i = 2
        while target.exists():
            target = dest_dir / (f.stem + f"-{i}" + f.suffix)
            i += 1
        shutil.move(str(f), str(target))
        moved.append(target.name)
        print(f"[info] Saved media: {target.name}", file=sys.stderr)

    return raw_jsonl, flat_jsonl, moved

def read_jsonl(path):
    objs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                objs.append(json.loads(line))
    objs.sort(key=lambda o: o.get("created_at",""))
    return objs

def short_name_from_objs(objs):
    text = (objs[0].get("text","") or "").strip() if objs else ""
    text = re.sub(r"\s+", " ", text)
    return (text[:120] + "…") if len(text) > 120 else text

def build_markdown(objs, url: str, local_media_files):
    if not objs:
        md = MD_DIR / (slugify(iso_now()+"-tweet") + ".md")
        md.write_text(f"---\ntitle: \"Tweet\"\ndate: \"{iso_now()}\"\nurl: \"{url}\"\n---\n", encoding="utf-8")
        return md, {}

    first   = objs[0]
    author  = extract_author(first)
    created = first.get("created_at") or iso_now()
    is_thread = len(objs) > 1
    title   = f"{'Thread' if is_thread else 'Tweet'} by {author}"
    md_slug = slugify(f"{created[:10]}-{title}") or "tweet"
    md_path = MD_DIR / f"{md_slug}.md"

    lines = [
        "---",
        f'title: "{title}"',
        f'author: "{author}"',
        f'date: "{created}"',
        f'url: "{url}"',
        f'tweet_id: "{tweet_id_from_url(url)}"',
        f'media_count: {len(local_media_files)}',
        "---",
        ""
    ]
    for o in objs:
        ttime = o.get("created_at","")
        text  = (o.get("text","") or "").replace("\r","").strip()
        lines.append(f"**{ttime}**  \n{text}\n")

    if local_media_files:
        lines.append("<details><summary>Local media</summary>\n")
        for name in local_media_files:
            if name.lower().endswith(".mp4"):
                lines.append(f"- [video](/static/vid/tweets/{name})")
            else:
                lines.append(f"- ![](/static/img/tweets/{name})")
        lines.append("\n</details>\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # Get full tweet text (combine all tweets if thread)
    full_text = "\n\n".join((o.get("text","") or "").strip() for o in objs)

    # Categorize media files
    images = [f for f in local_media_files if not f.lower().endswith(('.mp4', '.mov', '.m4v'))]
    videos = [f for f in local_media_files if f.lower().endswith(('.mp4', '.mov', '.m4v'))]

    meta = {
        "url": url,
        "author": author,
        "title": title,
        "name": short_name_from_objs(objs),  # Sheet "Name"
        "date": created,
        "md_path": str(md_path),
        "media_count": len(local_media_files),
        "tweet_id": tweet_id_from_url(url),
        "text": full_text,
        "images": images,
        "videos": videos,
    }
    return md_path, meta

def archive(url: str):
    """Archive a tweet to Markdown with media."""
    print(f"[info] Archiving: {url}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as td:
        workdir = pathlib.Path(td)
        _, flat, local_media = fetch_tweet(url, workdir)
        objs = read_jsonl(flat)

        if not objs:
            print(f"[warn] No tweet data found for {url}", file=sys.stderr)

        md_path, meta = build_markdown(objs, url, local_media)
        print(f"[info] Created: {md_path}", file=sys.stderr)

        # Output JSON for caller (process_bookmarks.py expects this on last line)
        print(json.dumps({"archived": meta, "md": str(md_path)}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/archive_tweet.py <tweet_url>", file=sys.stderr)
        print("Requires TWITTER_BEARER_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    try:
        archive(sys.argv[1])
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)