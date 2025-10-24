#!/usr/bin/env python3
"""
Archive a Tweet or an author's entire thread using snscrape (no API key).

Usage:
  python scripts/archive_tweet.py <tweet_or_thread_url>

Env:
  ARCHIVE_THREAD = "true" | "false"   (default: false)  → when true, fetch the author's whole thread
  SKIP_VIDEO     = "true" | "false"   (default: false)  → when true, do not download videos
"""

import os, re, sys, json, pathlib, datetime, shutil, requests
from slugify import slugify
import snscrape.modules.twitter as sntwitter

# ====== OUTPUT TREE ======
MD_DIR   = pathlib.Path("static/md/tweets")         # Markdown output
IMG_DIR  = pathlib.Path("static/img/tweets")        # images
VID_DIR  = pathlib.Path("static/vid/tweets")        # videos
for d in (MD_DIR, IMG_DIR, VID_DIR):
    d.mkdir(parents=True, exist_ok=True)
# =========================

ARCHIVE_THREAD = os.environ.get("ARCHIVE_THREAD", "false").lower() == "true"
SKIP_VIDEO     = os.environ.get("SKIP_VIDEO", "false").lower() == "true"

HTTP_TIMEOUT = 30  # seconds per media download


# ---------- helpers ----------
def iso_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def tweet_id_from_url(url: str):
    m = re.search(r"/status/(\d+)", url)
    return m.group(1) if m else None

def safe_write_bytes(path: pathlib.Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)

def pick_best_video_variant(media):
    """snscrape Video/Gif objects expose .variants (list with .contentType, .url, .bitrate(?) )"""
    variants = getattr(media, "variants", None) or []
    # Prefer mp4; if bitrate is present, pick highest
    mp4s = [v for v in variants if "mp4" in (getattr(v, "contentType", "") or "").lower()]
    if not mp4s:
        return None
    # Some variants have .bitrate; if missing, just pick the last
    def vb(v):
        b = getattr(v, "bitrate", None)
        try: return int(b) if b is not None else -1
        except: return -1
    mp4s.sort(key=vb, reverse=True)
    return mp4s[0].url

def download(url: str, dest_dir: pathlib.Path) -> str:
    # Create a simple filename from the tail of the URL
    tail = url.split("/")[-1].split("?")[0] or "media"
    # add extension if not present
    if "." not in tail:
        # guess from content-type
        head = requests.head(url, timeout=HTTP_TIMEOUT, allow_redirects=True)
        ct = head.headers.get("content-type", "")
        if "png" in ct: tail += ".png"
        elif "gif" in ct: tail += ".gif"
        elif "mp4" in ct: tail += ".mp4"
        else: tail += ".jpg"
    name = slugify(tail, lowercase=False)  # keep extension case
    target = dest_dir / name
    # avoid overwrite
    i = 2
    while target.exists():
        stem, ext = target.stem, target.suffix
        target = dest_dir / f"{stem}-{i}{ext}"
        i += 1

    r = requests.get(url, timeout=HTTP_TIMEOUT, stream=True)
    r.raise_for_status()
    with open(target, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 14):
            if chunk:
                f.write(chunk)
    return target.name

def extract_author(tweet_obj):
    u = tweet_obj.user.username if getattr(tweet_obj, "user", None) else None
    return f"@{u}" if u else "@unknown"

def short_name(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return (text[:120] + "…") if len(text) > 120 else text

# ---------- core scrape ----------
def fetch_head_tweet(tid: str):
    head = next(sntwitter.TwitterTweetScraper(tid).get_items(), None)
    if not head:
        raise RuntimeError("Tweet not found or not public.")
    return head

def fetch_author_thread(head_tweet):
    """Return only tweets by the same author in this conversation (ignores other users' replies)."""
    username = head_tweet.user.username
    conv_id  = head_tweet.conversationId
    query = f"conversation_id:{conv_id} from:{username}"
    items = list(sntwitter.TwitterSearchScraper(query).get_items())
    # Include head if not present (snscrape search may omit it)
    if not any(getattr(t, "id", None) == head_tweet.id for t in items):
        items.append(head_tweet)
    # Sort by date ascending
    items.sort(key=lambda t: t.date)
    return items

def collect_media(tweet_obj):
    imgs, vids = [], []
    media = getattr(tweet_obj, "media", None) or []
    for m in media:
        # Photo
        if hasattr(m, "fullUrl") and "video" not in (getattr(m, "type", "") or "").lower():
            try:
                name = download(m.fullUrl, IMG_DIR)
                imgs.append(name)
            except Exception:
                pass
        # Video/GIF (as MP4 if available)
        elif not SKIP_VIDEO and hasattr(m, "variants"):
            best = pick_best_video_variant(m)
            if best:
                try:
                    name = download(best, VID_DIR)
                    vids.append(name)
                except Exception:
                    pass
    return imgs, vids

# ---------- markdown ----------
def build_markdown(url: str, tweets: list):
    if not tweets:
        md = MD_DIR / (slugify(iso_now()+"-tweet") + ".md")
        md.write_text(f"---\ntitle: \"Tweet\"\ndate: \"{iso_now()}\"\nurl: \"{url}\"\n---\n", encoding="utf-8")
        return md, {}

    author   = extract_author(tweets[0])
    created  = tweets[0].date.isoformat()
    is_thread = len(tweets) > 1
    title    = f"{'Thread' if is_thread else 'Tweet'} by {author}"
    md_slug  = slugify(f"{created[:10]}-{title}") or "tweet"
    md_path  = MD_DIR / f"{md_slug}.md"

    # Gather media across all tweets
    all_img, all_vid = [], []
    lines = [
        "---",
        f'title: "{title}"',
        f'author: "{author}"',
        f'date: "{created}"',
        f'url: "{url}"',
        f'tweet_id: "{tweet_id_from_url(url) or ""}"',
        f'media_count: 0',  # patch later
        "---",
        ""
    ]

    for tw in tweets:
        ttime = tw.date.isoformat()
        text  = tw.rawContent or ""
        lines.append(f"**{ttime}**  \n{text}\n")
        imgs, vids = collect_media(tw)
        all_img.extend(imgs)
        all_vid.extend(vids)

    media_total = len(all_img) + len(all_vid)
    # Patch media count in front matter
    lines[6] = f"media_count: {media_total}"

    if media_total:
        lines.append("<details><summary>Local media</summary>\n")
        for name in all_img:
            lines.append(f"- ![](static/img/tweets/{name})")
        for name in all_vid:
            lines.append(f"- [video](static/vid/tweets/{name})")
        lines.append("\n</details>\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    # Minimal meta for the Sheet 'Learning' row
    name_for_sheet = short_name(tweets[0].rawContent or title)
    meta = {
        "url": url,
        "author": author,
        "title": title,
        "name": name_for_sheet,
        "date": created,
        "md_path": str(md_path),
        "media_count": media_total,
        "tweet_id": tweet_id_from_url(url) or "",
        "type": "thread" if is_thread else "tweet",
        "link": url,
    }
    return md_path, meta

# ---------- main ----------
def archive(url: str):
    tid = tweet_id_from_url(url)
    if not tid:
        raise SystemExit("Could not extract tweet ID from URL.")

    head = fetch_head_tweet(tid)
    tweets = [head]
    if ARCHIVE_THREAD:
        tweets = fetch_author_thread(head)

    md_path, meta = build_markdown(url, tweets)
    print(json.dumps({"archived": meta, "md": str(md_path)}))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/archive_tweet.py <tweet_or_thread_url>")
        sys.exit(1)
    archive(sys.argv[1])