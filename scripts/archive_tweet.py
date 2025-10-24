#!/usr/bin/env python3
import json, subprocess, tempfile, shutil, pathlib, re, datetime, sys, os
from slugify import slugify

# ====== YOUR TREE ======
MD_DIR   = pathlib.Path("static/md/tweets")         # Markdown output
IMG_DIR  = pathlib.Path("static/img/tweets")        # images (jpg/png/gif)
VID_DIR  = pathlib.Path("static/vid/tweets")        # videos (mp4)
for d in (MD_DIR, IMG_DIR, VID_DIR):
    d.mkdir(parents=True, exist_ok=True)
# =======================

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr}")
    return r

def run_check(cmd):
    # run a command and fail with helpful stderr/stdout
    r = subprocess.run(cmd, text=True, env=os.environ, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr or r.stdout}")
    return r

def run_to_file(cmd, outpath: pathlib.Path):
    # run a command that writes to stdout; save stdout to a file
    with open(outpath, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, text=True, env=os.environ, stdout=f, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr}")

def iso_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def tweet_id_from_url(url: str):
    m = re.search(r"/status/(\d+)", url)
    return m.group(1) if m else ""

def extract_author(obj):
    for key in ("author","user"):
        if key in obj and isinstance(obj[key], dict):
            u = obj[key].get("username") or obj[key].get("screen_name")
            if u: return f"@{u}"
    return f"@{obj.get('author_id','unknown')}"

def fetch_thread(url: str, workdir: pathlib.Path):
    raw_jsonl  = workdir / "thread.jsonl"
    flat_jsonl = workdir / "thread_flat.jsonl"

    tid = tweet_id_from_url(url) or url.strip()
    if not re.fullmatch(r"\d+", tid):
        raise RuntimeError(f"Could not extract numeric tweet ID from URL: {url}")

    # Prefer the full conversation (writes to stdout, we capture to file)
    try:
        run_to_file(["twarc2", "conversation", tid], raw_jsonl)
    except RuntimeError as e:
        print(f"[warn] conversation fetch failed, falling back to single tweet: {e}")
        run_to_file(["twarc2", "tweet", tid], raw_jsonl)

    # Flatten: reads raw_jsonl, writes to stdout; capture to flat_jsonl
    run_to_file(["twarc2", "flatten", str(raw_jsonl)], flat_jsonl)

    # Media: reads raw_jsonl and downloads to a directory
    tmp_media = workdir / "media"
    tmp_media.mkdir(exist_ok=True)
    run_check(["twarc2", "media", str(raw_jsonl), "--download-dir", str(tmp_media)])

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

    meta = {
        "url": url,
        "author": author,
        "title": title,
        "name": short_name_from_objs(objs),  # Sheet "Name"
        "date": created,
        "md_path": str(md_path),
        "media_count": len(local_media_files),
        "tweet_id": tweet_id_from_url(url),
    }
    return md_path, meta

def archive(url: str):
    with tempfile.TemporaryDirectory() as td:
        workdir = pathlib.Path(td)
        _, flat, local_media = fetch_thread(url, workdir)
        objs = read_jsonl(flat)
        md_path, meta = build_markdown(objs, url, local_media)
        print(json.dumps({"archived": meta, "md": str(md_path)}))  # for caller

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/archive_tweet.py <tweet_or_thread_url>")
        sys.exit(1)
    archive(sys.argv[1])