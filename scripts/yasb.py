#!/usr/bin/env python3

# Copyright (c) 2022 Peter Bui <pbui@nd.edu>

# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

""" Yet Another Static Blogger """

import collections
import csv
import io
import os
import itertools
import sys

import dateutil.parser
import tornado.template
import markdown
import markdown.extensions.codehilite
import markdown.extensions.toc
import markdown.extensions.footnotes
import yaml
import re

try:
    import requests  # type: ignore
except Exception:
    requests = None

# Page

PageFields = 'title prefix icon navigation internal external body'.split()
Page       = collections.namedtuple('Page', PageFields)

def _load_csv_to_resources_map(src: str):
    """
    Load CSV from a URL or file path and return a mapping:
        { lecture_id: [ {name, type, link, student?}, ... ] }

    The CSV should contain columns: lecture_id, name, link, [type], [student].
    Header names are case-insensitive and spaces become underscores.
    """
    def normalize_headers(headers):
        return [h.strip().lower().replace(' ', '_') for h in headers]

    def best_of(row, *cands):
        for c in cands:
            if c in row and row[c]:
                return str(row[c]).strip()
        return ''

    # Fetch content
    text = ''
    if src.startswith('http://') or src.startswith('https://'):
        if not requests:
            raise RuntimeError("requests module not available to fetch CSV")
        r = requests.get(src, timeout=30, headers={
            'User-Agent': 'nd-cse-site-bot/1.0 (+github actions)'
        })
        r.raise_for_status()
        # Handle BOM and odd encodings
        try:
            text = r.content.decode('utf-8-sig')
        except Exception:
            text = r.text
    else:
        # Local file path; if missing, try env fallback URL
        try:
            with open(src, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            fallback = os.environ.get('COURSE_RESOURCES_CSV_URL', '')
            if fallback:
                if not requests:
                    raise RuntimeError("requests module not available to fetch CSV")
                r = requests.get(fallback, timeout=30, headers={'User-Agent': 'nd-cse-site-bot/1.0'})
                r.raise_for_status()
                try:
                    text = r.content.decode('utf-8-sig')
                except Exception:
                    text = r.text
            else:
                raise

    reader = csv.DictReader(io.StringIO(text))
    reader.fieldnames = normalize_headers(reader.fieldnames or [])

    out = {}
    total_rows = 0
    kept_rows = 0
    for raw in reader:
        total_rows += 1
        row = {k: (v or '').strip() for k, v in raw.items()}

        lecture_id = best_of(row, 'lecture_id', 'lecture', 'lecture id', 'topic_id')
        name = best_of(row, 'name', 'title', 'resource', 'resource_name')
        link = best_of(row, 'link', 'url', 'href')
        rtype = best_of(row, 'type', 'category', 'format') or 'reading'
        student = best_of(
            row,
            'student',
            'student_name',
            'student_credit',
            'student_contributor',
            'submitted_by',
            'submittedby',
            'attribution',
            'credit',
        )
        if not student:
            # Fall back to any remaining column that looks like a student credit.
            EXCLUDE_STUDENT_KEYS = ('repository', 'id', 'email', 'netid', 'username', 'link')
            for key, value in row.items():
                if not value:
                    continue
                if 'student' not in key:
                    continue
                if any(ex in key for ex in EXCLUDE_STUDENT_KEYS):
                    continue
                student = value
                break
        is_primary_raw = best_of(row, 'is_primary', 'primary', 'required')

        def to_bool(s: str) -> bool:
            if not s:
                return False
            s = s.strip().lower()
            return s in ('1', 'true', 'yes', 'y', 'required')

        if not lecture_id or not name or not link:
            continue

        entry = {'name': name, 'type': rtype, 'link': link}
        if student:
            entry['student'] = student
        if to_bool(is_primary_raw):
            entry['primary'] = True

        out.setdefault(lecture_id, []).append(entry)
        kept_rows += 1

    # Deduplicate
    for k, items in list(out.items()):
        seen = set()
        deduped = []
        for it in items:
            sig = (it.get('type', ''), it.get('name', ''), it.get('link', ''))
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(it)
        out[k] = deduped
    # Basic debug to stderr to aid troubleshooting in Actions logs
    try:
        import sys
        sys.stderr.write(f"[yasb] CSV resources: rows={total_rows}, kept={kept_rows}, lectures={len(out)}\n")
    except Exception:
        pass
    return out


def load_page_from_yaml(path):
    data     = yaml.safe_load(open(path))
    external = data.get('external', {}) or {}

    for k, v in external.items():
        if isinstance(v, str) and v.startswith('csv:'):
            src = v[len('csv:'):]
            data['external'][k] = _load_csv_to_resources_map(src)
        else:
            data['external'][k] = yaml.safe_load(open(v))

    if 'prefix' not in data:
        data['prefix'] = ''

    return Page(**data)

def render_page(page):
    hilite = markdown.extensions.codehilite.CodeHiliteExtension(noclasses=True)
    toc    = markdown.extensions.toc.TocExtension(permalink=True)
    footnotes = markdown.extensions.footnotes.FootnoteExtension()
    loader = tornado.template.Loader('templates')
    layout = u'''
{{% extends "base.tmpl" %}}

{{% block body %}}
{}
{{% end %}}
'''.format(markdown.markdown(page.body, extensions=['extra', toc, hilite, footnotes], output_format='html5'))

    template = tornado.template.Template(layout, loader=loader)
    def slugify(s: str) -> str:
        s = (s or '').lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        s = s.strip('-')
        return s

    # Known aliases where the topic text doesn't match the desired slug
    LECTURE_ALIASES = {
        'syllabus, history of ai': 'introduction',
        'intro to ai': 'introduction',
    }

    def lecture_id_for(topic: str) -> str:
        key = (topic or '').strip().lower()
        slug = LECTURE_ALIASES.get(key, slugify(topic))
        return f"lec-{slug}" if slug else ''

    def resources_for(resources_map, topic_or_id: str):
        if not isinstance(resources_map, dict):
            return []
        key = (topic_or_id or '').strip()
        # If caller passed a full id like 'lec-...'
        if key.startswith('lec-'):
            return resources_map.get(key, [])
        # Otherwise compute from topic text
        lid = lecture_id_for(key)
        return resources_map.get(lid, [])

    def _normalized(value: str) -> str:
        return (value or '').strip().lower()

    def _search_resources(resources_map, ids_iter, target_name, keywords=None):
        for lid in ids_iter:
            if not lid:
                continue
            for resource in resources_map.get(lid, []):
                if _normalized(resource.get('name')) != target_name:
                    continue
                rtype = _normalized(resource.get('type'))
                if keywords:
                    if not rtype:
                        continue
                    if not any(keyword in rtype for keyword in keywords):
                        continue
                return resource
        return None

    def find_assignment_resource(resources_map, assignment_name: str, lecture_id: str = ''):
        if not isinstance(resources_map, dict):
            return None

        target_name = _normalized(assignment_name)
        if not target_name:
            return None

        preferred_keywords = (
            'assignment',
            'homework',
            'project',
            'exam',
            'quiz',
            'practice',
            'solution',
        )

        candidate_ids = []
        if lecture_id:
            candidate_ids.append(lecture_id)
        slug = lecture_id_for(assignment_name)
        if slug and slug not in candidate_ids:
            candidate_ids.append(slug)

        resource = _search_resources(resources_map, candidate_ids, target_name, preferred_keywords)
        if resource:
            return resource

        resource = _search_resources(resources_map, candidate_ids, target_name)
        if resource:
            return resource

        resource = _search_resources(resources_map, resources_map.keys(), target_name, preferred_keywords)
        if resource:
            return resource

        return _search_resources(resources_map, resources_map.keys(), target_name)

    settings = {
        'page'      : page,
        'dateutil'  : dateutil,
        'itertools' : itertools,
        'slugify'   : slugify,
        'lecture_id_for': lecture_id_for,
        'resources_for': resources_for,
        'find_assignment_resource': find_assignment_resource,
    }
    print(template.generate(**settings).decode())

# Main Execution
def main():
    for path in sys.argv[1:]:
        page = load_page_from_yaml(path)
        render_page(page)

if __name__ == '__main__':
    main()

# vim: set sts=4 sw=4 ts=8 expandtab ft=python:
