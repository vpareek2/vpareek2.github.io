#!/usr/bin/env python3
"""
Builds blog index and converts Markdown posts to HTML.

Usage:
  python scripts/build_posts.py

Requires:
  pip install markdown beautifulsoup4 python-frontmatter python-markdown-math
"""
from pathlib import Path
import datetime as dt
import frontmatter
import markdown
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MD_DIR = ROOT / 'blogs' / 'posts'
OUT_DIR = ROOT / 'blogs' / 'posts_html'
BLOG_HTML = ROOT / 'blog.html'
POST_TEMPLATE = ROOT / 'post_template.html'
INDEX_HTML = ROOT / 'index.html'
SKIP_SLUGS = { 'blog2', 'blog1' }  # skip duplicate/placeholder/removed

def _strip_leading_h1(md_text):
    lines = md_text.splitlines()
    i = 0
    # Skip leading empties
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith('#'):
        # drop this heading line
        i += 1
        # also drop a single blank right after, if present
        if i < len(lines) and lines[i].strip() == '':
            i += 1
        return '\n'.join(lines[i:])
    return md_text

def read_posts():
    posts = []
    for md_file in sorted(MD_DIR.glob('*.md')):
        fm = frontmatter.load(md_file)
        # Determine title and strip duplicate H1 from content
        raw = fm.content
        # If first non-empty line is markdown H1, capture it
        first_line = next((ln for ln in raw.splitlines() if ln.strip() != ''), '')
        derived_title = first_line.lstrip('# ').strip() if first_line.lstrip().startswith('#') else ''
        title = fm.get('title') or derived_title or f"Post: {md_file.stem}"
        content_body = _strip_leading_h1(raw)
        date_s = fm.get('date')
        try:
            date = dt.datetime.fromisoformat(str(date_s)) if date_s else dt.datetime.fromtimestamp(md_file.stat().st_mtime)
        except Exception:
            date = dt.datetime.fromtimestamp(md_file.stat().st_mtime)
        summary = fm.get('summary') or ''
        tags = fm.get('tags') or []
        slug = md_file.stem
        slug = md_file.stem
        if fm.get('draft') or slug in SKIP_SLUGS:
            continue
        posts.append({
            'title': title,
            'date': date,
            'summary': summary,
            'tags': tags,
            'slug': slug,
            'content': content_body,
        })
    posts.sort(key=lambda x: x['date'], reverse=True)
    return posts

def render_post_html(title, date, author, content_html):
    template = POST_TEMPLATE.read_text(encoding='utf-8')
    meta = f'<div class="post-meta">{date.strftime("%B %d, %Y")} • Written By {author}</div>'
    body = f'<h1 class="post-title">{title}</h1>{meta}<div class="post-content">{content_html}</div>'
    return template.replace('{{ title }}', title).replace('{{ content }}', body)

def build_posts(posts):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in posts:
        # Preserve LaTeX math delimiters so MathJax can render client-side.
        # pymdownx.arithmatex processes \(\) and \[\] (and $) ahead of markdown,
        # preventing backslash-escape removal by the parser.
        extensions = ['fenced_code', 'tables', 'toc', 'mdx_math']
        html = markdown.markdown(p['content'], extensions=extensions)
        out = OUT_DIR / f"{p['slug']}.html"
        out.write_text(render_post_html(p['title'], p['date'], 'Veer', html), encoding='utf-8')

def update_blog_index(posts):
    soup = BeautifulSoup(BLOG_HTML.read_text(encoding='utf-8'), 'html.parser')
    ul = soup.find(id='post-list')
    if not ul:
        return
    ul.clear()
    for p in posts:
        li = soup.new_tag('li', **{'class': 'post-card'})
        a = soup.new_tag('a', href=f"blogs/posts_html/{p['slug']}.html", **{'class': 'blog-link'})
        container = soup.new_tag('div')
        d = soup.new_tag('div', **{'class': 'pcard-date'})
        d.string = p['date'].strftime('%B %d, %Y')
        t = soup.new_tag('div', **{'class': 'pcard-title'})
        t.string = p['title']
        ex = soup.new_tag('div', **{'class': 'pcard-excerpt'})
        ex.string = (p['summary'] or '')[:220]
        container.append(d); container.append(t); container.append(ex)
        a.append(container)
        li.append(a)
        ul.append(li)
    BLOG_HTML.write_text(str(soup), encoding='utf-8')

def update_home_recent(posts):
    # Take top 3 (or fewer)
    top = posts[:3]
    soup = BeautifulSoup(INDEX_HTML.read_text(encoding='utf-8'), 'html.parser')
    grid = soup.find(id='home-recent')
    if not grid:
        return
    grid.clear()
    for p in top:
        a = soup.new_tag('a', href=f"blogs/posts_html/{p['slug']}.html", **{'class': 'post-card'})
        d = soup.new_tag('div', **{'class': 'pcard-date'})
        d.string = p['date'].strftime('%B %d, %Y')
        t = soup.new_tag('h3', **{'class': 'pcard-title'})
        t.string = p['title']
        ex = soup.new_tag('p', **{'class': 'pcard-excerpt'})
        ex.string = (p['summary'] or '')[:220]
        a.append(d); a.append(t); a.append(ex)
        grid.append(a)
    INDEX_HTML.write_text(str(soup), encoding='utf-8')

def main():
    posts = read_posts()
    build_posts(posts)
    update_blog_index(posts)
    update_home_recent(posts)
    print(f"Built {len(posts)} posts and updated blog index.")

if __name__ == '__main__':
    main()
