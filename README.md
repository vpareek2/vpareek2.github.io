# vpareek2.github.io

Local build notes
- Dependencies: `markdown`, `beautifulsoup4`, `python-frontmatter`, `python-markdown-math`
- Install: `python3 -m pip install -U markdown beautifulsoup4 python-frontmatter python-markdown-math`
- Build posts and indexes: `python3 scripts/build_posts.py`

Math rendering
- Posts are written in Markdown with LaTeX using `\(...\)` and `\[...\]` or `$...$` / `$$...$$`.
- The builder preserves math and pages load MathJax v3 to render equations.
