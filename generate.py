import markdown
from bs4 import BeautifulSoup
import os
from datetime import datetime
import sys

def convert_md_to_html(md_file):
    with open(md_file, 'r') as f:
        md_content = f.readlines()
    title = md_content[0].replace('#', '').strip()
    md_content = ''.join(md_content[1:])
    html_content = markdown.markdown(md_content)
    return title, html_content

def update_blog_html(blog_html, post_title, post_date, post_filename):
    with open(blog_html, 'r') as f:
        soup = BeautifulSoup(f, 'html.parser')
    blog_posts = soup.find('div', class_='blog-posts')
    new_post = soup.new_tag('div', **{'class': 'blog-post'})
    post_meta = soup.new_tag('div', **{'class': 'post-meta'})
    post_meta.string = f"Veer Pareek - {post_date}"
    new_post.append(post_meta)
    post_title_tag = soup.new_tag('h2')
    post_link = soup.new_tag('a', href=post_filename)
    post_link.string = post_title
    post_title_tag.append(post_link)
    new_post.append(post_title_tag)
    blog_posts.insert(0, new_post)
    with open(blog_html, 'w') as f:
        f.write(str(soup))

def create_post_html(content, title, date, author, template_file):
    with open(template_file, 'r') as f:
        template = f.read()
    post_meta = f'<div class="post-meta">{date} • Written By {author}</div>'
    post_content = f'<h1 class="post-title">{title}</h1>{post_meta}<div class="post-content">{content}</div>'
    post_html = template.replace('{{ title }}', title).replace('{{ content }}', post_content)
    return post_html

def main(md_file, blog_html, template_file):
    # Convert Markdown to HTML
    title, html_content = convert_md_to_html(md_file)
    
    # Generate filename for new post
    post_filename = f"./blogs/posts_html/{os.path.splitext(os.path.basename(md_file))[0]}.html"
    
    # Update blog.html (listing page)
    post_date = datetime.now().strftime("%B %d, %Y")
    update_blog_html(blog_html, title, post_date, post_filename)
    
    # Create new HTML file for full post
    post_date_full = datetime.now().strftime("%B %d, %Y")  # "May 24, 2024" format
    author = "Veer Pareek"
    post_html = create_post_html(html_content, title, post_date_full, author, template_file)
    with open(post_filename, 'w') as f:
        f.write(post_html)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate.py <markdown_file> <blog_html> <template_file>")
        sys.exit(1)
    
    md_file = sys.argv[1]
    blog_html = sys.argv[2]
    template_file = sys.argv[3]
    
    main(md_file, blog_html, template_file)