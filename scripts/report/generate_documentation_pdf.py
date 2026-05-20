from pathlib import Path
from markdown import markdown
from weasyprint import HTML, CSS

INPUT_MARKDOWN = Path('project_documentation.md')
OUTPUT_PDF = Path('project_documentation.pdf')
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

CSS_CONTENT = f'''
@font-face {{
    font-family: 'NotoSansCJK';
    src: url('file://{FONT_PATH}');
}}
body {{
    font-family: 'NotoSansCJK', sans-serif;
    line-height: 1.5;
    font-size: 12px;
    margin: 40px;
}}
h1 {{ font-size: 24px; margin-bottom: 18px; }}
h2 {{ font-size: 18px; margin-bottom: 14px; }}
h3 {{ font-size: 15px; margin-bottom: 12px; }}
pre, code {{ font-family: 'NotoSansCJK', monospace; background: #f4f4f4; padding: 6px; }}
ul, ol {{ margin-left: 20px; }}
'''


def main():
    markdown_text = INPUT_MARKDOWN.read_text(encoding='utf-8')
    html_body = markdown(markdown_text, extensions=['fenced_code', 'tables'])
    html = f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>{html_body}</body></html>'
    HTML(string=html).write_pdf(OUTPUT_PDF, stylesheets=[CSS(string=CSS_CONTENT)])


if __name__ == '__main__':
    main()
