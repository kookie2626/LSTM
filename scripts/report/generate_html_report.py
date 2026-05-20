import base64
import re
from pathlib import Path
import markdown

md_path = Path('/home/keun/.gemini/antigravity/brain/f323c41e-9aee-4de9-b081-8592b3b088cd/artifacts/team_report.md')
html_path = Path('/home/keun/workspace/project/ML/outputs/team_report.html')
md_text = md_path.read_text(encoding='utf-8')

def img_repl(match):
    alt_text = match.group(1)
    img_path = match.group(2)
    try:
        with open(img_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        ext = img_path.split('.')[-1]
        return f'![{alt_text}](data:image/{ext};base64,{b64})'
    except Exception as e:
        print(f"Failed to load image {img_path}: {e}")
        return match.group(0)

md_text_b64 = re.sub(r'!\[(.*?)\]\((.*?)\)', img_repl, md_text)

html_body = markdown.markdown(md_text_b64, extensions=['fenced_code', 'tables'])

# GitHub alerts (blockquotes) styling fix
html_body = html_body.replace('<blockquote>\n<p>[!TIP]', '<blockquote style="border-left: 5px solid #2ecc71; background-color: #eafaf1;"><p><strong>💡 팁 (TIP)</strong><br>')
html_body = html_body.replace('<blockquote>\n<p>[!WARNING]', '<blockquote style="border-left: 5px solid #f39c12; background-color: #fef5e7;"><p><strong>⚠️ 중요 (WARNING)</strong><br>')

html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>팀 공유용 분석 보고서</title>
    <style>
        body {{
            font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #f9f9f9;
        }}
        .container {{
            background-color: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-top: 30px; }}
        h1 {{ border-bottom: 2px solid #3498db; color: #1a5276; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        blockquote {{ border-left: 5px solid #3498db; background-color: #ebf5fb; padding: 15px; margin: 20px 0; border-radius: 0 4px 4px 0; }}
        hr {{ border: 0; border-top: 1px solid #eee; margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}
    </div>
</body>
</html>
"""

html_path.write_text(html_template, encoding='utf-8')
print(f"Generated {html_path}")
