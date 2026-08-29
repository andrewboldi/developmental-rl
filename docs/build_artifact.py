"""Build a single-file artifact page from index.html + data.js + app.js.

The artifact host wraps content in its own document skeleton, so we emit only:
<title> + <style> + font links + body markup + CDN script tags + inlined
local scripts. Output: docs/artifact.html (gitignored-size fine, ~400 KB).
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    html = (HERE / "index.html").read_text()
    head = re.search(r"<head>(.*?)</head>", html, re.S).group(1)
    body = re.search(r"<body>(.*?)</body>", html, re.S).group(1)

    title = re.search(r"<title>.*?</title>", head, re.S).group(0)
    style = re.search(r"<style>.*?</style>", head, re.S).group(0)
    links = "\n".join(re.findall(r'<link rel="(?:preconnect|stylesheet)"[^>]*>', head))

    def inline_local(m):
        src = m.group(1)
        if src.startswith("http"):
            return m.group(0)
        js = (HERE / src).read_text()
        return "<script>\n" + js + "\n</script>"

    body = re.sub(r'<script src="([^"]+)"></script>', inline_local, body)

    out = f"{title}\n{links}\n{style}\n{body}"
    (HERE / "artifact.html").write_text(out)
    print(f"wrote docs/artifact.html ({len(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
