"""Assemble paper_full.md: inject related work + render references.bib as text."""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bib_entries(bibtext):
    entries = []
    for m in re.finditer(r"@\w+\{[^,]+,(.*?)\n\}", bibtext, re.S):
        body = m.group(1)
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*\n", body, re.S):
            f[fm.group(1).lower()] = re.sub(r"[{}]", "", fm.group(2)).strip()
        if "title" in f:
            venue = f.get("journal") or f.get("booktitle") or f.get("howpublished") or ""
            venue = re.sub(r"\\url", "", venue)
            entries.append((f.get("author", "Unknown"), f.get("year", "n.d."), f["title"], venue))
    entries.sort(key=lambda e: e[0].lower())
    return entries


def main():
    paper = (HERE / "paper.md").read_text()
    rw = (HERE / "related_work.md").read_text()
    paper = paper.replace("<!--RELATED_WORK-->", rw)

    refs = ["# References", ""]
    for author, year, title, venue in bib_entries((HERE / "references.bib").read_text()):
        line = f"- {author} ({year}). *{title}*." + (f" {venue}." if venue else "")
        refs.append(line)
    paper = paper.replace("<!--REFERENCES-->", "\n".join(refs))

    # pdflatex-safe substitutions
    for a, b in [("−", "-"), ("≤", "<="), ("≥", ">="),
                 ("×", "x"), ("≈", "~"), ("θ", "theta"),
                 ("ε", "epsilon"), ("π", "pi"), ("→", "->")]:
        paper = paper.replace(a, b)
    (HERE / "paper_full.md").write_text(paper)
    print(f"assembled paper_full.md ({len(paper.splitlines())} lines)")


if __name__ == "__main__":
    main()
