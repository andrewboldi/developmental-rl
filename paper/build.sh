#!/usr/bin/env bash
# Build paper.pdf from paper.md via pandoc + pdflatex.
set -euo pipefail
cd "$(dirname "$0")"
pandoc paper.md \
  --from markdown+tex_math_dollars+raw_tex \
  --citeproc --bibliography references.bib --csl ieee.csl 2>/dev/null \
  -V geometry:margin=1.05in -V fontsize=10pt -V linkcolor=blue \
  -V colorlinks=true \
  --pdf-engine=pdflatex -o paper.pdf || \
pandoc paper.md \
  --from markdown+tex_math_dollars+raw_tex \
  --citeproc --bibliography references.bib \
  -V geometry:margin=1.05in -V fontsize=10pt -V colorlinks=true \
  --pdf-engine=pdflatex -o paper.pdf
echo "wrote paper/paper.pdf"
