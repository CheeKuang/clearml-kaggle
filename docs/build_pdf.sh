#!/bin/sh
# Rebuild docs/GUIDE.pdf from docs/GUIDE.md after editing it:   sh docs/build_pdf.sh
# Needs pandoc and a Chromium browser (Microsoft Edge or Google Chrome) for the print to PDF.
set -eu
cd "$(dirname "$0")"
python3 build_code_reference.py                           # refresh Part 6 from the current code
tmp=$(mktemp -d)
tail -n +2 GUIDE.md > "$tmp/body.md"                       # the first line (# title) becomes the cover page
cp guide_print.css "$tmp/print.css"
pandoc "$tmp/body.md" -f gfm -t html5 -s --toc --toc-depth=2 \
  --metadata title="Building-segmentation pipeline: beginner handbook" \
  --metadata subtitle="ClearML + Hydra + Kaggle. One model per class for the Chisel server." \
  --metadata date="Generated from docs/GUIDE.md, $(date +%Y-%m-%d)" --metadata toc-title="Contents" \
  -c print.css -o "$tmp/GUIDE.html"
for b in "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" \
         "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" chromium google-chrome; do
  if command -v "$b" >/dev/null 2>&1 || [ -x "$b" ]; then
    "$b" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/GUIDE.pdf" "file://$tmp/GUIDE.html" 2>/dev/null
    rm -rf "$tmp"; echo "written docs/GUIDE.pdf"; exit 0
  fi
done
echo "no Edge/Chrome found: open $tmp/GUIDE.html in a browser and print to PDF" >&2; exit 1
