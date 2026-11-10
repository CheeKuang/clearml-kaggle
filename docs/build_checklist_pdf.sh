#!/bin/sh
# Rebuild docs/DATASET_CHECKLIST.pdf from docs/DATASET_CHECKLIST.md after editing it:   sh docs/build_checklist_pdf.sh
# Needs pandoc and a Chromium browser (Microsoft Edge or Google Chrome) for the print to PDF.
set -eu
cd "$(dirname "$0")"
tmp=$(mktemp -d)
tail -n +2 DATASET_CHECKLIST.md > "$tmp/body.md"            # the first line (# title) becomes the cover page
cp guide_print.css "$tmp/print.css"
printf 'img { max-width: 100%%; height: auto; }\n' >> "$tmp/print.css"     # pictures fit the page width
pandoc "$tmp/body.md" -f gfm -t html5 -s --embed-resources --resource-path=. --toc --toc-depth=2 \
  --metadata title="Dataset checklist" \
  --metadata subtitle="What to look for before you train a model on a dataset: a beginner's guide" \
  --metadata date="Generated from docs/DATASET_CHECKLIST.md, $(date +%Y-%m-%d)" --metadata toc-title="Contents" \
  -c "$tmp/print.css" -o "$tmp/DATASET_CHECKLIST.html"
for b in "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" \
         "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" chromium google-chrome; do
  if command -v "$b" >/dev/null 2>&1 || [ -x "$b" ]; then
    "$b" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/DATASET_CHECKLIST.pdf" "file://$tmp/DATASET_CHECKLIST.html" 2>/dev/null
    rm -rf "$tmp"; echo "written docs/DATASET_CHECKLIST.pdf"; exit 0
  fi
done
echo "no Edge/Chrome found: open $tmp/DATASET_CHECKLIST.html in a browser and print to PDF" >&2; exit 1
