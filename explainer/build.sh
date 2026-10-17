#!/usr/bin/env bash
# Build the EN4554 Assignment 2 explainer PDF.
#   1. regenerate figures from results/*.json
#   2. concatenate the HTML parts
#   3. render with headless Chrome
set -e
cd "$(dirname "$0")"

PY="/c/Users/ASUS/.conda/envs/cnn_gpu/python.exe"
CHROME="/c/Program Files/Google/Chrome/Application/chrome.exe"
OUT="EN4554_a02_explained.pdf"

echo "[1/3] figures"
"$PY" -I figs.py

echo "[2/3] concatenating HTML"
cat src/p0_head.html src/p1.html src/p2.html src/p3.html src/p4.html > guide.html
echo "      guide.html  $(wc -c < guide.html) bytes"

echo "[3/3] rendering with Chrome (MathJax needs a moment)"
rm -f "$OUT"
"$CHROME" \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --no-pdf-header-footer \
  --virtual-time-budget=90000 \
  --run-all-compositor-stages-before-draw \
  --print-to-pdf="$(pwd -W 2>/dev/null || pwd)/$OUT" \
  "file://$(pwd -W 2>/dev/null || pwd)/guide.html" 2>/dev/null || true

if [ -f "$OUT" ]; then
  echo "      OK -> $OUT  ($(du -h "$OUT" | cut -f1))"
else
  echo "      FAILED: no PDF produced"
  exit 1
fi
