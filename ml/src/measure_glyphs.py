# グリフ切り出し方式 (方式A) の歩留まり測定。
# ラベル済みセルについて、幾何ガード付き抽出 (glyph_extract) で得た成分に
# quantity の各桁を安全に割り当てられる割合を測る。分類は5つ:
#   exact : × + 数字に割り当て成功
#   digits: 数字だけに割り当て成功 (× は検出できていない)
#   over  : 成分数 > 桁数 + 1 (切れ端やノイズの混入)
#   under : 成分数 < 桁数 (数字の結合・取りこぼし)
#   unsafe: 成分数は合うが × の背丈チェックで割り当てを棄却
# 使い方:
#   python src/measure_glyphs.py --cells-csv data/cells/labels_template.csv

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from glyph_extract import extract_glyph_boxes, label_glyphs
from preprocess import extract_digit_strip, load_config


def categorize(boxes, quantity):
  # 抽出成分と quantity から歩留まり分類を返す
  digits = len(str(quantity))
  if len(boxes) > digits + 1:
    return "over"
  if len(boxes) < digits:
    return "under"
  labeled = label_glyphs(boxes, quantity)
  if labeled is None:
    return "unsafe"
  if len(boxes) == digits + 1:
    return "exact"
  return "digits"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--cells-csv", default="data/cells/labels_template.csv")
  parser.add_argument("--config", default="config/preprocess.json")
  parser.add_argument("--examples", type=int, default=20,
                      help="失敗例として表示する件数")
  args = parser.parse_args()

  config = load_config(args.config)
  csv_path = Path(args.cells_csv)
  with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

  counts = Counter()
  by_resolution = defaultdict(Counter)
  failures = []
  total = 0
  for row in rows:
    quantity = (row.get("quantity") or "").strip()
    if not quantity:
      continue
    cell_path = csv_path.parent / row["file"]
    cell = Image.open(cell_path).convert("RGB")
    strip = extract_digit_strip(cell, config)
    boxes = extract_glyph_boxes(strip)
    category = categorize(boxes, int(quantity))
    counts[category] += 1
    total += 1
    resolution = (row.get("screenshot") or "?").rsplit("_", 1)[0]
    by_resolution[resolution][category] += 1
    if category in ("over", "under", "unsafe") and len(failures) < args.examples:
      failures.append(
        f"{row['file']} qty={quantity} グリフ数={len(boxes)} ({category})"
      )

  print(f"対象セル: {total}")
  for category in ("exact", "digits", "over", "under", "unsafe"):
    n = counts[category]
    print(f"  {category:6s}: {n:4d} ({n / total * 100:.1f}%)")
  usable = counts["exact"] + counts["digits"]
  print(f"グリフ学習に使える見込み: {usable} ({usable / total * 100:.1f}%)")
  print("解像度別:")
  for resolution in sorted(by_resolution):
    c = by_resolution[resolution]
    t = sum(c.values())
    print(f"  {resolution}: n={t} exact={c['exact']} digits={c['digits']} "
          f"over={c['over']} under={c['under']} unsafe={c['unsafe']}")
  if failures:
    print("失敗例 (over/under/unsafe):")
    for line in failures:
      print(f"  {line}")


if __name__ == "__main__":
  main()
