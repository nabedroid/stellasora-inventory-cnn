# グリフ分類器 (方式A) の学習データを作る。
# ラベル済みセル (labels_template.csv) の帯から数字グリフを切り出し、
# quantity の各桁を割り当てて「1文字ごとの画像 + ラベル」にする。
# 割り当ての安全策は glyph_extract.label_glyphs に集約してあり、
# 安全でないセルは理由別に数えて捨てる (誤ラベルを混ぜないため)。
# 水増しはしない (固定フォントで実グリフが足りる見込みのため。
# 不足が評価で見えたら足す)。split はセルの split をそのまま継承し、
# 同じセルのグリフが train/val に割れないようにする。
# 使い方:
#   python src/build_glyph_dataset.py --cells-csv data/cells/labels_template.csv

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from glyph_extract import GLYPH_CLASSES, crop_glyph, extract_glyph_boxes, label_glyphs
from preprocess import extract_digit_strip, load_config


def skip_reason(boxes, quantity):
  # label_glyphs が None を返した理由を概算で分類する (集計表示用)
  digits = len(str(quantity))
  if len(boxes) > digits + 1:
    return "over"
  if len(boxes) < digits:
    return "under"
  return "unsafe"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--cells-csv", default="data/cells/labels_template.csv")
  parser.add_argument("--config", default="config/preprocess.json")
  parser.add_argument("--out", default="data/glyph_dataset")
  parser.add_argument("--pad", type=int, default=2, help="グリフ切り出しの余白 (px)")
  parser.add_argument("--size", type=int, default=32, help="グリフ画像の一辺 (px)")
  parser.add_argument("--exclude", default=None,
                      help="除外するグリフのファイル名を列挙した CSV (列: file)")
  args = parser.parse_args()

  config = load_config(args.config)
  csv_path = Path(args.cells_csv)
  with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

  exclude = set()
  if args.exclude:
    with open(args.exclude, encoding="utf-8") as f:
      exclude = {r["file"] for r in csv.DictReader(f)}

  out_dir = Path(args.out)
  counts = Counter()
  skipped = Counter()
  excluded = 0
  for row in rows:
    quantity = (row.get("quantity") or "").strip()
    if not quantity:
      continue
    cell = Image.open(csv_path.parent / row["file"]).convert("RGB")
    strip = extract_digit_strip(cell, config)
    boxes = extract_glyph_boxes(strip)
    labeled = label_glyphs(boxes, int(quantity))
    if labeled is None:
      skipped[skip_reason(boxes, int(quantity))] += 1
      continue
    split = (row.get("split") or "train").strip() or "train"
    stem = Path(row["file"]).with_suffix("").as_posix().replace("/", "_")
    for i, (box, label) in enumerate(labeled):
      name = f"{stem}__{i}.png"
      if name in exclude:
        excluded += 1
        continue
      glyph = crop_glyph(strip, box, args.pad, args.size)
      dest = out_dir / split / label
      dest.mkdir(parents=True, exist_ok=True)
      glyph.save(dest / name)
      counts[(split, label)] += 1

  with open(out_dir / "glyph_meta.json", "w", encoding="utf-8") as f:
    json.dump({"size": args.size, "pad": args.pad, "classes": GLYPH_CLASSES},
              f, ensure_ascii=False, indent=2)

  for split in ("train", "val"):
    total = sum(counts[(split, c)] for c in GLYPH_CLASSES)
    per = ", ".join(f"{c}:{counts[(split, c)]}" for c in GLYPH_CLASSES)
    print(f"{split}: {total} グリフ ({per})")
  print(f"スキップしたセル: {dict(skipped)} / 除外したグリフ: {excluded}")
  print(f"出力先: {out_dir}")


if __name__ == "__main__":
  main()
