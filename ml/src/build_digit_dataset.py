# 数字モデル (所持数) の学習データセットを作る。
# crop_icons.py で切り出した実セルと、quantity 列を埋めた labels_template.csv から、
# セル下部の帯だけを切り出して並べる。種画像は無いので素材は実セルのみ。
# 出力は data/digit_dataset/ 以下:
#   train/<画像>.png, val/<画像>.png (帯画像)
#   train.csv, val.csv (file,quantity の対応表)
#   digit_meta.json (桁数や帯の切り出し条件)
# train の実セルは生 + 水増し (build_dataset.augment を流用)、val は生のみ。
# split は CSV の split 列に従う (アイコン側と同じ振り分けを使う)。
# 使い方:
#   python src/build_digit_dataset.py --cells-csv data/cells/labels_template.csv

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image

from build_dataset import augment
from preprocess import extract_digit_strip, load_config


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--cells-csv", default="data/cells/labels_template.csv")
  parser.add_argument("--cells-dir", default="data/cells", help="セル画像の置き場所")
  parser.add_argument("--out", default="data/digit_dataset")
  parser.add_argument("--config", default="config/preprocess.json")
  parser.add_argument("--augment", type=int, default=9, help="train の実セル1枚あたりの水増し枚数")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  config = load_config(args.config)
  rng = random.Random(args.seed)
  out_dir = Path(args.out)
  cells_dir = Path(args.cells_dir)

  rows = []
  skipped = 0
  with open(args.cells_csv, encoding="utf-8") as f:
    for row in csv.DictReader(f):
      raw_quantity = (row.get("quantity") or "").strip()
      if not raw_quantity:
        skipped += 1
        continue  # quantity 未記入の行は数字モデルでは使えない
      rows.append({
        "file": row["file"],
        "quantity": int(raw_quantity),
        "split": (row.get("split") or "train").strip() or "train",
      })
  if skipped:
    print(f"警告: quantity が空の行を {skipped} 件飛ばしました")
  if not rows:
    print("quantity 入りの行がありません。labels_template.csv の quantity 列を埋めてください")
    return

  manifests = {"train": [], "val": []}
  for row in rows:
    source = cells_dir / row["file"]
    if not source.exists():
      print(f"警告: セル画像がありません: {source}")
      continue
    cell = Image.open(source).convert("RGB")
    strip = extract_digit_strip(cell, config)
    split = row["split"]
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    # ファイル名はスクリーンショット名 + セル名で一意にする
    stem = row["file"].replace("/", "_").rsplit(".", 1)[0]
    strip.save(split_dir / f"{stem}.png")
    manifests[split].append({"file": f"{stem}.png", "quantity": row["quantity"]})
    if split == "train":
      for i in range(args.augment):
        name = f"{stem}_aug{i}.png"
        augment(strip, rng).save(split_dir / name)
        manifests[split].append({"file": name, "quantity": row["quantity"]})

  for split, entries in manifests.items():
    with open(out_dir / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=["file", "quantity"])
      writer.writeheader()
      writer.writerows(entries)
    print(f"{split}: {len(entries)} 枚")

  meta = {
    "digits": config["digit"]["digits"],
    "strip_top_ratio": config["digit"]["strip_top_ratio"],
    "input_width": config["digit"]["input_width"],
    "input_height": config["digit"]["input_height"],
  }
  with open(out_dir / "digit_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
  print(f"完了: {out_dir}")


if __name__ == "__main__":
  main()
