# labels_template.csv の split 列 (train/val) をアイテム単位で付け直す。
# crop_icons が出す CSV は全行 train で、review.html でも行ごとにしか変えられないため、
# ラベル確定後にこのスクリプトで一括して振り分ける。
# (付け直し忘れると val が 0 件になり、学習スクリプトが落ちる)
# 方針 (アイテムごと):
#   3セル以上 : 1セルを val にする
#   2セル     : item_id が偶数のときだけ 1セルを val にする (約半数のアイテム)
#   1セル     : train のみ (val にすると学習に出てこなくなるため)
# どのセルを val にするかは --seed (既定 42) で決定的に選ぶ。
# 既定では split 列を全て付け直す。--keep を付けると、既に val の行は残して
# val の無いアイテムにだけ補う。
# 使い方:
#   python src/assign_splits.py --csv data/cells/labels_template.csv

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--csv", default="data/cells/labels_template.csv")
  parser.add_argument("--out", default=None, help="出力先 (省略時は --csv を上書き)")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--keep", action="store_true",
                      help="既に val の行を残し、val が無いアイテムにだけ補う")
  args = parser.parse_args()

  csv_path = Path(args.csv)
  with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

  # item_id でグループ化する (ID 化できない行は分割対象外で train のまま)
  groups = defaultdict(list)
  skipped = 0
  for i, row in enumerate(rows):
    raw = (row.get("item_id") or "").strip()
    if raw.isdigit():
      groups[int(raw)].append(i)
    else:
      skipped += 1
      row["split"] = "train"
  if skipped:
    print(f"警告: item_id が空または数値でない行を {skipped} 件 train 固定にしました")

  rng = random.Random(args.seed)
  no_val_items = []
  for item_id in sorted(groups):
    idxs = groups[item_id]
    existing = []
    if args.keep:
      existing = [i for i in idxs if (rows[i].get("split") or "").strip() == "val"]
    if existing:
      # --keep のときは既存の val をそのまま採用する
      for i in idxs:
        rows[i]["split"] = "val" if i in existing else "train"
      continue
    for i in idxs:
      rows[i]["split"] = "train"
    if len(idxs) >= 3 or (len(idxs) == 2 and item_id % 2 == 0):
      rows[rng.choice(idxs)]["split"] = "val"
    else:
      no_val_items.append(item_id)

  out_path = Path(args.out) if args.out else csv_path
  with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  val_total = sum(1 for r in rows if r["split"] == "val")
  print(f"分割しました: train {len(rows) - val_total} / val {val_total} ({out_path})")
  print(f"対象アイテム {len(groups)} 種、val を付けたアイテム "
        f"{len(groups) - len(no_val_items)} 種")
  if no_val_items:
    print(f"val 無しアイテム (1セル、または2セルで奇数 ID): {len(no_val_items)} 種")


if __name__ == "__main__":
  main()
