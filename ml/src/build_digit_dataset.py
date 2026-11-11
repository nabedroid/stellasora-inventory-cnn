# 数字モデル (所持数) の学習データセットを作る。
# crop_icons.py で切り出した実セルと、quantity 列を埋めた labels_template.csv から、
# セル下部の帯だけを切り出して並べる。種画像は無いので素材は実セルのみ。
# 出力は data/digit_dataset/ 以下:
#   train/<画像>.png, val/<画像>.png (帯画像)
#   train.csv, val.csv (file,quantity の対応表)
#   digit_meta.json (桁数や帯の切り出し条件)
# train の実セルは生 + 水増し、val は生のみ。
# 数字の水増しは「セル全体をずらしてから帯を切り直す」方式にする: 検出枠のズレで
# 帯の中の数字がどう動くかを再現しつつ、桁同士の位置関係は描画どおりに保つため。
# 帯への劣化 (明るさ・ぼかし・JPEG) は build_dataset.augment を geometry=False で流用する。
# さらに水増しの一部では「背景スクランブル」を行う: 「×+数字」の保護枠の外側だけを
# 別のセルの帯で置き換え、アイコンの切れ端と数量の相関を学習データから断つ
# (切れ端から数量を当てる近道対策)。保護枠の中は元の画素のままなので数字は変わらない。
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


BAND_TOP = 0.38  # 帯の中で数字が現れる下部領域の開始位置 (帯に対する比率)
MARGIN = 3  # 保護枠をグリフの外側に広げる余白 (px)


def glyph_boxes(strip):
  # 帯の下部を二値化し、× と数字らしい連結成分の bbox (帯全体の座標) を x 順に返す。
  # label_assist の OCR 用グリフ検出と同じ方法。
  import cv2
  import numpy as np

  band_top = int(strip.height * BAND_TOP)
  band = strip.crop((0, band_top, strip.width, strip.height))
  arr = np.asarray(band.convert("L"))
  _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  ink = (bw == 0).astype(np.uint8)
  _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
  boxes = []
  for x, y, w, h, area in stats[1:]:
    if h >= 6 and w >= 2 and area >= 15:
      boxes.append((int(x), int(y) + band_top, int(w), int(h)))
  if not boxes:
    return []
  hmax = max(b[3] for b in boxes)
  return sorted(b for b in boxes if b[3] >= 0.5 * hmax)


def cluster_box(strip, quantity):
  # 「×+数字」の保護枠 (x0, y0, x1, y1) を返す。判定できなければ None。
  # 数字は右寄せ描画なので、検出したグリフのうち右端の (桁数+1) 個を本体とみなす
  # (左側にはアイコンの切れ端やノイズの箱が混ざるため、総数の一致は要求しない)。
  # 本体の左隣に近接した箱があれば取り込む (× の分割や破片の保護)。
  # 危険側の条件 (グリフ不足、枠が右端に届かない、太すぎる箱の混入) では None を返し、
  # 呼び出し側はスクランブル無しの水増しに倒す。
  boxes = glyph_boxes(strip)
  expected = len(str(quantity)) + 1
  if len(boxes) < expected:
    return None
  cluster = boxes[-expected:]
  idx = len(boxes) - expected - 1
  while idx >= 0:
    bx, _, bw, _ = boxes[idx]
    if cluster[0][0] - (bx + bw) <= MARGIN * 2 and bw <= strip.width * 0.25:
      cluster = [boxes[idx]] + cluster
      idx -= 1
    else:
      break
  if any(b[2] > strip.width * 0.25 for b in cluster):
    return None
  x0 = min(b[0] for b in cluster) - MARGIN
  y0 = min(b[1] for b in cluster) - MARGIN
  x1 = max(b[0] + b[2] for b in cluster) + MARGIN
  if x1 < strip.width * 0.9 or x1 - x0 > strip.width * 0.85:
    return None
  return (max(0, x0), max(0, y0), min(strip.width, x1), strip.height)


def scramble_strip(strip, donor, quantity):
  # 保護枠の外側だけを別セルの帯 (donor) で置き換えた帯を返す。
  # 保護枠の中は元の帯の画素をそのまま貼るので、数字は一切変わらない。
  box = cluster_box(strip, quantity)
  if box is None:
    return None
  x0, y0, x1, y1 = box
  base = donor.resize(strip.size, Image.LANCZOS).convert("RGB")
  base.paste(strip.crop((x0, y0, x1, y1)), (x0, y0))
  return base


def jitter_cell(cell, rng):
  # セル全体に拡縮とずらしをかける (数字用の幾何ジッター)。
  # アイコン用 augment より範囲を広く取り、縦方向を厚めにする
  # (検出枠の縦ズレが帯の切り出し位置にそのまま効くため)。
  width, height = cell.size
  factor = rng.uniform(0.93, 1.08)
  resized = cell.resize((round(width * factor), round(height * factor)), Image.LANCZOS)
  canvas = Image.new("RGB", (width, height), (124, 116, 104))
  x = (width - resized.width) // 2 + rng.randint(-4, 4)
  y = (height - resized.height) // 2 + rng.randint(-6, 6)
  canvas.paste(resized, (x, y))
  return canvas


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--cells-csv", default="data/cells/labels_template.csv")
  parser.add_argument("--cells-dir", default="data/cells", help="セル画像の置き場所")
  parser.add_argument("--out", default="data/digit_dataset")
  parser.add_argument("--config", default="config/preprocess.json")
  parser.add_argument("--augment", type=int, default=9, help="train の実セル1枚あたりの水増し枚数")
  parser.add_argument("--scramble-prob", type=float, default=0.6,
                      help="水増しに背景スクランブルを適用する確率 (0 で無効)")
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
  prepared = []
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
    prepared.append({"row": row, "cell": cell, "strip": strip, "stem": stem})

  # 背景スクランブルの donor は train の帯から取る (val の帯は混ぜない)
  donors = [p["strip"] for p in prepared if p["row"]["split"] == "train"]
  scramble_applied = 0
  scramble_skipped = 0
  for p in prepared:
    row = p["row"]
    if row["split"] != "train":
      continue
    for i in range(args.augment):
      name = f"{p['stem']}_aug{i}.png"
      done = False
      if donors and rng.random() < args.scramble_prob:
        # スクランブルは生の帯に対して先に行う (ジッター後の帯ではグリフ検出が
        # 不安定になり、適用率が大きく落ちるため)。その後は通常の水増し
        # (幾何 + 劣化) を丸ごとかけるので、数字の位置の揺れも付く。
        scrambled = scramble_strip(p["strip"], rng.choice(donors), row["quantity"])
        if scrambled is not None:
          augment(scrambled, rng).save(out_dir / "train" / name)
          scramble_applied += 1
          done = True
        else:
          scramble_skipped += 1
      if not done:
        # セルごとずらして帯を切り直し、帯には劣化だけを追加でかける
        shifted = extract_digit_strip(jitter_cell(p["cell"], rng), config)
        augment(shifted, rng, geometry=False).save(out_dir / "train" / name)
      manifests["train"].append({"file": name, "quantity": row["quantity"]})
  if args.scramble_prob > 0:
    print(f"背景スクランブル: {scramble_applied} 枚に適用、"
          f"{scramble_skipped} 枚は検出不一致でスキップ")

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
