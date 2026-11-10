# スクリーンショットからアイコンのセルを切り出す (学習データ作り専用)。
# 切り出しは preprocess.py の detect_cells (アイコン枠の検出方式) で行う。
# スクロールで縦位置がずれていても、枠が写っているアイコンだけを拾える。
# 本番の汎用検出は GitHub Pages 側 (TS) の責務。
# 使い方:
#   python src/crop_icons.py --screenshot shot1.png shot2.png --out data/cells
#   python src/crop_icons.py --screenshot data/shots --out data/cells
#   --screenshot にはファイル／ディレクトリの混在指定ができ、
#   ディレクトリの場合は直下の画像ファイルを全部処理する (再帰はしない)。
# 出力:
#   data/cells/<スクショ名>/r0_c0.png ... と labels_template.csv
#   labels_template.csv の item_id / quantity / split を埋めると
#   build_dataset.py が実セルとして取り込む。

import argparse
import csv
from pathlib import Path

from PIL import Image

from preprocess import detect_cells, load_config

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_screenshots(inputs):
  # ファイルはそのまま、ディレクトリは直下の画像を名前順に展開する
  shots = []
  for item in inputs:
    path = Path(item)
    if path.is_dir():
      for p in sorted(path.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
          shots.append(p)
    else:
      shots.append(path)
  return shots


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--screenshot", nargs="+", required=True)
  parser.add_argument("--out", default="data/cells")
  parser.add_argument("--config", default="config/preprocess.json")
  args = parser.parse_args()

  config = load_config(args.config)
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  rows = []
  for shot_path in collect_screenshots(args.screenshot):
    image = Image.open(shot_path).convert("RGB")
    stem = Path(shot_path).stem
    shot_dir = out_dir / stem
    shot_dir.mkdir(parents=True, exist_ok=True)
    cells = detect_cells(image, config)
    for row, col, cell in cells:
      name = f"r{row}_c{col}.png"
      cell.save(shot_dir / name)
      rows.append({
        "file": f"{stem}/{name}",
        "screenshot": stem,
        "row": row,
        "col": col,
        "item_id": "",
        "quantity": "",
        "split": "train",
      })
    print(f"{stem}: {len(cells)} セルを切り出しました")

  if not rows:
    print("対象の画像がありませんでした")
    return

  csv_path = out_dir / "labels_template.csv"
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"ラベル付け用 CSV を書きました: {csv_path}")
  print("item_id と quantity を埋め、検証用スクショの行は split を val に変えてください")


if __name__ == "__main__":
  main()
