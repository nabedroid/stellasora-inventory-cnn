# グリフ分類器 (方式A) の評価。
# 1. グリフ単位の精度と混同ペア (データセットの val)
# 2. セル単位の数量完全一致率 (labels_template.csv の val セルを
#    GlyphReader で読み直す。旧方式の evaluate_digits と同じ土俵の数字)
# 3. --train-check: 学習グリフをモデルで再判定し、自動ラベルと強く矛盾する
#    ものを label_suspects.csv に出す (誤ラベルの疑い。除外は
#    build_glyph_dataset.py の --exclude で行う)
# 使い方:
#   python src/evaluate_glyphs.py --run runs/glyph_v1 --dataset data/glyph_dataset

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image

from digit_read import GlyphReader
from glyph_extract import GLYPH_CLASSES
from preprocess import load_config
from train_glyphs import GlyphDataset, build_glyph_model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True)
  parser.add_argument("--dataset", default="data/glyph_dataset")
  parser.add_argument("--cells-csv", default="data/cells/labels_template.csv")
  parser.add_argument("--train-check", action="store_true",
                      help="学習グリフのラベル疑義チェックも行う")
  args = parser.parse_args()

  run_dir = Path(args.run)
  config = load_config(run_dir / "preprocess.json")
  with open(run_dir / "glyph_meta.json", encoding="utf-8") as f:
    meta = json.load(f)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(run_dir / "best.pt", map_location=device)
  model = build_glyph_model(meta["size"]).to(device)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()

  # 1. グリフ単位の評価
  val_data = GlyphDataset(args.dataset, "val", config)
  loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)
  correct = 0
  confusion = Counter()
  with torch.no_grad():
    for images, targets in loader:
      predicted = model(images.to(device)).argmax(dim=1).cpu()
      for p, t in zip(predicted.tolist(), targets.tolist()):
        if p == t:
          correct += 1
        else:
          confusion[(GLYPH_CLASSES[t], GLYPH_CLASSES[p])] += 1
  print(f"グリフ精度 (val): {correct}/{len(val_data)} = {correct / len(val_data):.4f}")
  for (truth, pred), n in confusion.most_common(10):
    print(f"  混同: {truth} -> {pred} x{n}")

  # 2. セル単位の完全一致 (val セルのみ)
  reader = GlyphReader(run_dir, device=device)
  csv_path = Path(args.cells_csv)
  with open(csv_path, encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f)
            if (r.get("split") or "").strip() == "val"
            and (r.get("quantity") or "").strip()]
  exact = 0
  misses = []
  for row in rows:
    cell = Image.open(csv_path.parent / row["file"]).convert("RGB")
    result = reader.read(cell)
    if result["quantity"] == int(row["quantity"]):
      exact += 1
    else:
      misses.append(f"{row['file']} 正解 {row['quantity']} / 推論 {result['quantity']}")
  print(f"セル完全一致 (val): {exact}/{len(rows)} = {exact / len(rows):.4f}")
  for line in misses[:20]:
    print(f"  不一致: {line}")

  # 3. 学習ラベルの疑義チェック (自己矛盾の抽出)
  if args.train_check:
    train_data = GlyphDataset(args.dataset, "train", config)
    suspects = collect_suspects(model, train_data, device)
    out_path = run_dir / "label_suspects.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(["file"])
      for name in suspects:
        writer.writerow([name])
    print(f"ラベル疑義: {len(suspects)} 件 -> {out_path}")


def collect_suspects(model, dataset, device):
  # 学習グリフを再判定し、自動ラベルと強く矛盾するファイル名を返す
  loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
  names = []
  offset = 0
  with torch.no_grad():
    for images, targets in loader:
      probs = model(images.to(device)).softmax(dim=1).cpu().tolist()
      for j, (p, t) in enumerate(zip(probs, targets.tolist())):
        predicted = max(range(len(p)), key=lambda k: p[k])
        if predicted != t and p[predicted] >= 0.9 and p[t] <= 0.1:
          names.append(dataset.samples[offset + j][0].name)
      offset += len(targets)
  return names


if __name__ == "__main__":
  main()
