# スクリーンショット1枚に対して推論を試す (確認実行用)。
# 枠検出で切り出し -> バッチでまとめて推論 -> セルごとに上位3候補を表示する。
# 本番 (ブラウザ) の代理確認なので、最終確認は export_onnx.py で書き出した
# モデルを GitHub Pages 側で動かして行うこと。
# 使い方:
#   python src/predict.py --run runs/icon_v1 --screenshot shot.png

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from preprocess import build_transform, detect_cells, load_config
from train import build_model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True)
  parser.add_argument("--screenshot", required=True)
  parser.add_argument("--out-json", default=None)
  args = parser.parse_args()

  run_dir = Path(args.run)
  config = load_config(run_dir / "preprocess.json")
  with open(run_dir / "labels.json", encoding="utf-8") as f:
    labels_data = json.load(f)
  classes = labels_data["classes"]
  name_by_class = labels_data.get("labels", {})

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  checkpoint = torch.load(run_dir / "best.pt", map_location=device)
  model = build_model(len(classes)).to(device)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()

  transform = build_transform(config, train=False)
  image = Image.open(args.screenshot).convert("RGB")
  cells = detect_cells(image, config)

  # 全セルを1つのバッチに詰めて1回で推論する (1セルずつ回さない)
  batch = torch.stack([transform(cell) for _, _, cell in cells]).to(device)
  with torch.no_grad():
    probs = model(batch).softmax(dim=1).cpu()

  results = []
  for (row, col, _), prob in zip(cells, probs):
    top = prob.topk(3)
    candidates = []
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
      class_name = classes[idx]
      info = name_by_class.get(class_name, {})
      candidates.append({
        "class": class_name,
        "id": info.get("id"),
        "name": info.get("name", ""),
        "prob": round(p, 4),
      })
    best = candidates[0]
    print(f"r{row}c{col}: {best['name'] or best['class']} ({best['prob']:.3f})"
          f" / 2位: {candidates[1]['name'] or candidates[1]['class']} ({candidates[1]['prob']:.3f})")
    results.append({"row": row, "col": col, "candidates": candidates})

  if args.out_json:
    with open(args.out_json, "w", encoding="utf-8") as f:
      json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON を書きました: {args.out_json}")


if __name__ == "__main__":
  main()
