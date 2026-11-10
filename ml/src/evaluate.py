# 学習済みモデルを検証データで評価する。
# 全体精度だけでなく「どのクラス同士を取り違えるか」を出すのが目的。
# 数字エリアの扱い (mask / cut / keep) を変えて比べたいときは、
# runs 側の preprocess.json ではなく --config で上書き指定もできる。
# 使い方:
#   python src/evaluate.py --run runs/icon_v1 --dataset data/dataset

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from preprocess import build_transform, load_config
from train import build_model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True, help="best.pt と labels.json のあるフォルダ")
  parser.add_argument("--dataset", default="data/dataset")
  parser.add_argument("--split", default="val")
  parser.add_argument("--config", default=None, help="指定すると run 側の設定より優先する")
  parser.add_argument("--batch", type=int, default=64)
  args = parser.parse_args()

  run_dir = Path(args.run)
  config = load_config(args.config) if args.config else load_config(run_dir / "preprocess.json")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  checkpoint = torch.load(run_dir / "best.pt", map_location=device)
  classes = checkpoint["classes"]
  model = build_model(len(classes)).to(device)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()

  data = datasets.ImageFolder(
    Path(args.dataset) / args.split, transform=build_transform(config, train=False))
  loader = DataLoader(data, batch_size=args.batch, shuffle=False)

  top1 = 0
  top5 = 0
  total = 0
  confusions = Counter()
  with torch.no_grad():
    for images, targets in loader:
      outputs = model(images.to(device))
      ranked = outputs.topk(min(5, len(classes)), dim=1).indices.cpu()
      for i, target in enumerate(targets):
        total += 1
        if ranked[i][0] == target:
          top1 += 1
        else:
          confusions[(classes[target], classes[ranked[i][0]])] += 1
        if target in ranked[i]:
          top5 += 1

  print(f"件数: {total}")
  print(f"Top1: {top1 / total:.4f} / Top5: {top5 / total:.4f}")
  print("取り違えの多いペア (正解 -> 誤判定, 件数):")
  for (truth, predicted), count in confusions.most_common(15):
    print(f"  {truth} -> {predicted}: {count}")

  report = {
    "total": total,
    "top1": top1 / total,
    "top5": top5 / total,
    "confusions": [
      {"truth": t, "predicted": p, "count": c}
      for (t, p), c in confusions.most_common()
    ],
  }
  report_path = run_dir / f"eval_{args.split}.json"
  with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
  print(f"レポートを書きました: {report_path}")


if __name__ == "__main__":
  main()
