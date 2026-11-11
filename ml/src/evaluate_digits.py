# 学習済みの数字モデルを検証データで評価する。
# 数量の完全一致率と桁ごとの精度に加えて、不一致の一覧を出す。
# 特に「x を数字と誤認して先頭に余分な桁が付く」型の誤り (幽霊桁) は
# 名指しで拾う。起きていないことの確認が目的なので、0 件ならその旨を出す。
# 使い方:
#   python src/evaluate_digits.py --run runs/digit_v1 --dataset data/digit_dataset

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from preprocess import digits_to_quantity, load_config
from train_digits import DigitDataset, build_digit_model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True, help="best.pt のあるフォルダ")
  parser.add_argument("--dataset", default="data/digit_dataset")
  parser.add_argument("--split", default="val")
  parser.add_argument("--config", default=None, help="指定すると run 側の設定より優先する")
  parser.add_argument("--batch", type=int, default=64)
  args = parser.parse_args()

  run_dir = Path(args.run)
  config = load_config(args.config) if args.config else load_config(run_dir / "preprocess.json")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  checkpoint = torch.load(run_dir / "best.pt", map_location=device)
  model = build_digit_model(config).to(device)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()

  data = DigitDataset(args.dataset, args.split, config, train=False)
  loader = DataLoader(data, batch_size=args.batch, shuffle=False)

  total = 0
  exact = 0
  digit_correct = 0
  digit_total = 0
  mismatches = []
  offset = 0
  with torch.no_grad():
    for images, targets in loader:
      outputs = model(images.to(device))
      predicted = outputs.argmax(dim=2).cpu()
      for i in range(len(targets)):
        truth_digits = targets[i].tolist()
        pred_digits = predicted[i].tolist()
        truth = digits_to_quantity(truth_digits)
        pred = digits_to_quantity(pred_digits)
        total += 1
        digit_correct += sum(1 for a, b in zip(truth_digits, pred_digits) if a == b)
        digit_total += len(truth_digits)
        if truth_digits == pred_digits:
          exact += 1
        else:
          # 幽霊桁: 予測の方が桁数が多く、正解が予測の末尾と一致する
          phantom = (
            pred is not None and truth is not None
            and len(str(pred)) > len(str(truth))
            and str(pred).endswith(str(truth))
          )
          mismatches.append({
            "file": data.rows[offset + i]["file"],
            "truth": truth,
            "predicted": pred,
            "phantom_leading_digit": phantom,
          })
      offset += len(targets)

  print(f"件数: {total}")
  print(f"完全一致: {exact / total:.4f} / 桁ごと: {digit_correct / digit_total:.4f}")
  phantoms = [m for m in mismatches if m["phantom_leading_digit"]]
  print(f"幽霊桁 (x の誤認が疑われる型): {len(phantoms)} 件")
  print("不一致の一覧 (正解 -> 予測):")
  for m in mismatches:
    mark = " [幽霊桁]" if m["phantom_leading_digit"] else ""
    print(f"  {m['truth']} -> {m['predicted']}{mark} ({m['file']})")

  report = {
    "total": total,
    "exact": exact / total,
    "digit_accuracy": digit_correct / digit_total,
    "phantom_leading_digit": len(phantoms),
    "mismatches": mismatches,
  }
  report_path = run_dir / f"eval_{args.split}.json"
  with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
  print(f"レポートを書きました: {report_path}")


if __name__ == "__main__":
  main()
