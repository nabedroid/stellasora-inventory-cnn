# アイコン分類モデルを学習する。
# MobileNetV3-Small の ImageNet 学習済み重みを使い、最終層をクラス数に差し替える。
# 最初の数 epoch は特徴抽出部を凍結して分類層だけを温め、その後全体を解凍する。
# 使い方:
#   python src/train.py --dataset data/dataset --epochs 30 --out runs/icon_v1

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

from preprocess import build_transform, load_config


def build_model(num_classes):
  # 学習済みモデルを取り、最終層だけ差し替える
  weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
  model = models.mobilenet_v3_small(weights=weights)
  in_features = model.classifier[3].in_features
  model.classifier[3] = nn.Linear(in_features, num_classes)
  return model


def run_epoch(model, loader, device, optimizer=None):
  # 1 epoch 分を回す。optimizer があれば学習、なければ評価のみ。
  training = optimizer is not None
  model.train(training)
  total_loss = 0.0
  correct = 0
  total = 0
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  with torch.set_grad_enabled(training):
    for images, targets in loader:
      images = images.to(device)
      targets = targets.to(device)
      if training:
        optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, targets)
      if training:
        loss.backward()
        optimizer.step()
      total_loss += loss.item() * len(targets)
      correct += (outputs.argmax(dim=1) == targets).sum().item()
      total += len(targets)
  return total_loss / total, correct / total


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="data/dataset")
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--freeze-epochs", type=int, default=3, help="最初に特徴抽出部を凍結する epoch 数")
  parser.add_argument("--batch", type=int, default=64)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--out", default="runs/icon_v1")
  parser.add_argument("--config", default="config/preprocess.json")
  args = parser.parse_args()

  config = load_config(args.config)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"使用デバイス: {device}")

  train_data = datasets.ImageFolder(
    Path(args.dataset) / "train", transform=build_transform(config, train=True))
  val_data = datasets.ImageFolder(
    Path(args.dataset) / "val", transform=build_transform(config, train=False))
  print(f"クラス数: {len(train_data.classes)}, 学習: {len(train_data)} 枚, 検証: {len(val_data)} 枚")

  train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=2)

  model = build_model(len(train_data.classes)).to(device)
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  best_accuracy = 0.0
  for epoch in range(args.epochs):
    # 凍結フェーズが終わったら全体を解凍し、学習率を少し下げる
    frozen = epoch < args.freeze_epochs
    for param in model.features.parameters():
      param.requires_grad = not frozen
    lr = args.lr if not frozen else args.lr * 3
    optimizer = torch.optim.AdamW(
      [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, device)
    print(f"epoch {epoch + 1}/{args.epochs} "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} / "
          f"val loss {val_loss:.4f} acc {val_acc:.4f}")

    if val_acc > best_accuracy:
      best_accuracy = val_acc
      torch.save({"state_dict": model.state_dict(), "classes": train_data.classes},
                 out_dir / "best.pt")
      print(f"  -> best.pt を更新しました (val acc {val_acc:.4f})")

  # 付属ファイル (ラベル対応表と前処理設定) を成果物に同梱する。
  # この2つが無いと推論側でクラス番号とアイテムを対応付けられない。
  labels = {"classes": train_data.classes}
  meta_path = Path(args.dataset) / "dataset_meta.json"
  if meta_path.exists():
    with open(meta_path, encoding="utf-8") as f:
      labels["labels"] = json.load(f)["labels"]
  with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)
  with open(out_dir / "preprocess.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
  print(f"完了。最良の検証精度: {best_accuracy:.4f} / 成果物: {out_dir}")


if __name__ == "__main__":
  main()
