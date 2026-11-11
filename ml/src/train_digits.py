# 数字モデル (所持数) を学習する。
# セル下部の帯を1枚として入力し、桁ごとのヘッド (6桁 x 11クラス) で読む。
# 本体は小さな CNN で、最後は平坦化して全結合する。全体平均プーリングは使わない
# (桁の位置情報が消えてしまうため、各ヘッドが帯のどこを見るかが学習の肝)。
# 使い方:
#   python src/train_digits.py --dataset data/digit_dataset --epochs 40 --out runs/digit_v1

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from preprocess import (
  build_digit_transform,
  digit_input_size,
  load_config,
  quantity_to_digits,
)


class DigitDataset(Dataset):
  # build_digit_dataset.py が出した帯画像と対応表 (file,quantity) を読む
  def __init__(self, dataset_dir, split, config, train=False):
    self.dir = Path(dataset_dir) / split
    self.transform = build_digit_transform(config, train=train)
    with open(Path(dataset_dir) / f"{split}.csv", encoding="utf-8") as f:
      self.rows = list(csv.DictReader(f))
    self.config = config

  def __len__(self):
    return len(self.rows)

  def __getitem__(self, index):
    row = self.rows[index]
    image = Image.open(self.dir / row["file"]).convert("RGB")
    digits = quantity_to_digits(int(row["quantity"]), self.config)
    return self.transform(image), torch.tensor(digits, dtype=torch.long)


def build_digit_model(config):
  # 数字モデル本体。入力サイズは config の digit 節から決まる。
  width, height = digit_input_size(config)
  digits = config["digit"]["digits"]

  class DigitNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.features = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        nn.MaxPool2d((1, 2)),  # 縦は潰しすぎない (数字の上下の形を残す)
      )
      with torch.no_grad():
        flat = self.features(torch.zeros(1, 3, height, width)).numel()
      self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flat, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, digits * 11),
      )

    def forward(self, x):
      out = self.classifier(self.features(x))
      return out.view(-1, digits, 11)

  return DigitNet()


def run_epoch(model, loader, device, optimizer=None):
  # 1 epoch 分を回す。戻り値は (loss, 桁ごとの精度, 数量の完全一致率)。
  training = optimizer is not None
  model.train(training)
  total_loss = 0.0
  digit_correct = 0
  digit_total = 0
  exact_correct = 0
  total = 0
  criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
  with torch.set_grad_enabled(training):
    for images, targets in loader:
      images = images.to(device)
      targets = targets.to(device)
      if training:
        optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs.reshape(-1, 11), targets.reshape(-1))
      if training:
        loss.backward()
        optimizer.step()
      predicted = outputs.argmax(dim=2)
      total_loss += loss.item() * len(targets)
      digit_correct += (predicted == targets).sum().item()
      digit_total += targets.numel()
      exact_correct += (predicted == targets).all(dim=1).sum().item()
      total += len(targets)
  return total_loss / total, digit_correct / digit_total, exact_correct / total


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="data/digit_dataset")
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--batch", type=int, default=64)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--out", default="runs/digit_v1")
  parser.add_argument("--config", default="config/preprocess.json")
  args = parser.parse_args()

  config = load_config(args.config)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"使用デバイス: {device}")

  train_data = DigitDataset(args.dataset, "train", config, train=True)
  val_data = DigitDataset(args.dataset, "val", config, train=False)
  print(f"学習: {len(train_data)} 枚, 検証: {len(val_data)} 枚")

  train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=2)

  model = build_digit_model(config).to(device)
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  best_exact = 0.0
  for epoch in range(args.epochs):
    train_loss, train_digit_acc, train_exact = run_epoch(model, train_loader, device, optimizer)
    val_loss, val_digit_acc, val_exact = run_epoch(model, val_loader, device)
    print(f"epoch {epoch + 1}/{args.epochs} "
          f"train loss {train_loss:.4f} digit {train_digit_acc:.4f} exact {train_exact:.4f} / "
          f"val loss {val_loss:.4f} digit {val_digit_acc:.4f} exact {val_exact:.4f}")

    if val_exact > best_exact:
      best_exact = val_exact
      torch.save({"state_dict": model.state_dict()}, out_dir / "best.pt")
      print(f"  -> best.pt を更新しました (val exact {val_exact:.4f})")

  with open(out_dir / "preprocess.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
  print(f"完了。最良の検証完全一致率: {best_exact:.4f} / 成果物: {out_dir}")


if __name__ == "__main__":
  main()
