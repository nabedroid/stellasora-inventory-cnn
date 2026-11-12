# グリフ分類器 (方式A の数字モデル) を学習する。
# 入力は帯から切り出した1文字 (32x32)、出力は 0〜9 + 棄却 (×) の11クラス。
# 全ての桁で同じモデルを使うため、桁位置ごとの学習量の偏りがない。
# 単一グリフの分類なので、全体平均プーリングで位置を潰してよい。
# 水増しはしない (build_glyph_dataset 参照)。
# 使い方:
#   python src/train_glyphs.py --dataset data/glyph_dataset --epochs 30 --out runs/glyph_v1

import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms

from glyph_extract import GLYPH_CLASSES
from preprocess import load_config


class GlyphDataset(Dataset):
  # build_glyph_dataset.py が出した <split>/<ラベル>/*.png を読む
  def __init__(self, dataset_dir, split, config):
    self.samples = []
    mean = config["normalize"]["mean"]
    std = config["normalize"]["std"]
    self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
    ])
    base = Path(dataset_dir) / split
    for class_index, label in enumerate(GLYPH_CLASSES):
      for path in sorted((base / label).glob("*.png")):
        self.samples.append((path, class_index))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    path, class_index = self.samples[index]
    image = Image.open(path).convert("RGB")
    return self.transform(image), class_index


def build_glyph_model(size):
  # グリフ分類器本体。size は入力画像の一辺。
  class GlyphNet(nn.Module):
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
        nn.AdaptiveAvgPool2d(1),
      )
      self.classifier = nn.Linear(96, len(GLYPH_CLASSES))

    def forward(self, x):
      out = self.features(x).flatten(1)
      return self.classifier(out)

  return GlyphNet()


def run_epoch(model, loader, device, optimizer=None):
  # 1 epoch 分を回す。戻り値は (loss, 精度)。
  training = optimizer is not None
  model.train(training)
  total_loss = 0.0
  correct = 0
  total = 0
  criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
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
  parser.add_argument("--dataset", default="data/glyph_dataset")
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--batch", type=int, default=64)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--out", default="runs/glyph_v1")
  parser.add_argument("--config", default="config/preprocess.json")
  args = parser.parse_args()

  config = load_config(args.config)
  with open(Path(args.dataset) / "glyph_meta.json", encoding="utf-8") as f:
    meta = json.load(f)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"使用デバイス: {device}")

  train_data = GlyphDataset(args.dataset, "train", config)
  val_data = GlyphDataset(args.dataset, "val", config)
  print(f"学習: {len(train_data)} グリフ, 検証: {len(val_data)} グリフ")

  train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False, num_workers=2)

  model = build_glyph_model(meta["size"]).to(device)
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  best_acc = 0.0
  for epoch in range(args.epochs):
    train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, device)
    print(f"epoch {epoch + 1}/{args.epochs} "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} / "
          f"val loss {val_loss:.4f} acc {val_acc:.4f}")
    if val_acc > best_acc:
      best_acc = val_acc
      torch.save({"state_dict": model.state_dict()}, out_dir / "best.pt")
      print(f"  -> best.pt を更新しました (val acc {val_acc:.4f})")

  # 読み取り側 (digit_read) が run だけで完結するよう、必要な定義を同梱する
  shutil.copy(Path(args.dataset) / "glyph_meta.json", out_dir / "glyph_meta.json")
  with open(out_dir / "preprocess.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
  print(f"完了。最良の検証精度: {best_acc:.4f} / 成果物: {out_dir}")


if __name__ == "__main__":
  main()
