# 方式A の数量読み取り (predict / label_assist / evaluate で共有)。
# セル画像 → 帯 → グリフ切り出し (幾何ガード) → 共有分類器で各グリフを判定
# → 左端の棄却 (×) を落として数字を連結し、数量にする。
# 途中の棄却やグリフ不在は「不明」(None) に倒す (誤読より安全なため)。
# 確信度は数字グリフの最小確率で代表する (旧方式のヘッド最小値と同じ流儀)。

import json
from pathlib import Path

import torch
from torchvision import transforms

from glyph_extract import GLYPH_CLASSES, REJECT_LABEL, crop_glyph, extract_glyph_boxes
from preprocess import extract_digit_strip, load_config
from train_glyphs import build_glyph_model


class GlyphReader:
  def __init__(self, run_dir, device=None):
    run_dir = Path(run_dir)
    self.config = load_config(run_dir / "preprocess.json")
    with open(run_dir / "glyph_meta.json", encoding="utf-8") as f:
      self.meta = json.load(f)
    self.device = device or torch.device(
      "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(run_dir / "best.pt", map_location=self.device)
    self.model = build_glyph_model(self.meta["size"]).to(self.device)
    self.model.load_state_dict(checkpoint["state_dict"])
    self.model.eval()
    mean = self.config["normalize"]["mean"]
    std = self.config["normalize"]["std"]
    self.transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean, std),
    ])

  def _predict_probs(self, glyph_images):
    # グリフ画像のリストに対するクラス確率 (リストのリスト) を返す
    batch = torch.stack([self.transform(g) for g in glyph_images])
    with torch.no_grad():
      probs = self.model(batch.to(self.device)).softmax(dim=1).cpu()
    return probs.tolist()

  def read(self, cell):
    # セル画像から数量を読む。戻り値は dict:
    # quantity (int|None), prob (float|None), debug (内訳文字列)
    strip = extract_digit_strip(cell, self.config)
    boxes = extract_glyph_boxes(strip)
    if not boxes:
      return {"quantity": None, "prob": None, "debug": "グリフ内訳: (検出なし)"}
    if len(boxes) > 7:
      return {"quantity": None, "prob": None,
              "debug": f"グリフ内訳: (グリフ過多 {len(boxes)} 個)"}
    glyphs = [crop_glyph(strip, box, self.meta["pad"], self.meta["size"])
              for box in boxes]
    probs_list = self._predict_probs(glyphs)

    labels = []
    parts = []
    for probs in probs_list:
      order = sorted(range(len(probs)), key=lambda i: -probs[i])
      labels.append(GLYPH_CLASSES[order[0]])
      labels_top2 = []
      for i in order[:2]:
        name = "棄" if GLYPH_CLASSES[i] == REJECT_LABEL else GLYPH_CLASSES[i]
        labels_top2.append(f"{name} {probs[i]:.2f}")
      parts.append(" / ".join(labels_top2))
    debug = "グリフ内訳 (左→右): " + " | ".join(parts)

    # 左端の棄却は × として消費する。それ以外の棄却が混ざれば不明に倒す
    digit_labels = labels
    digit_probs = [max(p) for p in probs_list]
    if labels[0] == REJECT_LABEL and len(labels) >= 2:
      digit_labels = labels[1:]
      digit_probs = digit_probs[1:]
    if REJECT_LABEL in digit_labels or not digit_labels:
      return {"quantity": None, "prob": None, "debug": debug}
    quantity = int("".join(digit_labels))
    return {"quantity": quantity, "prob": round(min(digit_probs), 4), "debug": debug}
