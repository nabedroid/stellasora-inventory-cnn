# 帯から数字グリフを切り出す共通部品 (方式A: グリフ分類)。
# 生の連結成分 (build_digit_dataset.glyph_boxes) に幾何ガードをかけ、
# 「数字グリフらしい成分」だけを x 順に返す。ガードの根拠は実測
# (2026-11-13、未見スクショ25セルで検証):
#   - セルの枠の影が帯の左右端に細い成分として出る → 端に触れる成分は除外
#   - アイコンの破片は数字よりずっと幅広い → 幅の上限で除外
#   - 数字と × は下端 (ベースライン) が揃い、破片は上に浮く
#     → 最右の背の高い成分の下端をベースラインとし、揃わない成分は除外
# ラベル付け (label_glyphs) は右端アンカー: 数量は右寄せ描画なので、
# 右端の成分が常に一の位。× は一番左の余りとして扱う。

import numpy as np
from PIL import Image

from build_digit_dataset import glyph_boxes

# グリフ分類器のクラス。10 は棄却 (× や数字でない成分)。
GLYPH_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "reject"]
REJECT_LABEL = "reject"

WIDTH_CAP_RATIO = 0.16   # 成分の幅の上限 (帯の幅に対する比率)。数字は約 0.10
TALL_H_RATIO = 0.35      # 「背が高い」とみなす高さ (帯の高さに対する比率)
BASELINE_TOL_RATIO = 0.08  # ベースラインからの下端のずれの許容 (帯の高さ比)


def extract_glyph_boxes(strip):
  # 帯から数字グリフらしい成分の bbox (帯座標, x 順) を返す
  boxes = glyph_boxes(strip)
  # 1. 帯の左右端に触れる成分 (セルの枠の影) を落とす
  boxes = [b for b in boxes if b[0] > 0 and b[0] + b[2] < strip.width]
  # 2. 幅広すぎる成分 (アイコンの破片) を落とす
  boxes = [b for b in boxes if b[2] <= strip.width * WIDTH_CAP_RATIO]
  if not boxes:
    return []
  # 3. ベースライン: 背の高い成分のうち最右のものの下端
  #    (最右は必ず数字なので、その下端が数字の並びの下端になる)
  tall = [b for b in boxes if b[3] >= strip.height * TALL_H_RATIO]
  reference = tall[-1] if tall else boxes[-1]
  baseline = reference[1] + reference[3]
  tol = max(2, round(strip.height * BASELINE_TOL_RATIO))
  boxes = [b for b in boxes if abs((b[1] + b[3]) - baseline) <= tol]
  return boxes


def label_glyphs(boxes, quantity):
  # 検出成分に quantity の各桁を割り当てる。割り当てが安全でなければ None。
  # 戻り値は [(box, ラベル)] で、ラベルは "0"〜"9" か REJECT_LABEL (×)。
  # × が検出できていない場合 (成分数 = 桁数) は数字だけを割り当てる。
  digits = str(quantity)
  if len(boxes) == len(digits):
    return list(zip(boxes, digits))
  if len(boxes) == len(digits) + 1:
    # 一番左が × のはず。× は数字より背が低いことで裏を取る
    # (背が同じなら、それは数字かもしれず割り当てがずれるので捨てる)
    rest = boxes[1:]
    heights = sorted(b[3] for b in rest)
    median_h = heights[len(heights) // 2]
    if boxes[0][3] >= median_h:
      return None
    return [(boxes[0], REJECT_LABEL)] + list(zip(rest, digits))
  return None


def crop_glyph(strip, box, pad, size):
  # 成分を pad 分広げて切り出し、縦横比を保って size x size にレターボックスする。
  # 余白の色は切り出し画像の縁の中央値 (背景になじませるため)。
  x, y, w, h = box
  x0 = max(0, x - pad)
  y0 = max(0, y - pad)
  x1 = min(strip.width, x + w + pad)
  y1 = min(strip.height, y + h + pad)
  crop = strip.crop((x0, y0, x1, y1)).convert("RGB")
  arr = np.asarray(crop)
  border = np.concatenate([
    arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :],
  ])
  fill = tuple(int(v) for v in np.median(border, axis=0))
  scale = min(size / crop.width, size / crop.height)
  resized = crop.resize(
    (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
    Image.LANCZOS,
  )
  canvas = Image.new("RGB", (size, size), fill)
  canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
  return canvas
