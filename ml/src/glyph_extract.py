# 帯から数字グリフを切り出す共通部品 (方式A: グリフ分類)。
# 生の連結成分 (build_digit_dataset.glyph_boxes) に幾何ガードをかけ、
# 「数字グリフらしい成分」だけを x 順に返す。ガードの根拠は実測
# (2026-11-13、未見スクショ25セルで検証):
#   - セルの枠の影が帯の左右端に細い成分として出る → 端に触れる成分は除外
#   - アイコンの破片は数字よりずっと幅広い → 幅の上限で除外
#   - 数字と × は下端 (ベースライン) が揃い、破片は上に浮く
#     → 最右の背の高い成分の下端をベースラインとし、揃わない成分は除外
# 二値化は2段構え (カスケード): 通常は大域的な Otsu で足りるが、
# 低画質のスマホスクショでは帯のグラデーションや JPEG のにじみで
# 数字が周囲と融合し、Otsu では分離できないことがある (実測 2026-11-14)。
# 割り当てが安全でない場合だけ適応的しきい値で切り直す。
# 適応的を全面適用すると綺麗なセルで過分割が増えるため、再試行に留める。
# ラベル付け (label_glyphs) は右端アンカー: 数量は右寄せ描画なので、
# 右端の成分が常に一の位。× は一番左の余りとして扱う。

import cv2
import numpy as np
from PIL import Image

from build_digit_dataset import BAND_TOP, glyph_boxes

# グリフ分類器のクラス。10 は棄却 (× や数字でない成分)。
GLYPH_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "reject"]
REJECT_LABEL = "reject"

WIDTH_CAP_RATIO = 0.16   # 成分の幅の上限 (帯の幅に対する比率)。数字は約 0.10
MIN_WIDTH_RATIO = 0.03   # 成分の幅の下限 (同)。数字は細い「1」でも約 0.07
TALL_H_RATIO = 0.35      # 「背が高い」とみなす高さ (帯の高さに対する比率)
BASELINE_TOL_RATIO = 0.08  # ベースラインからの下端のずれの許容 (帯の高さ比)
X_HEIGHT_RATIO = 0.85    # × とみなす背丈の上限 (他の成分の中央値に対する比率)
ADAPTIVE_BLOCK = 31      # 適応的しきい値の近傍サイズ (カスケード再試行用)
ADAPTIVE_C = 10          # 適応的しきい値の減算定数


def raw_boxes_adaptive(strip):
  # glyph_boxes の二値化だけを適応的しきい値に替えた生抽出。
  # 後段の成分フィルタ (最小サイズと 0.5 x hmax) は glyph_boxes と同一。
  band_top = int(strip.height * BAND_TOP)
  band = strip.crop((0, band_top, strip.width, strip.height))
  arr = np.asarray(band.convert("L"))
  bw = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
  ink = (bw == 0).astype(np.uint8)
  _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
  boxes = []
  for x, y, w, h, area in stats[1:]:
    if h >= 6 and w >= 2 and area >= 15:
      boxes.append((int(x), int(y) + band_top, int(w), int(h)))
  if not boxes:
    return []
  hmax = max(b[3] for b in boxes)
  return sorted(b for b in boxes if b[3] >= 0.5 * hmax)


def _guarded_boxes(raw_boxes, strip):
  # 生の連結成分に幾何ガードをかけ、数字グリフらしい成分だけを x 順に返す
  boxes = list(raw_boxes)
  # 1. 帯の左右端に触れる成分 (セルの枠の影) を落とす
  boxes = [b for b in boxes if b[0] > 0 and b[0] + b[2] < strip.width]
  # 2. 幅で落とす: 広すぎる成分 (アイコンの破片) と、
  #    細すぎる成分 (枠の影や二値化のノイズ) は数字ではありえない
  boxes = [b for b in boxes
           if strip.width * MIN_WIDTH_RATIO <= b[2] <= strip.width * WIDTH_CAP_RATIO]
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


def extract_glyph_boxes(strip):
  # 帯から数字グリフらしい成分の bbox (帯座標, x 順) を返す (Otsu のみ)
  return _guarded_boxes(glyph_boxes(strip), strip)


def extract_glyph_boxes_cascade(strip, quantity=None):
  # カスケード抽出。まず Otsu で抽出し、割り当てが安全でない (または
  # 数字らしい成分がガードで落ちている) 場合だけ適応的しきい値で再試行する。
  # quantity が分かっている学習側では label_glyphs の安全性で判定し、
  # 推論側では「背の高い成分がガードで落ちた = 融合の疑い」を合図にする。
  raw = glyph_boxes(strip)
  boxes = _guarded_boxes(raw, strip)
  if quantity is not None:
    if label_glyphs(boxes, quantity) is not None:
      return boxes
    boxes2 = _guarded_boxes(raw_boxes_adaptive(strip), strip)
    if label_glyphs(boxes2, quantity) is not None:
      return boxes2
    return boxes
  tall = strip.height * TALL_H_RATIO
  dropped = [b for b in raw if b[3] >= tall and b not in boxes]
  if boxes and not dropped:
    return boxes
  boxes2 = _guarded_boxes(raw_boxes_adaptive(strip), strip)
  return boxes2 if len(boxes2) > len(boxes) else boxes


def is_x_like(box, other_boxes):
  # box が × らしい背丈か (他の成分の背丈の中央値より十分に低いか)。
  # × の背丈は数字の 0.67〜0.73 倍程度 (実測) で、数字同士はほぼ揃う。
  if not other_boxes:
    return False
  heights = sorted(b[3] for b in other_boxes)
  median_h = heights[len(heights) // 2]
  return box[3] < median_h * X_HEIGHT_RATIO


def label_glyphs(boxes, quantity):
  # 検出成分に quantity の各桁を割り当てる。割り当てが安全でなければ None。
  # 戻り値は [(box, ラベル)] で、ラベルは "0"〜"9" か REJECT_LABEL (×)。
  # × が検出できていない場合 (成分数 = 桁数) は数字だけを割り当てる。
  digits = str(quantity)
  if len(boxes) == len(digits):
    # 左端が × らしい背丈なら、× が数字として数えられていて実際は
    # 1桁欠けている疑いが強い。誤ラベルを混ぜないため捨てる
    if len(boxes) >= 2 and is_x_like(boxes[0], boxes[1:]):
      return None
    return list(zip(boxes, digits))
  if len(boxes) == len(digits) + 1:
    # 一番左が × のはず。× は数字より背が低いことで裏を取る
    # (背が数字並みなら、それは数字かもしれず割り当てがずれるので捨てる)
    if not is_x_like(boxes[0], boxes[1:]):
      return None
    return [(boxes[0], REJECT_LABEL)] + list(zip(boxes[1:], digits))
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
