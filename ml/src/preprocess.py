# 学習側と推論側で共有する前処理。
# ここの定数は config/preprocess.json に切り出してある。
# 将来 GitHub Pages 側 (TypeScript) に移植するときも、この JSON を正として
# 両側の数値を一致させること。二重管理を避けるための要のファイル。

import json
from pathlib import Path

from PIL import Image


def load_config(path):
  # 設定 JSON を読み込んで dict で返す
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def input_size(config):
  # モデルへの入力サイズ (幅, 高さ) を返す。
  # mask / keep モードは正方形のまま、cut モードは帯の分だけ縦が短くなる。
  width = config["input_width"]
  if config["number_area"]["mode"] == "cut":
    ratio = config["number_area"]["strip_ratio"]
    return width, round(width * (1 - ratio))
  return width, width


def apply_number_area(image, config):
  # 所持数エリアへの対処を行う。
  #   mask: 帯全体を正規化の平均色で塗りつぶす (デフォルト)
  #   cut : 帯を切り落として長方形にする
  #   keep: 何もしない (対照実験用。数字が手がかり化するので常用しない)
  # 帯は「幅全体・高さ固定」なので、桁数などの情報は一切残らない。
  mode = config["number_area"]["mode"]
  if mode == "keep":
    return image
  width, height = image.size
  ratio = config["number_area"]["strip_ratio"]
  strip_top = round(height * (1 - ratio))
  if mode == "cut":
    return image.crop((0, 0, width, strip_top))
  # mode == "mask"
  fill = tuple(config["number_area"]["fill_rgb"])
  result = image.copy()
  result.paste(fill, (0, strip_top, width, height))
  return result


def apply_frame(image, config):
  # セル外周の枠リングを正規化の平均色で塗りつぶす。
  # カーソル枠はアニメーションで収縮し、位相によって見た目も
  # 検出矩形も変わるため、枠を学習させるのではなく領域ごと消す。
  # mode が mask 以外、または frame 設定自体が無い場合は何もしない。
  frame = config.get("frame") or {}
  if frame.get("mode") != "mask":
    return image
  width, height = image.size
  ring_x = round(width * frame["ring_ratio"])
  ring_y = round(height * frame["ring_ratio"])
  fill = tuple(frame["fill_rgb"])
  result = image.copy()
  result.paste(fill, (0, 0, width, ring_y))
  result.paste(fill, (0, height - ring_y, width, height))
  result.paste(fill, (0, 0, ring_x, height))
  result.paste(fill, (width - ring_x, 0, width, height))
  return result


def crop_cells(image, config):
  # スクリーンショットから格子状のセルを切り出して返す。
  # 戻り値は (行, 列, PIL.Image) のリスト。
  # 格子形状は 1920x1080 基準の値を、実際の画像幅に比例させて使う。
  # お試し版なので他解像度への本格対応はしていない (本番は TS 側の責務)。
  grid = config["grid"]
  scale = image.width / grid["reference_width"]
  origin_x = grid["origin_x"] * scale
  origin_y = grid["origin_y"] * scale
  pitch_x = grid["pitch_x"] * scale
  pitch_y = grid["pitch_y"] * scale
  size = grid["cell_size"] * scale
  cells = []
  for row in range(grid["rows"]):
    for col in range(grid["cols"]):
      x = round(origin_x + pitch_x * col)
      y = round(origin_y + pitch_y * row)
      s = round(size)
      cells.append((row, col, image.crop((x, y, x + s, y + s))))
  return cells


def detect_cell_rects(image, config):
  # アイコン枠を検出して切り出す。戻り値は (行, 列, x, y, w, h, 切り出し画像)。
  # (OCR 版 TS の IconExtractService と同じ方式)。
  # グレースケール -> 二値化 -> 輪郭 -> 正方形に近い外接矩形だけを残す。
  # スクロールで縦位置がずれていても、枠が写っているアイコンだけを拾える。
  # 画面端で見切れたアイコンは矩形が正方形にならないため自動的に除外される。
  import cv2
  import numpy as np

  grid = config["grid"]
  scale = image.width / grid["reference_width"]
  expected = grid["cell_size"] * scale

  gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
  # 閾値は 180。選択カーソルの緑はグレースケールで約189しかなく、200 では黒に落ちて
  # セルの上端を食い、矩形が潰れて検出から漏れる (2026-11-11 に実測で確認。180 では
  # スクショ16枚で喪失0・誤検出0のままカーソルセルだけが回収できた)
  _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  rects = []
  for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # 幅の許容は 0.85〜1.15 倍。画面左上のボタン類 (105px 等) を拾わないため
    if w < expected * 0.85 or w > expected * 1.15:
      continue
    if not 0.95 <= w / h <= 1.05:
      continue
    rects.append((x, y, w, h))

  # 二重検出の除去 (TS 版と同じく、近すぎる矩形は片方だけ残す)
  rects.sort(key=lambda r: (r[1], r[0]))
  kept = []
  for x, y, w, h in rects:
    too_close = False
    for kx, ky, kw, kh in kept:
      if abs(kx - x) < w * 0.8 and abs(ky - y) < h * 0.8:
        too_close = True
        break
    if not too_close:
      kept.append((x, y, w, h))

  # 行にまとめてから (行, 列, 画像) にする
  rows = []
  for rect in kept:
    x, y, w, h = rect
    if rows and abs(y - rows[-1][0][1]) < h / 2:
      rows[-1].append(rect)
    else:
      rows.append([rect])
  cells = []
  for row_index in range(len(rows)):
    row_rects = sorted(rows[row_index], key=lambda r: r[0])
    for col_index in range(len(row_rects)):
      x, y, w, h = row_rects[col_index]
      cells.append((row_index, col_index, x, y, w, h, image.crop((x, y, x + w, y + h))))
  return cells


def detect_cells(image, config):
  # 座標が不要な呼び出し用に、(行, 列, 切り出し画像) だけを返す
  return [
    (row, col, cell)
    for row, col, _x, _y, _w, _h, cell in detect_cell_rects(image, config)
  ]


def build_transform(config, train=False):
  # PIL 画像 -> モデル入力テンソルへの変換を返す (torchvision 使用)。
  # 学習時と推論時で必ず同じ変換を通すこと。ここがズレると精度が静かに落ちる。
  # 注意: 左右反転 (flip) は絶対に入れない。アイコンには文字や向きの情報があり、
  # 反転すると実在しないアイコンを学習してしまう。
  import torch
  from torchvision import transforms

  size = input_size(config)
  mean = config["normalize"]["mean"]
  std = config["normalize"]["std"]
  steps = []
  if train:
    # 色味だけ軽く揺らす (幾何学的な水増しは build_dataset.py 側で実施済み)
    steps.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1))
  steps.append(transforms.Lambda(lambda img: apply_number_area(img, config)))
  steps.append(transforms.Lambda(lambda img: apply_frame(img, config)))
  steps.append(transforms.Resize(size[::-1]))  # Resize は (高さ, 幅) 指定
  steps.append(transforms.ToTensor())
  steps.append(transforms.Normalize(mean, std))
  return transforms.Compose(steps)
