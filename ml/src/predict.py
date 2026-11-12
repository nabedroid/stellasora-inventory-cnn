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
from PIL import Image, ImageDraw, ImageFont

from preprocess import (
  BLANK_DIGIT,
  build_digit_transform,
  build_transform,
  detect_cell_rects,
  digits_to_quantity,
  extract_digit_strip,
  load_config,
)
from train import build_model
from train_digits import build_digit_model


def load_font(size):
  # 日本語ラベル用のフォントを探す (同梱 -> システム -> 既定 の順)
  candidates = [
    Path(__file__).resolve().parent.parent / "fonts" / "MPLUS1p-Regular.ttf",
    Path("/usr/local/share/fonts/MPLUS1p-Regular.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
  ]
  for path in candidates:
    if path.exists():
      return ImageFont.truetype(str(path), size)
  return ImageFont.load_default()


def confidence_color(prob):
  # 確信度で枠色を変える (低いセルを目視で拾いやすくするため)
  if prob >= 0.9:
    return (46, 204, 64)
  if prob >= 0.5:
    return (255, 165, 0)
  return (255, 65, 65)


def fit_label(draw, prefix, name, suffix, max_width):
  # ラベルがセルの横幅 (+少しのはみ出し) に収まるフォントと文字列を返す。
  # 長い名前はフォントを段階的に縮め、それでも入らなければ名前の途中を省略する
  # (ID と確率は常に見えるようにする)。
  for size in (17, 15, 13, 12):
    font = load_font(size)
    label = prefix + name + suffix
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    if right - left <= max_width:
      return font, label, right - left, bottom - top
  font = load_font(12)
  text = name
  while text and draw.textbbox((0, 0), prefix + text + "…" + suffix, font=font)[2] > max_width:
    text = text[:-1]
  label = prefix + text + "…" + suffix
  left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
  return font, label, right - left, bottom - top


def draw_predictions(image, cells, results, out_path):
  # スクリーンショットに検出枠と Top1 ラベルを描いて保存する
  canvas = image.copy()
  draw = ImageDraw.Draw(canvas)
  for (row, col, x, y, w, h, _cell), result in zip(cells, results):
    best = result["candidates"][0]
    color = confidence_color(best["prob"])
    draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    if best["id"] is not None:
      prefix, name = f"{best['id']} ", best["name"]
    else:
      prefix, name = "", best["class"]
    suffix = f" {best['prob']:.2f}"
    font, label, text_w, text_h = fit_label(
      draw, prefix, name, suffix, w + 16
    )
    # ラベルは2行に分ける: 1行目 = ID + 名前 + 確信度、2行目 = 数量。
    # 1行にまとめると横に長くなり、隣のセルや画面端で見切れるため。
    lines = [label]
    line2_w = 0
    if result.get("quantity") is not None:
      lines.append(f"x{result['quantity']}")
      left, _, right, _ = draw.textbbox((0, 0), lines[1], font=font)
      line2_w = right - left
    block_w = max(text_w, line2_w) + 8
    block_h = text_h * len(lines) + 3 * (len(lines) + 1)  # 行間と上下の余白は 3px
    # ラベルはセルの上端から内側へ下方向に伸ばす。セルの上に伸ばすと、
    # 上の段のセルの数字部分 (下部) に被ってしまうため。
    # 自分のセルの数字部分は下端にあるので、これで数字は隠れなくなる。
    label_y = y + 3
    draw.rectangle([x, label_y, x + block_w, label_y + block_h], fill=color)
    for i, line in enumerate(lines):
      draw.text((x + 4, label_y + 3 + i * (text_h + 3)), line,
                fill=(0, 0, 0), font=font)
  canvas.save(out_path)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True)
  parser.add_argument("--digit-run", default=None,
                      help="数字モデルの run フォルダ。指定すると所持数も推論する")
  parser.add_argument("--screenshot", required=True)
  parser.add_argument("--out-json", default=None)
  parser.add_argument("--plot-out", default="data/predict_plot.png",
                      help="プロット画像の保存先 (data/ は Git 管理外)")
  parser.add_argument("--no-plot", action="store_true", help="プロット画像を出さない")
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
  cells = detect_cell_rects(image, config)

  # 全セルを1つのバッチに詰めて1回で推論する (1セルずつ回さない)
  batch = torch.stack([transform(cell) for *_xywh, cell in cells]).to(device)
  with torch.no_grad():
    probs = model(batch).softmax(dim=1).cpu()

  # 数字モデルが指定されていれば、同じセルの帯から所持数も推論する
  quantities = [None] * len(cells)
  quantity_probs = [None] * len(cells)
  digit_debug = [None] * len(cells)
  if args.digit_run:
    digit_dir = Path(args.digit_run)
    digit_config = load_config(digit_dir / "preprocess.json")
    digit_checkpoint = torch.load(digit_dir / "best.pt", map_location=device)
    digit_model = build_digit_model(digit_config).to(device)
    digit_model.load_state_dict(digit_checkpoint["state_dict"])
    digit_model.eval()
    digit_transform = build_digit_transform(digit_config, train=False)
    strips = torch.stack([
      digit_transform(extract_digit_strip(cell, digit_config))
      for *_xywh, cell in cells
    ]).to(device)
    with torch.no_grad():
      digit_probs = digit_model(strips).softmax(dim=2).cpu()
    for i, head_probs in enumerate(digit_probs):
      digits = head_probs.argmax(dim=1).tolist()
      quantities[i] = digits_to_quantity(digits)
      # 数量の確信度は一番弱い桁の確率で代表する (1桁でも怪しければ全体が怪しい)
      quantity_probs[i] = round(head_probs.max(dim=1).values.min().item(), 4)
      if quantities[i] is None:
        # 数量が決められなかったセルは、原因切り分け用に各桁ヘッドの
        # 上位2候補 (空欄か数字か、確率) を記録しておき、結果表示で出す
        parts = []
        for hp in head_probs:
          top2 = hp.topk(2)
          labels = []
          for p, cls in zip(top2.values.tolist(), top2.indices.tolist()):
            name = "空" if cls == BLANK_DIGIT else str(cls)
            labels.append(f"{name} {p:.2f}")
          parts.append(" / ".join(labels))
        digit_debug[i] = "桁内訳 (左→右): " + " | ".join(parts)

  results = []
  for i, ((row, col, *_rest), prob) in enumerate(zip(cells, probs)):
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
    line = (f"r{row}c{col}: {best['name'] or best['class']} ({best['prob']:.3f})"
            f" / 2位: {candidates[1]['name'] or candidates[1]['class']} ({candidates[1]['prob']:.3f})")
    if args.digit_run:
      if quantities[i] is not None:
        line += f" / 数量: {quantities[i]} ({quantity_probs[i]:.3f})"
      else:
        line += " / 数量: 不明"
    print(line)
    if digit_debug[i]:
      print(f"    {digit_debug[i]}")
    results.append({
      "row": row, "col": col, "candidates": candidates,
      "quantity": quantities[i], "quantity_prob": quantity_probs[i],
    })

  if args.out_json:
    with open(args.out_json, "w", encoding="utf-8") as f:
      json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON を書きました: {args.out_json}")

  if not args.no_plot:
    plot_path = Path(args.plot_out)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    draw_predictions(image, cells, results, plot_path)
    print(f"プロット画像を書きました: {plot_path}")


if __name__ == "__main__":
  main()
