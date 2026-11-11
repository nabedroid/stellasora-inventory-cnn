# ラベル付けの半自動化。
# crop_icons.py が出した labels_template.csv の各セルに、仮ラベル (候補 Top3) を付けて
#   - labels_suggested.csv : item_id に第1候補を先埋めした CSV
#   - review.html          : 目視確認・修正用の単体ページ (サーバー不要)
# を出力する。CSV に記録するラベルは ID で、名前は確認ページの表示と選択のためだけに使う
# (ID ⇔ 名前の対応はマスター辞書から引く)。
# review.html で修正して「CSVをダウンロード」すると labels_template.csv が得られる。
#
# 仮ラベルの生成器 (backend) は2つあり、成果物の形式は共通:
#   template: 種画像 (マスター辞書) とのテンプレート照合。モデル未作成の初回用
#   model   : 学習済みモデルの推論。2周目以降はこちらが速くて正確
# 使い方:
#   python src/label_assist.py --cells-dir data/cells --master ../common/public/master-dictionary.json
#   python src/label_assist.py --cells-dir data/cells --backend model --run runs/icon_v1 \
#     --master ../common/public/master-dictionary.json
# 数量の下書きも付けるとき (Tesseract 使用、要イメージ再ビルド):
#   python src/label_assist.py --cells-dir data/cells --quantity-backend tesseract \
#     --master ../common/public/master-dictionary.json

import argparse
import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from item_map import load_item_map
from preprocess import apply_frame, apply_number_area, extract_digit_strip, load_config

INNER = 64  # 照合用に縮小する一辺 (px)
SCALES = (0.85, 1.0, 1.18)  # 種画像の配置スケール候補


def decode_seed(item):
  # マスター辞書の iconDataUrl を RGBA 画像で返す
  payload = item["iconDataUrl"].split(",", 1)[1]
  return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGBA")


def thumb_data_url(image, size=96, quality=82):
  # HTML 埋め込み用に小さな JPEG の data URL を作る
  img = image.convert("RGB")
  img.thumbnail((size, size), Image.LANCZOS)
  buffer = io.BytesIO()
  img.save(buffer, format="JPEG", quality=quality)
  return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def suggest_by_template(cells, master, config):
  # 種画像との照合で候補を返す。
  # 種画像は「絵柄のみ・背景透過」なので、アルファがある画素だけで誤差を測る。
  # セル側は数字マスクと枠マスクを済ませた姿で比べる (学習時の入力に寄せる)。

  def seed_parts(item):
    im = decode_seed(item).resize((INNER, INNER), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32)
    return arr[:, :, :3], (arr[:, :, 3] > 128).astype(np.float32)

  def scaled(seed_rgb, seed_mask, scale):
    # 種画像を scale 倍して中央に置いたキャンバスとマスクを返す
    size = int(INNER * scale)
    rgb = Image.fromarray(seed_rgb.astype(np.uint8)).resize((size, size), Image.LANCZOS)
    msk = Image.fromarray((seed_mask * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    canvas = np.zeros((INNER, INNER, 3), np.float32)
    weight = np.zeros((INNER, INNER), np.float32)
    off = (INNER - size) // 2
    x0, y0 = max(0, off), max(0, off)
    x1, y1 = min(INNER, off + size), min(INNER, off + size)
    canvas[y0:y1, x0:x1] = np.asarray(rgb)[y0 - off:y1 - off, x0 - off:x1 - off]
    weight[y0:y1, x0:x1] = np.asarray(msk)[y0 - off:y1 - off, x0 - off:x1 - off] / 255.0
    return canvas, weight

  seeds = []
  for item in master:
    rgb, mask = seed_parts(item)
    variants = [scaled(rgb, mask, s) for s in SCALES]
    seeds.append((int(item["id"]), item["name"], variants))

  results = []
  for cell in cells:
    im = apply_frame(apply_number_area(cell, config), config).resize((INNER, INNER), Image.LANCZOS)
    ct = np.asarray(im).astype(np.float32)
    scored = []
    for item_id, name, variants in seeds:
      best = None
      for canvas, weight in variants:
        w = weight.sum()
        if w < 50:
          continue
        dist = float((np.abs(ct - canvas) * weight[:, :, None]).sum() / (w * 3))
        best = dist if best is None else min(best, dist)
      if best is not None:
        scored.append((best, item_id, name))
    scored.sort()
    top = scored[:3]
    # 確信度は「2位との差の比率」で出す (絶対距離は絵柄でブレるため)
    margin = (top[1][0] - top[0][0]) / top[1][0] if len(top) > 1 and top[1][0] > 0 else 1.0
    results.append({
      "candidates": [{"id": i, "name": n, "score": round(1.0 - d / 255.0, 4)} for d, i, n in top],
      "confidence": round(float(margin), 4),
    })
  return results


def suggest_by_model(cells, run_dir):
  # 学習済みモデルで候補を返す。読み込み手順は predict.py と同じ。
  import torch

  from preprocess import build_transform
  from train import build_model

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
  batch = torch.stack([transform(cell) for cell in cells]).to(device)
  with torch.no_grad():
    probs = model(batch).softmax(dim=1).cpu()

  results = []
  for prob in probs:
    top = prob.topk(3)
    candidates = []
    for p, idx in zip(top.values.tolist(), top.indices.tolist()):
      class_name = classes[idx]
      info = name_by_class.get(class_name, {})
      candidates.append({
        "id": info.get("id", int(class_name)),
        "name": info.get("name", class_name),
        "score": round(float(p), 4),
      })
    results.append({"candidates": candidates, "confidence": candidates[0]["score"]})
  return results


OCR_CROP = 0.62  # 帯の下部だけを使う (上部の絵柄の残骸を OCR に見せない)
OCR_X_CORR = 0.45  # 最左グリフを × とみなす相関の下限


def _ocr_band(strip):
  # 帯の下部を切り出して二値化 (白地に黒文字) した numpy 配列を返す
  import cv2

  band = strip.crop((0, int(strip.height * (1 - OCR_CROP)), strip.width, strip.height))
  arr = np.asarray(band.convert("L"))
  _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return bw


def _glyph_boxes(bw):
  # 文字らしい連結成分の bbox を x 順に返す
  import cv2

  ink = (bw == 0).astype(np.uint8)
  _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
  boxes = []
  for x, y, w, h, area in stats[1:]:
    if h >= 6 and w >= 2 and area >= 15:
      boxes.append((int(x), int(y), int(w), int(h)))
  if not boxes:
    return []
  hmax = max(b[3] for b in boxes)
  return sorted(b for b in boxes if b[3] >= 0.5 * hmax)


def _glyph_img(bw, box, size=20):
  import cv2

  x, y, w, h = box
  g = (bw[y:y + h, x:x + w] == 0).astype(np.float32)
  return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def suggest_quantities(cells, config):
  # Tesseract で数量の下書きを読む。前処理が精度をほぼ決める (実測で素読み 46% → 78%):
  #   帯の下部だけを二値化し、× を見つけてその右側だけを OCR する。
  # × は数字に化けて紛れ込むため、バッチ全帯の最左グリフの中央値を × テンプレとして
  # 照合する (× はどの帯でも必ず最左にある)。confidence が低いセルは review で重点確認。
  try:
    import pytesseract
  except ImportError:
    raise SystemExit("pytesseract がありません。イメージを再ビルドしてください")

  bands = [_ocr_band(extract_digit_strip(cell, config)) for cell in cells]

  # × テンプレ (バッチが小さすぎるときは作らず、× 除去なしで読む)
  template = None
  if len(bands) >= 5:
    glyphs = []
    for bw in bands:
      boxes = _glyph_boxes(bw)
      if boxes:
        glyphs.append(_glyph_img(bw, boxes[0]))
    if len(glyphs) >= 5:
      template = np.median(np.stack(glyphs), axis=0)

  results = []
  for bw in bands:
    img_arr = bw
    boxes = _glyph_boxes(bw)
    x_matched = False
    rest = len(boxes)  # × を除いた残りグリフ数 (読み取り桁数と一致するはず)
    if boxes and template is not None:
      g = _glyph_img(bw, boxes[0])
      denom = float(np.sqrt((g * g).sum() * (template * template).sum())) or 1.0
      if float((g * template).sum() / denom) >= OCR_X_CORR:
        x_matched = True
        rest = len(boxes) - 1
        # × の右端から左を切り落とし、数字クラスタの上下にそろえる
        x, _, w, _ = boxes[0]
        img_arr = bw[:, x + w:]
        ys = [b[1] for b in boxes]
        ye = [b[1] + b[3] for b in boxes]
        img_arr = img_arr[max(0, min(ys) - 2):min(img_arr.shape[0], max(ye) + 2), :]
    scale = 5 if bw.shape[0] < 24 else 3
    img = Image.fromarray(img_arr).resize(
      (img_arr.shape[1] * scale, img_arr.shape[0] * scale), Image.LANCZOS)
    img = ImageOps.expand(img, border=12, fill=255)
    try:
      data = pytesseract.image_to_data(
        img, config="--psm 7 -c tessedit_char_whitelist=0123456789",
        output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
      raise SystemExit(
        "tesseract の実行ファイルが見つかりません。イメージを再ビルドしてください "
        "(docker compose build ml)")
    text = ""
    confs = []
    for token, conf in zip(data["text"], data["conf"]):
      digits = "".join(ch for ch in token if ch.isdigit())
      if digits:
        text += digits
        if int(conf) >= 0:
          confs.append(int(conf))
    qconf = round(sum(confs) / len(confs) / 100.0, 4) if confs else 0.0
    # 構造の整合チェック: × が見つかり、読み取り桁数が残りグリフ数と一致するか。
    # tesseract の confidence は正読でも低く出がちで当てにならないため、
    # review の強調はこの整合フラグで行う
    # (実測: 整合した群の正解率 90%、不整合の群に誤読の 8 割が集まる)。
    consistent = bool(x_matched and text and len(text) == rest)
    results.append({
      "quantity": int(text) if text else None,
      "qconf": qconf,
      "qwarn": not consistent,
    })
  return results


REVIEW_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>ラベル確認</title>
<style>
  body { font-family: sans-serif; margin: 16px; }
  .row { display: flex; align-items: center; gap: 12px; padding: 6px 0; border-bottom: 1px solid #ddd; }
  .row.dubious { background: #fff4e5; }
  .row img.cell { width: 96px; height: 96px; object-fit: contain; }
  .cand { display: inline-block; text-align: center; width: 92px; }
  .cand img { width: 56px; height: 56px; object-fit: contain; }
  .cand span { display: block; font-size: 11px; }
  button { font-size: 15px; padding: 6px 18px; }
  select, input[type=text] { font-size: 14px; }
  .qty { width: 92px; font-size: 15px; }
  .qty.warn { background: #ffd9a0; }
  .qnote { font-size: 11px; color: #666; }
</style>
</head>
<body>
<h1>ラベル確認</h1>
<p>確信度が低い順に並んでいます (背景が橙の行は要確認)。
候補の画像と見比べて選び直し、「なし」は空きスロット用です。
数量欄には OCR の下書きが入っています (欄の背景が橙なら構造チェックで引っかかった読みで要確認、空欄は読み取れなかったセル)。
最後に CSV をダウンロードして、data/cells/labels_template.csv に上書きしてください。</p>
<p><button id="export">CSVをダウンロード</button></p>
<div id="list"></div>
<p><button id="export2">CSVをダウンロード</button></p>
<script>
const DATA = __DATA__;
const ITEMS = __ITEMS__;
const list = document.getElementById('list');
DATA.forEach((row, ri) => {
  const div = document.createElement('div');
  div.className = 'row' + (row.confidence < 0.15 ? ' dubious' : '');
  let html = `<img class="cell" src="${row.img}"><div>${row.file}<br>確信度 ${row.confidence}</div>`;
  row.candidates.forEach((c, ci) => {
    const checked = (row.item_id === '' ? ci === 0 : String(c.id) === row.item_id) ? 'checked' : '';
    html += `<label class="cand"><input type="radio" name="r${ri}" value="${c.id}" ${checked}>
      <img src="${ITEMS[c.id].img}"><span>${c.name}<br>${c.score}</span></label>`;
  });
  const noneChecked = row.item_id === '__none__' ? 'checked' : '';
  html += `<label class="cand"><input type="radio" name="r${ri}" value="" ${noneChecked}><span>なし</span></label>`;
  html += `<select id="other${ri}"><option value="">その他から選ぶ…</option>` +
    Object.keys(ITEMS).map(id => `<option value="${id}">${ITEMS[id].name}</option>`).join('') + `</select>`;
  html += `<label>split <select id="split${ri}">
    <option value="train" ${row.split !== 'val' ? 'selected' : ''}>train</option>
    <option value="val" ${row.split === 'val' ? 'selected' : ''}>val</option></select></label>`;
  const qwarn = row.qwarn ? ' warn' : '';
  const qnote = row.qconf !== null ? `<span class="qnote">OCR ${row.qconf}</span>` : '';
  html += `<label>数量 <input type="text" inputmode="numeric" class="qty${qwarn}" id="qty${ri}" value="${row.quantity}"> ${qnote}</label>`;
  div.innerHTML = html;
  list.appendChild(div);
});
function exportCsv() {
  const lines = ['file,screenshot,row,col,item_id,quantity,split'];
  DATA.forEach((row, ri) => {
    const other = document.getElementById('other' + ri).value;
    const picked = document.querySelector(`input[name="r${ri}"]:checked`);
    const itemId = other !== '' ? other : (picked ? picked.value : '');
    const split = document.getElementById('split' + ri).value;
    const qty = document.getElementById('qty' + ri).value.trim();
    lines.push([row.file, row.screenshot, row.row, row.col, itemId, qty, split].join(','));
  });
  const blob = new Blob([lines.join('\\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'labels_template.csv';
  a.click();
}
document.getElementById('export').addEventListener('click', exportCsv);
document.getElementById('export2').addEventListener('click', exportCsv);
</script>
</body>
</html>
"""


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--cells-dir", default="data/cells")
  parser.add_argument("--master", required=True, help="master-dictionary.json のパス")
  parser.add_argument("--backend", choices=["template", "model"], default="template")
  parser.add_argument("--run", default=None, help="backend=model のときの学習 run ディレクトリ")
  parser.add_argument("--quantity-backend", choices=["tesseract", "none"], default="none",
                      help="数量の下書き生成 (tesseract は要イメージ再ビルド)")
  parser.add_argument("--config", default="config/preprocess.json")
  args = parser.parse_args()

  if args.backend == "model" and not args.run:
    parser.error("--backend model には --run が必要です")

  cells_dir = Path(args.cells_dir)
  csv_path = cells_dir / "labels_template.csv"
  with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
  with open(args.master, encoding="utf-8") as f:
    master = json.load(f)
  config = load_config(args.config)

  images = []
  for row in rows:
    images.append(Image.open(cells_dir / row["file"]).convert("RGB"))

  if args.backend == "model":
    suggestions = suggest_by_model(images, Path(args.run))
  else:
    suggestions = suggest_by_template(images, master, config)

  # 既存の記入を ID にそろえる (数字ならそのまま、名前ならマスター辞書で引く)
  name_to_id, _ = load_item_map(args.master)

  def existing_id(row):
    raw = (row.get("item_id") or "").strip()
    if raw.isdigit():
      return int(raw)
    name = (row.get("item_name") or "").strip()
    return name_to_id.get(name) if name else None

  # 数量の下書き (既存の記入があるセルには付けない)
  drafts = [None] * len(rows)
  if args.quantity_backend == "tesseract":
    drafts = suggest_quantities(images, config)

  # 候補つき CSV (第1候補の ID を item_id に先埋め。既存の記入は上書きしない)
  out_rows = []
  for i, (row, sug) in enumerate(zip(rows, suggestions)):
    quantity = (row.get("quantity") or "").strip()
    quantity_conf = ""
    quantity_warn = ""
    if quantity == "" and drafts[i] is not None:
      if drafts[i]["quantity"] is not None:
        quantity = str(drafts[i]["quantity"])
      quantity_conf = drafts[i]["qconf"]
      quantity_warn = 1 if drafts[i]["qwarn"] else 0
    merged = {
      "file": row["file"], "screenshot": row["screenshot"],
      "row": row["row"], "col": row["col"],
      "item_id": existing_id(row), "quantity": quantity,
      "quantity_conf": quantity_conf, "quantity_warn": quantity_warn,
      "split": row["split"],
    }
    if merged["item_id"] is None and sug["candidates"]:
      merged["item_id"] = sug["candidates"][0]["id"]
    if merged["item_id"] is None:
      merged["item_id"] = ""
    merged["cand2_id"] = sug["candidates"][1]["id"] if len(sug["candidates"]) > 1 else ""
    merged["cand3_id"] = sug["candidates"][2]["id"] if len(sug["candidates"]) > 2 else ""
    merged["confidence"] = sug["confidence"]
    out_rows.append(merged)
  suggested_path = cells_dir / "labels_suggested.csv"
  with open(suggested_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    writer.writeheader()
    writer.writerows(out_rows)

  # 確認用 HTML (確信度が低い順に並べる)
  items = {}
  for item in master:
    items[int(item["id"])] = {"name": item["name"], "img": thumb_data_url(decode_seed(item))}
  data = []
  for row, sug, image, merged in zip(rows, suggestions, images, out_rows):
    eid = existing_id(row)
    data.append({
      "file": row["file"], "screenshot": row["screenshot"],
      "row": row["row"], "col": row["col"], "split": row["split"],
      "item_id": "" if eid is None else str(eid), "img": thumb_data_url(image, size=130),
      "candidates": sug["candidates"], "confidence": sug["confidence"],
      "quantity": merged["quantity"],
      "qconf": merged["quantity_conf"] if merged["quantity_conf"] != "" else None,
      "qwarn": merged["quantity_warn"] == 1,
    })
  data.sort(key=lambda d: d["confidence"])
  html = REVIEW_HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
  html = html.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
  html_path = cells_dir / "review.html"
  html_path.write_text(html, encoding="utf-8")

  low = sum(1 for s in suggestions if s["confidence"] < 0.15)
  print(f"仮ラベルを付けました: {suggested_path} / {html_path}")
  print(f"全 {len(rows)} 件、確信度 0.15 未満は {low} 件 (review.html で重点確認)")
  if args.quantity_backend == "tesseract":
    filled = sum(1 for m in out_rows if m["quantity_conf"] != "" and m["quantity"] != "")
    missed = sum(1 for m in out_rows if m["quantity_conf"] != "" and m["quantity"] == "")
    print(f"数量の下書き: {filled} 件に値を入れ、{missed} 件は読み取れませんでした (review.html で記入)")


if __name__ == "__main__":
  main()
