# 学習データセットを作る。
# 素材は2系統:
#   1) マスター辞書 (master-dictionary.json) の iconDataUrl を種画像として水増し
#   2) crop_icons.py で切り出してラベル付けした実セル (あれば)
# 出力は ImageFolder 形式: data/dataset/train/<ID3桁>/, data/dataset/val/<ID3桁>/
# 注意: 水増し画像同士は似すぎているので、本来の検証は「実セルの val」で見ること。
# 実セルが無い段階では、val は水増しのホールドアウトになる (精度は甘めに出る)。
# 使い方:
#   python src/build_dataset.py --master ../common/public/master-dictionary.json
#   python src/build_dataset.py --master ... --cells-csv data/cells/labels_template.csv

import argparse
import base64
import csv
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from item_map import load_item_map


def augment(image, rng):
  # スクリーンショットの劣化を模した水増しを1回適用する。
  # ここでは前処理 (数字マスク等) は行わない。マスクはモデル入力時に一括適用する。
  img = image.copy()
  width, height = img.size

  # 拡大縮小して中心に戻す (切り出し位置のズレを模す)
  factor = rng.uniform(0.94, 1.06)
  if factor != 1.0:
    resized = img.resize((round(width * factor), round(height * factor)), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), (124, 116, 104))
    x = (width - resized.width) // 2 + rng.randint(-3, 3)
    y = (height - resized.height) // 2 + rng.randint(-3, 3)
    canvas.paste(resized, (x, y))
    img = canvas

  # 明るさ・コントラスト・彩度
  img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
  img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
  img = ImageEnhance.Color(img).enhance(rng.uniform(0.9, 1.1))

  # 縮小して戻す (解像度低下の模倣)
  if rng.random() < 0.4:
    small = rng.uniform(0.5, 0.85)
    img = img.resize((round(width * small), round(height * small)), Image.LANCZOS)
    img = img.resize((width, height), Image.LANCZOS)

  # ぼかし
  if rng.random() < 0.3:
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))

  # JPEG 圧縮 (スクショの保存劣化の模倣)
  if rng.random() < 0.7:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=rng.randint(65, 100))
    buffer.seek(0)
    img = Image.open(buffer).convert("RGB")

  return img


def decode_data_url(data_url):
  # マスター辞書の iconDataUrl (data:image/png;base64,...) を PIL 画像にする
  payload = data_url.split(",", 1)[1]
  return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def save_variant(image, path, rng, do_augment):
  # 1枚保存する。水増しするかはフラグで切り替える
  img = augment(image, rng) if do_augment else image
  img.save(path)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--master", required=True, help="master-dictionary.json のパス")
  parser.add_argument("--cells-csv", default=None, help="ラベル付け済み labels_template.csv")
  parser.add_argument("--items", default="config/items.json",
                      help="ID ⇔ 名前の対応表 (item_name 列の変換に使う)")
  parser.add_argument("--cells-dir", default="data/cells", help="セル画像の置き場所")
  parser.add_argument("--out", default="data/dataset")
  parser.add_argument("--per-class", type=int, default=300, help="種画像から作る学習枚数/クラス")
  parser.add_argument("--val-ratio", type=float, default=0.1)
  parser.add_argument("--limit-classes", type=int, default=None, help="動作確認用にクラス数を絞る")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  rng = random.Random(args.seed)
  out_dir = Path(args.out)

  with open(args.master, encoding="utf-8") as f:
    master = json.load(f)
  if args.limit_classes:
    master = master[:args.limit_classes]

  labels = {}
  for item in master:
    class_name = f"{int(item['id']):03d}"
    labels[class_name] = {"id": int(item["id"]), "name": item["name"]}

  # 種画像からの水増し
  for item in master:
    class_name = f"{int(item['id']):03d}"
    seed_image = decode_data_url(item["iconDataUrl"])
    train_dir = out_dir / "train" / class_name
    val_dir = out_dir / "val" / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    val_count = max(1, round(args.per_class * args.val_ratio))
    for i in range(args.per_class):
      target = val_dir if i < val_count else train_dir
      save_variant(seed_image, target / f"seed_{i:04d}.png", rng, do_augment=True)
    print(f"{class_name} {item['name']}: 種画像から {args.per_class} 枚生成")

  # 実セルの取り込み (あれば)
  if args.cells_csv:
    # ラベルは ID (item_id) でも名前 (item_name) でもよい。名前は対応表で ID に変換する
    name_to_id = {}
    if Path(args.items).exists():
      name_to_id, _ = load_item_map(args.items)
    cells_dir = Path(args.cells_dir)
    with open(args.cells_csv, encoding="utf-8") as f:
      for row in csv.DictReader(f):
        raw_id = (row.get("item_id") or "").strip()
        raw_name = (row.get("item_name") or "").strip()
        if raw_id:
          item_id = int(raw_id)
        elif raw_name:
          item_id = name_to_id.get(raw_name)
          if item_id is None:
            print(f"警告: 対応表に無い名前なので飛ばします: {raw_name}")
            continue
        else:
          continue  # ラベル未記入の行は飛ばす
        class_name = f"{item_id:03d}"
        source = cells_dir / row["file"]
        if not source.exists():
          print(f"警告: セル画像がありません: {source}")
          continue
        image = Image.open(source).convert("RGB")
        split = row["split"] if row["split"] in ("train", "val") else "train"
        target_dir = out_dir / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = f"real_{row['screenshot']}_{Path(row['file']).stem}"
        if split == "val":
          # 検証用は水増ししない生画像だけを置く
          image.save(target_dir / f"{stem}.png")
        else:
          # 学習用は生1枚 + 水増し9枚
          image.save(target_dir / f"{stem}.png")
          for i in range(9):
            save_variant(image, target_dir / f"{stem}_aug{i}.png", rng, do_augment=True)
    print("実セルを取り込みました")

  meta = {"labels": labels}
  with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
  print(f"データセットを書きました: {out_dir}")


if __name__ == "__main__":
  main()
