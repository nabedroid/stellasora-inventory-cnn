# ml (学習側)

アイコン分類モデルの学習一式。コンテナは ml の1個だけ
(切り出し・データ作成・学習・評価・書き出しは全部この中)。

## 手順

前提: スクリーンショットは 1920x1080 の PNG。
切り出しはアイコン枠の検出方式 (detect_cells) なので、スクロールした画面でもよい。
config の grid は検出のサイズ基準 (セル 130px @1920幅) として使う。

```bash
# リポジトリのルートでコンテナ起動と入室
docker compose up -d ml
docker compose exec ml bash
# 以下はコンテナ内 (/app/ml) で実行

# 1. スクショからセルを切り出す (ラベル付け用 CSV も出る)
# --screenshot にはディレクトリも指定でき、直下の画像を全部処理する
python src/crop_icons.py --screenshot data/shots --out data/cells

# 2. 仮ラベル付け (初回はテンプレート照合、モデル作成後はモデル推論)
python src/label_assist.py --cells-dir data/cells \
  --master ../common/public/master-dictionary.json
# python src/label_assist.py --cells-dir data/cells --backend model \
#   --run runs/icon_v1 --master ../common/public/master-dictionary.json

# 3. ラベル確認 (目視)
# data/cells/review.html をブラウザで開き、候補と見比べて修正する
# (確信度が低い順に並ぶ。空きスロットは「なし」を選ぶ)
# 最後に CSV をダウンロードして data/cells/labels_template.csv に上書きする
# ※ CSV に記録されるラベルは item_id。具体物の確認と選択は名前と画像で行い、
#    名前 ⇔ ID の変換は config/items.json の対応表で引く
#    (build_dataset.py は item_name 列で書かれた CSV も対応表で変換して読める。
#     対応表に無い名前は警告して飛ばす)
# ※ ラベルが空の行は build_dataset が無言で飛ばすので注意
# ※ quantity は今は使わない (数字モデルを作る段階で埋める)

# 4. データセット生成 (種画像の水増し + 実セル取り込み)
python src/build_dataset.py \
  --master ../common/public/master-dictionary.json \
  --cells-csv data/cells/labels_template.csv

# 5. 学習
python src/train.py --dataset data/dataset --epochs 30 --out runs/icon_v1

# 6. 評価 (取り違えペアまで出る)
python src/evaluate.py --run runs/icon_v1 --dataset data/dataset

# 7. スクショ1枚で確認実行
python src/predict.py --run runs/icon_v1 --screenshot data/shots/shot1.png

# 8. ONNX 書き出し (デフォルトで ../common/public/models/ へ出力)
python src/export_onnx.py --run runs/icon_v1
```

動作確認だけなら `build_dataset.py` に `--limit-classes 5 --per-class 30`、
`train.py` に `--epochs 2` を付けると全工程を数分で通せる。

## 注意

- 検証精度は「実セルの val」で見ること。水増し同士の val は甘く出る
- 水増しに左右反転は入れていない (文字・向きのあるアイコンが壊れるため)
- 前処理の定数は config/preprocess.json が正。Web 側も同じファイルを
  モデルと一緒に読み込むので、片方だけ変えないこと
- ID ⇔ 名前の対応表は config/items.json が正 (定義は利用者が管理する)。
  同梱の初期値はマスター辞書から起こしたもの。名前はマスター辞書と
  一致させること (種画像や候補表示は名前で結合しているため)
- 数字エリアの扱い (mask / cut / keep) は config の number_area.mode で
  切り替えて evaluate で比較できる。デフォルトは mask
- 数字エリアのマスクと枠リングのマスク (frame、カーソル枠対策) は、
  データセット作成時ではなくモデル入力時 (学習・評価・推論の transform)
  に自動でかかる。データセット内の PNG は無加工のままなので、
  設定を変えても作り直しは不要で、学習からやり直せばよい
