# ml (学習側)

アイコン分類モデルの学習一式。コンテナは ml の1個だけ
(切り出し・データ作成・学習・評価・書き出しは全部この中)。

## 手順

前提: スクリーンショットは PC 版の解像度 (1920x1080 / 1600x900 / 1280x720) の PNG。
切り出しはアイコン枠の検出方式 (detect_cells) なので、スクロールした画面でもよい。
config の grid は検出のサイズ基準 (セル 130px @1920幅) として使う。

### 0. コンテナに入る

リポジトリのルートで実行。以降の手順は全部コンテナ内 (/app/ml) で行う。

```bash
docker compose up -d ml
docker compose exec ml bash
```

### 1. セル切り出し

スクショからセルを切り出す。ラベル付け用 CSV も出る。
`--screenshot` にはディレクトリも指定でき、直下の画像を全部処理する。

```bash
python src/crop_icons.py --screenshot data/shots --out data/cells
```

### 2. 仮ラベル付け

初回はテンプレート照合、モデル作成後はモデル推論に切り替える。

```bash
# 初回 (テンプレート照合)
python src/label_assist.py --cells-dir data/cells \
  --master ../common/public/master-dictionary.json

# モデル作成後 (モデル推論)
python src/label_assist.py --cells-dir data/cells --backend model \
  --run runs/icon_v1 --master ../common/public/master-dictionary.json
```

数量の下書きも同時に付けられる。quantity が空のセルだけが対象で、結果は review.html の数量欄に先埋めされる。生成器は2つある。

Tesseract OCR で下書きする (要イメージ再ビルド: Dockerfile に tesseract-ocr を追加済み)。

```bash
python src/label_assist.py --cells-dir data/cells --quantity-backend tesseract \
  --master ../common/public/master-dictionary.json
```

数字モデルで下書きする (digit run がある場合の本命。一番弱い桁の確信度が 0.9 未満のセルと、桁並びが不正なセルは review.html で強調される)。

```bash
python src/label_assist.py --cells-dir data/cells --quantity-backend model \
  --digit-run runs/digit_v1 --master ../common/public/master-dictionary.json
```

両方のモデルを併用する (2周目以降の標準。アイコンの候補と数量の下書きを1回で付ける)。

```bash
python src/label_assist.py --cells-dir data/cells --backend model \
  --run runs/icon_v1 --quantity-backend model --digit-run runs/digit_v1 \
  --master ../common/public/master-dictionary.json
```

### 3. ラベル確認 (目視)

- `data/cells/review.html` をブラウザで開き、候補と見比べて修正する
  (確信度が低い順に並ぶ。空きスロットは「なし」を選ぶ)
- 最後に CSV をダウンロードして `data/cells/labels_template.csv` に上書きする
- CSV に記録されるラベルは item_id。具体物の確認と選択は名前と画像で行い、
  名前 ⇔ ID の変換はマスター辞書から引く
  (build_dataset.py は item_name 列で書かれた CSV も変換して読める。
  辞書に無い名前は警告して飛ばす)
- ラベルが空の行は build_dataset が無言で飛ばすので注意
- quantity は今は使わない (数字モデルを作る段階で埋める)

### 4. データセット生成

種画像の水増し + 実セル取り込み。

```bash
python src/build_dataset.py \
  --master ../common/public/master-dictionary.json \
  --cells-csv data/cells/labels_template.csv
```

### 5. 学習

```bash
python src/train.py --dataset data/dataset --epochs 30 --out runs/icon_v1
```

### 6. 評価

取り違えペアまで出る。

```bash
python src/evaluate.py --run runs/icon_v1 --dataset data/dataset
```

### 7. スクショ1枚で確認実行

既定で `data/predict_plot.png` に枠と Top1 ラベルを描いた画像も出る
(保存先を変える場合は `--plot-out <path>`、画像が不要なら `--no-plot`)。

```bash
python src/predict.py --run runs/icon_v1 --screenshot data/shots/shot1.png
```

数字モデル (手順 10 で学習) がある場合は `--digit-run` を付けると、
所持数も推論して表示とプロットに含める。

```bash
python src/predict.py --run runs/icon_v1 --digit-run runs/digit_v1 --screenshot data/shots/shot1.png
```

### 8. ONNX 書き出し

デフォルトで `../common/public/models/` へ出力する。

```bash
python src/export_onnx.py --run runs/icon_v1
```

動作確認だけなら `build_dataset.py` に `--limit-classes 5 --per-class 30`、
`train.py` に `--epochs 2` を付けると全工程を数分で通せる。

### 9. 数字データセット生成

labels_template.csv の quantity 列が埋まっている行だけを使い、
セル下部の帯を切り出したデータセット (data/digit_dataset) を作る。
split はアイコン側と同じ列を使う。train は生 + 水増し 9 枚、val は生のみ。
水増しは「セル全体をずらして (縦方向厚め) から帯を切り直す」方式で、
検出枠のズレによる数字の動きを再現する。桁の組み替え合成はしない
(右寄せ描画では各桁の位置が文字列全体で決まるため、合成すると
実際には現れない位置関係を覚えてしまう)。

```bash
python src/build_digit_dataset.py --cells-csv data/cells/labels_template.csv
```

### 10. 数字モデルの学習

帯を 6 桁のヘッド (空欄 + 0〜9) で読むモデルを学習する。

```bash
python src/train_digits.py --dataset data/digit_dataset --epochs 40 --out runs/digit_v1
```

### 11. 数字モデルの評価

数量の完全一致率と桁ごとの精度、不一致の一覧を出す。
「x を数字と誤認して先頭に余分な桁が付く」型 (幽霊桁) の件数も数える。

```bash
python src/evaluate_digits.py --run runs/digit_v1 --dataset data/digit_dataset
```

### 12. 数字モデルの ONNX 書き出し

デフォルトで `../common/public/models/` へ digit_model.onnx、
digit_preprocess.json、digit_meta.json を出力する。

```bash
python src/export_digits_onnx.py --run runs/digit_v1
```

## 注意

- 検証精度は「実セルの val」で見ること。水増し同士の val は甘く出る
- 水増しに左右反転は入れていない (文字・向きのあるアイコンが壊れるため)
- 前処理の定数は config/preprocess.json が正。Web 側も同じファイルを
  モデルと一緒に読み込むので、片方だけ変えないこと
- ID ⇔ 名前の対応はマスター辞書が唯一の正 (旧 config/items.json の
  対応表は廃止して一本化した)。マスター辞書の読み込み時に ID・名前の
  重複があればエラーで止まる
- 数字エリアの扱い (mask / cut / keep) は config の number_area.mode で
  切り替えて evaluate で比較できる。デフォルトは mask
- 数字エリアのマスクと枠リングのマスク (frame、カーソル枠対策) は、
  データセット作成時ではなくモデル入力時 (学習・評価・推論の transform)
  に自動でかかる。データセット内の PNG は無加工のままなので、
  設定を変えても作り直しは不要で、学習からやり直せばよい
