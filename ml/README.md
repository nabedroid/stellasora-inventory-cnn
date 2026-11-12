# ml (学習側)

アイコン分類モデルと数字モデルの学習一式。コンテナは ml の1個だけ
(切り出し・データ作成・学習・評価・書き出しは全部この中)。

手順は「共通 (アイコン・数字の土台)」「アイコンモデル」
「数字モデル (新方式: グリフ分類)」「数字モデル (旧方式: 6ヘッド)」
の4部構成。旧方式は比較とフォールバックのために残してある。

## 共通 (アイコン・数字の土台)

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

数量の下書きも同時に付けられる。quantity が空のセルだけが対象で、結果は review.html の数量欄に先埋めされる。生成器は3つある。

グリフ分類器 (新方式、手順 12 で学習) で下書きする (glyph run がある場合の本命。数字グリフの最小確率が 0.9 未満のセルと、読めなかったセルは review.html で強調される)。

```bash
python src/label_assist.py --cells-dir data/cells --quantity-backend glyph \
  --glyph-run runs/glyph_v1 --master ../common/public/master-dictionary.json
```

Tesseract OCR で下書きする (要イメージ再ビルド: Dockerfile に tesseract-ocr を追加済み)。

```bash
python src/label_assist.py --cells-dir data/cells --quantity-backend tesseract \
  --master ../common/public/master-dictionary.json
```

旧方式の数字モデルで下書きする (digit run がある場合)。

```bash
python src/label_assist.py --cells-dir data/cells --quantity-backend model \
  --digit-run runs/digit_v1 --master ../common/public/master-dictionary.json
```

両方のモデルを併用する (2周目以降の標準。アイコンの候補と数量の下書きを1回で付ける)。

```bash
python src/label_assist.py --cells-dir data/cells --backend model \
  --run runs/icon_v1 --quantity-backend glyph --glyph-run runs/glyph_v1 \
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

### 4. train/val 分割

ラベルが確定したら、split 列をアイテム単位で付け直す。
crop_icons の出力は全行 train で、review.html でも行ごとにしか変えられないため、
ここで一括して振り分ける (付け直さないと val が 0 件になり学習が落ちる)。
方針はアイテムごとに「3セル以上は1セルを val、2セルは約半数 (偶数 ID)、
1セルは train のみ」。どのセルを選ぶかは seed 42 で決定的に決まる。

```bash
python src/assign_splits.py --csv data/cells/labels_template.csv
```

## アイコンモデル

### 5. データセット生成

種画像の水増し + 実セル取り込み。

```bash
python src/build_dataset.py \
  --master ../common/public/master-dictionary.json \
  --cells-csv data/cells/labels_template.csv
```

### 6. 学習

```bash
python src/train.py --dataset data/dataset --epochs 30 --out runs/icon_v1
```

### 7. 評価

取り違えペアまで出る。

```bash
python src/evaluate.py --run runs/icon_v1 --dataset data/dataset
```

### 8. スクショ1枚で確認実行

既定で `data/predict_plot.png` に枠と Top1 ラベルを描いた画像も出る
(保存先を変える場合は `--plot-out <path>`、画像が不要なら `--no-plot`)。

```bash
python src/predict.py --run runs/icon_v1 --screenshot data/shots/shot1.png
```

数字モデル (新方式、手順 12 で学習) がある場合は `--glyph-run` を付けると、
所持数も推論して表示とプロットに含める。
読めなかったセルは、原因切り分け用にグリフごとの内訳 (上位2候補) も表示する。

```bash
python src/predict.py --run runs/icon_v1 --glyph-run runs/glyph_v1 --screenshot data/shots/shot1.png
```

旧方式の数字モデルは `--digit-run runs/digit_v1` で付けられる
(両方付けた場合は新方式が優先)。

### 9. ONNX 書き出し

デフォルトで `../common/public/models/` へ出力する。

```bash
python src/export_onnx.py --run runs/icon_v1
```

動作確認だけなら `build_dataset.py` に `--limit-classes 5 --per-class 30`、
`train.py` に `--epochs 2` を付けると全工程を数分で通せる。

## 数字モデル (新方式: グリフ分類)

帯を二値化して連結成分に分け、幾何ガード (端の成分の除外、幅の上限、
ベースラインの一致) で数字グリフだけを残し、1つの分類器で
0〜9 + 棄却 (×) の11クラスを判定して連結する方式。
桁位置を持たないので、ある数量でしか出ない桁という偏りが起きない。
抽出と割り当ての共通部品は src/glyph_extract.py、読み取りは src/digit_read.py。

### 10. グリフ切り出しの歩留まり測定

ラベル済みセルのうち、グリフへの切り分けと数量の割り当てが安全にできる
割合を測る (exact = × + 数字、digits = 数字のみ、over / under / unsafe = 不採用)。
不採用のセルは学習データから落ちるだけで、推論時に読めないとは限らない。

```bash
python src/measure_glyphs.py --cells-csv data/cells/labels_template.csv
```

### 11. グリフデータセット生成

labels_template.csv の quantity 列が埋まっている行だけを使い、
グリフ1文字ごとの画像データセット (data/glyph_dataset) を作る。
割り当ては右端アンカー (数量は右寄せ描画なので右端が一の位) で、
安全でないセルは丸ごと捨てる。水増しはしない
(固定フォントで、複数解像度の実グリフがそのまま集まるため)。
split はセルの split を継承する (同じセルのグリフが train/val に割れない)。

```bash
python src/build_glyph_dataset.py --cells-csv data/cells/labels_template.csv
```

### 12. 学習

グリフ分類器を学習する。成果物の run フォルダには best.pt の他に
glyph_meta.json と preprocess.json が同梱され、読み取り側が run だけで完結する。

```bash
python src/train_glyphs.py --dataset data/glyph_dataset --epochs 30 --out runs/glyph_v1
```

### 13. 評価

グリフ単位の精度と混同ペア、および val セルでの数量の完全一致率
(旧方式の evaluate と同じ土俵の数字) を出す。

```bash
python src/evaluate_glyphs.py --run runs/glyph_v1
```

`--train-check` を付けると、学習グリフをモデルで再判定して自動ラベルと
強く矛盾するものを runs/<run>/label_suspects.csv に出す (誤ラベルの疑い)。
疑いのあるグリフを除いてデータセットを作り直す場合は、その CSV を
build_glyph_dataset.py の `--exclude` に渡す。

```bash
python src/evaluate_glyphs.py --run runs/glyph_v1 --train-check
python src/build_glyph_dataset.py --cells-csv data/cells/labels_template.csv \
  --exclude runs/glyph_v1/label_suspects.csv
```

### 14. ONNX 書き出し

デフォルトで `../common/public/models/` へ glyph_model.onnx、
glyph_preprocess.json、glyph_meta.json を出力する。
glyph_meta.json にはグリフ抽出の幾何ガードの定数も含めてあり、
Web 側 (TS) が抽出を再現できるようにしてある。

```bash
python src/export_glyph_onnx.py --run runs/glyph_v1
```

## 数字モデル (旧方式: 6ヘッド)

帯全体を1枚として入力し、桁ごとのヘッド (6桁 × 11クラス) で読む方式。
アイコンの切れ端と数量の相関を覚えてしまう問題 (切れ端から数量を当てる
近道) が見つかったため、新方式に置き換えた。比較とフォールバック用に残す。

### 旧1. 数字データセット生成

labels_template.csv の quantity 列が埋まっている行だけを使い、
セル下部の帯を切り出したデータセット (data/digit_dataset) を作る。
split はアイコン側と同じ列を使う。train は生 + 水増し 9 枚、val は生のみ。
水増しは「セル全体をずらして (縦方向厚め) から帯を切り直す」方式で、
検出枠のズレによる数字の動きを再現する。桁の組み替え合成はしない
(右寄せ描画では各桁の位置が文字列全体で決まるため、合成すると
実際には現れない位置関係を覚えてしまう)。
加えて水増しの一部 (既定 60%、--scramble-prob で調整) では背景スクランブルを行う:
「×+数字」の保護枠の外側だけを別のセルの帯で置き換え、アイコンの切れ端と
数量の相関を断つ (切れ端から数量を当てる近道対策)。保護枠の中は元の画素の
ままなので数字自体は変わらない。グリフ検出が quantity の桁数と整合しない
帯はスクランブルせず、通常の水増しに倒れる。

```bash
python src/build_digit_dataset.py --cells-csv data/cells/labels_template.csv
```

### 旧2. 数字モデルの学習

帯を 6 桁のヘッド (空欄 + 0〜9) で読むモデルを学習する。

```bash
python src/train_digits.py --dataset data/digit_dataset --epochs 40 --out runs/digit_v1
```

### 旧3. 数字モデルの評価

数量の完全一致率と桁ごとの精度、不一致の一覧を出す。
「x を数字と誤認して先頭に余分な桁が付く」型 (幽霊桁) の件数も数える。

```bash
python src/evaluate_digits.py --run runs/digit_v1 --dataset data/digit_dataset
```

### 旧4. 数字モデルの ONNX 書き出し

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
