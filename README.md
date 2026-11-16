# 概要
ステラソラの物資画面のスクリーンショットから素材数を解析し、
各素材の所持数を表示する (CNN 版)。

アイコン判定と所持数判定の両方を CNN (ONNX) で行う単体リポジトリ。
旧版 `stellasora-inventory-ocr` (ORB 特徴量方式 + Tesseract) とは別管理で、
あちらは今後アップデートしない。

- アイコン判定: CNN (MobileNetV3-Small、ONNX Runtime Web、WebGPU / WASM)
- 所持数: CNN (グリフ分類。数字の帯を連結成分に分けて1文字ずつ判定する)
- 処理はブラウザ内で完結し、スクリーンショットは外部に送信されない

公開ページ: https://nabedroid.github.io/stellasora-inventory-cnn/

# 使い方 (利用者向け)
1. GitHub Pages のページを開く
2. 物資画面のスクリーンショットを選択する (複数枚可)
3. 「解析開始」を押す

公開ページはモデル一式込みでデプロイされている。
ローカルで動かす場合は、モデル一式が `common/public/models/` に
配置されている必要がある (下記「モデルの学習と配置」参照)。
未配置だとエラーになる。

# 開発環境
- Windows 11
- Docker

# ローカルでの実行
```
# Web アプリ
docker compose up main
# マスターデータを自作する場合
docker compose up manager
```
ブラウザで http://localhost:3000/ を開く (manager は 3001)。

# モデルの学習と配置
学習は ml コンテナで行う。詳細は `ml/README.md` を参照。

流れだけ書くと:
1. スクショからセル切り出しとラベル付け
2. データセット生成 (マスター辞書の種画像 + 実セル)
3. 学習 (アイコン: MobileNetV3-Small / 所持数: グリフ分類器)
4. ONNX 書き出しで `common/public/models/` へ配置 (計6ファイル)
   - アイコン: `export_onnx.py`
     (icon_model.onnx / labels.json / preprocess.json)
   - 所持数: `export_glyph_onnx.py`
     (glyph_model.onnx / glyph_meta.json / glyph_preprocess.json)

## 数字エリアの扱い
分類の入力では所持数の帯 (下部20%) をマスクする (平均色で塗りつぶし)。
設定は `ml/config/preprocess.json` が正で、書き出し時にモデル側の
preprocess.json として同梱され、Web 側はそちらを読む。
理由: 学習データではアイテムごとに所持数がほぼ固定のため、
数字を残すとモデルが数字を識別の手がかりに覚えてしまう (ショートカット学習)。
