# 概要
ステラソラの物資画面のスクリーンショットから素材数を解析し、
各素材の所持数を表示する (CNN 版)。

アイコンの判定を CNN (ONNX) で行う単体リポジトリ。
旧版 `stellasora-inventory-ocr` (ORB 特徴量方式) とは別管理で、
あちらは今後アップデートしない。

- アイコン判定: CNN (ONNX Runtime Web、WebGPU / WASM)
- 所持数: Tesseract.js (数字 CNN は今後対応予定)
- 処理はブラウザ内で完結し、スクリーンショットは外部に送信されない

# 使い方 (利用者向け)
1. GitHub Pages のページを開く
2. 物資画面のスクリーンショットを選択する (複数枚可)
3. 「解析開始」を押す

前提として、モデル一式が `common/public/models/` に配置されている必要がある
(下記「モデルの学習と配置」参照)。未配置だとエラーになる。

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
3. 学習 (MobileNetV3-Small)
4. `export_onnx.py` で `common/public/models/` へ書き出し
   (icon_model.onnx / labels.json / preprocess.json の3点セット)

## 数字エリアの扱い
分類の入力では所持数の帯 (下部20%) をマスクする (平均色で塗りつぶし)。
設定は `ml/config/preprocess.json` が正で、Web 側も同じ JSON を読む。
理由: 学習データではアイテムごとに所持数がほぼ固定のため、
数字を残すとモデルが数字を識別の手がかりに覚えてしまう (ショートカット学習)。
