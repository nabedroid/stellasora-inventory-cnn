# モデル一式の置き場所

ml 側で学習・書き出しした以下の6ファイルをここに置く。

アイコン判定用 (export_onnx.py が出力):
- icon_model.onnx
- labels.json
- preprocess.json

所持数のグリフ判定用 (export_glyph_onnx.py が出力):
- glyph_model.onnx
- glyph_meta.json
- glyph_preprocess.json

置くだけで Web アプリ (CNN 判定) が動く。未配置の場合はアプリ側に
「モデル一式が配置されていません」というエラーが出る。
