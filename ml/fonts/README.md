# プロット用フォントについて

predict.py のプロット画像で日本語ラベルを描くには日本語フォントが要る。
フォントファイルは **Git には入れず**、ml コンテナのイメージビルド時に
Dockerfile が `/usr/local/share/fonts/MPLUS1p-Regular.ttf` へ取得する
(`/app/ml` はホストの bind mount で隠れるため、マウント範囲外に置く)。

手動で `ml/fonts/MPLUS1p-Regular.ttf` に置いた場合も、そちらが優先で使われる。

- フォント: M PLUS 1p (Regular)
- ライセンス: SIL Open Font License 1.1
- 配布元: https://fonts.google.com/specimen/M+PLUS+1p (Google Fonts)

フォントが無い環境では predict.py は既定フォントにフォールバックする
(その場合、日本語ラベルは正しく描画されない)。
