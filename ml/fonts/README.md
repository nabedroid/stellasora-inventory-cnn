# プロット用フォントについて

predict.py のプロット画像で日本語ラベルを描くには日本語フォントが要る。
配置先は `ml/fonts/MPLUS1p-Regular.ttf` で、**Git には入れず**、
ml コンテナのイメージビルド時に Dockerfile がダウンロードして配置する
(手動で同名ファイルを置いても、そのまま使われる)。

- フォント: M PLUS 1p (Regular)
- ライセンス: SIL Open Font License 1.1
- 配布元: https://fonts.google.com/specimen/M+PLUS+1p (Google Fonts)

フォントが無い環境では predict.py は既定フォントにフォールバックする
(その場合、日本語ラベルは正しく描画されない)。
