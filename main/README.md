# 概要
ステラソラの物資画面のスクリーンショットから素材数を解析し、一覧表示およびCSV出力する。
アイコン判定・所持数判定とも CNN (ONNX) で行う。

公開ページ: https://nabedroid.github.io/stellasora-inventory-cnn/
(main への push で自動デプロイされる)

# 開発環境
- Windows 11
- Docker

# ローカル実行

1. Docker がインストールされた Windows の ターミナル（powershell）で以下のコマンドを実行

```
# ビルド及び開発サーバーでアプリが起動します
docker compose up main --build
# または watch モードで起動します
docker compose up main --watch
```

2. ブラウザで http://localhost:3000/ にアクセス

前提: モデル一式が `common/public/models/` に配置されていること
(未配置だとアプリ側にエラーが出る。配置方法はルートの README を参照)。
