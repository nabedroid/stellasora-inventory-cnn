# 概要
ステラソラの物資画面のスクリーンショットから素材数を解析し、一覧表示およびCSV出力する。

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
