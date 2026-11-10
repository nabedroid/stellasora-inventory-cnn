# 概要
アイテムアイコンからアイテムの名前を抽出する際に使用するマスターデータを作成します。

# 開発環境
- Windows 11
- Docker

# ローカル実行

1. Docker がインストールされた Windows の ターミナル（powershell）で以下のコマンドを実行

```
# ビルド及び開発サーバーでアプリが起動します
docker compose up manager
```

2. ブラウザで http://localhost:3001/ にアクセス

3. 物資画面のスクリーンショットをアップロードする

4. 物資画面で選択されたアイテムのアイコン画像と名前が辞書に取り込まれます

# 補足：run で起動する場合

正式な手順は上記 `up` だが、`up` は終了後にコンテナの stop / 削除が別途必要になる。
使い捨てでよければ `run --rm` でも起動でき、終了すればコンテナが自動で削除される。

```
docker compose run --rm --service-ports manager
```

注意点：

- `run` は既定では compose の ports 設定を公開しないため、`--service-ports` が必須。
  付け忘れるとサーバーは起動してもブラウザから接続できない (ERR_CONNECTION_REFUSED)
- アクセス先は起動時に表示される URL (http://localhost:3001/stellasora-inventory-ocr/manager/)