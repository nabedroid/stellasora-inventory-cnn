# 学習・評価・推論を全部この1コンテナで行う。
# ステップごとにコンテナを分けない (イメージが同一になるだけなので)。
# torch / torchvision はこのベースイメージに同梱されている。
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app/ml

COPY ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ./

# predict.py のプロット描画用に日本語フォントをビルド時に取得する
# (バイナリは Git 管理しない。/app/ml はホストの bind mount で隠れるため、
#  マウント範囲外の /usr/local/share/fonts に置く)
RUN mkdir -p /usr/local/share/fonts && python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/google/fonts/main/ofl/mplus1p/MPLUS1p-Regular.ttf', '/usr/local/share/fonts/MPLUS1p-Regular.ttf')"

# 常駐させて docker compose exec で各スクリプトを実行する使い方を想定
CMD ["sleep", "infinity"]
