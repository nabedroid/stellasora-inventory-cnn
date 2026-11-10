# 学習・評価・推論を全部この1コンテナで行う。
# ステップごとにコンテナを分けない (イメージが同一になるだけなので)。
# torch / torchvision はこのベースイメージに同梱されている。
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app/ml

COPY ml/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ./

# 常駐させて docker compose exec で各スクリプトを実行する使い方を想定
CMD ["sleep", "infinity"]
