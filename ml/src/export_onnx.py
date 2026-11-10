# 学習済みモデルを ONNX に書き出す。
# labels.json と preprocess.json も同じフォルダにコピーし、
# GitHub Pages 側がこの3点セットだけで推論できるようにする。
# 出力先のデフォルトは Web アプリが読む common/public/models/。
# 使い方:
#   python src/export_onnx.py --run runs/icon_v1

import argparse
import json
import shutil
from pathlib import Path

import torch

from preprocess import input_size, load_config
from train import build_model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True)
  parser.add_argument("--out", default=None)
  args = parser.parse_args()

  run_dir = Path(args.run)
  # デフォルトの出力先はリポジトリ内の common/public/models/ (Web アプリの読込先)
  repo_root = Path(__file__).resolve().parents[2]
  out_dir = Path(args.out) if args.out else repo_root / "common" / "public" / "models"
  out_dir.mkdir(parents=True, exist_ok=True)

  config = load_config(run_dir / "preprocess.json")
  with open(run_dir / "labels.json", encoding="utf-8") as f:
    classes = json.load(f)["classes"]

  device = torch.device("cpu")  # 書き出しは CPU で十分
  checkpoint = torch.load(run_dir / "best.pt", map_location=device)
  model = build_model(len(classes)).to(device)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()

  width, height = input_size(config)
  dummy = torch.randn(1, 3, height, width)
  onnx_path = out_dir / "icon_model.onnx"
  # dynamo=False を明示する。新しい torch ではデフォルトの書き出し方式が
  # 変わっていて追加パッケージ (onnxscript) が必要になるため、従来方式に固定する。
  torch.onnx.export(
    model, dummy, str(onnx_path),
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
    dynamo=False,
  )
  print(f"ONNX を書きました: {onnx_path} (入力: batch x 3 x {height} x {width})")

  # 付属ファイルを同梱する
  shutil.copy(run_dir / "labels.json", out_dir / "labels.json")
  shutil.copy(run_dir / "preprocess.json", out_dir / "preprocess.json")
  print("labels.json と preprocess.json をコピーしました")

  # 書き出したモデルが onnxruntime で読めるか軽く確認する (入っていれば)
  try:
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"input": np.zeros((2, 3, height, width), dtype=np.float32)})
    print(f"onnxruntime での確認 OK: 出力形状 {output[0].shape}")
  except ImportError:
    print("onnxruntime が無いので読み込み確認はスキップしました")


if __name__ == "__main__":
  main()
