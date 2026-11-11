# 学習済みの数字モデルを ONNX に書き出す。
# digit_preprocess.json と digit_meta.json も同じフォルダに置き、
# GitHub Pages 側が帯の切り出し条件と合わせて推論できるようにする。
# (アイコン側の preprocess.json と名前が被らないよう、数字側は digit_ 接頭辞)
# 出力先のデフォルトは Web アプリが読む common/public/models/。
# 使い方:
#   python src/export_digits_onnx.py --run runs/digit_v1

import argparse
import json
import shutil
from pathlib import Path

import torch

from preprocess import digit_input_size, load_config
from train_digits import build_digit_model


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", required=True)
  parser.add_argument("--out", default=None)
  args = parser.parse_args()

  run_dir = Path(args.run)
  repo_root = Path(__file__).resolve().parents[2]
  out_dir = Path(args.out) if args.out else repo_root / "common" / "public" / "models"
  out_dir.mkdir(parents=True, exist_ok=True)

  config = load_config(run_dir / "preprocess.json")
  device = torch.device("cpu")
  checkpoint = torch.load(run_dir / "best.pt", map_location=device)
  model = build_digit_model(config).to(device)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()

  width, height = digit_input_size(config)
  dummy = torch.randn(1, 3, height, width)
  onnx_path = out_dir / "digit_model.onnx"
  # アイコン側と同理由で dynamo=False に固定する
  torch.onnx.export(
    model, dummy, str(onnx_path),
    input_names=["input"], output_names=["digits"],
    dynamic_axes={"input": {0: "batch"}, "digits": {0: "batch"}},
    opset_version=17,
    dynamo=False,
  )
  print(f"ONNX を書きました: {onnx_path} (入力: batch x 3 x {height} x {width})")

  shutil.copy(run_dir / "preprocess.json", out_dir / "digit_preprocess.json")
  meta = {
    "digits": config["digit"]["digits"],
    "blank": 10,
    "strip_top_ratio": config["digit"]["strip_top_ratio"],
    "input_width": width,
    "input_height": height,
  }
  with open(out_dir / "digit_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
  print("digit_preprocess.json と digit_meta.json を書きました")

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
