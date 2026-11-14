# 学習済みのグリフ分類器 (方式A) を ONNX に書き出す。
# glyph_preprocess.json と glyph_meta.json も同じフォルダに置く。
# meta には帯の切り出し条件に加えて、グリフ抽出の幾何ガードの定数と
# 切り出し条件 (pad / size) を含める。Web 側 (TS) が同じ抽出と
# レターボックスを再現するために必要で、値は glyph_extract の定数が正。
# (アイコン・旧数字と名前が被らないよう、ファイル名は glyph_ 接頭辞)
# 出力先のデフォルトは Web アプリが読む common/public/models/。
# 使い方:
#   python src/export_glyph_onnx.py --run runs/glyph_v1

import argparse
import json
import shutil
from pathlib import Path

import torch

import glyph_extract
from build_digit_dataset import BAND_TOP
from preprocess import load_config
from train_glyphs import build_glyph_model


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
  with open(run_dir / "glyph_meta.json", encoding="utf-8") as f:
    glyph_meta = json.load(f)
  size = glyph_meta["size"]
  device = torch.device("cpu")
  checkpoint = torch.load(run_dir / "best.pt", map_location=device)
  model = build_glyph_model(size).to(device)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()

  dummy = torch.randn(1, 3, size, size)
  onnx_path = out_dir / "glyph_model.onnx"
  # アイコン側と同理由で dynamo=False に固定する
  torch.onnx.export(
    model, dummy, str(onnx_path),
    input_names=["input"], output_names=["glyph"],
    dynamic_axes={"input": {0: "batch"}, "glyph": {0: "batch"}},
    opset_version=17,
    dynamo=False,
  )
  print(f"ONNX を書きました: {onnx_path} (入力: batch x 3 x {size} x {size})")

  shutil.copy(run_dir / "preprocess.json", out_dir / "glyph_preprocess.json")
  meta = {
    "classes": glyph_meta["classes"],
    "input_size": size,
    "pad": glyph_meta["pad"],
    "strip_top_ratio": config["digit"]["strip_top_ratio"],
    # 帯の中で数字が現れる下部領域の開始位置 (build_digit_dataset.BAND_TOP)
    "band_top": BAND_TOP,
    # グリフ抽出の幾何ガード (glyph_extract.py の定数と一致させる)
    "width_cap_ratio": glyph_extract.WIDTH_CAP_RATIO,
    "min_width_ratio": glyph_extract.MIN_WIDTH_RATIO,
    "tall_h_ratio": glyph_extract.TALL_H_RATIO,
    "baseline_tol_ratio": glyph_extract.BASELINE_TOL_RATIO,
    "x_height_ratio": glyph_extract.X_HEIGHT_RATIO,
    # カスケード再試行 (適応的しきい値) の条件
    "adaptive_block": glyph_extract.ADAPTIVE_BLOCK,
    "adaptive_c": glyph_extract.ADAPTIVE_C,
  }
  with open(out_dir / "glyph_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
  print("glyph_preprocess.json と glyph_meta.json を書きました")

  try:
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"input": np.zeros((2, 3, size, size), dtype=np.float32)})
    print(f"onnxruntime での確認 OK: 出力形状 {output[0].shape}")
  except ImportError:
    print("onnxruntime が無いので読み込み確認はスキップしました")


if __name__ == "__main__":
  main()
