// CNN (ONNX) でアイコンを判定するサービス。
// モデル一式は common/public/models/ に置く:
//   icon_model.onnx / labels.json / preprocess.json (ml 側の export_onnx.py が出力)
// 前処理 (数字エリアのマスク・入力サイズ・正規化) は preprocess.json が正で、
// 学習側 (Python) と数値を一致させること。

import * as ort from 'onnxruntime-web';
import type { CnnCandidate } from '@common/types';

declare const cv: any;

const MODEL_BASE_URL = `${import.meta.env.BASE_URL}models/`;

// onnxruntime-web の wasm ファイルの参照先。
// GitHub Pages 単体で動かすための簡易指定 (CDN)。
// 完全オフライン化する場合は、node_modules/onnxruntime-web/dist の
// *.wasm を public にコピーし、ここを相対パスに変えること。
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

interface PreprocessConfig {
  input_width: number;
  number_area: {
    mode: 'mask' | 'cut' | 'keep';
    strip_ratio: number;
    fill_rgb: number[];
  };
  frame?: {
    mode: 'mask' | 'keep';
    ring_ratio: number;
    fill_rgb: number[];
  };
  normalize: {
    mean: number[];
    std: number[];
  };
}

interface LabelsData {
  classes: string[];
  labels?: Record<string, { id: number; name: string }>;
}

export class IconCnnService {
  private static _instance: IconCnnService | null = null;

  private readonly session: ort.InferenceSession;
  private readonly config: PreprocessConfig;
  private readonly labels: LabelsData;

  private constructor(session: ort.InferenceSession, config: PreprocessConfig, labels: LabelsData) {
    this.session = session;
    this.config = config;
    this.labels = labels;
  }

  public static async getInstanceAsync(): Promise<IconCnnService> {
    if (this._instance) {
      return this._instance;
    }

    // 付属ファイル (前処理設定とラベル対応表) を先に読む
    const [configResponse, labelsResponse] = await Promise.all([
      fetch(`${MODEL_BASE_URL}preprocess.json`),
      fetch(`${MODEL_BASE_URL}labels.json`),
    ]);
    if (!configResponse.ok || !labelsResponse.ok) {
      throw new Error(
        'モデル一式が配置されていません。ml 側で学習と書き出しを行い、' +
        'icon_model.onnx / labels.json / preprocess.json を common/public/models/ に置いてください。');
    }
    const config = await configResponse.json() as PreprocessConfig;
    const labels = await labelsResponse.json() as LabelsData;

    // WebGPU が使えれば WebGPU、ダメなら WASM に落とす
    let session: ort.InferenceSession;
    try {
      session = await ort.InferenceSession.create(
        `${MODEL_BASE_URL}icon_model.onnx`, { executionProviders: ['webgpu'] });
    } catch {
      session = await ort.InferenceSession.create(
        `${MODEL_BASE_URL}icon_model.onnx`, { executionProviders: ['wasm'] });
    }

    this._instance = new IconCnnService(session, config, labels);
    return this._instance;
  }

  // 入力テンソルの (幅, 高さ)。cut モードだけ帯の分だけ縦が短い。
  private inputSize(): { width: number; height: number } {
    const width = this.config.input_width;
    if (this.config.number_area.mode === 'cut') {
      return { width, height: Math.round(width * (1 - this.config.number_area.strip_ratio)) };
    }
    return { width, height: width };
  }

  // アイコン画像 (cv.Mat) をモデル入力の数値列 (CHW, 正規化済み) に変換する。
  // 学習側 preprocess.py の build_transform と同じ手順にすること。
  private toInputData(iconMat: any, out: Float32Array, offset: number): void {
    const { mode, strip_ratio, fill_rgb } = this.config.number_area;
    const { mean, std } = this.config.normalize;
    const { width, height } = this.inputSize();

    // Mat をそのままの大きさで canvas 化する
    const srcCanvas = document.createElement('canvas');
    cv.imshow(srcCanvas, iconMat);
    const srcCtx = srcCanvas.getContext('2d')!;

    // 数字エリアへの対処 (mask / cut / keep)
    let sourceHeight = srcCanvas.height;
    if (mode === 'mask') {
      srcCtx.fillStyle = `rgb(${fill_rgb[0]}, ${fill_rgb[1]}, ${fill_rgb[2]})`;
      srcCtx.fillRect(0, srcCanvas.height * (1 - strip_ratio), srcCanvas.width, srcCanvas.height * strip_ratio);
    } else if (mode === 'cut') {
      sourceHeight = Math.round(srcCanvas.height * (1 - strip_ratio));
    }

    // 枠リングへの対処 (カーソル枠のアニメーション位相差を消す)
    // 学習側 preprocess.py の apply_frame と同じ処理にすること。
    if (this.config.frame?.mode === 'mask') {
      const frame = this.config.frame;
      const ringX = Math.round(srcCanvas.width * frame.ring_ratio);
      const ringY = Math.round(srcCanvas.height * frame.ring_ratio);
      srcCtx.fillStyle = `rgb(${frame.fill_rgb[0]}, ${frame.fill_rgb[1]}, ${frame.fill_rgb[2]})`;
      srcCtx.fillRect(0, 0, srcCanvas.width, ringY);
      srcCtx.fillRect(0, srcCanvas.height - ringY, srcCanvas.width, ringY);
      srcCtx.fillRect(0, 0, ringX, srcCanvas.height);
      srcCtx.fillRect(srcCanvas.width - ringX, 0, ringX, srcCanvas.height);
    }

    // 入力サイズへ縮小して画素を取り出す
    const inputCanvas = document.createElement('canvas');
    inputCanvas.width = width;
    inputCanvas.height = height;
    const inputCtx = inputCanvas.getContext('2d')!;
    inputCtx.drawImage(srcCanvas, 0, 0, srcCanvas.width, sourceHeight, 0, 0, width, height);
    const pixels = inputCtx.getImageData(0, 0, width, height).data;

    // RGB の CHW 並びで正規化しながら詰める (alpha は使わない)
    const planeSize = width * height;
    for (let i = 0; i < planeSize; i++) {
      for (let channel = 0; channel < 3; channel++) {
        const value = pixels[i * 4 + channel] / 255;
        out[offset + channel * planeSize + i] = (value - mean[channel]) / std[channel];
      }
    }
  }

  // アイコン群をバッチで判定し、アイコンごとに候補上位3件を返す。
  // 1個ずつ推論を回さず、必ずバッチで1回にまとめること。
  public async classifyAsync(iconMats: any[]): Promise<CnnCandidate[][]> {
    if (iconMats.length === 0) {
      return [];
    }
    const { width, height } = this.inputSize();
    const planeSize = width * height;
    const data = new Float32Array(iconMats.length * 3 * planeSize);
    iconMats.forEach((iconMat, index) => {
      this.toInputData(iconMat, data, index * 3 * planeSize);
    });

    const inputName = this.session.inputNames[0];
    const tensor = new ort.Tensor('float32', data, [iconMats.length, 3, height, width]);
    const output = await this.session.run({ [inputName]: tensor });
    const logits = output[this.session.outputNames[0]].data as Float32Array;
    const classCount = this.labels.classes.length;

    return iconMats.map((_, index) => {
      const row = logits.subarray(index * classCount, (index + 1) * classCount);
      // softmax で確率に直す
      const max = Math.max(...row);
      const exps = Array.from(row, value => Math.exp(value - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(value => value / sum);
      // 確率の高い順に上位3件
      const order = probs
        .map((prob, classIndex) => ({ prob, classIndex }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 3);
      return order.map(({ prob, classIndex }) => {
        const className = this.labels.classes[classIndex];
        const info = this.labels.labels?.[className];
        return { id: info?.id, name: info?.name ?? className, prob };
      });
    });
  }
}
