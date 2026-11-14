// CNN (ONNX) で所持数を読むサービス (方式A: グリフ分類)。
// モデル一式は common/public/models/ に置く:
//   glyph_model.onnx / glyph_meta.json / glyph_preprocess.json
//   (ml 側の export_glyph_onnx.py が出力)
// 処理は ml 側の digit_read.py と同じ手順に揃えること:
//   帯の切り出し → 二値化 (Otsu、融合の疑いがあれば適応的しきい値で再試行)
//   → 連結成分に幾何ガード → グリフを一括推論 → × の消費と数量の組み立て。
// 抽出定数 (ガード・適応的しきい値の条件) は glyph_meta.json が正。

import * as ort from 'onnxruntime-web';
import { crop, MatManager } from '@common/utils/mat';

declare const cv: any;

const MODEL_BASE_URL = `${import.meta.env.BASE_URL}models/`;

// 帯の中で数字が現れる下部領域の開始位置 (帯に対する比率)。
// ml 側 build_digit_dataset.BAND_TOP と一致させること。
const BAND_TOP = 0.38;

// 連結成分の最小サイズ (ml 側 glyph_boxes と同じ)
const MIN_COMPONENT_HEIGHT = 6;
const MIN_COMPONENT_WIDTH = 2;
const MIN_COMPONENT_AREA = 15;

interface GlyphMeta {
  classes: string[];
  input_size: number;
  pad: number;
  strip_top_ratio: number;
  width_cap_ratio: number;
  min_width_ratio: number;
  tall_h_ratio: number;
  baseline_tol_ratio: number;
  x_height_ratio: number;
  adaptive_block: number;
  adaptive_c: number;
  band_top?: number;
}

interface GlyphPreprocess {
  normalize: {
    mean: number[];
    std: number[];
  };
}

// グリフ成分の矩形 (帯座標)
interface GlyphBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export class GlyphCnnService {
  private static _instance: GlyphCnnService | null = null;

  private readonly session: ort.InferenceSession;
  private readonly meta: GlyphMeta;
  private readonly preprocess: GlyphPreprocess;

  private constructor(session: ort.InferenceSession, meta: GlyphMeta, preprocess: GlyphPreprocess) {
    this.session = session;
    this.meta = meta;
    this.preprocess = preprocess;
  }

  public static async getInstanceAsync(): Promise<GlyphCnnService> {
    if (this._instance) {
      return this._instance;
    }

    const [metaResponse, preprocessResponse] = await Promise.all([
      fetch(`${MODEL_BASE_URL}glyph_meta.json`),
      fetch(`${MODEL_BASE_URL}glyph_preprocess.json`),
    ]);
    if (!metaResponse.ok || !preprocessResponse.ok) {
      throw new Error(
        'グリフモデル一式が配置されていません。ml 側で学習と書き出しを行い、' +
        'glyph_model.onnx / glyph_meta.json / glyph_preprocess.json を common/public/models/ に置いてください。');
    }
    const meta = await metaResponse.json() as GlyphMeta;
    const preprocess = await preprocessResponse.json() as GlyphPreprocess;

    let session: ort.InferenceSession;
    try {
      session = await ort.InferenceSession.create(
        `${MODEL_BASE_URL}glyph_model.onnx`, { executionProviders: ['webgpu'] });
    } catch {
      session = await ort.InferenceSession.create(
        `${MODEL_BASE_URL}glyph_model.onnx`, { executionProviders: ['wasm'] });
    }

    this._instance = new GlyphCnnService(session, meta, preprocess);
    return this._instance;
  }

  /**
   * アイコン画像 (セル全体の Mat) の配列から所持数を一括で読む。
   * 読めなかったセルは null。
   */
  public async readQuantitiesAsync(iconMats: any[]): Promise<(number | null)[]> {
    // セルごとにグリフを切り出し、全セル分をまとめて1回で推論する
    const perCell: { boxes: GlyphBox[]; glyphCount: number }[] = [];
    const glyphInputs: Float32Array[] = [];

    for (const iconMat of iconMats) {
      const { boxes, inputs } = this.extractGlyphs(iconMat);
      perCell.push({ boxes, glyphCount: inputs.length });
      glyphInputs.push(...inputs);
    }

    const probsList = await this.predictProbsAsync(glyphInputs);

    const results: (number | null)[] = [];
    let cursor = 0;
    for (const cell of perCell) {
      const probs = probsList.slice(cursor, cursor + cell.glyphCount);
      cursor += cell.glyphCount;
      results.push(this.assembleQuantity(cell.boxes, probs));
    }
    return results;
  }

  // セルから帯を切り出し、グリフ成分の矩形とモデル入力を返す
  private extractGlyphs(iconMat: any): { boxes: GlyphBox[]; inputs: Float32Array[] } {
    using matManager = new MatManager();
    const stripTop = Math.round(iconMat.rows * this.meta.strip_top_ratio);
    const stripMat = matManager.add(
      crop(iconMat, 0, stripTop, iconMat.cols, iconMat.rows - stripTop));
    const stripWidth = stripMat.cols;
    const stripHeight = stripMat.rows;

    const grayMat = matManager.add(new cv.Mat());
    cv.cvtColor(stripMat, grayMat, cv.COLOR_RGBA2GRAY);

    const boxes = this.extractBoxesCascade(grayMat, stripWidth, stripHeight);
    if (boxes.length === 0) {
      return { boxes, inputs: [] };
    }

    // グリフの切り出しは RGB の帯に対して行う (ml 側 crop_glyph と同じ)
    const stripCanvas = document.createElement('canvas');
    cv.imshow(stripCanvas, stripMat);
    const inputs = boxes.map(box => this.toGlyphInput(stripCanvas, box));
    return { boxes, inputs };
  }

  // カスケード抽出: まず Otsu で抽出し、数字らしい背丈の成分が
  // ガードで落ちていたら (融合の疑い) 適応的しきい値で切り直す。
  // ml 側 extract_glyph_boxes_cascade の推論側と同じ判定。
  private extractBoxesCascade(grayMat: any, stripWidth: number, stripHeight: number): GlyphBox[] {
    const raw = this.rawBoxes(grayMat, stripHeight, false);
    const boxes = this.guardedBoxes(raw, stripWidth, stripHeight);
    const tallHeight = stripHeight * this.meta.tall_h_ratio;
    const dropped = raw.some(b => b.height >= tallHeight && !boxes.includes(b));
    if (boxes.length > 0 && !dropped) {
      return boxes;
    }
    const rawAdaptive = this.rawBoxes(grayMat, stripHeight, true);
    const boxesAdaptive = this.guardedBoxes(rawAdaptive, stripWidth, stripHeight);
    return boxesAdaptive.length > boxes.length ? boxesAdaptive : boxes;
  }

  // 帯の下部を二値化し、連結成分の生の矩形 (帯座標, x 順) を返す。
  // adaptive = true のとき二値化を適応的しきい値に替える。
  private rawBoxes(grayMat: any, stripHeight: number, adaptive: boolean): GlyphBox[] {
    using matManager = new MatManager();
    const bandTop = Math.floor(stripHeight * (this.meta.band_top ?? BAND_TOP));
    const bandMat = matManager.add(
      crop(grayMat, 0, bandTop, grayMat.cols, stripHeight - bandTop));

    const binaryMat = matManager.add(new cv.Mat());
    if (adaptive) {
      cv.adaptiveThreshold(
        bandMat, binaryMat, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, this.meta.adaptive_block, this.meta.adaptive_c);
    } else {
      cv.threshold(bandMat, binaryMat, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);
    }
    // 数字は暗い側 (二値化で 0 になった画素) がインク
    const inkMat = matManager.add(new cv.Mat());
    cv.threshold(binaryMat, inkMat, 0, 1, cv.THRESH_BINARY_INV);

    const labelsMat = matManager.add(new cv.Mat());
    const statsMat = matManager.add(new cv.Mat());
    const centroidsMat = matManager.add(new cv.Mat());
    const count = cv.connectedComponentsWithStats(
      inkMat, labelsMat, statsMat, centroidsMat, 8, cv.CV_32S);

    const boxes: GlyphBox[] = [];
    const stats = statsMat.data32S;
    for (let i = 1; i < count; i++) {
      const x = stats[i * 5];
      const y = stats[i * 5 + 1];
      const width = stats[i * 5 + 2];
      const height = stats[i * 5 + 3];
      const area = stats[i * 5 + 4];
      if (height >= MIN_COMPONENT_HEIGHT && width >= MIN_COMPONENT_WIDTH && area >= MIN_COMPONENT_AREA) {
        boxes.push({ x, y: y + bandTop, width, height });
      }
    }
    if (boxes.length === 0) {
      return [];
    }
    const hmax = Math.max(...boxes.map(b => b.height));
    return boxes
      .filter(b => b.height >= 0.5 * hmax)
      .sort((a, b) => (a.x - b.x) || (a.y - b.y));
  }

  // 生の連結成分に幾何ガードをかける (ml 側 _guarded_boxes と同じ)
  private guardedBoxes(rawBoxes: GlyphBox[], stripWidth: number, stripHeight: number): GlyphBox[] {
    // 帯の左右端に触れる成分 (セルの枠の影) を落とす
    let boxes = rawBoxes.filter(b => b.x > 0 && b.x + b.width < stripWidth);
    // 幅で落とす: 広すぎる成分 (アイコンの破片) と細すぎる成分 (ノイズ)
    boxes = boxes.filter(b =>
      stripWidth * this.meta.min_width_ratio <= b.width &&
      b.width <= stripWidth * this.meta.width_cap_ratio);
    if (boxes.length === 0) {
      return [];
    }
    // ベースライン: 背の高い成分のうち最右のものの下端
    const tall = boxes.filter(b => b.height >= stripHeight * this.meta.tall_h_ratio);
    const reference = tall.length > 0 ? tall[tall.length - 1] : boxes[boxes.length - 1];
    const baseline = reference.y + reference.height;
    const tol = Math.max(2, Math.round(stripHeight * this.meta.baseline_tol_ratio));
    return boxes.filter(b => Math.abs((b.y + b.height) - baseline) <= tol);
  }

  // グリフ成分を pad 分広げて切り出し、縦横比を保って入力サイズに
  // レターボックスし、正規化済みの数値列 (CHW) に変換する。
  // 余白の色は切り出し画像の縁の中央値 (ml 側 crop_glyph と同じ)。
  private toGlyphInput(stripCanvas: HTMLCanvasElement, box: GlyphBox): Float32Array {
    const size = this.meta.input_size;
    const pad = this.meta.pad;
    const x0 = Math.max(0, box.x - pad);
    const y0 = Math.max(0, box.y - pad);
    const x1 = Math.min(stripCanvas.width, box.x + box.width + pad);
    const y1 = Math.min(stripCanvas.height, box.y + box.height + pad);
    const cropWidth = x1 - x0;
    const cropHeight = y1 - y0;

    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = cropWidth;
    cropCanvas.height = cropHeight;
    const cropCtx = cropCanvas.getContext('2d')!;
    cropCtx.drawImage(stripCanvas, x0, y0, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);
    const pixels = cropCtx.getImageData(0, 0, cropWidth, cropHeight).data;

    // 縁の画素の中央値を余白の色にする
    const border: number[][] = [[], [], []];
    const pushPixel = (px: number, py: number) => {
      const offset = (py * cropWidth + px) * 4;
      for (let channel = 0; channel < 3; channel++) {
        border[channel].push(pixels[offset + channel]);
      }
    };
    for (let px = 0; px < cropWidth; px++) {
      pushPixel(px, 0);
      pushPixel(px, cropHeight - 1);
    }
    for (let py = 0; py < cropHeight; py++) {
      pushPixel(0, py);
      pushPixel(cropWidth - 1, py);
    }
    const fill = border.map(values => {
      const sorted = [...values].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      const median = sorted.length % 2 === 1
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2;
      return Math.trunc(median);
    });

    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = `rgb(${fill[0]}, ${fill[1]}, ${fill[2]})`;
    ctx.fillRect(0, 0, size, size);
    const scale = Math.min(size / cropWidth, size / cropHeight);
    const resizedWidth = Math.max(1, Math.round(cropWidth * scale));
    const resizedHeight = Math.max(1, Math.round(cropHeight * scale));
    ctx.drawImage(
      cropCanvas,
      Math.floor((size - resizedWidth) / 2), Math.floor((size - resizedHeight) / 2),
      resizedWidth, resizedHeight);
    const inputPixels = ctx.getImageData(0, 0, size, size).data;

    const { mean, std } = this.preprocess.normalize;
    const out = new Float32Array(3 * size * size);
    const planeSize = size * size;
    for (let i = 0; i < planeSize; i++) {
      for (let channel = 0; channel < 3; channel++) {
        const value = inputPixels[i * 4 + channel] / 255;
        out[channel * planeSize + i] = (value - mean[channel]) / std[channel];
      }
    }
    return out;
  }

  // グリフ群をバッチで推論し、グリフごとのクラス確率を返す
  private async predictProbsAsync(inputs: Float32Array[]): Promise<number[][]> {
    if (inputs.length === 0) {
      return [];
    }
    const size = this.meta.input_size;
    const planeSize = 3 * size * size;
    const data = new Float32Array(inputs.length * planeSize);
    inputs.forEach((input, index) => data.set(input, index * planeSize));

    const inputName = this.session.inputNames[0];
    const tensor = new ort.Tensor('float32', data, [inputs.length, 3, size, size]);
    const output = await this.session.run({ [inputName]: tensor });
    const logits = output[this.session.outputNames[0]].data as Float32Array;
    const classCount = this.meta.classes.length;

    return inputs.map((_, index) => {
      const row = logits.subarray(index * classCount, (index + 1) * classCount);
      const max = Math.max(...row);
      const exps = Array.from(row, value => Math.exp(value - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map(value => value / sum);
    });
  }

  // box が × らしい背丈か (他の成分の背丈の中央値より十分に低いか)
  private isXLike(box: GlyphBox, others: GlyphBox[]): boolean {
    if (others.length === 0) {
      return false;
    }
    const heights = others.map(b => b.height).sort((a, b) => a - b);
    const medianHeight = heights[Math.floor(heights.length / 2)];
    return box.height < medianHeight * this.meta.x_height_ratio;
  }

  // グリフの判定列から数量を組み立てる (ml 側 digit_read と同じ)。
  // 左端の棄却は背丈が × らしい場合だけ × として消費し、数字並みの
  // 背丈なら棄却以外の最上位候補をその桁として採用する。
  // それ以外の位置に棄却が混ざれば不明 (null) に倒す。
  private assembleQuantity(boxes: GlyphBox[], probsList: number[][]): number | null {
    if (boxes.length === 0) {
      return null;
    }
    const rejectIndex = this.meta.classes.indexOf('reject');
    const labels = probsList.map(probs => {
      let best = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[best]) {
          best = i;
        }
      }
      return best;
    });

    let digitLabels = labels;
    if (labels[0] === rejectIndex && labels.length >= 2) {
      if (this.isXLike(boxes[0], boxes.slice(1))) {
        digitLabels = labels.slice(1);
      } else {
        const probs0 = probsList[0];
        let best = -1;
        for (let i = 0; i < probs0.length; i++) {
          if (i !== rejectIndex && (best < 0 || probs0[i] > probs0[best])) {
            best = i;
          }
        }
        digitLabels = [best, ...labels.slice(1)];
      }
    }

    if (digitLabels.includes(rejectIndex)) {
      return null;
    }
    const text = digitLabels.map(i => this.meta.classes[i]).join('');
    return parseInt(text, 10);
  }
}
