declare const cv: any;

export class MatManager {
  private mats: any[] = [];

  public add(mat: any): any {
    this.mats.push(mat);
    return mat;
  }

  public delete(): void {
    this.mats.forEach(mat => mat.delete());
    this.mats = [];
  }

  [Symbol.dispose]() {
    this.delete();
  }
}

/**
 * 画像を指定された矩形で切り出す
 */
export const crop = (src: any, x: number, y: number, width: number, height: number): any => {

  const roundX = Math.round(x)
  const roundY = Math.round(y)
  const roundW = Math.round(width)
  const roundH = Math.round(height)

  if (roundX < 0 || roundY < 0 || roundW <= 0 || roundH <= 0 ||
    roundX + roundW > src.cols || roundY + roundH > src.rows) {
    console.error(`Invalid crop region: src size(${src.cols}, ${src.rows}) crop rect(${roundX}, ${roundY}, ${roundW}, ${roundH})`)
    throw new Error('Invalid crop region')
  }

  const rect = new cv.Rect(roundX, roundY, roundW, roundH);
  return src.roi(rect);
}

/**
 * マスクを作成する
 */
export const mask = (src: any, lowerRgba: number[], upperRgba: number[]): any => {
  const lowerMat = new cv.matFromArray(1, lowerRgba.length, cv.CV_8UC1, lowerRgba);
  const upperMat = new cv.matFromArray(1, upperRgba.length, cv.CV_8UC1, upperRgba);
  const mask = new cv.Mat();
  cv.inRange(src, lowerMat, upperMat, mask);
  lowerMat.delete();
  upperMat.delete();
  return mask;
}

/**
 * 色を反転する
 */
export const invert = (src: any): any => {
  const dst = new cv.Mat();
  cv.bitwise_not(src, dst);
  return dst;
}

/**
 * Mat を ImageData に変換する
 */
export const toImageData = (mat: any): ImageData => {
  const canvas = document.createElement('canvas');
  canvas.width = mat.cols;
  canvas.height = mat.rows;
  cv.imshow(canvas, mat);
  return canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * Mat を DataUrl に変換する
 */
export const toDataUrl = (mat: any): string => {
  const canvas = document.createElement('canvas');
  cv.imshow(canvas, mat);
  return canvas.toDataURL('image/png');
}

/**
 * Mat を Base64 に変換する
 */
export const toBase64 = (mat: any): string => {
  const rows = mat.rows;
  const cols = mat.cols;
  const type = mat.type();
  const data = mat.data;
  const uint8Array = new Uint8Array(data);

  // 構造化データとして保存 (rows, cols, type, data)
  const meta = { rows, cols, type };
  const binary = String.fromCharCode(...uint8Array);
  const base64Data = btoa(binary);

  return JSON.stringify({ ...meta, data: base64Data });
}

/**
 * Base64 文字列から Mat に変換する
 * @param base64 Base64 文字列
 * @returns Mat オブジェクト
 */
export const fromBase64 = (base64: string): any => {
  if (!base64) return new cv.Mat();
  try {
    const { rows, cols, type, data } = JSON.parse(base64);
    const binary = atob(data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return cv.matFromArray(rows, cols, type, Array.from(bytes));
  } catch (e) {
    console.error('Failed to restore Mat from base64', e);
    return new cv.Mat();
  }
}

/**
 * File から Mat を作成する
 * @param file File オブジェクト
 * @returns Mat オブジェクト
 */
export const fromFile = (file: File): Promise<any> => {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      try {
        const mat = cv.imread(image);
        resolve(mat);
      } catch (e) {
        reject(e);
      } finally {
        URL.revokeObjectURL(url);
      }
    };
    image.onerror = reject;
    image.src = url;
  });
}