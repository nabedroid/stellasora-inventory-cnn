declare const cv: any;

let orb: any = null;
let matcher: any = null;

const _init = () => {
  if (!orb) {
    // nfeatures: 100
    orb = new cv.ORB(100);
  }
  if (!matcher) {
    matcher = new cv.BFMatcher(cv.NORM_HAMMING, true);
  }
}

export class ComputeResult {

  private _isDisposed: boolean = false;

  constructor(
    public readonly descriptors: any,
    public readonly keypoints: any,
  ) { }

  public delete(): void {
    if (this._isDisposed) return;
    this.descriptors?.delete();
    this.keypoints?.delete();
    this._isDisposed = true;
  }

  [Symbol.dispose](): void {
    this.delete();
  }
}


/**
 * 画像から ORB 特徴量を計算する
 */
export const compute = (mat: any, mask: any = null): ComputeResult | null => {
  _init();
  const keypoints = new cv.KeyPointVector();
  const descriptors = new cv.Mat();

  orb.detectAndCompute(mat, mask, keypoints, descriptors);
  if (descriptors.empty()) return null;
  // Mat のデータを Base64 文字列に変換
  return new ComputeResult(descriptors, keypoints);
}

/**
 * 2つの特徴量を比較する
 * @param descriptors1 特徴量1
 * @param descriptors2 特徴量2
 * @param minDistance 最小距離
 * @returns 類似度
 */
export const compare = (descriptors1: any, descriptors2: any, minDistance: number = 30): number => {
  _init();
  if (descriptors1.empty() || descriptors2.empty()) return 0;

  const matches = new cv.DMatchVector();
  try {
    matcher.match(descriptors1, descriptors2, matches);

    let goodMatches = 0;

    for (let i = 0; i < matches.size(); i++) {
      // distance はハミング距離、小さいほど類似度が高い
      // 特徴量間の距離が閾値以下なら一致とみなす
      if (matches.get(i).distance < minDistance) {
        goodMatches++;
      }
    }

    return goodMatches;
  } finally {
    matches.delete();
  }
}