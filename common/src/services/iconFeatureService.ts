
import {
  compute as computeFeatures,
  compare as compareDescriptors,
} from "../utils/feature";
import { MatManager, toBase64, crop } from "../utils/mat";

declare const cv: any;

/**
 * アイコンの特徴量を計算し、比較するサービス
 * FeatureMatcher をラップして、アイコンに特化した機能を提供する
 */
export class IconFeatureService {

  private static _instance: IconFeatureService | null = null;

  private constructor() { }

  public static getInstance(): IconFeatureService {
    if (!this._instance) {
      this._instance = new IconFeatureService();
    }
    return this._instance;
  }

  private static preprocessCompute(mat: any): any {
    using matManager = new MatManager();

    // 短い辺に合わせて、中央を正方形に切り取る
    const size = Math.min(mat.cols, mat.rows);
    const x = Math.floor((mat.cols - size) / 2);
    const y = Math.floor((mat.rows - size) / 2);
    const croppedMat = matManager.add(crop(mat, x, y, size, size));
    // 100x100にリサイズ
    const resizedMat = new cv.Mat();
    cv.resize(croppedMat, resizedMat, new cv.Size(100, 100), 0, 0, size > 100 ? cv.INTER_AREA : cv.INTER_LINEAR);

    return resizedMat;
  }

  /**
   * アイコンの特徴量を計算する
   */
  public computeFeatures(mat: any): string | null {
    // アイコンの中心部分のみを特徴量計算に使う
    using matManager = new MatManager();
    const grayMat = matManager.add(new cv.Mat());
    if (mat.channels() === 4) {
      cv.cvtColor(mat, grayMat, cv.COLOR_RGBA2GRAY);
    } else if (mat.channels() === 3) {
      cv.cvtColor(mat, grayMat, cv.COLOR_RGB2GRAY);
    } else {
      mat.copyTo(grayMat);
    }
    // 前処理
    const preprocessedMat = matManager.add(IconFeatureService.preprocessCompute(grayMat));
    // マスク作成
    const mask = matManager.add(new cv.Mat.zeros(100, 100, cv.CV_8UC1));
    // 枠線、所持数部分以外を有効にする
    mask.roi(new cv.Rect(10, 10, 80, 70)).setTo(new cv.Scalar(255));

    try {
      using result = computeFeatures(preprocessedMat, mask);
      return result ? toBase64(result.descriptors) : null;
    } catch (e) {
      console.error(e);
      return null;
    }
  }

  /**
   * アイコンから色情報を計算する(中央部分の3x3 HSV配列)
   */
  public computeColorHash(mat: any): number[] {
    using matManager = new MatManager();
    const preprocessMat = matManager.add(IconFeatureService.preprocessCompute(mat));
    // 枠線、所持数部分を除外する
    const cropMat = matManager.add(crop(preprocessMat, 20, 20, 60, 60));
    const resizedMat = matManager.add(new cv.Mat());
    cv.resize(cropMat, resizedMat, new cv.Size(3, 3));

    const hsvMat = matManager.add(new cv.Mat());
    if (resizedMat.channels() === 4) {
      const rgbMat = matManager.add(new cv.Mat());
      cv.cvtColor(resizedMat, rgbMat, cv.COLOR_RGBA2RGB);
      cv.cvtColor(rgbMat, hsvMat, cv.COLOR_RGB2HSV);
    } else if (resizedMat.channels() === 3) {
      cv.cvtColor(resizedMat, hsvMat, cv.COLOR_RGB2HSV);
    } else {
      // 万が一チャンネル数が合わない場合はそのままコピー
      resizedMat.copyTo(hsvMat);
    }

    const data = Array.from(new Uint8Array(hsvMat.data));
    return data;
  }

  /**
   * 2つのアイコンの特徴量を比較し、一致するかどうかを返す
   */
  public compareFeatures(descriptors1: any, descriptors2: any): boolean {
    const score = compareDescriptors(descriptors1, descriptors2);
    return score >= 20;
  }

  /**
   * 色情報のハッシュを比較し、ある程度の誤差(遊び)を持たせて一致判定を行う
   * 互換性チェックは行わず、確実にデータが存在する前提
   */
  public compareColor(hash1: number[], hash2: number[], threshold: number = 30): boolean {
    if (hash1.length !== hash2.length) return false;

    const channels = hash1.length / 9; // 3x3 = 9ピクセル

    if (channels === 3) {
      // HSV空間での比較とみなす
      let diffH = 0;
      let diffS = 0;
      let diffV = 0;

      for (let i = 0; i < hash1.length; i += 3) {
        const h1 = hash1[i];
        const s1 = hash1[i + 1];
        const v1 = hash1[i + 2];
        const h2 = hash2[i];
        const s2 = hash2[i + 1];
        const v2 = hash2[i + 2];

        // Hueは0-179の円環なので最短距離を求める
        const dH = Math.abs(h1 - h2);
        diffH += Math.min(dH, 180 - dH);
        diffS += Math.abs(s1 - s2);
        diffV += Math.abs(v1 - v2);
      }

      const avgDiffH = diffH / 9;
      const avgDiffS = diffS / 9;
      const avgDiffV = diffV / 9;

      // H(色相)の差はよりクリティカルなため重みをつけ、V(明度)の違いにはある程度寛容にする
      // H:1.5, S:1.0, V:0.5 など重み付けした平均差分を計算
      const avgDiff = (avgDiffH * 1.5 + avgDiffS * 1.0 + avgDiffV * 0.5) / 3;
      return avgDiff <= threshold;
    } else {
      // 予期せぬチャンネル数の場合は単純な差分計算フォールバック
      let diff = 0;
      for (let i = 0; i < hash1.length; i += channels) {
        for (let j = 0; j < channels; j++) {
          diff += Math.abs(hash1[i + j] - hash2[i + j]);
        }
      }
      const avgDiff = diff / (9 * channels);
      return avgDiff <= threshold;
    }
  }
}