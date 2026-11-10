import type { IconRegion, DictionaryEntry } from '@common/types';
import { TesseractOcrService } from '@common/services/tesseractOcrService';
import { IconFeatureService } from '@common/services/iconFeatureService';
import { IconExtractService } from '@common/services/iconExtractService';
import { MatManager, crop, toDataUrl } from '@common/utils/mat';

declare const cv: any;

/**
 * 物資画面のスクショからアイテムの情報を抽出する
 */
export class ItemDataExtractService {

  private readonly iconFeatureService: IconFeatureService;
  private readonly tesseractOcrService: TesseractOcrService;

  private constructor(tesseractOcrService: TesseractOcrService) {
    this.iconFeatureService = IconFeatureService.getInstance();
    this.tesseractOcrService = tesseractOcrService;
  }

  public static async getInstanceAsync(): Promise<ItemDataExtractService> {
    const tesseractOcrService = await TesseractOcrService.getInstanceAsync({ lang: 'jpn' });
    return new ItemDataExtractService(tesseractOcrService);
  }

  public dispose(): void {
    this.tesseractOcrService.dispose();
  }

  /**
   * 画像からアイテム情報を抽出する
   * @param src OpenCV Mat
   */
  public async extractAsync(
    src: any,
  ): Promise<DictionaryEntry> {

    // アイコン領域を大まかに切り出す
    const iconAreaRect = { x: src.cols * 0.68, y: src.rows * 0.11, w: src.cols * 0.1, h: src.rows * 0.18 };
    const iconAreaMat = crop(src, iconAreaRect.x, iconAreaRect.y, iconAreaRect.w, iconAreaRect.h);
    // アイコン領域からアイコンを検出する
    const iconRegions = await IconExtractService.extractAsync(iconAreaMat, { threshold: 200 });
    if (iconRegions.length === 0) {
      throw new Error('アイコンが見つかりませんでした。');
    }
    // 複数見つかった場合は一番大きいアイコンを選択
    const iconRegion = iconRegions.sort((a: IconRegion, b: IconRegion) => b.width - a.width)[0];
    // アイコンを切り出す
    const iconMat = crop(iconAreaMat, iconRegion.x, iconRegion.y, iconRegion.width, iconRegion.height);
    const iconDataUrl = toDataUrl(iconMat);

    // 特徴量と色の計算
    const features = this.iconFeatureService.computeFeatures(iconMat);
    const colorHash = this.iconFeatureService.computeColorHash(iconMat);

    // 名前の OCR
    // アイテムの名前欄のみを切り出し、OCRを実行
    const nameAreaRect = { x: src.cols * 0.785, y: src.rows * 0.12, w: src.cols * 0.19, h: src.rows * 0.048 };
    const nameAreaMat = crop(src, nameAreaRect.x, nameAreaRect.y, nameAreaRect.w, nameAreaRect.h);
    const ocrText = await this.tesseractOcrService.recognizeAsync(nameAreaMat);

    // OCR 結果を補正する
    const correctName = ItemDataExtractService.toCorrectName(ocrText);

    // 後始末
    iconAreaMat.delete();
    iconMat.delete();
    nameAreaMat.delete();

    return {
      id: null,
      iconDataUrl: iconDataUrl,
      name: correctName,
      features: features,
      colorHash: colorHash,
    };
  }

  private static toCorrectName(name: string): string {
    // 空白を除去
    const trimName = name.replace(/\s/g, '');
    // 共通の誤認識の補正
    const correctName = trimName.replace(/糧金/g, '煌金')
      .replace(/関銀/g, '閃銀')
      .replace(/翠糧/g, '翠煌')
      .replace(/残津/g, '残滓')
      .replace(/残洲/g, '残滓')
      .replace(/\(大\)/g, '（大）')
      .replace(/\(小\)/g, '（小）')
      .replace(/かき水機/g, 'かき氷機')
      .replace(/永皆/g, '永昏');
    // 個別の補正
    const errorMap: Record<string, string> = {
      'テクニックの銘': 'テクニックの駒',
      '太5ロスレコセレクション': '★5ロスレコセレクション',
      '大4ロスレコセレクション有': '★4ロスレコセレクションⅡ',
      '太4ロスレコセレクション則|': '★4ロスレコセレクションⅢ',
      '上厳選ロスレコ昇格箱': '厳選ロスレコ昇格箱',
      'ドボの残党': 'ドボの残滓',
      '太4ロスレコセレクション有1': '★4ロスレコセレクションⅡ',
      '太4ロスレコセレクション山|': '★4ロスレコセレクションⅢ',
    };
    // 個別の補正を適用する
    return errorMap[correctName] || correctName;
  }
}
