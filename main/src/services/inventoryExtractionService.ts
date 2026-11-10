import type { AnalyzedItem } from '@common/types';
import { IconExtractService } from '@common/services/iconExtractService';
import { crop, MatManager, toDataUrl, invert, fromFile } from '@common/utils/mat';
import { TesseractOcrService } from '@common/services/tesseractOcrService';
import { IconCnnService } from './iconCnnService';

declare const cv: any;

export class InventoryExtractionService {

  private readonly tesseractOcrService: TesseractOcrService;
  private readonly iconCnnService: IconCnnService;

  private constructor(tesseractOcrService: TesseractOcrService, iconCnnService: IconCnnService) {
    this.tesseractOcrService = tesseractOcrService;
    this.iconCnnService = iconCnnService;
  }

  public static async getInstanceAsync(): Promise<InventoryExtractionService> {
    const tesseractOcrService = await TesseractOcrService.getInstanceAsync({ lang: 'eng', whitelist: '0123456789x' });
    const iconCnnService = await IconCnnService.getInstanceAsync();
    const instance = new InventoryExtractionService(tesseractOcrService, iconCnnService);
    return instance;
  }

  public dispose(): void {
    this.tesseractOcrService?.dispose();
  }

  public [Symbol.dispose](): void {
    this.dispose();
  }

  /**
   * 画像ファイル配列を読み込み、解析から重複排除、ソートまですべて行う
   */
  public async extractAllAsync(
    images: File[],
    onProgress: (percent: number, message: string) => void
  ): Promise<AnalyzedItem[]> {
    const allExtractedItems: AnalyzedItem[] = [];

    for (let i = 0; i < images.length; i++) {
      const file = images[i];
      const imgIndex = i + 1;

      onProgress(
        (i / images.length) * 100,
        `画像読み込み中 (${imgIndex}/${images.length}): ${file.name}`
      );

      const src = await fromFile(file);

      const items = await this.analyzeAsync(src, i, (percent, message) => {
        onProgress(
          (i / images.length) * 100 + (percent * (1 / images.length)),
          `画面 ${imgIndex}/${images.length}: ${message}`
        );
      });

      allExtractedItems.push(...items);
      src.delete();
    }

    onProgress(95, '重複アイテムを統合中...');
    const uniqueItems = this.deduplicate(allExtractedItems);

    // マスターデータのID順にソート (IDが不明なものは末尾)
    uniqueItems.sort((a, b) => {
      const idA = a.id ?? Number.MAX_SAFE_INTEGER;
      const idB = b.id ?? Number.MAX_SAFE_INTEGER;
      return idA - idB;
    });

    onProgress(100, '解析完了！');
    return uniqueItems;
  }

  /**
   * 画像からアイコンを検出し、所持数OCRとCNN判定からアイテム情報を抽出する
   * @param src 解析対象の cv.Mat
   * @param sourceImageIndex 画像のインデックス番号
   * @param onProgress プログレス通知コールバック
   * @returns 解析抽出されたアイテム結果のリスト
   */
  public async analyzeAsync(
    src: any,
    sourceImageIndex: number,
    onProgress?: (percent: number, message: string) => void
  ): Promise<AnalyzedItem[]> {
    using matManager = new MatManager();
    // アイコン領域を検出
    const iconRegions = await IconExtractService.extractAsync(
      src,
      { minIconWidth: 50, maxIconWidth: 150 }
    );

    // まず切り出しと所持数OCRまでを済ませ、判定対象を集める
    const pending: { iconMat: any; iconDataUrl: string; quantity: number }[] = [];
    for (let j = 0; j < iconRegions.length; j++) {
      const region = iconRegions[j];

      if (onProgress) {
        onProgress((j / iconRegions.length) * 100, `アイテム ${j + 1}/${iconRegions.length}`);
      }

      // アイコン領域を切り出し
      const iconMat = matManager.add(crop(src, region.x, region.y, region.width, region.height));
      // DataUrlに変換
      const iconDataUrl = toDataUrl(iconMat);
      // OCR して所持数を抽出
      const quantity = await this.ocrNumberAsync(iconMat);
      if (quantity === null) {
        // 所持数が読み取れない場合はスキップ (現行仕様を踏襲)
        continue;
      }

      pending.push({ iconMat, iconDataUrl, quantity });
    }

    // アイコンの判定は CNN でバッチ一括に行う
    const candidatesList = await this.iconCnnService.classifyAsync(
      pending.map(item => item.iconMat));

    return pending.map((item, index) => {
      const candidates = candidatesList[index];
      const best = candidates[0];
      return {
        id: best?.id,
        iconDataUrl: item.iconDataUrl,
        name: best?.name ?? '',
        quantity: item.quantity,
        sourceImageIndex,
        confidence: best?.prob,
        candidates,
      };
    });
  }

  /**
   * アイコン画像から所持数を抽出する
   * @param iconMat アイコン画像
   * @returns 所持数
   */
  private async ocrNumberAsync(iconMat: any): Promise<number | null> {
    using matManager = new MatManager();
    // アイコン下部（数字が描かれている領域）を切り出す
    const iconLowerMat = matManager.add(crop(iconMat, 0, iconMat.rows * 0.8, iconMat.cols, iconMat.rows * 0.2));
    // 数字の領域を二値化
    const grayMat = matManager.add(new cv.Mat());
    cv.cvtColor(iconLowerMat, grayMat, cv.COLOR_RGBA2GRAY);
    const binaryMat = matManager.add(new cv.Mat());
    cv.threshold(grayMat, binaryMat, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU);
    // 輪郭検出
    const contoursMat = matManager.add(new cv.MatVector());
    const hierarchyMat = matManager.add(new cv.Mat());
    cv.findContours(binaryMat, contoursMat, hierarchyMat, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    // 検出された輪郭からノイズを除去する
    const rects: any[] = [];
    for (let i = 0; i < contoursMat.size(); i++) {
      const cnt = matManager.add(contoursMat.get(i));
      const rect = cv.boundingRect(cnt);
      // 明らかに小さい、細い輪郭を除去
      if (rect.width > 2 && iconLowerMat.rows * 0.3 <= rect.height && rect.height <= iconLowerMat.rows * 0.8) {
        rects.push(rect);
      }
    }
    // 1桁目、2桁目の数字の底辺のy座標を算出
    const rightLimit = iconLowerMat.cols * 0.8;
    const numberBottomY = rects
      .filter(r => r.x > rightLimit)
      .reduce((max, r) => Math.max(max, r.y + r.height), 0);
    // 共通の底辺を持つ輪郭（＝数字の輪郭の可能性が高い）を抽出
    const yThreshold = iconLowerMat.rows * 0.1;
    const targetRects = rects
      .filter(r => Math.abs(r.y + r.height - numberBottomY) <= yThreshold);
    // 数字らしき輪郭がなければ null を返す
    if (targetRects.length === 0) {
      return null;
    }
    // 全ての数字を内包する矩形を算出
    const minX = Math.min(...targetRects.map(r => r.x));
    const minY = Math.min(...targetRects.map(r => r.y));
    const maxX = Math.max(...targetRects.map(r => r.x + r.width));
    const maxY = Math.max(...targetRects.map(r => r.y + r.height));
    // 少し余裕をもって切り出す
    const padding = 2;
    const x = minX - padding < 0 ? 0 : minX - padding;
    const y = minY - padding < 0 ? 0 : minY - padding;
    const width = x + maxX - minX + padding * 2 > binaryMat.cols ? binaryMat.cols - minX : maxX - minX + padding * 2;
    const height = y + maxY - minY + padding * 2 > binaryMat.rows ? binaryMat.rows - minY : maxY - minY + padding * 2;
    const numberMat = matManager.add(crop(binaryMat, x, y, width, height));
    // 反転
    const invertedMat = matManager.add(invert(numberMat));
    const recognizedText = await this.tesseractOcrService.recognizeAsync(invertedMat);
    // 数字のみを抽出
    const digits = recognizedText.replace(/\D/g, '');
    if (digits === null || digits === '') {
      return null;
    }

    return parseInt(digits);
  }

  /**
   * 複数枚の解析結果を結合し、重複を排除する
   * CNN は必ず名前を返すので、名前が分かるものは名前で統合する。
   * (旧方式にあった特徴量ベースの重複判定は CNN 化に伴い廃止)
   */
  private deduplicate(items: AnalyzedItem[]): AnalyzedItem[] {
    const uniqueItems: AnalyzedItem[] = [];
    const seenNames = new Set<string>();

    for (const item of items) {
      if (item.name) {
        if (seenNames.has(item.name)) {
          continue;
        }
        seenNames.add(item.name);
      }
      uniqueItems.push(item);
    }

    return uniqueItems;
  }
}
