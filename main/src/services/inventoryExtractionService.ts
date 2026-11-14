import type { AnalyzedItem } from '@common/types';
import { IconExtractService } from '@common/services/iconExtractService';
import { crop, MatManager, toDataUrl, fromFile } from '@common/utils/mat';
import { GlyphCnnService } from './glyphCnnService';
import { IconCnnService } from './iconCnnService';

export class InventoryExtractionService {

  private readonly glyphCnnService: GlyphCnnService;
  private readonly iconCnnService: IconCnnService;

  private constructor(glyphCnnService: GlyphCnnService, iconCnnService: IconCnnService) {
    this.glyphCnnService = glyphCnnService;
    this.iconCnnService = iconCnnService;
  }

  public static async getInstanceAsync(): Promise<InventoryExtractionService> {
    const glyphCnnService = await GlyphCnnService.getInstanceAsync();
    const iconCnnService = await IconCnnService.getInstanceAsync();
    const instance = new InventoryExtractionService(glyphCnnService, iconCnnService);
    return instance;
  }

  public dispose(): void {
    // CNN セッションはシングルトンで保持するため、ここで破棄するものはない
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
   * 画像からアイコンを検出し、所持数のグリフ判定とアイコンのCNN判定から
   * アイテム情報を抽出する
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

    // アイコン領域を切り出す
    const pending: { iconMat: any; iconDataUrl: string }[] = [];
    for (let j = 0; j < iconRegions.length; j++) {
      const region = iconRegions[j];

      if (onProgress) {
        onProgress((j / iconRegions.length) * 50, `アイテム ${j + 1}/${iconRegions.length}`);
      }

      const iconMat = matManager.add(crop(src, region.x, region.y, region.width, region.height));
      const iconDataUrl = toDataUrl(iconMat);
      pending.push({ iconMat, iconDataUrl });
    }

    // 所持数はグリフ分類でバッチ一括に読む
    if (onProgress) {
      onProgress(50, '所持数を読み取り中...');
    }
    const quantities = await this.glyphCnnService.readQuantitiesAsync(
      pending.map(item => item.iconMat));

    // 所持数が読み取れたものだけを判定対象にする (現行仕様を踏襲)
    const readable = pending
      .map((item, index) => ({ ...item, quantity: quantities[index] }))
      .filter(item => item.quantity !== null);

    // アイコンの判定は CNN でバッチ一括に行う
    if (onProgress) {
      onProgress(75, 'アイテムを判定中...');
    }
    const candidatesList = await this.iconCnnService.classifyAsync(
      readable.map(item => item.iconMat));

    return readable.map((item, index) => {
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
