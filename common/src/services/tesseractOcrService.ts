import { createWorker } from 'tesseract.js';
import { toImageData } from '../utils/mat';

/**
 * Tesseract.js を用いた OCR サービス実装
 */
export class TesseractOcrService {
  private worker: any = null;

  private constructor(worker: any) {
    this.worker = worker;
  }

  public static async getInstanceAsync({
    lang = 'eng',
    whitelist = null,
    psm = null,
  }: {
    lang?: string;
    whitelist?: string | null;
    psm?: number | null;
  } = {}): Promise<TesseractOcrService> {
    const worker = await createWorker(lang);
    const params: Record<string, any> = {};

    if (whitelist !== null) params.tessedit_char_whitelist = whitelist;
    if (psm !== null) params.tessedit_pageseg_mode = psm;

    if (Object.keys(params).length > 0) {
      await worker.setParameters(params);
    }

    return new TesseractOcrService(worker);
  }

  public async recognizeAsync(
    mat: any,
    rectangle: {
      x: number, y: number, width: number, height: number
    } | null = null,
  ): Promise<string> {

    const imageData = toImageData(mat);
    const canvas = document.createElement('canvas');
    canvas.width = mat.cols;
    canvas.height = mat.rows;
    canvas.getContext('2d')?.putImageData(imageData, 0, 0);

    let result: string;
    if (rectangle) {
      const { data: { text } } = await this.worker.recognize(canvas, {
        rectangle: { top: rectangle.y, left: rectangle.x, width: rectangle.width, height: rectangle.height },
      });
      result = text;
    } else {
      const { data: { text } } = await this.worker.recognize(canvas);
      result = text;
    }

    return result.trim();
  }

  public async dispose(): Promise<void> {
    if (this.worker) {
      await this.worker.terminate();
      this.worker = null;
    }
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.dispose();
  }
}
