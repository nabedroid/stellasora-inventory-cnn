import { fromBase64 } from '../utils/mat';

declare const cv: any;

const MASTER_DICT_URL = `${import.meta.env.BASE_URL}master-dictionary.json`;

export interface ItemMasterDataJson {
  id: number;
  features: string;
  colorHash: number[];
  name: string;
  iconDataUrl: string;
}

export class ItemMasterData {
  /** インデックス (Dicter用) */
  public id: number;
  /** 特徴量データ (Base64化した記述子 Mat) */
  public features: string;
  /** 色情報 (3x3グリッドRGB) */
  public colorHash: number[];
  /** アイテム名 */
  public name: string;
  /** アイコン画像（Base64 Data URL） */
  public iconDataUrl: string;
  /** 特徴量データ (Base64化した記述子 Mat) */
  private _featuresMat: any | null;

  constructor(param: {
    id: number,
    features: string,
    colorHash: number[],
    name: string,
    iconDataUrl: string
  }) {
    this.id = param.id;
    this.features = param.features;
    this.colorHash = param.colorHash;
    this.name = param.name;
    this.iconDataUrl = param.iconDataUrl;
    this._featuresMat = null;
  }

  public featuresMat(): any {
    if (this._featuresMat === null) {
      this._featuresMat = fromBase64(this.features);
    }
    return this._featuresMat;
  }

  public static fromJson(json: ItemMasterDataJson): ItemMasterData {
    return new ItemMasterData(json);
  }

  public toJSON(): ItemMasterDataJson {
    return {
      id: this.id,
      features: this.features,
      colorHash: this.colorHash,
      name: this.name,
      iconDataUrl: this.iconDataUrl
    }
  }

  public dispose(): void {
    if (this._featuresMat !== null) {
      this._featuresMat.delete();
      this._featuresMat = null;
    }
  }
}

/**
 * マスター辞書データを管理するサービス
 */
export class ItemMasterService {
  private static _instance: ItemMasterService | null = null;
  private _master: ItemMasterData[] = [];
  private constructor() {
  }

  /**
   * マスターデータの全件を取得する (サマリービューアイコン表示などのため)
   */
  public get masterData(): ItemMasterData[] {
    return this._master;
  }

  /**
   * シングルトンインスタンスを取得し、未初期化ならマスターデータを読み込む
   */
  public static async getInstanceAsync(): Promise<ItemMasterService> {
    if (!this._instance) {
      this._instance = new ItemMasterService();
      await this._instance.loadMasterDataAsync();
    }
    return this._instance;
  }

  public dispose(): void {
    // メモリ解放のため、キャッシュしている特徴量データを破棄する
    // (instance は破棄しない)
    this._master.forEach(entry => entry.dispose());
    this._master = [];
  }

  /**
   * マスターデータの非同期ロード
   */
  private async loadMasterDataAsync(): Promise<void> {
    try {
      const response = await fetch(MASTER_DICT_URL);
      if (response.ok) {
        // 既存のデータを破棄
        this.dispose();
        const items = await response.json();
        items.forEach((item: ItemMasterDataJson) => {
          this._master.push(ItemMasterData.fromJson(item));
        });
      }
    } catch (e) {
      console.warn('マスターデータの読み込みに失敗しました。', e);
    }
  }

}
