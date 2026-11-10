/**
 * アイテム解析結果
 */
export interface AnalyzedItem {
  /** マスターデータのID（ソート用） */
  id?: number;
  /** アイコン画像（Base64 Data URL） */
  iconDataUrl: string;
  /** アイテム名（CNNの判定結果、未知の場合は空） */
  name: string;
  /** 所持数 */
  quantity: number | null;
  /** 元画像のインデックス */
  sourceImageIndex: number;
  /** CNN の最上位候補の確率 (0-1) */
  confidence?: number;
  /** CNN の候補上位3件 */
  candidates?: CnnCandidate[];
}

/**
 * CNN の判定候補1件
 */
export interface CnnCandidate {
  /** マスターデータのID */
  id?: number;
  /** アイテム名 */
  name: string;
  /** 確率 (0-1) */
  prob: number;
}

/**
 * グリッド検出結果のアイコン領域
 */
export interface IconRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * OCR結果
 */
export interface OcrResult {
  /** 認識テキスト（"×287" 等） */
  text: string;
  /** 信頼度 (0-1) */
  confidence: number;
}

/**
 * 解析進捗
 */
export interface AnalysisProgress {
  /** 現在のステップ */
  step: 'loading' | 'grid-detection' | 'icon-extraction' | 'ocr' | 'identification' | 'dedup' | 'done';
  /** 進捗率 (0-100) */
  percent: number;
  /** メッセージ */
  message: string;
}

