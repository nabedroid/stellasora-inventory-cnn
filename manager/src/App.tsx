import React, { useState, useCallback } from 'react'
import { ImageUploader } from '@common/components/ImageUploader'
import { AnalysisProgress } from '@common/components/AnalysisProgress'
import { ItemDataExtractService } from './services/itemDataExtractService'
import { ItemMasterData, ItemMasterDataJson } from '@common/services/itemMasterService';
import { IconFeatureService } from '@common/services/iconFeatureService'
import type { AnalysisProgress as ProgressType } from '@common/types'

declare const cv: any;

function App() {
  const [images, setImages] = useState<File[]>([])
  const [entries, setEntries] = useState<ItemMasterData[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState<ProgressType | null>(null)

  // Drag and Drop
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null)
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null)

  // 辞書 JSON のインポート
  const handleImportJSON = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      try {
        const data = JSON.parse(event.target?.result as string) as ItemMasterDataJson[]
        setEntries(data.map((entry) => ItemMasterData.fromJson(entry)))
      } catch (err) {
        alert('JSON のパースに失敗しました')
      }
    }
    reader.readAsText(file)
  }, [])

  const handleImagesSelected = useCallback((files: File[]) => {
    setImages((prev: File[]) => [...prev, ...files])
  }, [])

  const handleRemoveImage = useCallback((index: number) => {
    setImages((prev: File[]) => prev.filter((_, i) => i !== index))
  }, [])

  // 解析処理 (1枚から1アイテム)
  const handleAnalyze = useCallback(async () => {
    const itemDataExtractService = await ItemDataExtractService.getInstanceAsync()
    if (images.length === 0) return
    setIsAnalyzing(true)

    try {
      const newEntries: ItemMasterData[] = []

      // 画像ごとに処理
      for (let i = 0; i < images.length; i++) {
        const file = images[i]
        setProgress({
          step: 'loading',
          percent: (i / images.length) * 100,
          message: `画像読み込み中: ${file.name}`
        })

        // 画像の読み込み
        const img = await loadImage(file)
        const src = cv.imread(img)

        setProgress({
          step: 'ocr',
          percent: (i / images.length) * 100 + 20,
          message: `アイテム情報抽出中...`
        })

        // アイテム情報の抽出
        const result = await itemDataExtractService.extractAsync(src)

        // 抽出結果の追加
        newEntries.push(new ItemMasterData({
          id: entries.length + newEntries.length + 1,
          name: result.name,
          iconDataUrl: result.iconDataUrl,
          features: result.features,
          colorHash: result.colorHash
        }))

        src.delete()
      }

      setEntries((prev) => [...prev, ...newEntries])
      setImages([]) // 解析後はクリア
      setProgress({ step: 'done', percent: 100, message: '完了' })
    } catch (e: any) {
      console.error(e)
      setProgress({ step: 'done', percent: 0, message: `エラー: ${e.message}` })
    } finally {
      setIsAnalyzing(false)
      itemDataExtractService?.dispose()
    }
  }, [images, entries.length])

  const handleUpdateEntry = (index: number, field: keyof ItemMasterData, value: any) => {
    setEntries((prev) => {
      const updated = [...prev]
      updated[index] = { ...updated[index], [field]: value }
      return updated
    })
  }

  const handleDragStart = (index: number) => {
    setDraggedIndex(index)
  }

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault()
    setDragOverIndex(index)
  }

  const handleDrop = (index: number) => {
    if (draggedIndex === null || draggedIndex === index) {
      setDraggedIndex(null)
      setDragOverIndex(null)
      return
    }
    setEntries((prev: ItemMasterData[]) => {
      const updated = [...prev]
      const temp = updated[draggedIndex]
      updated.splice(draggedIndex, 1)
      updated.splice(index, 0, temp)
      return updated
    })
    setDraggedIndex(null)
    setDragOverIndex(null)
  }

  const handleDragEnd = () => {
    setDraggedIndex(null)
    setDragOverIndex(null)
  }

  const handleRemoveEntry = (index: number) => {
    setEntries((prev: ItemMasterData[]) => prev.filter((_, i) => i !== index))
  }

  const handleRecalculateFeatures = useCallback(async () => {
    const iconFeatureService = IconFeatureService.getInstance();
    if (entries.length === 0) return;
    setIsAnalyzing(true);
    setProgress({ step: 'loading', percent: 0, message: '全アイテムの特徴量と色情報を再計算中...' });

    try {
      const updatedEntries = [...entries];
      for (let i = 0; i < updatedEntries.length; i++) {
        const entry = updatedEntries[i];

        setProgress({
          step: 'ocr',
          percent: (i / updatedEntries.length) * 100,
          message: `再計算中: ${entry.name || `アイテム #${i + 1}`}`
        });

        const img = await imageFromDataUrl(entry.iconDataUrl);
        const mat = cv.imread(img);
        const features = iconFeatureService.computeFeatures(mat);
        const colorHash = iconFeatureService.computeColorHash(mat);
        mat.delete();

        entry.features = features;
        entry.colorHash = colorHash;
      }

      setEntries(updatedEntries);
      setProgress({ step: 'done', percent: 100, message: '再計算完了！' });
    } catch (e: any) {
      console.error(e);
      setProgress({ step: 'done', percent: 0, message: `エラー: ${e.message}` });
    } finally {
      setIsAnalyzing(false);
    }
  }, [entries]);

  const handleExportJSON = useCallback(() => {
    const json = JSON.stringify(entries, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'master-dictionary.json'
    a.click()
    URL.revokeObjectURL(url)
  }, [entries])

  return (
    <div className="app" style={{ fontSize: '0.9rem' }}>
      <header style={{
        padding: '8px 16px',
        backgroundColor: '#264278',
        color: 'white',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        position: 'sticky',
        top: 0,
        zIndex: 100
      }}>
        <h1 style={{ margin: 0, fontSize: '1.1rem' }}>ステラソラ アイテムマスター作成ツール</h1>
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ fontSize: '0.8rem', opacity: 0.8 }}>Import:</span>
            <input type="file" accept=".json" onChange={handleImportJSON} style={{ fontSize: '0.8rem', width: '180px' }} />
          </div>
          <button
            onClick={handleRecalculateFeatures}
            disabled={entries.length === 0 || isAnalyzing}
            style={{
              padding: '4px 12px',
              backgroundColor: '#e1b12c', // yellow/gold for regenerate
              color: 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: entries.length === 0 || isAnalyzing ? 'not-allowed' : 'pointer',
              fontWeight: 'bold',
              fontSize: '0.8rem',
              opacity: entries.length === 0 || isAnalyzing ? 0.5 : 1
            }}
          >
            特徴量再計算
          </button>
          <button
            onClick={handleExportJSON}
            style={{
              padding: '4px 12px',
              backgroundColor: '#4a69bd',
              color: 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: 'pointer',
              fontWeight: 'bold',
              fontSize: '0.8rem'
            }}
          >
            Export JSON ({entries.length})
          </button>
        </div>
      </header>

      <main style={{ padding: '12px' }}>
        {/* 操作エリア */}
        <section style={{
          display: 'flex',
          gap: '15px',
          marginBottom: '15px',
          padding: '10px',
          backgroundColor: '#f8f9fa',
          borderRadius: '4px',
          alignItems: 'flex-start',
          border: '1px solid #dee2e6'
        }}>
          <div style={{ flex: 1 }}>
            <ImageUploader
              images={images}
              onImagesSelected={handleImagesSelected}
              onRemoveImage={handleRemoveImage}
            />
          </div>
          <div style={{ width: '220px' }}>
            <button
              onClick={handleAnalyze}
              disabled={images.length === 0 || isAnalyzing}
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#e55039',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold',
                marginBottom: '5px'
              }}
            >
              {isAnalyzing ? '分析中...' : '画像から抽出'}
            </button>
            {progress && <AnalysisProgress progress={progress} />}
          </div>
        </section>

        {/* 編集グリッド */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
          {entries.map((entry, idx) => (
            <div
              key={idx}
              draggable
              onDragStart={() => handleDragStart(idx)}
              onDragOver={(e) => handleDragOver(e, idx)}
              onDrop={() => handleDrop(idx)}
              onDragEnd={handleDragEnd}
              style={{
                border: draggedIndex === idx ? '2px dashed #4a69bd' : (dragOverIndex === idx ? '2px solid #e55039' : '1px solid #dee2e6'),
                opacity: draggedIndex === idx ? 0.5 : 1,
                padding: '8px',
                borderRadius: '4px',
                backgroundColor: dragOverIndex === idx ? '#f8d7da' : 'white',
                width: '120px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                cursor: 'grab',
                transition: 'all 0.2s ease'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginBottom: '5px' }}>
                <span style={{ fontSize: '0.8rem', color: '#666', fontWeight: 'bold' }}>#{idx + 1}</span>
                <button
                  onClick={() => handleRemoveEntry(idx)}
                  style={{
                    color: '#e55039',
                    border: 'none',
                    background: 'none',
                    cursor: 'pointer',
                    fontSize: '1rem',
                    lineHeight: 1,
                    padding: 0
                  }}
                  title="削除"
                >
                  ×
                </button>
              </div>
              <img src={entry.iconDataUrl} alt="" style={{ width: '48px', height: '48px', display: 'block', border: '1px solid #ccc', margin: '5px 0' }} />
              <input
                type="text"
                value={entry.name}
                onChange={(e) => handleUpdateEntry(idx, 'name', e.target.value)}
                style={{ width: '100%', padding: '2px 4px', fontSize: '0.8rem', textAlign: 'center', boxSizing: 'border-box' }}
              />
            </div>
          ))}
        </div>
      </main>
    </div>
  )
}

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = reject
      img.src = e.target?.result as string
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

function imageFromDataUrl(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataUrl;
  });
}

export default App
