import { useState, useCallback, useEffect } from 'react'
import { ImageUploader } from '@common/components/ImageUploader'
import { AnalysisProgress } from '@common/components/AnalysisProgress'
import { InventoryExtractionService } from './services/inventoryExtractionService'
import { ItemMasterService } from '../../common/src/services/itemMasterService'
import { SummaryView } from './components/SummaryView'
import type { AnalyzedItem, AnalysisProgress as ProgressType } from '@common/types'

declare const cv: any;

function App() {
  const [images, setImages] = useState<File[]>([])
  const [results, setResults] = useState<AnalyzedItem[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState<ProgressType | null>(null)
  const [viewMode, setViewMode] = useState<'list' | 'summary'>('list')
  const [masterData, setMasterData] = useState<any[]>([])

  useEffect(() => {
    ItemMasterService.getInstanceAsync().then(service => {
      setMasterData(service.masterData);
    });
  }, []);

  const handleImagesSelected = useCallback((files: File[]) => {
    setImages((prev: File[]) => [...prev, ...files])
  }, [])

  const handleRemoveImage = useCallback((index: number) => {
    setImages((prev: File[]) => prev.filter((_, i) => i !== index))
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (images.length === 0) return
    setIsAnalyzing(true)
    setResults([])

    const inventoryExtractionService = await InventoryExtractionService.getInstanceAsync()

    try {

      const items = await inventoryExtractionService.extractAllAsync(images, (percent, message) => {
        setProgress({ step: 'ocr', percent, message })
      })

      setResults(items)

      setProgress({ step: 'done', percent: 100, message: '解析完了！' })
    } catch (e: any) {
      console.error(e)
      setProgress({ step: 'done', percent: 0, message: `エラーが発生しました: ${e.message}` })
    } finally {
      setIsAnalyzing(false)
      inventoryExtractionService.dispose()
    }
  }, [images])

  const handleExportCSV = useCallback(() => {
    const csvRows = ['アイテム名,所持数']
    results.forEach((item: AnalyzedItem) => {
      csvRows.push(`${item.name || '未登録'},${item.quantity}`)
    })
    const csvContent = '\uFEFF' + csvRows.join('\n') // BOM for Excel
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `inventory_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }, [results])

  const handleReset = useCallback(() => {
    setImages([])
    setResults([])
    setProgress(null)
  }, [])

  const handleUpdateResult = useCallback((index: number, field: keyof AnalyzedItem, value: any) => {
    setResults((prev) => {
      const updated = [...prev]
      updated[index] = { ...updated[index], [field]: value }
      return updated
    })
  }, [])

  const handleRemoveResult = useCallback((index: number) => {
    setResults((prev) => prev.filter((_, i) => i !== index))
  }, [])

  return (
    <div className="app">
      <header className="app-header" style={{ padding: '20px', backgroundColor: '#ffffff', color: '#000000', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ margin: 0 }}>ステラソラ アイテム所持数チェッカー（CNN版）</h1>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => setViewMode(prev => prev === 'list' ? 'summary' : 'list')}
            style={{
              padding: '8px 16px',
              backgroundColor: viewMode === 'list' ? '#f39c12' : '#4a69bd',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            {viewMode === 'list' ? 'サマリー表示に切替' : 'リスト表示に戻る'}
          </button>
        </div>
      </header>

      <main className="app-main" style={{ padding: '20px' }}>
        <section className="upload-section">
          <ImageUploader
            images={images}
            onImagesSelected={handleImagesSelected}
            onRemoveImage={handleRemoveImage}
          />
          <div className="actions" style={{ marginTop: '15px' }}>
            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={images.length === 0 || isAnalyzing}
              style={{
                padding: '10px 24px',
                backgroundColor: '#264278',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              {isAnalyzing ? '解析中...' : `解析開始（${images.length}枚）`}
            </button>
          </div>
        </section>

        {progress && <AnalysisProgress progress={progress} />}

        {viewMode === 'summary' && (
          <section className="summary-section" style={{ marginTop: '30px' }}>
            {results.length > 0 && (
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginBottom: '15px' }}>
                <button
                  onClick={handleExportCSV}
                  style={{ padding: '8px 16px', backgroundColor: '#4a69bd', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
                >
                  CSVエクスポート
                </button>
                <button
                  onClick={handleReset}
                  style={{ padding: '8px 16px', backgroundColor: '#e55039', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
                >
                  リセット
                </button>
              </div>
            )}
            <SummaryView results={results} masterData={masterData} />
          </section>
        )}

        {viewMode === 'list' && results.length > 0 && (
          <section className="results-user" style={{ marginTop: '30px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
              <h2 style={{ margin: 0 }}>在庫抽出結果 ({results.length} 件)</h2>
              <div style={{ display: 'flex', gap: '10px' }}>
                <button
                  onClick={handleExportCSV}
                  style={{ padding: '8px 16px', backgroundColor: '#4a69bd', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
                >
                  CSVエクスポート
                </button>
                <button
                  onClick={handleReset}
                  style={{ padding: '8px 16px', backgroundColor: '#e55039', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
                >
                  リセット
                </button>
              </div>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '20px' }}>
              {results.map((item: AnalyzedItem, idx: number) => (
                <div key={idx} style={{
                  border: '1px solid var(--border)',
                  padding: '8px',
                  borderRadius: '4px',
                  backgroundColor: 'var(--bg-panel)',
                  width: '120px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                  position: 'relative'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginBottom: '5px' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)', fontWeight: 'bold' }}>#{idx + 1}</span>
                    <button
                      onClick={() => handleRemoveResult(idx)}
                      style={{
                        color: 'var(--danger)',
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
                  <img src={item.iconDataUrl} alt="icon" style={{ width: '48px', height: '48px', borderRadius: '4px', marginBottom: '8px', border: '1px solid var(--border)' }} />
                  <input
                    type="text"
                    value={item.name}
                    onChange={(e) => handleUpdateResult(idx, 'name', e.target.value)}
                    style={{
                      width: '100%',
                      padding: '2px 4px',
                      fontSize: '0.8rem',
                      textAlign: 'center',
                      boxSizing: 'border-box',
                      border: '1px solid var(--border)',
                      borderRadius: '3px',
                      color: 'var(--text-main)',
                      backgroundColor: 'transparent'
                    }}
                  />
                  <input
                    type="text"
                    value={item.quantity ?? ''}
                    onChange={(e: any) => handleUpdateResult(idx, 'quantity', e.target.value)}
                    style={{
                      width: '60%',
                      marginTop: '5px',
                      padding: '2px 4px',
                      fontSize: '1rem',
                      fontWeight: 'bold',
                      color: 'var(--primary)',
                      textAlign: 'center',
                      border: '1px solid var(--border)',
                      borderRadius: '3px',
                      backgroundColor: 'transparent'
                    }}
                  />
                </div>
              ))}
            </div>
          </section>
        )}
      </main>

    </div>
  )
}

export default App
