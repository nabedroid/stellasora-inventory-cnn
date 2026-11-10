import type { AnalysisProgress as ProgressType } from '../types'

interface AnalysisProgressProps {
  progress: ProgressType
}

export function AnalysisProgress({ progress }: AnalysisProgressProps) {
  return (
    <div className="analysis-progress">
      <div className="progress-info">
        <span className="step-label">{progress.message}</span>
        <span className="percent-label">{Math.round(progress.percent)}%</span>
      </div>
      <div className="progress-bar-container">
        <div
          className="progress-bar-fill"
          style={{ width: `${progress.percent}%` }}
        ></div>
      </div>
    </div>
  )
}
