import { X, Sparkles } from 'lucide-react'

export interface ZoneDetail {
  zone_id: string
  severity: 'low' | 'moderate' | 'high' | 'extreme'
  mean_relative_heat: number
  top_contributors: string[]
  top_recommendations: string[]
  gemini_summary: string
}

const SEVERITY_STYLES: Record<ZoneDetail['severity'], string> = {
  extreme: 'bg-red-700 text-white',
  high:    'bg-orange-500 text-white',
  moderate:'bg-yellow-400 text-gray-900',
  low:     'bg-green-600 text-white',
}

interface DetailPanelProps {
  zone: ZoneDetail | null
  loading: boolean
  error: string | null
  onClose: () => void
}

export default function DetailPanel({ zone, loading, error, onClose }: DetailPanelProps) {
  return (
    <div className="absolute top-4 right-4 w-80 bg-gray-900 border border-gray-700 rounded-xl shadow-2xl text-gray-100 overflow-y-auto max-h-[calc(100vh-2rem)]">
      <div className="flex items-center justify-between p-4 border-b border-gray-700">
        <h2 className="text-sm font-semibold tracking-wide uppercase text-gray-400">Zone Detail</h2>
        <button onClick={onClose} className="text-gray-500 hover:text-gray-200 transition-colors">
          <X size={16} />
        </button>
      </div>

      <div className="p-4 space-y-4">
        {loading && (
          <>
            <div className="h-5 w-3/4 bg-gray-700 rounded animate-pulse" />
            <div className="h-16 bg-gray-800 rounded-lg animate-pulse" />
            <div className="space-y-2">
              <div className="h-3 w-1/3 bg-gray-700 rounded animate-pulse" />
              <div className="h-4 w-full bg-gray-800 rounded animate-pulse" />
              <div className="h-4 w-5/6 bg-gray-800 rounded animate-pulse" />
            </div>
            <div className="space-y-2">
              <div className="h-3 w-1/3 bg-gray-700 rounded animate-pulse" />
              <div className="h-4 w-full bg-gray-800 rounded animate-pulse" />
              <div className="h-4 w-4/5 bg-gray-800 rounded animate-pulse" />
            </div>
            <div className="border-t border-gray-700 pt-4 space-y-2">
              <div className="h-3 w-1/4 bg-gray-700 rounded animate-pulse" />
              <div className="h-4 w-full bg-gray-800 rounded animate-pulse" />
              <div className="h-4 w-full bg-gray-800 rounded animate-pulse" />
              <div className="h-4 w-2/3 bg-gray-800 rounded animate-pulse" />
            </div>
          </>
        )}

        {error && (
          <p className="text-red-400 text-sm text-center py-4">{error}</p>
        )}

        {zone && !loading && (
          <>
            <div className="flex items-center gap-3">
              <span className={`text-xs font-bold uppercase px-2.5 py-1 rounded-full ${SEVERITY_STYLES[zone.severity]}`}>
                {zone.severity}
              </span>
              <span className="text-xs text-gray-400 font-mono">{zone.zone_id}</span>
            </div>

            <div className="bg-gray-800 rounded-lg p-3">
              <p className="text-xs text-gray-400 mb-1">Mean heat above city median</p>
              <p className="text-2xl font-bold text-white">
                {zone.mean_relative_heat >= 0 ? '+' : ''}{zone.mean_relative_heat.toFixed(1)}
                <span className="text-base font-normal text-gray-400"> °C</span>
              </p>
            </div>

            <div>
              <h3 className="text-xs font-semibold uppercase text-gray-400 mb-2">Top Contributors</h3>
              <ul className="space-y-1">
                {zone.top_contributors.map((c) => (
                  <li key={c} className="text-sm text-gray-200 flex items-start gap-2">
                    <span className="text-orange-400 mt-0.5">•</span>
                    {c}
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <h3 className="text-xs font-semibold uppercase text-gray-400 mb-2">Recommendations</h3>
              <ul className="space-y-1">
                {zone.top_recommendations.map((r) => (
                  <li key={r} className="text-sm text-gray-200 flex items-start gap-2">
                    <span className="text-green-400 mt-0.5">•</span>
                    {r}
                  </li>
                ))}
              </ul>
            </div>

            <div className="border-t border-gray-700 pt-4">
              <div className="flex items-center gap-1.5 mb-3">
                <Sparkles size={13} className="text-violet-400" />
                <h3 className="text-xs font-semibold uppercase text-violet-400">Gemini Summary</h3>
              </div>
              {zone.gemini_summary ? (
                <div className="rounded-lg border border-violet-800/50 bg-violet-950/30 p-3">
                  <p className="text-sm text-gray-200 leading-relaxed">{zone.gemini_summary}</p>
                </div>
              ) : (
                <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-3">
                  <p className="text-sm text-gray-500 italic">No summary available for this zone.</p>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
