import { useCallback, useEffect, useRef, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import ZoneLayer from './ZoneLayer'
import DetailPanel, { ZoneDetail } from './DetailPanel'

export default function Map() {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<mapboxgl.Map | null>(null)
  const [mapReady, setMapReady] = useState(false)
  const [selectedZone, setSelectedZone] = useState<ZoneDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return

    mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN ?? ''

    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: 'mapbox://styles/mapbox/dark-v11',
      center: [-79.3946, 43.6444],
      zoom: 13,
    })

    map.addControl(new mapboxgl.NavigationControl(), 'bottom-right')

    map.on('load', () => {
      map.fitBounds([[-79.4071, 43.6354], [-79.3821, 43.6624]], { padding: 40 })
      mapRef.current = map
      setMapReady(true)
    })

    return () => {
      map.remove()
      mapRef.current = null
      setMapReady(false)
    }
  }, [])

  const handleZoneClick = useCallback(async function handleZoneClick(
    zoneId: string,
    fallback: Omit<ZoneDetail, 'gemini_summary'>
  ) {
    setLoading(true)
    setError(null)
    setSelectedZone(null)

    try {
      const res = await fetch(`/api/backend/zones/${zoneId}`)
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data: ZoneDetail = await res.json()
      setSelectedZone(data)
    } catch {
      // Show what we already have from the map layer, just without Gemini summary
      setSelectedZone({ ...fallback, gemini_summary: '' })
    } finally {
      setLoading(false)
    }
  }, [])

  function handleClose() {
    setSelectedZone(null)
    setError(null)
  }

  const showPanel = selectedZone !== null || loading

  return (
    <div className="relative w-full h-screen">
      <div ref={containerRef} className="w-full h-full" />

      {mapReady && mapRef.current && (
        <ZoneLayer map={mapRef.current} onZoneClick={handleZoneClick} />
      )}

      {showPanel && (
        <DetailPanel
          zone={selectedZone}
          loading={loading}
          error={error}
          onClose={handleClose}
        />
      )}
    </div>
  )
}
