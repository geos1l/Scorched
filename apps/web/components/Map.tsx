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
      center: [-79.3832, 43.6532],
      zoom: 11,
    })

    map.addControl(new mapboxgl.NavigationControl(), 'bottom-right')

    map.on('load', () => {
      mapRef.current = map
      setMapReady(true)
    })

    return () => {
      map.remove()
      mapRef.current = null
      setMapReady(false)
    }
  }, [])

  const handleZoneClick = useCallback(async function handleZoneClick(zoneId: string) {
    setLoading(true)
    setError(null)
    setSelectedZone(null)

    try {
      const base = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
      const res = await fetch(`${base}/zones/${zoneId}`)
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const data: ZoneDetail = await res.json()
      setSelectedZone(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load zone')
    } finally {
      setLoading(false)
    }
  }, [])

  function handleClose() {
    setSelectedZone(null)
    setError(null)
  }

  const showPanel = selectedZone !== null || loading || error !== null

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
