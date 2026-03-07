import { useEffect } from 'react'
import mapboxgl from 'mapbox-gl'

const SEVERITY_COLORS: Record<string, string> = {
  extreme: '#C0392B',
  high: '#E67E22',
  moderate: '#F1C40F',
  low: '#27AE60',
}

interface ZoneLayerProps {
  map: mapboxgl.Map
  onZoneClick: (zoneId: string) => void
}

export default function ZoneLayer({ map, onZoneClick }: ZoneLayerProps) {
  useEffect(() => {
    let mounted = true

    async function addLayers() {
      const res = await fetch('/dummy-zones.geojson')
      if (!mounted) return
      const data = await res.json()

      map.addSource('zones', { type: 'geojson', data })

      map.addLayer({
        id: 'zones-fill',
        type: 'fill',
        source: 'zones',
        paint: {
          'fill-color': [
            'match',
            ['get', 'severity'],
            'extreme', SEVERITY_COLORS.extreme,
            'high',    SEVERITY_COLORS.high,
            'moderate',SEVERITY_COLORS.moderate,
            'low',     SEVERITY_COLORS.low,
            '#888888',
          ],
          'fill-opacity': 0.55,
        },
      })

      map.addLayer({
        id: 'zones-outline',
        type: 'line',
        source: 'zones',
        paint: {
          'line-color': '#ffffff',
          'line-width': 1,
          'line-opacity': 0.35,
        },
      })

      const handleClick = (e: mapboxgl.MapMouseEvent & { features?: mapboxgl.MapboxGeoJSONFeature[] }) => {
        const zoneId = e.features?.[0]?.properties?.zone_id
        if (zoneId) onZoneClick(zoneId)
      }

      map.on('click', 'zones-fill', handleClick)
      map.on('mouseenter', 'zones-fill', () => { map.getCanvas().style.cursor = 'pointer' })
      map.on('mouseleave', 'zones-fill', () => { map.getCanvas().style.cursor = '' })
    }

    addLayers().catch(console.error)

    return () => {
      mounted = false
      if (map.getLayer('zones-fill')) map.removeLayer('zones-fill')
      if (map.getLayer('zones-outline')) map.removeLayer('zones-outline')
      if (map.getSource('zones')) map.removeSource('zones')
    }
  }, [map, onZoneClick])

  return null
}
