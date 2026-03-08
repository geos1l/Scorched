import { useEffect } from 'react'
import mapboxgl from 'mapbox-gl'

const SEVERITY_COLORS: Record<string, string> = {
  extreme: '#C0392B',
  high: '#E67E22',
  moderate: '#F1C40F',
  low: '#27AE60',
}

import { ZoneDetail } from './DetailPanel'

interface ZoneLayerProps {
  map: mapboxgl.Map
  onZoneClick: (zoneId: string, fallback: Omit<ZoneDetail, 'gemini_summary'>) => void
}

export default function ZoneLayer({ map, onZoneClick }: ZoneLayerProps) {
  useEffect(() => {
    let mounted = true
    let cleanup: (() => void) | undefined

    async function addLayers() {
      const res = await fetch(`/api/backend/zones?city_id=toronto`)
      if (!mounted) return
      const json = await res.json()
      const data = json.zones

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
        const props = e.features?.[0]?.properties
        if (!props?.zone_id) return
        const fallback: Omit<ZoneDetail, 'gemini_summary'> = {
          zone_id: props.zone_id,
          severity: props.severity,
          mean_relative_heat: props.mean_relative_heat ?? 0,
          top_contributors: typeof props.top_contributors === 'string' ? JSON.parse(props.top_contributors) : (props.top_contributors ?? []),
          top_recommendations: typeof props.top_recommendations === 'string' ? JSON.parse(props.top_recommendations) : (props.top_recommendations ?? []),
        }
        onZoneClick(props.zone_id, fallback)
      }

      const handleMouseEnter = () => { map.getCanvas().style.cursor = 'pointer' }
      const handleMouseLeave = () => { map.getCanvas().style.cursor = '' }

      map.on('click', 'zones-fill', handleClick)
      map.on('mouseenter', 'zones-fill', handleMouseEnter)
      map.on('mouseleave', 'zones-fill', handleMouseLeave)

      cleanup = () => {
        map.off('click', 'zones-fill', handleClick)
        map.off('mouseenter', 'zones-fill', handleMouseEnter)
        map.off('mouseleave', 'zones-fill', handleMouseLeave)
        if (map.getLayer('zones-fill')) map.removeLayer('zones-fill')
        if (map.getLayer('zones-outline')) map.removeLayer('zones-outline')
        if (map.getSource('zones')) map.removeSource('zones')
      }
    }

    addLayers().catch(console.error)

    return () => {
      mounted = false
      cleanup?.()
    }
  }, [map, onZoneClick])

  return null
}
