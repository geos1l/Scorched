/**
 * SegOverlay — adds a semi-transparent segmentation mask mosaic over the AOI.
 *
 * Fetches /api/backend/tiles/aoi/info for WGS84 bounds, then loads
 * /api/backend/tiles/aoi/mosaic as a georeferenced image source.
 * Only visible within the AOI bbox; the rest of the map is unaffected.
 *
 * Colors: building=red, road=gray, vegetation=green, water=blue
 */

import { useEffect } from 'react'
import mapboxgl from 'mapbox-gl'

interface SegOverlayProps {
  map: mapboxgl.Map
  visible: boolean
}

export default function SegOverlay({ map, visible }: SegOverlayProps) {
  // Add source + layer once on mount
  useEffect(() => {
    let mounted = true
    let cleanup: (() => void) | undefined

    async function addOverlay() {
      const res = await fetch('/api/backend/tiles/aoi/info')
      if (!mounted || !res.ok) return

      const { bounds } = await res.json()
      const [minLon, minLat, maxLon, maxLat] = bounds as [number, number, number, number]

      if (!mounted) return

      map.addSource('seg-mosaic', {
        type: 'image',
        url: '/api/backend/tiles/aoi/mosaic',
        coordinates: [
          [minLon, maxLat], // NW
          [maxLon, maxLat], // NE
          [maxLon, minLat], // SE
          [minLon, minLat], // SW
        ],
      })

      map.addLayer({
        id: 'seg-mosaic-layer',
        type: 'raster',
        source: 'seg-mosaic',
        paint: {
          'raster-opacity': 0.6,
          'raster-fade-duration': 0,
        },
      })

      cleanup = () => {
        if (map.getLayer('seg-mosaic-layer')) map.removeLayer('seg-mosaic-layer')
        if (map.getSource('seg-mosaic')) map.removeSource('seg-mosaic')
      }
    }

    addOverlay().catch(console.error)

    return () => {
      mounted = false
      cleanup?.()
    }
  }, [map])

  // Toggle visibility whenever `visible` changes
  useEffect(() => {
    if (!map.getLayer('seg-mosaic-layer')) return
    map.setLayoutProperty('seg-mosaic-layer', 'visibility', visible ? 'visible' : 'none')
  }, [map, visible])

  return null
}
