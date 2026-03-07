import dynamic from 'next/dynamic'

// Mapbox GL requires browser APIs — load client-side only
const Map = dynamic(() => import('../components/Map'), { ssr: false })

export default function Home() {
  return (
    <main className="w-full h-screen bg-gray-900">
      <Map />
    </main>
  )
}
