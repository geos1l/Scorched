import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://45.63.18.135:8000/:path*',
      },
    ]
  },
}

export default nextConfig
