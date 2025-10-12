/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ['localhost', '127.0.0.1'],
    unoptimized: true
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
  },
  webpack: (config, { isServer }) => {
    // Fix for modules that can't be used on the client side
    if (!isServer) {
      config.resolve.fallback = {
        fs: false,
        net: false,
        tls: false,
        crypto: false
      }
    }
    return config
  },
  // Enable source maps in development
  productionBrowserSourceMaps: false,
  
  // Optimize for Windows development
  swcMinify: true,
  
  // Headers for CORS during development
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Access-Control-Allow-Origin',
            value: 'http://localhost:8000'
          }
        ]
      }
    ]
  }
}

module.exports = nextConfig