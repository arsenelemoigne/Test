/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ['@mendable/firecrawl-js'],
  },
}

export default nextConfig
