import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    // Proxy /api requests to the local FastAPI backend during development
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
