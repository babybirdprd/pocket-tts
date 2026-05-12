import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

const apiTarget = process.env.POCKET_TTS_API_BASE || "http://localhost:8000"

// https://vite.dev/config/
export default defineConfig({
  base: "./",
  plugins: [tailwindcss(), react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      '/stream': apiTarget,
      '/generate': apiTarget,
      '/health': apiTarget,
      '/wasm': apiTarget,
    }
  }
})
