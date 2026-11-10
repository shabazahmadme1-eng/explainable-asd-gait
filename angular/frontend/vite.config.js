import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // GitHub Pages serves the app under /<repo>/; the deploy workflow sets VITE_BASE.
  base: process.env.VITE_BASE || '/',
})
