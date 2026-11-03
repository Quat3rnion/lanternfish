import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { mockApi } from './mock/api.ts';

// TRAVERSE_API=http://host:port npm run dev  -> proxy /api to a running backend
// (no TRAVERSE_API)                          -> serve the mock API from fixtures/
const backend = process.env.TRAVERSE_API;

export default defineConfig({
  base: './',
  plugins: [svelte(), ...(backend ? [] : [mockApi()])],
  server: {
    port: 5179,
    strictPort: true,
    proxy: backend ? { '/api': { target: backend, changeOrigin: true } } : undefined,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
  },
});
