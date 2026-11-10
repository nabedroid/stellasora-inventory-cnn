import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import react from '@vitejs/plugin-react';

// __dirname を解決する
import path from 'path'
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default defineConfig({
  // リポジトリ名に合わせて変更
  base: '/stellasora-inventory-ocr/manager/',
  plugins: [
    react(),
    // common/public をルートにコピー
    viteStaticCopy({
      targets: [
        {
          src: '../common/public/*',
          dest: '',
          rename: { stripBase: 2 },
        },
      ],
    }),
  ],
  resolve: {
    alias: {
      '@common': path.resolve(__dirname, '../common/src'),
    },
  },
  server: {
    host: true,
    port: 3001,
    watch: {
      usePolling: true,
    },
  },
})
