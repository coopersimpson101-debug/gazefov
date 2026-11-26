import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: './',
  build: { target: 'es2022', outDir: 'dist', rollupOptions: { input: { demo: 'index.html' } } },
  test: { include: ['tests/**/*.test.ts'], environment: 'node', testTimeout: 120000 },
});
