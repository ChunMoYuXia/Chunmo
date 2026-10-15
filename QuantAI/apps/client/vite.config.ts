import { sveltekit } from '@sveltejs/kit/vite';
import { svelteTesting } from '@testing-library/svelte/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [sveltekit(), svelteTesting()],
  server: { strictPort: true, port: 5173 },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/zrender/')) return 'chart-renderer';
          if (id.includes('/node_modules/echarts/')) return 'chart-engine';
        },
      },
    },
  },
  test: { include: ['src/**/*.test.ts'], environment: 'node', setupFiles: ['./src/test-setup.ts'] },
});
