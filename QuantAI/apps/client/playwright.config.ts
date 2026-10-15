import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  fullyParallel: false,
  timeout: 120000,
  globalSetup: './e2e/global-setup.ts',
  use: {
    browserName: 'chromium',
    channel: process.env.CI ? undefined : 'msedge',
    viewport: { width: 1440, height: 1000 },
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  reporter: [['list'], ['html', { open: 'never' }]],
});
