import { test as base, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
const connection = () =>
  JSON.parse(readFileSync(resolve('../../.cache/e2e/connection.json'), 'utf8')) as {
    url: string;
    token: string;
  };
export const test = base.extend<{ localSession: void }>({
  baseURL: async ({}, use) => {
    await use(connection().url);
  },
  localSession: [
    async ({ context }, use) => {
      const { url, token } = connection();
      await context.addCookies([
        { name: 'portfolio_session', value: token, url, httpOnly: true, sameSite: 'Strict' },
      ]);
      await context.addInitScript(
        (url) => sessionStorage.setItem('portfolio-backend-url', url),
        url,
      );
      await use();
    },
    { auto: true },
  ],
});
export { expect };
