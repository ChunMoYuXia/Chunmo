import js from '@eslint/js';
import ts from 'typescript-eslint';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import svelteConfig from './svelte.config.js';

export default ts.config(
  {
    ignores: [
      'build/**',
      '.svelte-kit/**',
      'node_modules/**',
      'playwright-report/**',
      'test-results/**',
      'src-tauri/resources/**',
      'src-tauri/gen/**',
      'src-tauri/target/**',
    ],
  },
  js.configs.recommended,
  ...ts.configs.recommended,
  ...svelte.configs['flat/recommended'],
  { languageOptions: { globals: { ...globals.browser, ...globals.node } } },
  { files:['e2e/fixtures.ts'], rules:{'no-empty-pattern':['error',{allowObjectPatternsAsParameters:true}]} },
  {
    files: ['**/*.svelte'],
    languageOptions: { parserOptions: { parser: ts.parser, svelteConfig } },
  },
);
