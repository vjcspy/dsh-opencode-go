import { defineConfig } from 'vitest/config'

// The Agent shell may export NODE_ENV=production, which fails the React client
// specs: Vite then resolves Node builds whose `node:` imports the browser
// runtime cannot load. The suite owns its own environment, set before Vite
// resolves any module.
process.env.NODE_ENV = 'test'

export default defineConfig({
  test: {
    setupFiles: ['tests/setup.ts'],
    include: ['tests/**/*.spec.ts', 'tests/**/*.spec.tsx'],
    testTimeout: 15000,
    server: { deps: { inline: ['@deepseek-ai/dsh-client-ui-primitives'] } },
  },
})
