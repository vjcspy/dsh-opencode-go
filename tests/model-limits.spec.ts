import { afterEach, expect, it, vi } from 'vitest'
import { OpencodeGoAdapter } from '../src/adapter.ts'
import { discoverCatalogModels, discoverSettingsModels } from '../src/catalog.ts'
import { configOf } from './config-of.ts'
import { closeMockGateways, listingBody, mockGateway, textEvents } from './mock-gateway.ts'
import { bootProfile, ENTRY_ID } from './support/profile-fixture.ts'
import { MODELS_METADATA_URL } from './support/model-metadata.ts'

afterEach(closeMockGateways)

it('keeps original catalog references when runtime capacities are overridden', async () => {
  const gateway = await mockGateway({ status: 200, body: listingBody(['deepseek-v4.1-flash']) })
  const config = configOf(gateway.url, { modelLimits: { 'deepseek-v4.1-flash': { contextWindow: 123456, maxTokens: 5432 } } })
  const adapter = new OpencodeGoAdapter({ config: () => config, resolveApiKey: async () => 'test-key' })
  expect((await adapter.resolveModel('opencode-go', 'deepseek-v4.1-flash')).context?.contextWindow).toBe(123456)
  expect(await discoverCatalogModels(adapter.catalogOf(config))).toContainEqual(expect.objectContaining({
    id: 'deepseek-v4.1-flash', contextWindow: 262144, maxTokens: 131072,
  }))
})

it.each([undefined, 8192, 512])('caps explicit and default request output (%s)', async (requested) => {
  const gateway = await mockGateway({ status: 200, body: listingBody(['deepseek-v4.1-flash']) })
  gateway.pushCompletions({ events: textEvents })
  const config = configOf(gateway.url, { modelLimits: { 'deepseek-v4.1-flash': { maxTokens: 1024 } } })
  const adapter = new OpencodeGoAdapter({ config: () => config, resolveApiKey: async () => 'test-key' })
  for await (const chunk of adapter.stream({ provider: 'opencode-go', model: 'deepseek-v4.1-flash', messages: [],
    ...(requested === undefined ? {} : { maxTokens: requested }),
  })) void chunk
  const body = gateway.bodies[0] as Record<string, number>
  expect(body.max_tokens ?? body.max_completion_tokens).toBe(Math.min(requested ?? 1024, 1024))
})

it('retains last known online metadata when limits change during an outage', async () => {
  const gateway = await mockGateway({ status: 200, body: listingBody(['union-alpha']) })
  let config = configOf(gateway.url)
  const adapter = new OpencodeGoAdapter({ config: () => config, resolveApiKey: async () => 'test-key' })
  expect((await adapter.resolveModel('opencode-go', 'union-alpha')).context?.contextWindow).toBe(262144)
  const priorFetch = globalThis.fetch
  vi.stubGlobal('fetch', (input: string | URL | Request, init?: RequestInit) =>
    String(input) === MODELS_METADATA_URL ? Promise.resolve(new Response('', { status: 503 })) : priorFetch(input, init))
  config = { ...config, modelLimits: { 'union-alpha': { contextWindow: 123456 } } }
  await expect(adapter.resolveModel('opencode-go', 'union-alpha')).resolves.toMatchObject({ context: { contextWindow: 123456 } })
  // An explicit refresh must preserve the learned model too, not just the TTL hit.
  expect((await discoverSettingsModels(adapter.catalogOf(config))).models).toContainEqual(expect.objectContaining({
    id: 'union-alpha', contextWindow: 262144,
  }))
  expect(gateway.modelListings).toBe(2)
  config = { ...config, modelLimits: {} }
  expect((await adapter.resolveModel('opencode-go', 'union-alpha')).context?.contextWindow).toBe(262144)
})

it('keeps capacities captured before an in-flight fetch across a settings change', async () => {
  const gateway = await mockGateway({ status: 200, body: listingBody(['deepseek-v4.1-flash']) })
  let config = configOf(gateway.url, { modelLimits: { 'deepseek-v4.1-flash': { contextWindow: 123456 } } })
  const adapter = new OpencodeGoAdapter({ config: () => config, resolveApiKey: async () => 'test-key' })
  const priorFetch = globalThis.fetch
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  vi.stubGlobal('fetch', async (input: string | URL | Request, init?: RequestInit) => {
    if (String(input) === MODELS_METADATA_URL) await gate
    return priorFetch(input, init)
  })
  const pending = adapter.resolveModel('opencode-go', 'deepseek-v4.1-flash')
  config = { ...config, modelLimits: { 'deepseek-v4.1-flash': { contextWindow: 42 } } }
  release()
  expect((await pending).context?.contextWindow).toBe(123456)
  expect((await adapter.resolveModel('opencode-go', 'deepseek-v4.1-flash')).context?.contextWindow).toBe(42)
})

it('can explicitly return a single inherited capacity to the catalog', async () => {
  const gateway = await mockGateway({ status: 200, body: listingBody(['deepseek-v4.1-flash']) })
  const { ctx, dispose } = await bootProfile({
    config: { baseURL: gateway.url, modelLimits: { example: { contextWindow: 123456, maxTokens: 1024 } } },
    credentials: { OPENCODE_API_KEY: 'test-key' },
  })
  try {
    const limits = () => ctx.settings.describe().find(row => row.ns === ENTRY_ID)?.value.modelLimits
    expect(limits()).toEqual({ example: { contextWindow: 123456, maxTokens: 1024 } })

    // `null` selects the catalog value for one field without disturbing its sibling.
    await ctx.settings.mutate(ENTRY_ID, [{ op: 'set', path: ['modelLimits', 'example', 'contextWindow'], value: null }])
    expect(limits()).toEqual({ example: { contextWindow: null, maxTokens: 1024 } })

    // A whole null entry selects both catalog values.
    await ctx.settings.mutate(ENTRY_ID, [{ op: 'set', path: ['modelLimits', 'example'], value: null }])
    expect(limits()).toEqual({ example: null })
  } finally {
    await dispose()
  }
})
