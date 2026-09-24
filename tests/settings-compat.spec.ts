/**
 * DSH 0.1.5/0.1.6 expose `installSection` and no `configure`. The host half
 * must still install its settings section and route a section edit into the
 * live configuration, so a declared older host is not silently broken; 0.1.7+
 * must prefer `configure` and never touch the removed method.
 */
import { describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime from '@deepseek-ai/dsh-llm'
import { apply, NS } from '../src/index.ts'
import { PlainConfig } from '../src/config.ts'
import { configOf } from './config-of.ts'

const BASE_URL = 'https://opencode.ai/zen/go/v1'

/** `ctx.inject` callbacks settle on a later microtask/timer turn than `apply`. */
const tick = async (): Promise<void> => { await new Promise(resolve => setTimeout(resolve, 0)) }

describe('settings service across host generations', () => {
  it('installs a section when the settings service has no configure', async () => {
    const installSection = vi.fn()
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    ctx.provide('credentials', {
      describe: () => Promise.resolve({ configured: false, writable: true }),
      resolve: () => Promise.resolve(undefined),
    } as never)
    ctx.provide('settings', { installSection } as never)

    apply(ctx, configOf(BASE_URL))
    await tick()

    expect(installSection).toHaveBeenCalledOnce()
    const [owner, ns, schema, entry, hooks] = installSection.mock.calls[0]!
    expect(owner).toBe(ctx)
    expect(ns).toBe(NS)
    expect(schema).toBe(PlainConfig)
    expect(entry.baseURL).toBe(BASE_URL)
    expect(hooks).toMatchObject({
      validate: expect.any(Function) as unknown,
      setSource: expect.any(Function) as unknown,
      onChange: expect.any(Function) as unknown,
    })
  })

  it('drives the route from the section source sink and change hook', async () => {
    const installSection = vi.fn()
    const configured = new Set<string>()
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    ctx.provide('credentials', {
      describe: (ref: string) => Promise.resolve({ configured: configured.has(String(ref)), writable: true }),
      resolve: () => Promise.resolve(undefined),
    } as never)
    ctx.provide('settings', { installSection } as never)

    apply(ctx, configOf(BASE_URL))
    await tick()
    expect(ctx.llm.listProviders()).toEqual([])

    const hooks = installSection.mock.calls[0]![4] as {
      setSource: (source: () => ReturnType<typeof configOf>) => void
      onChange: () => void
    }
    configured.add('SWITCHED_REF')
    hooks.setSource(() => configOf(BASE_URL, { apiKeyEnv: 'SWITCHED_REF' }))
    hooks.onChange()

    await expect.poll(() => ctx.llm.listProviders(), { timeout: 10_000 })
      .toContainEqual({ id: 'opencode-go', name: 'OpenCode Go' })
  })

  it('prefers configure and never calls the removed installSection', async () => {
    const configure = vi.fn(() => () => {})
    const installSection = vi.fn()
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    ctx.provide('settings', { configure, installSection } as never)

    apply(ctx, configOf(BASE_URL))
    await tick()

    expect(configure).toHaveBeenCalledWith({ auto: false }, ctx.fiber)
    expect(installSection).not.toHaveBeenCalled()
  })
})
