/**
 * Mount the built plugin through the real Loader and Settings service with an
 * in-memory profile patch, so a configuration spec exercises the same
 * entry-form write path the Web UI uses. The pattern mirrors the harness's own
 * `packages/settings/settings/tests/configuration-fixture.ts` and the plugin's
 * `tests/fixtures/profile-compatibility.mjs`.
 */
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import Loader from '@deepseek-ai/cordis-plugin-loader'
import Settings from '@deepseek-ai/dsh-settings'

/** Profile entry id the plugin's configuration form is keyed by. */
export const ENTRY_ID = 'opencode-go'

/** The managed credential document's current layout: a versioned refs map. */
export function credentialsYaml(refs: Record<string, string>): string {
  const rows = Object.entries(refs).map(([ref, value]) => `  ${ref}: ${value}`).join('\n')
  return `version: 1\nrefs:\n${rows}\n`
}

export interface ProfileFixture {
  /** Mounted context: `ctx.llm`, `ctx.settings`, `ctx.credentials`, `ctx.typertGateway` are live. */
  ctx: Context
  /** Temporary DSH home holding the credential document. */
  home: string
  /** Dispose the context and remove the temporary home. */
  dispose(): Promise<void>
}

export interface ProfileFixtureOptions {
  /** Composition entry config the plugin starts from. */
  config?: Record<string, unknown>
  /** Credential refs written to the managed document before boot. */
  credentials?: Record<string, string>
  /** Extra plugins mounted after the runtime and before the plugin entry. */
  before?: (ctx: Context) => Promise<void>
}

/**
 * Boot the plugin through the Loader with the real settings service.
 * @param options - entry config, credential refs, and optional pre-mount hook.
 * @returns the fixture handle.
 */
export async function bootProfile(options: ProfileFixtureOptions = {}): Promise<ProfileFixture> {
  const home = await mkdtemp(join(tmpdir(), 'opencode-go-profile-'))
  const credentialsPath = join(home, '.credentials.yaml')
  await writeFile(credentialsPath, credentialsYaml(options.credentials ?? {}), { mode: 0o600 })
  const ctx = new Context()
  ctx.baseUrl = new URL('../../package.json', import.meta.url).href
  await ctx.plugin(Loader)
  await ctx.loader.create({ name: '@deepseek-ai/dsh-llm' })
  await ctx.loader.create({ name: '@deepseek-ai/dsh-credentials-local', config: { path: credentialsPath, watch: false } })
  if (options.before !== undefined) await options.before(ctx)
  const id = await ctx.loader.create({
    id: ENTRY_ID,
    name: new URL('../../lib/index.js', import.meta.url).href,
    config: options.config ?? {},
  })
  await ctx.loader.await()
  const entry = ctx.loader.resolve(id)
  ctx.provide('profileContext', { home })
  // The real settings service owns the write path; this editor supplies an
  // in-memory profile and delegates updates to the real Loader, including
  // validation and live Config references.
  ctx.provide('configEditor', {
    documentPath: join(home, 'cordis.patch.yml'),
    entries: () => [entry],
    configuration: () => [{ entry, inherited: {}, override: entry.options.config }],
    edit: async (
      _entry: unknown,
      change: (current: Record<string, unknown>, inherited: Record<string, unknown>) => Record<string, unknown>,
    ) => {
      const next = change(entry.options.config as Record<string, unknown>, {})
      const fiber = entry.fiber!
      const validated = fiber.ctx.waterfall(fiber, 'internal/config', next, () => next)
      await entry.update({ config: validated })
    },
  })
  await ctx.plugin(Settings)
  await ctx.loader.await()
  return {
    ctx,
    home,
    dispose: async () => {
      await ctx.fiber.dispose()
      await rm(home, { recursive: true, force: true })
    },
  }
}
