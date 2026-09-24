/**
 * Materialize hoisted fixture dependencies as version-checked symlinks.
 *
 * Why this exists: the root devDependencies intentionally pin the SAME
 * versions as the `tests/hosts/v017-rc1` fixture (the plugin typechecks
 * against the baseline it runs on). npm therefore dedupes those packages
 * to the root, and `tests/host-compatibility.spec.ts` /
 * `tests/artifact.client.spec.ts` deep imports through
 * `tests/hosts/<name>/node_modules/…` fail at resolve time — even with
 * `install-strategy=nested`, which only nests *conflicting* versions.
 * Upstream never hits this because its root pins a generation no fixture
 * uses. See register row #2 and the fork architecture doc.
 *
 * The rule is strict: link ONLY when the root copy's version is EXACTLY
 * the fixture's pin. A mismatch fails loud — silently substituting a
 * wrong host generation would falsify the version matrix the fixtures
 * exist to test. Runs on `postinstall`, so every `npm install`/`npm ci`
 * self-heals (npm may prune the links on reinstall).
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, symlinkSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'

const root = resolve(dirname(new URL(import.meta.url).pathname), '..')
const hostsDir = join(root, 'tests', 'hosts')

const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'))
const versionOf = (dir) => {
  const f = join(dir, 'package.json')
  return existsSync(f) ? readJson(f).version : null
}

let linked = 0
for (const host of readdirSync(hostsDir, { withFileTypes: true })) {
  if (!host.isDirectory()) continue
  const manifestFile = join(hostsDir, host.name, 'package.json')
  if (!existsSync(manifestFile)) continue
  const manifest = readJson(manifestFile)
  const pins = { ...manifest.dependencies, ...manifest.devDependencies }
  for (const [name, pin] of Object.entries(pins)) {
    if (!pin || /^[~^<>=*| ]/.test(pin)) continue // exact pins only
    const fixturePath = join(hostsDir, host.name, 'node_modules', ...name.split('/'))
    const have = versionOf(fixturePath)
    if (have !== null) {
      if (have !== pin) {
        throw new Error(`fixture drift: ${host.name} needs ${name}@${pin} but nested copy is ${have}`)
      }
      continue
    }
    const rootPath = join(root, 'node_modules', ...name.split('/'))
    const rootVersion = versionOf(rootPath)
    if (rootVersion === null) continue // installed later / optional: npm fills it, postinstall re-runs
    if (rootVersion !== pin) {
      throw new Error(`refusing to link: root has ${name}@${rootVersion} but ${host.name} pins ${pin}`)
    }
    mkdirSync(dirname(fixturePath), { recursive: true })
    symlinkSync(relative(dirname(fixturePath), rootPath), fixturePath, 'junction')
    linked++
  }
}
console.log(`materialize-fixture-links: ${linked} link(s)`)
