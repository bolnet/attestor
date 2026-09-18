export const meta = {
  name: 'build-bench-token-savings',
  description: 'Build software step-by-step via subagents; record attestor recall token savings per step (single-run counterfactual)',
  whenToUse: 'Live token-savings benchmark: drive a real multi-step software build through subagents, comparing attestor recall vs full-history context at every step.',
  phases: [{ title: 'Build', detail: 'recall -> build subagent -> capture+account, sequentially' }],
}

// ── Config (override via Workflow args) ───────────────────────────────────
const ATTESTOR = '/Users/aarjay/Documents/attestor'
const STORE     = args?.store     || '~/.attestor'
const NS        = args?.namespace || 'bench-fitness-build'
const LEDGER    = args?.ledger    || '/tmp/build_bench_fitness.json'
const REPO      = args?.repo      || '/Users/aarjay/Documents/fitness_backend'
const BRANCH    = args?.branch    || 'bench/token-savings'
const BUDGET    = args?.budget    || 4000   // realistic injected-context cap; naive history is unbounded
const PY        = `cd ${ATTESTOR} && .venv/bin/python`

// Each step: { id, query (what to recall), task (what to build), entity }
// Pass args.steps to scale to ~100; default = Phase 0 M0.3 (3 real steps, smoke).
const steps = args?.steps || [
  { id: '0.3.1', entity: 'AuthProvider',
    query: 'auth provider interface, JWT verification, Clerk adapter, tenant resolution',
    task: 'Implement an AuthProvider interface + Clerk adapter (verify session JWT against JWKS, extract user + org). Files: apps/api/src/identity/auth.provider.ts, clerk.adapter.ts. TDD against a signed fixture JWT: valid token -> claims; expired/forged -> 401. Keep Clerk-specific code isolated behind the interface.' },
  { id: '0.3.2', entity: 'RequestContext',
    query: 'request context, tenant guard, async local storage, withTenant, RLS tenant_id',
    task: 'Implement a NestJS guard + RequestContext (tenantId, actorId, role) bound to async-local-storage. Files: apps/api/src/common/request-context.ts, tenant.guard.ts. TDD: authed request with org->tenant mapping populates context; missing org -> 403; context not leaked across requests; DB calls auto-use withTenant(ctx.tenantId).' },
  { id: '0.3.3', entity: 'Membership',
    query: 'org to tenant mapping, provisioning on first login, membership table, idempotent',
    task: 'Implement org->tenant mapping table + provisioning on first login. Files: packages/db/src/schema/tenant.ts (membership), identity/membership.service.ts. TDD: first login for a new org creates tenant + membership atomically; second login is idempotent; users get the right role.' },
]

const RECALL_SCHEMA = {
  type: 'object',
  properties: {
    recall_tokens: { type: 'integer' },
    n_memories: { type: 'integer' },
    context: { type: 'string' },
  },
  required: ['recall_tokens', 'n_memories', 'context'],
}
const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    decisions_file: { type: 'string', description: 'Absolute path the agent wrote its full decision record to' },
    summary: { type: 'string' },
    tests_pass: { type: 'boolean' },
  },
  required: ['decisions_file', 'summary', 'tests_pass'],
}
const ACCT_SCHEMA = {
  type: 'object',
  properties: { row: { type: 'object' }, summary: { type: 'object' } },
  required: ['row', 'summary'],
}

phase('Build')
log(`build-bench: ${steps.length} steps, repo=${REPO} branch=${BRANCH} ns=${NS} budget=${BUDGET}`)

const ledger = []
for (const step of steps) {
  // 1. RECALL — attestor surfaces only the relevant prior decisions (flat, capped at BUDGET).
  const recall = await agent(
    `Run exactly this command and return ONLY the JSON it prints on stdout:\n\n` +
    `${PY} -m evals.build_bench.recall_step ${STORE} ${NS} ${JSON.stringify(step.query)} --budget ${BUDGET}`,
    { label: `recall:${step.id}`, phase: 'Build', schema: RECALL_SCHEMA }
  )

  // 2. BUILD — the subagent gets ONLY the recalled context (by design), does the real work.
  const decFile = `/tmp/build_bench_${step.id.replace(/\./g, '_')}_decisions.md`
  const build = await agent(
    `You are implementing ONE step of a real, ongoing software build.\n\n` +
    `Repo: ${REPO}\n` +
    `Git: work on branch \`${BRANCH}\` (create it from main if it does not exist). Commit your work with a conventional-commit message when tests pass.\n\n` +
    `RELEVANT PRIOR DECISIONS (recalled from attestor memory — this is ALL the prior context you get, by design; do not ask for more):\n` +
    `${recall?.context || '(none — this is an early step)'}\n\n` +
    `STEP ${step.id}: ${step.task}\n\n` +
    `Work test-first, run the tests, keep the diff minimal and consistent with the recalled conventions. ` +
    `Then WRITE a concise but complete record of what you built + every decision a later step would need (file paths, signatures, conventions, gotchas) to this exact file: ${decFile}\n\n` +
    `Return JSON: { decisions_file: "${decFile}", summary: "<one-line>", tests_pass: <bool> }.`,
    { label: `build:${step.id}`, phase: 'Build', schema: BUILD_SCHEMA }
  )

  // 3. CAPTURE + ACCOUNT — persist the decisions to attestor and record the ledger row.
  const acct = await agent(
    `Run exactly this command and return ONLY the JSON it prints on stdout:\n\n` +
    `${PY} -m evals.build_bench.capture_step ${STORE} ${NS} ` +
    `--decisions-file ${build?.decisions_file || decFile} ` +
    `--step ${step.id} --recall-tokens ${recall?.recall_tokens ?? 0} ` +
    `--n-memories ${recall?.n_memories ?? 0} --ledger ${LEDGER} ` +
    `--category build --entity ${JSON.stringify(step.entity || '')}`,
    { label: `account:${step.id}`, phase: 'Build', schema: ACCT_SCHEMA }
  )

  const row = acct?.row || {}
  ledger.push(row)
  log(`step ${step.id}: recall=${row.recall_tokens}t  naive=${row.full_history_tokens}t  savings=${row.savings_ratio}x  (tests_pass=${build?.tests_pass})`)
}

return { steps: ledger, ledger_file: LEDGER, summary: ledger.length ? ledger[ledger.length - 1] : null }
