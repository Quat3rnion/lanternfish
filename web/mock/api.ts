// Development-only mock of the traverse HTTP API. It serves the files in
// fixtures/ and simulates trace runs, so the UI works without the Rust backend.
// It is not part of the production bundle.
import { readFileSync, readdirSync } from 'node:fs';
import { join, basename } from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';

type Json = Record<string, any>;

interface MockModel {
  path: string;
  name: string;
  size_bytes: number;
  fixture: string; // trace name whose data a finished run copies
}

interface MockRun {
  id: string;
  name: string;
  model: string;
  prompt: string;
  n: number;
  raw: boolean;
  started: number;
  fail: boolean;
  registered: boolean;
}

const fixturesDir = join(import.meta.dirname, '..', 'fixtures');

function loadFixtures(): Map<string, Json> {
  const out = new Map<string, Json>();
  for (const f of readdirSync(fixturesDir).filter((f) => f.endsWith('.json'))) {
    const viz = JSON.parse(readFileSync(join(fixturesDir, f), 'utf8')) as Json;
    out.set(viz.name ?? basename(f, '.json'), viz);
  }
  return out;
}

// Deterministic perturbation of an MoE trace: a second trace for the same
// model, so compare mode has something to compare in development.
function variantOf(viz: Json, name: string): Json {
  const v = structuredClone(viz);
  v.name = name;
  v.created = '2026-11-02T19:30:00';
  const ne = v.n_experts || 0;
  const hash = (a: number, b: number) => ((a * 2654435761) ^ (b * 40503)) >>> 0;
  for (const layer of v.layers) {
    for (const [key, rec] of Object.entries<Json>(layer.tokens)) {
      const i = Number(key);
      const h = hash(layer.layer + 1, i + 7);
      if (rec.experts && ne) {
        rec.experts = rec.experts.map((e: number, k: number) =>
          k >= 4 && (h >> k) % 3 === 0 ? (e * 7 + layer.layer * 13 + i) % ne : e,
        );
      }
      if (typeof rec.next_rank === 'number' && rec.next_rank > 0) {
        rec.next_rank = Math.round(rec.next_rank * (0.25 + (h % 100) / 40));
      }
    }
  }
  return v;
}

function summary(viz: Json) {
  return {
    name: viz.name,
    model: {
      name: viz.model?.name ?? '?',
      arch: viz.model?.arch ?? '?',
      n_layers: viz.n_layers,
      n_experts: viz.n_experts ?? 0,
    },
    tokens: viz.tokens?.length ?? 0,
    created: viz.created ?? '',
  };
}

const PHASES = ['loading model', 'tokenizing', 'prefill', 'decode', 'writing viz'];
const RUN_MS = 9000;

function runView(r: MockRun) {
  const t = Date.now() - r.started;
  if (t < 1200) {
    return { id: r.id, name: r.name, model: r.model, state: 'queued', phase: 'waiting for a slot', progress: 0, log_tail: '' };
  }
  const progress = Math.min(1, (t - 1200) / RUN_MS);
  if (r.fail && progress >= 0.5) {
    return {
      id: r.id, name: r.name, model: r.model, state: 'failed', phase: 'decode', progress: 0.5,
      log_tail: 'decode step 3/…\nllama_decode: failed to allocate compute buffer',
      error: 'mock failure: the prompt contains the word "fail"',
    };
  }
  if (progress >= 1) {
    return {
      id: r.id, name: r.name, model: r.model, state: 'done', phase: 'done', progress: 1,
      log_tail: `wrote traces/${r.name}/viz.json\ndone in ${(RUN_MS / 1000).toFixed(1)} s`,
    };
  }
  const phase = PHASES[Math.min(PHASES.length - 1, Math.floor(progress * PHASES.length))];
  const step = Math.floor(progress * r.n);
  return {
    id: r.id, name: r.name, model: r.model, state: 'running', phase, progress,
    log_tail: `${phase}…\nstep ${step}/${r.n}  captured 9 tensors x ${Math.max(1, step)} rows`,
  };
}

function send(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status;
  res.setHeader('content-type', 'application/json');
  res.setHeader('cache-control', 'no-store');
  res.end(JSON.stringify(body));
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let s = '';
    req.on('data', (c) => (s += c));
    req.on('end', () => resolve(s));
    req.on('error', reject);
  });
}

export function mockApi(): Plugin {
  const traces = loadFixtures();
  const moe = [...traces.values()].find((v) => (v.n_experts ?? 0) > 0);
  if (moe) traces.set(`${moe.name}-variant`, variantOf(moe, `${moe.name}-variant`));
  const dense = [...traces.values()].find((v) => !v.n_experts);

  const models: MockModel[] = [];
  if (moe) models.push({ path: moe.model?.path ?? '/models/moe.gguf', name: moe.model?.name ?? 'MoE', size_bytes: 3_946_000_000, fixture: moe.name });
  if (dense) models.push({ path: dense.model?.path ?? '/models/dense.gguf', name: dense.model?.name ?? 'Dense', size_bytes: 2_019_000_000, fixture: dense.name });

  const runs: MockRun[] = [];
  let seq = 0;

  function settle() {
    for (const r of runs) {
      const v = runView(r);
      if (v.state === 'done' && !r.registered) {
        r.registered = true;
        const m = models.find((m) => m.path === r.model);
        const src = m && traces.get(m.fixture);
        if (src) {
          const copy = structuredClone(src);
          copy.name = r.name;
          copy.prompt = r.prompt;
          copy.created = new Date().toISOString().slice(0, 19);
          traces.set(r.name, copy);
        }
      }
    }
  }

  return {
    name: 'traverse-mock-api',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url ?? '/', 'http://x');
        const p = url.pathname;
        if (!p.startsWith('/api/')) return next();
        settle();
        try {
          if (p === '/api/traces' && req.method === 'GET') {
            const list = [...traces.values()].map(summary).sort((a, b) => b.created.localeCompare(a.created));
            return send(res, 200, list);
          }
          const m = p.match(/^\/api\/traces\/(.+)$/);
          if (m && req.method === 'GET') {
            const viz = traces.get(decodeURIComponent(m[1]));
            return viz ? send(res, 200, viz) : send(res, 404, { error: 'no such trace' });
          }
          if (p === '/api/models' && req.method === 'GET') {
            return send(res, 200, models.map(({ path, name, size_bytes }) => ({ path, name, size_bytes })));
          }
          if (p === '/api/runs' && req.method === 'GET') {
            return send(res, 200, runs.map(runView).reverse());
          }
          if (p === '/api/runs' && req.method === 'POST') {
            let body: Json;
            try {
              body = JSON.parse(await readBody(req));
            } catch {
              return send(res, 400, { error: 'body is not JSON' });
            }
            const name = String(body.name ?? '').trim();
            if (!models.some((m) => m.path === body.model)) return send(res, 400, { error: 'unknown model' });
            if (!/^[A-Za-z0-9._-]{1,64}$/.test(name)) return send(res, 400, { error: 'name must be 1-64 chars of A-Z a-z 0-9 . _ -' });
            if (traces.has(name) || runs.some((r) => r.name === name)) return send(res, 409, { error: `trace "${name}" already exists` });
            if (!String(body.prompt ?? '').trim()) return send(res, 400, { error: 'prompt is empty' });
            const n = Number(body.n);
            if (!Number.isInteger(n) || n < 1 || n > 4096) return send(res, 400, { error: 'n must be an integer 1-4096' });
            const id = `run-${++seq}`;
            runs.push({
              id, name, model: body.model, prompt: String(body.prompt), n, raw: !!body.raw,
              started: Date.now(), fail: /\bfail\b/i.test(String(body.prompt)), registered: false,
            });
            return send(res, 200, { id });
          }
          return send(res, 404, { error: 'not found' });
        } catch (e) {
          return send(res, 500, { error: String(e) });
        }
      });
    },
  };
}
