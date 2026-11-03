// Shareable URL state: ?t=trace&tok=index&view=lens&cmp=other&cmode=diff
import type { CompareMode, ViewId } from './types';

export interface UrlState {
  t: string | null;
  tok: number | null;
  view: ViewId | null;
  cmp: string | null;
  cmode: CompareMode | null;
}

const VIEWS: ViewId[] = ['route', 'lens', 'write', 'usage'];

export function readUrl(): UrlState {
  const q = new URLSearchParams(location.search);
  const tok = q.get('tok');
  const view = q.get('view') as ViewId | null;
  const cmode = q.get('cmode');
  return {
    t: q.get('t'),
    tok: tok != null && /^\d+$/.test(tok) ? Number(tok) : null,
    view: view && VIEWS.includes(view) ? view : null,
    cmp: q.get('cmp'),
    cmode: cmode === 'side' || cmode === 'diff' ? cmode : null,
  };
}

export function writeUrl(s: UrlState) {
  const q = new URLSearchParams();
  if (s.t) q.set('t', s.t);
  if (s.tok != null) q.set('tok', String(s.tok));
  if (s.view) q.set('view', s.view);
  if (s.cmp) q.set('cmp', s.cmp);
  if (s.cmp && s.cmode) q.set('cmode', s.cmode);
  const next = `${location.pathname}${q.size ? '?' + q.toString() : ''}${location.hash}`;
  if (next !== `${location.pathname}${location.search}${location.hash}`) history.replaceState(null, '', next);
}
