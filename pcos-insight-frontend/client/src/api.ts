export type Field = { name: string; label: string; tier: string; choices: number[] | null;
  min_observed: number; max_observed: number };
export type Stage = { id: string; title: string; model: string; n_features: number;
  threshold: number; fields: Field[]; cv_metrics: Record<string, number> };
export type Schema = { stages: Stage[]; evaluation_label: string; selection_rule: string;
  versions: Record<string, string>; version_warnings: string[]; fingerprint: string; scope: string };
export type Prediction = { stage: string; model: string; score: number; threshold: number;
  screening_class: string; warnings: string[]; fingerprint: string };
export type Contribution = { feature: string; value: number; contribution_pp: number };
export type Explanation = { base_score: number; score: number; contributions: Contribution[]; fingerprint: string };
export type Comparison = { analysis: string; feature_set: string; model: string; [key: string]: string | number | null };
export async function api<T>(url: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/${url}`, { method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body), signal });
  let data;
  try { data = await response.json(); } catch { throw new Error('The model service did not return JSON. Check that the Python service is running.'); }
  if (!response.ok) throw new Error(data.error || `Model service error (${response.status}).`);
  return data;
}
