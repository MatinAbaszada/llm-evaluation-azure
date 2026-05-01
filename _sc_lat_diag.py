import json, pathlib, statistics, os
def load(p):
    out = []
    for l in open(p, encoding='utf-8'):
        r = json.loads(l)
        if 'elapsed_s' in r and 'task_id' in r:
            out.append(r)
    return out
DATASETS = ['humaneval','mbpp','mmlu_pro','gpqa','gsm8k']

print('=== SC selfcons__gpt41mini__N3 elapsed stats per dataset ===')
for ds in DATASETS:
    recs = load(pathlib.Path('optimization_results/selfcons__gpt41mini__N3')/f'{ds}.jsonl')
    el = [r['elapsed_s'] for r in recs]
    print(f'  {ds:10s} n={len(recs)} mean={statistics.mean(el):.3f} median={statistics.median(el):.3f} max={max(el):.3f} min={min(el):.3f}')

print()
print('=== April-9 standalone gpt-4.1-mini elapsed stats per dataset ===')
sa = sorted([d for d in os.listdir('results') if 'gpt-4.1-mini' in d])[-1]
print('folder:', sa)
for ds in DATASETS:
    recs = load(pathlib.Path('results')/sa/f'{ds}.jsonl')
    el = [r['elapsed_s'] for r in recs]
    print(f'  {ds:10s} n={len(recs)} mean={statistics.mean(el):.3f} median={statistics.median(el):.3f} max={max(el):.3f} min={min(el):.3f}')

print()
print('=== SC sample records (gpqa first 5) ===')
recs = load('optimization_results/selfcons__gpt41mini__N3/gpqa.jsonl')
for r in recs[:5]:
    tid = r.get('task_id','?')
    print(f"  task={tid[:35]:35s} elapsed={r['elapsed_s']:7.3f}  ts={r.get('timestamp')}  agree={r.get('vote_agreement')}  n_votes={r.get('n_votes')}")

print()
print('=== Latency >120s outlier counts ===')
for ds in DATASETS:
    sc_el = [r['elapsed_s'] for r in load(pathlib.Path('optimization_results/selfcons__gpt41mini__N3')/f'{ds}.jsonl')]
    sa_el = [r['elapsed_s'] for r in load(pathlib.Path('results')/sa/f'{ds}.jsonl')]
    print(f'  {ds:10s} SC outliers>120s={sum(1 for x in sc_el if x>120)}  Standalone outliers>120s={sum(1 for x in sa_el if x>120)}')

print()
print('=== Standalone elapsed without long-tail (excluding >120s) ===')
for ds in DATASETS:
    sa_el = [r['elapsed_s'] for r in load(pathlib.Path('results')/sa/f'{ds}.jsonl')]
    trimmed = [x for x in sa_el if x <= 120]
    print(f'  {ds:10s} trimmed_mean={statistics.mean(trimmed):.3f}  trimmed_median={statistics.median(trimmed):.3f}  kept={len(trimmed)}/{len(sa_el)}')
