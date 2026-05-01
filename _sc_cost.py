import json, pathlib, statistics
DATASETS = ['humaneval','mbpp','mmlu_pro','gpqa','gsm8k']

def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip() and 'cost_usd' in l]

def per_ds_cost(folder):
    out = {}
    for ds in DATASETS:
        recs = load(pathlib.Path(folder) / f'{ds}.jsonl')
        out[ds] = statistics.mean(r['cost_usd'] for r in recs) * 1000  # m$
    return out

sa41 = per_ds_cost('results/20260409_023804_gpt-4.1-mini')
sa54 = per_ds_cost('results/20260409_023806_gpt-5.4-mini')
sc41 = per_ds_cost('optimization_results/selfcons__gpt41mini__N3')
sc54 = per_ds_cost('optimization_results/selfcons__gpt54mini__N3')

print(f"{'dataset':<10} {'4.1m std':>10} {'SC 4.1m':>10} {'mult':>6}    {'5.4m std':>10} {'SC 5.4m':>10} {'mult':>6}")
for ds in DATASETS:
    print(f"{ds:<10} {sa41[ds]:10.3f} {sc41[ds]:10.3f} {sc41[ds]/sa41[ds]:5.2f}x   {sa54[ds]:10.3f} {sc54[ds]:10.3f} {sc54[ds]/sa54[ds]:5.2f}x")
m41 = statistics.mean(sa41.values()); m41sc = statistics.mean(sc41.values())
m54 = statistics.mean(sa54.values()); m54sc = statistics.mean(sc54.values())
print(f"{'Macro':<10} {m41:10.3f} {m41sc:10.3f} {m41sc/m41:5.2f}x   {m54:10.3f} {m54sc:10.3f} {m54sc/m54:5.2f}x")
