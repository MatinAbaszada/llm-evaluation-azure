import json, pathlib, statistics

DATASETS = ['humaneval','mbpp','mmlu_pro','gpqa','gsm8k']
MANIFEST = json.load(open('_latency_swap_revert.json', encoding='utf-8'))

def load(p, revert_key=None):
    out = []
    rev = MANIFEST.get(revert_key, {}) if revert_key else {}
    for i, line in enumerate(open(p, encoding='utf-8')):
        s = line.strip()
        if not s: continue
        try: r = json.loads(s)
        except: continue
        if r.get('type') == 'dataset_summary': continue
        if 'elapsed_s' not in r: continue
        orig = rev.get(str(i))
        if orig and orig.get('elapsed_s') is not None:
            r['elapsed_s'] = orig['elapsed_s']
        out.append(r)
    return out

def per_ds_lat(folder, with_revert=False):
    out = {}
    for ds in DATASETS:
        rel = f'{folder}/{ds}.jsonl'
        recs = load(rel, revert_key=rel if with_revert else None)
        out[ds] = statistics.mean(r['elapsed_s'] for r in recs)
    return out

# Standalone (April-9, throttled big sweep)
sa41 = per_ds_lat('results/20260409_023804_gpt-4.1-mini')
sa54 = per_ds_lat('results/20260409_023806_gpt-5.4-mini')
# SC current (post-swap, clipped)
sc41_now = per_ds_lat('optimization_results/selfcons__gpt41mini__N3')
sc54_now = per_ds_lat('optimization_results/selfcons__gpt54mini__N3')
# SC reverted (real measurement)
sc41_real = per_ds_lat('optimization_results/selfcons__gpt41mini__N3', with_revert=True)
sc54_real = per_ds_lat('optimization_results/selfcons__gpt54mini__N3', with_revert=True)

print('=== gpt-4.1-mini latency (s) ===')
print(f"{'dataset':<10} {'standalone':>11} {'SC (now)':>10} {'SC (real)':>11} {'real/std':>10}")
for ds in DATASETS:
    print(f"{ds:<10} {sa41[ds]:11.2f} {sc41_now[ds]:10.3f} {sc41_real[ds]:11.3f} {sc41_real[ds]/sa41[ds]:10.2%}")
print(f"{'Macro':<10} {statistics.mean(sa41.values()):11.2f} {statistics.mean(sc41_now.values()):10.3f} {statistics.mean(sc41_real.values()):11.3f}")

print()
print('=== gpt-5.4-mini latency (s) ===')
print(f"{'dataset':<10} {'standalone':>11} {'SC (now)':>10} {'SC (real)':>11} {'real/std':>10}")
for ds in DATASETS:
    print(f"{ds:<10} {sa54[ds]:11.2f} {sc54_now[ds]:10.3f} {sc54_real[ds]:11.3f} {sc54_real[ds]/sa54[ds]:10.2%}")
print(f"{'Macro':<10} {statistics.mean(sa54.values()):11.2f} {statistics.mean(sc54_now.values()):10.3f} {statistics.mean(sc54_real.values()):11.3f}")
