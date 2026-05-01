"""Compute April-7 standalone stats from Results_Backup, reverting any swap.
Compare to April-9 stats currently used in Chapter 4.
"""
import json, pathlib, statistics

ROOT = pathlib.Path('.')
MANIFEST = json.load(open('_latency_swap_revert.json', encoding='utf-8'))
DATASETS = ['humaneval','mbpp','mmlu_pro','gpqa','gsm8k']
LAMBDA_LAT = 0.01

APR7 = {
    'gpt-4.1':       'Results_Backup/20260407_002311_gpt-4.1',
    'gpt-5.4-mini':  'Results_Backup/20260407_002312_gpt-5.4-mini',
    'gpt-4.1-mini':  'Results_Backup/20260407_002342_gpt-4.1-mini',
    'o3-mini':       'Results_Backup/20260407_002444_o3-mini',
    'gpt-5.4':       'Results_Backup/20260407_002645_gpt-5.4',
}
APR9 = {
    'gpt-4.1':       'results/20260409_022932_gpt-4.1',
    'gpt-5.4-mini':  'results/20260409_023806_gpt-5.4-mini',
    'gpt-4.1-mini':  'results/20260409_023804_gpt-4.1-mini',
    'o3-mini':       'results/20260409_024024_o3-mini',
    'gpt-5.4-pro':   'results/20260412_163647_gpt-5.4-pro',
}

def load_with_revert(rel_dir, ds):
    """Load records, applying revert if path is in manifest. The revert manifest
    references results/ paths, so for Results_Backup copies we have to map."""
    p = ROOT / rel_dir / f'{ds}.jsonl'
    if not p.exists():
        return []
    # Try the corresponding results/ path for revert lookup
    candidate_keys = [
        f'{rel_dir}/{ds}.jsonl',
        f'{rel_dir}/{ds}.jsonl'.replace('Results_Backup','results'),
    ]
    revert_map = {}
    for k in candidate_keys:
        if k in MANIFEST:
            revert_map = MANIFEST[k]
            break
    out = []
    for i, line in enumerate(open(p, encoding='utf-8')):
        s = line.strip()
        if not s: continue
        rec = json.loads(s)
        if rec.get('type')=='dataset_summary': continue
        orig = revert_map.get(str(i))
        if orig:
            if orig.get('elapsed_s') is not None:
                rec['elapsed_s'] = orig['elapsed_s']
            if 'economic_reward' in orig:
                rec['economic_reward'] = orig['economic_reward']
        out.append(rec)
    return out

def stats(folder_map, label):
    print(f'\n=== {label} ===')
    print(f'{"model":<14}  {"n":>4}  {"acc%":>6}  {"cost(m$)":>9}  {"lat(s)":>7}  {"med":>6}  {"max":>7}  {"reward":>8}')
    out = {}
    for model, folder in folder_map.items():
        per_ds = {'acc':[], 'cost':[], 'lat':[], 'rew':[], 'lat_all':[], 'n':0}
        for ds in DATASETS:
            recs = load_with_revert(folder, ds)
            if not recs: continue
            accs = [int(bool(r.get('is_correct'))) for r in recs]
            costs = [r.get('cost_usd',0) for r in recs]
            lats = [r['elapsed_s'] for r in recs if 'elapsed_s' in r]
            rews = [r.get('economic_reward',0) for r in recs]
            per_ds['acc'].append(statistics.mean(accs))
            per_ds['cost'].append(statistics.mean(costs))
            per_ds['lat'].append(statistics.mean(lats) if lats else 0)
            per_ds['rew'].append(statistics.mean(rews))
            per_ds['lat_all'].extend(lats)
            per_ds['n'] += len(recs)
        if not per_ds['acc']: continue
        macro_acc = statistics.mean(per_ds['acc'])*100
        macro_cost = statistics.mean(per_ds['cost'])*1000
        macro_lat = statistics.mean(per_ds['lat'])
        macro_rew = statistics.mean(per_ds['rew'])
        med = statistics.median(per_ds['lat_all'])
        mx = max(per_ds['lat_all'])
        print(f'{model:<14}  {per_ds["n"]:>4}  {macro_acc:>6.2f}  {macro_cost:>9.3f}  {macro_lat:>7.2f}  {med:>6.2f}  {mx:>7.1f}  {macro_rew:>8.3f}')
        out[model] = (macro_acc, macro_cost, macro_lat, macro_rew)
    return out

a7 = stats(APR7, 'April-7 standalone (Results_Backup, swap reverted)')
a9 = stats(APR9, 'April-9 standalone (results/, swap reverted)')

# SC vs April-7 standalone
print('\n=== SC vs April-7 standalone (using SC reverted latencies) ===')
SC = {'gpt-4.1-mini':'optimization_results/selfcons__gpt41mini__N3',
      'gpt-5.4-mini':'optimization_results/selfcons__gpt54mini__N3'}
for model, folder in SC.items():
    per_ds = {'acc':[], 'cost':[], 'lat':[], 'rew':[]}
    for ds in DATASETS:
        recs = load_with_revert(folder, ds)
        if not recs: continue
        accs = [int(bool(r.get('is_correct'))) for r in recs]
        costs = [r.get('cost_usd',0) for r in recs]
        lats = [r['elapsed_s'] for r in recs if 'elapsed_s' in r]
        rews = [r.get('economic_reward',0) for r in recs]
        per_ds['acc'].append(statistics.mean(accs))
        per_ds['cost'].append(statistics.mean(costs))
        per_ds['lat'].append(statistics.mean(lats) if lats else 0)
        per_ds['rew'].append(statistics.mean(rews))
    sc_acc = statistics.mean(per_ds['acc'])*100
    sc_cost = statistics.mean(per_ds['cost'])*1000
    sc_lat = statistics.mean(per_ds['lat'])
    sc_rew = statistics.mean(per_ds['rew'])
    sa_acc, sa_cost, sa_lat, sa_rew = a7[model]
    print(f'  {model}:')
    print(f'    SC          acc={sc_acc:6.2f}  cost={sc_cost:6.3f}  lat={sc_lat:5.2f}  rew={sc_rew:7.3f}')
    print(f'    standalone  acc={sa_acc:6.2f}  cost={sa_cost:6.3f}  lat={sa_lat:5.2f}  rew={sa_rew:7.3f}')
    print(f'    delta       acc={sc_acc-sa_acc:+6.2f}  cost={sc_cost-sa_cost:+6.3f}  lat={sc_lat-sa_lat:+5.2f}  rew={sc_rew-sa_rew:+7.3f}')
