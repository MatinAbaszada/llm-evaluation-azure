"""Compute post-revert latency means for optimization files (cascade/router/SC)
and compare them against the current chapter numbers, WITHOUT modifying any
file on disk. Uses _latency_swap_revert.json to reconstruct original elapsed_s.
"""
import json, pathlib, statistics
from collections import defaultdict

ROOT = pathlib.Path('.')
MANIFEST = json.load(open('_latency_swap_revert.json', encoding='utf-8'))
DATASETS = ['humaneval','mbpp','mmlu_pro','gpqa','gsm8k']
LAMBDA_LAT = 0.01

def load_with_revert(rel_path: str):
    """Yield records with elapsed_s/economic_reward restored from manifest if present."""
    p = ROOT / rel_path
    if not p.exists():
        return []
    out = []
    revert_map = MANIFEST.get(rel_path, {})
    for i, line in enumerate(open(p, encoding='utf-8')):
        s = line.strip()
        if not s:
            continue
        rec = json.loads(s)
        if rec.get('type') == 'dataset_summary':
            continue
        orig = revert_map.get(str(i))
        if orig:
            if orig.get('elapsed_s') is not None:
                rec['elapsed_s'] = orig['elapsed_s']
            if 'economic_reward' in orig:
                rec['economic_reward'] = orig['economic_reward']
        out.append(rec)
    return out

def macro_stats(folder: str):
    """Return (macro_lat, macro_cost, macro_acc, macro_reward) using post-revert data."""
    per_ds_lat, per_ds_cost, per_ds_acc, per_ds_rew = {}, {}, {}, {}
    for ds in DATASETS:
        rel = f'optimization_results/{folder}/{ds}.jsonl'
        recs = load_with_revert(rel)
        if not recs:
            continue
        lats = [r['elapsed_s'] for r in recs if 'elapsed_s' in r]
        costs = [r.get('cost_usd',0) for r in recs]
        accs = [int(bool(r.get('is_correct'))) for r in recs]
        rews = [r.get('economic_reward',0) for r in recs]
        per_ds_lat[ds] = statistics.mean(lats) if lats else 0
        per_ds_cost[ds] = statistics.mean(costs)*1000  # m$
        per_ds_acc[ds] = statistics.mean(accs)*100
        per_ds_rew[ds] = statistics.mean(rews)
    if not per_ds_lat:
        return None
    return (
        statistics.mean(per_ds_lat.values()),
        statistics.mean(per_ds_cost.values()),
        statistics.mean(per_ds_acc.values()),
        statistics.mean(per_ds_rew.values()),
        per_ds_lat,
    )

# Now also compute current (pre-revert) for comparison
def macro_stats_current(folder: str):
    per_ds_lat, per_ds_rew = {}, {}
    for ds in DATASETS:
        p = ROOT / f'optimization_results/{folder}/{ds}.jsonl'
        if not p.exists(): continue
        recs = []
        for line in open(p, encoding='utf-8'):
            s=line.strip()
            if not s: continue
            r=json.loads(s)
            if r.get('type')=='dataset_summary': continue
            recs.append(r)
        lats = [r['elapsed_s'] for r in recs if 'elapsed_s' in r]
        rews = [r.get('economic_reward',0) for r in recs]
        per_ds_lat[ds] = statistics.mean(lats) if lats else 0
        per_ds_rew[ds] = statistics.mean(rews)
    if not per_ds_lat: return None
    return statistics.mean(per_ds_lat.values()), statistics.mean(per_ds_rew.values())

CONFIGS = [
    # cascades
    'cascade__small-gpt41mini__large-gpt41__T60',
    'cascade__small-gpt41mini__large-gpt41__T75',
    'cascade__small-gpt41mini__large-gpt41__T90',
    'cascade__small-gpt41mini__large-gpt54__T60',
    'cascade__small-gpt41mini__large-gpt54__T75',
    'cascade__small-gpt41mini__large-gpt54__T90',
    'cascade__small-gpt54mini__large-gpt41__T60',
    'cascade__small-gpt54mini__large-gpt41__T75',
    'cascade__small-gpt54mini__large-gpt41__T90',
    'cascade__small-gpt54mini__large-gpt54__T60',
    'cascade__small-gpt54mini__large-gpt54__T75',
    'cascade__small-gpt54mini__large-gpt54__T90',
    # routers
    'router__rtr-gpt41mini__small-gpt41mini__large-gpt41',
    'router__rtr-gpt41mini__small-gpt54mini__large-gpt54',
    'router__rtr-gpt54mini__small-gpt41mini__large-gpt41',
    'router__rtr-gpt54mini__small-gpt54mini__large-gpt54',
    # SC
    'selfcons__gpt41mini__N3',
    'selfcons__gpt54mini__N3',
]

print(f'{"config":70s} {"cur_lat":>9s} {"new_lat":>9s} {"cur_rew":>9s} {"new_rew":>9s}')
for cfg in CONFIGS:
    cur = macro_stats_current(cfg)
    new = macro_stats(cfg)
    if cur is None or new is None:
        print(f'{cfg:70s}  (missing)')
        continue
    cur_lat, cur_rew = cur
    new_lat, new_cost, new_acc, new_rew, _per = new
    print(f'{cfg:70s} {cur_lat:9.3f} {new_lat:9.3f} {cur_rew:9.3f} {new_rew:9.3f}')
