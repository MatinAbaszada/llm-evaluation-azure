import json, pathlib, statistics
DATASETS = ['humaneval','mbpp','mmlu_pro','gpqa','gsm8k']
for cfg in ['selfcons__gpt41mini__N3','selfcons__gpt54mini__N3']:
    print(f'=== {cfg} (after scaling) ===')
    per_lat, per_acc, per_rew, per_cost = [], [], [], []
    for ds in DATASETS:
        recs = [json.loads(l) for l in open(pathlib.Path('optimization_results')/cfg/f'{ds}.jsonl', encoding='utf-8') if l.strip() and 'elapsed_s' in l]
        lat  = statistics.mean(r['elapsed_s'] for r in recs)
        acc  = statistics.mean(int(bool(r.get('is_correct'))) for r in recs) * 100
        rew  = statistics.mean(r['economic_reward'] for r in recs)
        cost = statistics.mean(r['cost_usd'] for r in recs) * 1000
        per_lat.append(lat); per_acc.append(acc); per_rew.append(rew); per_cost.append(cost)
        print(f'  {ds:10s} acc={acc:6.2f}  cost={cost:6.3f}m$  lat={lat:6.2f}s  rew={rew:7.3f}')
    print(f'  MACRO      acc={statistics.mean(per_acc):6.2f}  cost={statistics.mean(per_cost):6.3f}m$  lat={statistics.mean(per_lat):6.2f}s  rew={statistics.mean(per_rew):7.3f}')
    print()
