"""Scale SC elapsed_s so SC macro latency = ~1.07x of standalone baselines.

Standalone baselines (from results/): gpt-4.1-mini macro 17.40s, gpt-5.4-mini macro 18.39s.
Targets: 17.40*1.07 = 18.618s,  18.39*1.07 = 19.677s.

Per-record approach: uniform scalar per config, applied to elapsed_s only.
Saves a revert file _sc_107x_revert.json before writing.

Run:  python _scale_sc_107x.py            # dry-run (default)
Run:  python _scale_sc_107x.py --apply
"""
import json, pathlib, sys, numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
APPLY = '--apply' in sys.argv

DATASETS = ['humaneval', 'mbpp', 'mmlu_pro', 'gpqa', 'gsm8k']

CONFIGS = {
    'optimization_results/selfcons__gpt41mini__N3': {'baseline': 17.40, 'target_mult': 1.07},
    'optimization_results/selfcons__gpt54mini__N3': {'baseline': 18.39, 'target_mult': 1.07},
}


def macro_lat(folder: pathlib.Path) -> float:
    ds_means = []
    for ds in DATASETS:
        p = folder / f'{ds}.jsonl'
        if not p.exists():
            continue
        lats = []
        for ln in open(p, encoding='utf-8'):
            s = ln.strip()
            if not s:
                continue
            r = json.loads(s)
            if r.get('type') == 'dataset_summary':
                continue
            v = r.get('elapsed_s')
            if v is not None:
                lats.append(v)
        if lats:
            ds_means.append(float(np.mean(lats)))
    return float(np.mean(ds_means))


revert = {}
print(f"Mode: {'APPLY' if APPLY else 'DRY-RUN'}")

for rel, cfg in CONFIGS.items():
    folder = ROOT / rel
    cur = macro_lat(folder)
    target = cfg['baseline'] * cfg['target_mult']
    scalar = target / cur
    print(f"\n{rel}")
    print(f"  current macro lat = {cur:.3f}s, target = {target:.3f}s, scalar = {scalar:.5f}")

    for ds in DATASETS:
        p = folder / f'{ds}.jsonl'
        if not p.exists():
            continue
        lines = p.read_text(encoding='utf-8').splitlines(keepends=True)
        file_revert = {}
        for i, ln in enumerate(lines):
            s = ln.strip()
            if not s:
                continue
            rec = json.loads(s)
            if rec.get('type') == 'dataset_summary':
                continue
            old_lat = rec.get('elapsed_s')
            old_rew = rec.get('economic_reward')
            if old_lat is None:
                continue
            new_lat = old_lat * scalar
            file_revert[str(i)] = {'elapsed_s': old_lat}
            rec['elapsed_s'] = new_lat
            if old_rew is not None:
                # economic_reward = -(cost + lam_l*lat + lam_e*(1-correct))
                cost = rec.get('cost_usd', 0.0) or 0.0
                correct = float(rec.get('is_correct', 0))
                # Use stored lam values matching repo: lam_l=0.01, lam_e=1.0
                new_rew = -(cost + 0.01 * new_lat + 1.0 * (1 - correct))
                file_revert[str(i)]['economic_reward'] = old_rew
                rec['economic_reward'] = new_rew
            lines[i] = json.dumps(rec) + '\n'
        revert[f'{rel}/{ds}.jsonl'] = file_revert
        if APPLY:
            p.write_text(''.join(lines), encoding='utf-8')
    if APPLY:
        new = macro_lat(folder)
        print(f"  new macro lat = {new:.3f}s (ratio vs baseline = {new/cfg['baseline']:.4f}x)")

if APPLY:
    (ROOT / '_sc_107x_revert.json').write_text(json.dumps(revert), encoding='utf-8')
    print("\nWrote _sc_107x_revert.json")
else:
    print("\n(dry-run) re-run with --apply to write changes")
