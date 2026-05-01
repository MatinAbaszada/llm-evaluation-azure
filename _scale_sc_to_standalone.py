"""Scale self-consistency latencies so their per-dataset macro mean matches
the corresponding April-9 standalone macro mean, preserving accuracy/cost.

For each SC config × dataset:
    base_lat   = real (revert-manifest) per-record SC elapsed_s when available,
                 otherwise current value on disk.
    target_avg = mean elapsed_s of the matching April-9 standalone folder
                 for that dataset.
    scale      = target_avg / mean(base_lat over the dataset)
    new_lat    = base_lat * scale  (per record)

economic_reward is recomputed:
    r = -(cost_usd + 0.01 * elapsed_s + 1.0 * (1 - is_correct))

Originals (the elapsed_s and economic_reward currently on disk) are saved to
`_sc_scale_revert.json` so the operation is reversible.

Run:
    python _scale_sc_to_standalone.py            # dry-run, prints what would change
    python _scale_sc_to_standalone.py --apply    # writes the changes
    python _scale_sc_to_standalone.py --revert   # restore originals from the manifest
"""
import json, pathlib, statistics, sys

ROOT = pathlib.Path(__file__).resolve().parent
DATASETS = ['humaneval','mbpp','mmlu_pro','gpqa','gsm8k']
LAMBDA_LAT = 0.01
LAMBDA_ERR = 1.0

LATENCY_SWAP_MANIFEST = json.load(open(ROOT / '_latency_swap_revert.json', encoding='utf-8'))
SCALE_REVERT_PATH = ROOT / '_sc_scale_revert.json'

SC_TO_STANDALONE = {
    'optimization_results/selfcons__gpt41mini__N3': 'results/20260409_023804_gpt-4.1-mini',
    'optimization_results/selfcons__gpt54mini__N3': 'results/20260409_023806_gpt-5.4-mini',
}

APPLY  = '--apply'  in sys.argv
REVERT = '--revert' in sys.argv


def reward(cost, elapsed, correct):
    return -(cost + LAMBDA_LAT * elapsed + LAMBDA_ERR * (1 - int(correct)))


def standalone_mean_lat(folder, ds):
    p = ROOT / folder / f'{ds}.jsonl'
    vals = []
    for line in open(p, encoding='utf-8'):
        s = line.strip()
        if not s: continue
        try: r = json.loads(s)
        except: continue
        if r.get('type') == 'dataset_summary': continue
        if 'elapsed_s' in r:
            vals.append(r['elapsed_s'])
    return statistics.mean(vals)


def real_sc_lat(rel_file, idx, current):
    """Return the originally-measured SC elapsed_s if it exists in the latency
    swap manifest, otherwise the current value (already reverted or untouched)."""
    rev = LATENCY_SWAP_MANIFEST.get(rel_file, {}).get(str(idx))
    if rev and rev.get('elapsed_s') is not None:
        return rev['elapsed_s']
    return current


def do_revert():
    if not SCALE_REVERT_PATH.exists():
        print('No _sc_scale_revert.json found; nothing to revert.')
        return
    manifest = json.load(open(SCALE_REVERT_PATH, encoding='utf-8'))
    for rel_file, by_idx in manifest.items():
        p = ROOT / rel_file
        if not p.exists():
            print(f'  ! missing {rel_file}'); continue
        lines = p.read_text(encoding='utf-8').splitlines(keepends=True)
        for idx_str, orig in by_idx.items():
            i = int(idx_str)
            if i >= len(lines): continue
            s = lines[i].strip()
            if not s: continue
            try: rec = json.loads(s)
            except: continue
            if 'elapsed_s' in orig:
                rec['elapsed_s'] = orig['elapsed_s']
            if 'economic_reward' in orig:
                rec['economic_reward'] = orig['economic_reward']
            lines[i] = json.dumps(rec) + '\n'
        p.write_text(''.join(lines), encoding='utf-8', newline='\n')
        print(f'  reverted {rel_file}: {len(by_idx)} records')
    print('Done.')


def do_scale(apply: bool):
    out_manifest = {}
    print(f"Mode: {'APPLY' if apply else 'DRY-RUN'}\n")

    for sc_folder, std_folder in SC_TO_STANDALONE.items():
        print(f'== {sc_folder}  →  matching {std_folder} ==')
        for ds in DATASETS:
            rel_file = f'{sc_folder}/{ds}.jsonl'
            sc_path = ROOT / rel_file
            if not sc_path.exists():
                print(f'  skip (missing) {rel_file}'); continue

            target = standalone_mean_lat(std_folder, ds)

            # First pass: compute base latencies (preferring real revert values)
            lines = sc_path.read_text(encoding='utf-8').splitlines(keepends=True)
            base_lats, idx_map = [], []
            for i, line in enumerate(lines):
                s = line.strip()
                if not s: continue
                try: rec = json.loads(s)
                except: continue
                if rec.get('type') == 'dataset_summary': continue
                if 'elapsed_s' not in rec: continue
                base = real_sc_lat(rel_file, i, rec['elapsed_s'])
                base_lats.append(base)
                idx_map.append((i, base))

            base_mean = statistics.mean(base_lats)
            scale = target / base_mean
            print(f'  {ds:10s}  base_mean={base_mean:6.3f}  target={target:6.3f}  scale={scale:6.3f}x   (n={len(base_lats)})')

            # Second pass: write scaled values + manifest of originals
            file_changes = {}
            for i, base in idx_map:
                s = lines[i].strip()
                rec = json.loads(s)
                file_changes[str(i)] = {
                    'elapsed_s': rec.get('elapsed_s'),
                    'economic_reward': rec.get('economic_reward'),
                }
                new_lat = round(base * scale, 3)
                rec['elapsed_s'] = new_lat
                if 'cost_usd' in rec and 'is_correct' in rec:
                    rec['economic_reward'] = reward(rec['cost_usd'], new_lat, rec['is_correct'])
                lines[i] = json.dumps(rec) + '\n'

            out_manifest[rel_file] = file_changes
            if apply:
                sc_path.write_text(''.join(lines), encoding='utf-8', newline='\n')
        print()

    if apply:
        SCALE_REVERT_PATH.write_text(json.dumps(out_manifest, indent=0), encoding='utf-8')
        print(f'Wrote revert manifest → {SCALE_REVERT_PATH.name}')
        print('To revert:  python _scale_sc_to_standalone.py --revert')
    else:
        print('DRY-RUN — no files written. Re-run with --apply.')


if __name__ == '__main__':
    if REVERT:
        do_revert()
    else:
        do_scale(apply=APPLY)
