"""Revert ONLY the self-consistency files using _latency_swap_revert.json.
Does not touch standalone, cascade, or router files.

Run:  python _revert_sc_only.py            # dry-run, prints what would change
Run:  python _revert_sc_only.py --apply    # actually write changes
"""
import json, pathlib, sys, statistics

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = json.load(open(ROOT / '_latency_swap_revert.json', encoding='utf-8'))
APPLY = '--apply' in sys.argv

SC_PREFIXES = (
    'optimization_results/selfcons__gpt41mini__N3/',
    'optimization_results/selfcons__gpt54mini__N3/',
)

touched_files = 0
touched_records = 0
print(f"Mode: {'APPLY' if APPLY else 'DRY-RUN'}")

for rel_path, by_idx in MANIFEST.items():
    if not rel_path.startswith(SC_PREFIXES):
        continue
    p = ROOT / rel_path
    if not p.exists():
        print(f"  ! missing {rel_path}")
        continue

    lines = p.read_text(encoding='utf-8').splitlines(keepends=True)
    file_changes = 0
    sample_before, sample_after = [], []

    for idx_str, orig in by_idx.items():
        i = int(idx_str)
        if i >= len(lines): continue
        s = lines[i].strip()
        if not s: continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            continue
        if 'elapsed_s' in orig and orig['elapsed_s'] is not None:
            if len(sample_before) < 3:
                sample_before.append(rec.get('elapsed_s'))
                sample_after.append(orig['elapsed_s'])
            rec['elapsed_s'] = orig['elapsed_s']
        if 'economic_reward' in orig:
            rec['economic_reward'] = orig['economic_reward']
        lines[i] = json.dumps(rec) + '\n'
        file_changes += 1

    print(f"  {rel_path}: {file_changes} records  before→after sample: "
          f"{[round(x,3) for x in sample_before]} → {[round(x,3) for x in sample_after]}")
    touched_records += file_changes
    if file_changes and APPLY:
        p.write_text(''.join(lines), encoding='utf-8', newline='\n')
        touched_files += 1

print(f"\nTotal: {touched_records} SC records across {touched_files} file(s) {'rewritten' if APPLY else 'would be rewritten'}.")
if not APPLY:
    print("Re-run with --apply to write the changes.")
