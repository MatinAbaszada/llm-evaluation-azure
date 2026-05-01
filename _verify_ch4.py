"""Recompute every number cited in Chapter 4 and print for visual diff."""
import sys, json, statistics
sys.path.insert(0, '.')
import visualize as v

DS = v.DATASETS
LAM_L, LAM_E = 0.01, 1.0


def reward(r):
    return -((r.get('cost_usd', 0) or 0)
             + LAM_L * (r.get('elapsed_s', 0) or 0)
             + LAM_E * (1 - r.get('is_correct', 0)))


def macro(per_ds_vals):
    vals = [x for x in per_ds_vals if x is not None]
    return sum(vals) / len(vals) if vals else None


def per_ds(folder, fn):
    out = {}
    for ds in DS:
        recs = v._load_records(folder, ds)
        out[ds] = fn(recs) if recs else None
    return out


# ============================================================================
print('=' * 80)
print('§4.1 STANDALONE')
print('=' * 80)
folders = v._find_model_folders()
print('Latest run per model:')
for m, f in sorted(folders.items()):
    print(f'  {m:14s} -> {f.name}')
print()

print(f'{"model":14s} {"acc%":>6s} {"cost m$":>9s} {"lat_mean":>9s} {"lat_med":>8s} {"out>120":>8s} {"max_s":>7s} {"reward":>8s}')
for m, f in sorted(folders.items()):
    accs = per_ds(f, lambda rs: 100 * sum(r.get('is_correct', 0) for r in rs) / len(rs))
    costs = per_ds(f, lambda rs: 1000 * sum((r.get('cost_usd', 0) or 0) for r in rs) / len(rs))
    lats = per_ds(f, lambda rs: sum((r.get('elapsed_s', 0) or 0) for r in rs) / len(rs))
    rews = per_ds(f, lambda rs: sum(reward(r) for r in rs) / len(rs))
    all_lats, n_out, mx = [], 0, 0.0
    for ds in DS:
        recs = v._load_records(f, ds)
        ls = [r.get('elapsed_s', 0) or 0 for r in recs]
        all_lats += ls
        n_out += sum(1 for x in ls if x > 120)
        mx = max(mx, max(ls) if ls else 0)
    print(f'{m:14s} {macro(accs.values()):6.2f} {macro(costs.values()):9.3f} '
          f'{statistics.mean(all_lats):9.2f} {statistics.median(all_lats):8.2f} '
          f'{n_out:8d} {mx:7.1f} {macro(rews.values()):+8.3f}')

print()
print('Per-dataset accuracies (for §4.1.1 sentences):')
for m, f in sorted(folders.items()):
    a = per_ds(f, lambda rs: 100 * sum(r.get('is_correct', 0) for r in rs) / len(rs))
    print(f'  {m:14s} HE={a["humaneval"]:.1f}  MBPP={a["mbpp"]:.1f}  MMLU={a["mmlu_pro"]:.1f}  GPQA={a["gpqa"]:.1f}  GSM={a["gsm8k"]:.1f}')

print()
print('Per-dataset costs m$ (for §4.1.3 GPQA column):')
for m, f in sorted(folders.items()):
    c = per_ds(f, lambda rs: 1000 * sum((r.get('cost_usd', 0) or 0) for r in rs) / len(rs))
    print(f'  {m:14s} HE={c["humaneval"]:.3f}  MBPP={c["mbpp"]:.3f}  MMLU={c["mmlu_pro"]:.3f}  GPQA={c["gpqa"]:.3f}  GSM={c["gsm8k"]:.3f}')

print()
print('Per-dataset mean latency (for §4.1.2 prose):')
for m, f in sorted(folders.items()):
    l = per_ds(f, lambda rs: sum((r.get('elapsed_s', 0) or 0) for r in rs) / len(rs))
    print(f'  {m:14s} HE={l["humaneval"]:.1f}  MBPP={l["mbpp"]:.1f}  MMLU={l["mmlu_pro"]:.1f}  GPQA={l["gpqa"]:.1f}  GSM={l["gsm8k"]:.1f}')

print()
print('MMLU-Pro completion-token check (gpt-4.1 vs gpt-4.1-mini):')
for m in ['gpt-4.1', 'gpt-4.1-mini']:
    f = folders[m]
    recs = v._load_records(f, 'mmlu_pro')
    toks = [r.get('completion_tokens', 0) for r in recs if r.get('completion_tokens') is not None]
    if toks:
        print(f'  {m:14s} mean completion_tokens on MMLU-Pro = {statistics.mean(toks):.1f}')

# ============================================================================
print()
print('=' * 80)
print('§4.2 CASCADE')
print('=' * 80)
cdata = v._load_cascade_data()
for k, s in sorted(cdata.items()):
    sm, lg, t = k
    print(f'{sm:14s} -> {lg:8s} T={t:2d}  esc={s["escalation_pct"]:5.2f}%  acc={s["macro_accuracy"]:5.2f}%  cost={s["avg_cost"]*1000:6.3f} m$  reward={s["avg_reward"]:+.4f}')

print()
print('§4.2.2 latency ranges (macro):')
import pathlib
for folder in v._find_cascade_folders():
    parsed = v._parse_cascade_name(folder.name)
    if not parsed: continue
    sm, lg, t = parsed
    lats = []
    for ds in DS:
        recs = v._load_records(folder, ds)
        if recs:
            lats.append(sum((r.get('elapsed_s', 0) or 0) for r in recs) / len(recs))
    print(f'  {sm:14s} -> {lg:8s} T={t}  macro_latency={macro(lats):.2f}s')

print()
print('§4.2.2 GPQA cost for gpt-5.4-mini -> gpt-5.4 at T=60 vs T=90:')
for t in (60, 90):
    folder = v.OPT_RESULTS_DIR / f'cascade__small-gpt54mini__large-gpt54__T{t}'
    recs = v._load_records(folder, 'gpqa')
    esc = sum(1 for r in recs if r.get('escalated'))
    cost_mean = 1000 * sum((r.get('cost_usd', 0) or 0) for r in recs) / len(recs) if recs else None
    print(f'  T={t}: gpqa_cost={cost_mean:.3f} m$  esc_rate={100*esc/len(recs):.2f}%  n={len(recs)}')

# ============================================================================
print()
print('=' * 80)
print('§4.3 ROUTER')
print('=' * 80)
rdata = v._load_router_data()
for k, s in sorted(rdata.items()):
    rtr, sm, lg = k
    print(f'rtr={rtr:14s} {sm:14s} -> {lg:8s}  acc={s["macro_accuracy"]:5.2f}%  cost={s["avg_cost"]*1000:6.3f} m$  lat={s["avg_latency"]:5.2f}s  reward={s["avg_reward"]:+.4f}  pct_large={s["pct_large"]:.2f}%')
    print(f'    per-ds pct_large: {dict((k, round(v_, 1)) for k, v_ in s["ds_pct_large"].items())}')
    print(f'    per-ds accuracy : {dict((k, round(v_, 1)) for k, v_ in s["ds_accuracy"].items())}')

print()
print('GPQA router cost (gpt-5.4-mini, 5.4-mini -> 5.4):')
folder = v.OPT_RESULTS_DIR / 'router__rtr-gpt54mini__small-gpt54mini__large-gpt54'
recs = v._load_records(folder, 'gpqa')
print(f'  cost={1000*sum((r.get("cost_usd",0) or 0) for r in recs)/len(recs):.3f} m$  n={len(recs)}')

# ============================================================================
print()
print('=' * 80)
print('§4.4 SELF-CONSISTENCY')
print('=' * 80)
sdata = v._load_selfcons_data()
for k, s in sorted(sdata.items()):
    m, n = k
    print(f'{m:14s} N={n}  acc={s["macro_accuracy"]:5.2f}%  cost={s["avg_cost"]*1000:6.3f} m$  lat={s["avg_latency"]:5.2f}s  reward={s["avg_reward"]:+.4f}  agreement={s["agreement_rate"]:.2f}%')
    print(f'    per-ds accuracy : {dict((k, round(v_, 1)) for k, v_ in s["ds_accuracy"].items())}')
    print(f'    per-ds agreement: {dict((k, round(v_, 1)) for k, v_ in s["ds_agreement"].items())}')
    print(f'    per-ds cost m$  : {dict((k, round(v_*1000, 3)) for k, v_ in s["ds_cost"].items())}')
