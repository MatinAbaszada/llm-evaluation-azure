"""
One-off variant of chartUNI_unified_best_strategy_heatmap requested by the
professor: keep the macro decision map exactly as produced by visualize.py,
then overwrite ~40 % of the cells closest to the main diagonal with a router
colour so we can see how an expanded router band would look.

Run:
    $env:PYTHONIOENCODING="utf-8"
    & "c:/Repos/Bachelor Thesis Project/.venv/Scripts/python.exe" chartUNI_router_diagonal_variant.py
Output: charts/chartUNI_unified_best_strategy_heatmap_router_diagonal.png
"""

from __future__ import annotations

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import visualize as v


# Fraction of the main-diagonal LENGTH that will be overridden with the
# router colour (band starts at the bottom-left corner where the original
# router region already sits, and widens as it moves toward the centre).
DIAGONAL_LENGTH_FRACTION = 0.35
# Maximum half-width (in cells) at the WIDE end of the band (the corner side).
# The band starts at this half-width and tapers smoothly down to 0 at the far
# end, giving a long triangular wedge whose tip fades to a single cell.
DIAGONAL_BAND_MAX_HALF_WIDTH = 6


def _build_unified_macro_grid():
    """Replicate the option-set + macro grid construction from
    chart_unified_best_strategy_heatmap so we can reuse it here."""
    model_folders = v._find_model_folders()
    if not model_folders:
        raise SystemExit("No standalone model folders found.")

    cascade_data = v._load_cascade_data()
    sc_data      = v._load_selfcons_data()
    router_data  = v._load_router_data()

    standalone_models = sorted(model_folders.keys())

    cascade_folders: dict[tuple, pathlib.Path] = {}
    for folder in v._find_cascade_folders():
        parsed = v._parse_cascade_name(folder.name)
        if parsed is not None:
            cascade_folders[parsed] = folder
    cascade_configs = [
        cfg for cfg in sorted(cascade_folders.keys())
        if any(
            r.get("escalated", False)
            for stem in v.DATASETS
            for r in v._load_records(cascade_folders[cfg], stem)
        )
    ]
    sc_configs     = sorted((sc_data or {}).keys())
    router_configs = sorted((router_data or {}).keys())

    sa_records = {
        m: {stem: v._load_records(folder, stem) for stem in v.DATASETS}
        for m, folder in model_folders.items()
    }
    cascade_records = {
        cfg: {stem: v._load_records(cascade_folders[cfg], stem) for stem in v.DATASETS}
        for cfg in cascade_configs
    }
    sc_records = {
        cfg: {stem: v._load_records(v._find_selfcons_folder(cfg), stem) for stem in v.DATASETS}
        for cfg in sc_configs
    }
    sc_base_lats = {cfg: v._base_avg_latency(cfg[0], model_folders) for cfg in sc_configs}
    router_records = {
        cfg: {stem: v._load_records(v._find_router_folder(cfg), stem) for stem in v.DATASETS}
        for cfg in router_configs
    }

    sa_colors = v._model_color_map(standalone_models)
    cascade_palette = [
        "#B71C1C", "#D32F2F", "#EF5350",
        "#BF360C", "#E64A19", "#FF7043",
        "#4E342E", "#6D4C41", "#A1887F",
        "#880E4F", "#C2185B", "#F06292",
    ]
    sc_palette     = ["#00695C", "#00ACC1"]
    router_palette = ["#4527A0", "#00838F", "#AD1457", "#FFB300"]

    cascade_colors = {cfg: cascade_palette[i % len(cascade_palette)]
                      for i, cfg in enumerate(cascade_configs)}
    sc_colors      = {cfg: sc_palette[i % len(sc_palette)]
                      for i, cfg in enumerate(sc_configs)}
    router_colors  = {cfg: router_palette[i % len(router_palette)]
                      for i, cfg in enumerate(router_configs)}

    lam_errors = v.HEATMAP_LAM_ERROR
    lam_lats   = v.HEATMAP_LAM_LAT

    options_macro = []
    for m in standalone_models:
        s = v._option_stats_macro(sa_records[m])
        if s is not None:
            options_macro.append((("standalone", m), *s))
    for cfg in cascade_configs:
        s = v._option_stats_macro(cascade_records[cfg])
        if s is not None:
            options_macro.append((("cascade", cfg), *s))
    for cfg in sc_configs:
        s = v._option_stats_macro_sc(sc_records[cfg], sc_base_lats[cfg])
        if s is not None:
            options_macro.append((("sc", cfg), *s))
    for cfg in router_configs:
        s = v._option_stats_macro(router_records[cfg])
        if s is not None:
            options_macro.append((("router", cfg), *s))

    grid = v._build_best_grid(options_macro, lam_errors, lam_lats)

    all_keys = (
        [("standalone", m) for m in standalone_models] +
        [("cascade", cfg) for cfg in cascade_configs] +
        [("sc",      cfg) for cfg in sc_configs] +
        [("router",  cfg) for cfg in router_configs]
    )
    key_to_color = {}
    for k in all_keys:
        fam, ident = k
        if fam == "standalone":
            key_to_color[k] = sa_colors[ident]
        elif fam == "cascade":
            key_to_color[k] = cascade_colors[ident]
        elif fam == "sc":
            key_to_color[k] = sc_colors[ident]
        elif fam == "router":
            key_to_color[k] = router_colors[ident]

    return {
        "grid": grid,
        "all_keys": all_keys,
        "key_to_color": key_to_color,
        "router_configs": router_configs,
        "lam_errors": lam_errors,
        "lam_lats": lam_lats,
    }


def _override_diagonal(grid: np.ndarray,
                       router_key,
                       length_fraction: float,
                       max_band_half_width: int) -> np.ndarray:
    """Override a widening band along the bottom-left part of the diagonal.

    - Starts at the (0,0) corner with a single cell.
    - Grows linearly to ``max_band_half_width`` half-width at the far end.
    - Covers the central ``length_fraction`` of the diagonal length.
    - On every column the band touches, any non gpt-5.4 cells sitting between
      the band and the existing gpt-5.4 region are also recoloured router so
      there is no other colour wedged between them.
    """
    nrows, ncols = grid.shape
    if nrows == 0 or ncols == 0:
        return grid.copy()

    diag_len = min(nrows, ncols)
    n_diag = max(1, int(round(length_fraction * diag_len)))

    # Identify the gpt-5.4 standalone cells in the ORIGINAL grid so we can
    # close the gap between the router band and the gpt-5.4 region.
    gpt54_mask = np.zeros((nrows, ncols), dtype=bool)
    # Any standalone cells (used to close gaps on the lower side of the band
    # so no other colour sits between router and the standalone region).
    standalone_mask = np.zeros((nrows, ncols), dtype=bool)
    # Identify pre-existing router cells in the original grid so we can also
    # close the gap on the corner side (no SC patches between the original
    # router blob and the start of the new widening band).
    orig_router_mask = np.zeros((nrows, ncols), dtype=bool)
    for i in range(nrows):
        for j in range(ncols):
            k = grid[i, j]
            if k is None:
                continue
            if k[0] == "standalone":
                standalone_mask[i, j] = True
                if k[1] == "gpt-5.4":
                    gpt54_mask[i, j] = True
            elif k[0] == "router":
                orig_router_mask[i, j] = True

    out = grid.copy()
    band_mask = np.zeros((nrows, ncols), dtype=bool)

    # Build a wedge that sits ON TOP of the gpt-5.4 standalone region:
    # its flat base is the upper boundary of gpt-5.4, and its point juts
    # upward into the SC area. The "thickness" (number of rows above the
    # gpt-5.4 boundary) tapers linearly from `max_band_half_width*2` at the
    # corner side down to 0 at the centre side.
    n_cols_band = max(1, int(round(length_fraction * ncols)))
    max_height = 2 * max_band_half_width
    for jj in range(n_cols_band):
        if n_cols_band > 1:
            t = jj / (n_cols_band - 1)      # 0 at the corner, 1 at the tip
            height = int(round(max_height * (1.0 - t)))
        else:
            height = 0
        if height <= 0:
            continue
        # Find the top of the gpt-5.4 region in this column (highest row index
        # that is gpt-5.4). If none, fall back to row 0 (corner side).
        gpt_rows = np.where(gpt54_mask[:, jj])[0]
        if gpt_rows.size == 0:
            base_row = 0
        else:
            base_row = int(gpt_rows.max())
        # Paint `height` rows starting just above the gpt-5.4 boundary.
        for h in range(1, height + 1):
            r = base_row + h
            if 0 <= r < nrows:
                out[r, jj] = router_key
                band_mask[r, jj] = True

    # No further per-column gap-fill: keep the triangular silhouette intact.

    # Close the gap on the corner side: for every column where the original
    # grid already contains router cells, fill anything between those cells
    # and the new band so the start of the band is continuous (no SC patches
    # remaining in between).
    new_router_mask = np.zeros((nrows, ncols), dtype=bool)
    for i in range(nrows):
        for j in range(ncols):
            k = out[i, j]
            if k is not None and k[0] == "router":
                new_router_mask[i, j] = True
    for j in range(ncols):
        rr = np.where(new_router_mask[:, j])[0]
        if rr.size == 0:
            continue
        lo, hi = int(rr.min()), int(rr.max())
        for r in range(lo, hi + 1):
            out[r, j] = router_key
    for i in range(nrows):
        cc = np.where(new_router_mask[i, :])[0]
        if cc.size == 0:
            continue
        lo, hi = int(cc.min()), int(cc.max())
        for c in range(lo, hi + 1):
            # Don't overwrite standalone cells — we want the band to TOUCH
            # the standalone region but not eat into it.
            if not standalone_mask[i, c]:
                out[i, c] = router_key

    # Final safety pass: restore EVERY standalone cell from the original grid
    # so the standalone regions keep their original colours and the router
    # band only overlays cascade / SC / empty cells.
    for i in range(nrows):
        for j in range(ncols):
            if standalone_mask[i, j]:
                out[i, j] = grid[i, j]

    # Remove any leftover ORIGINAL router cells that are no longer connected
    # to the new wedge — restore them to their original colour from `grid`
    # only if they would otherwise leave a disconnected speck. Concretely:
    # if a cell was originally router but is not part of the contiguous wedge
    # (band_mask + per-column fill), revert it to its original value… except
    # the original value IS router, so instead repaint specks of size <=2
    # that are not adjacent to the wedge with the surrounding SC colour.
    final_router_mask = np.zeros((nrows, ncols), dtype=bool)
    for i in range(nrows):
        for j in range(ncols):
            k = out[i, j]
            if k is not None and k[0] == "router":
                final_router_mask[i, j] = True

    # Connected-components flood fill (4-connectivity) on final_router_mask.
    visited = np.zeros((nrows, ncols), dtype=bool)
    for si in range(nrows):
        for sj in range(ncols):
            if not final_router_mask[si, sj] or visited[si, sj]:
                continue
            stack = [(si, sj)]
            comp = []
            while stack:
                i, j = stack.pop()
                if i < 0 or i >= nrows or j < 0 or j >= ncols:
                    continue
                if visited[i, j] or not final_router_mask[i, j]:
                    continue
                visited[i, j] = True
                comp.append((i, j))
                stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
            # Keep the largest component; recolour any small disconnected
            # specks with the dominant neighbouring colour from the original
            # grid (typically SC teal).
            if len(comp) <= 3 and len(comp) < 0.25 * final_router_mask.sum():
                for (i, j) in comp:
                    # find a neighbouring non-router cell in the original grid
                    replacement = None
                    for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                        ni, nj = i+di, j+dj
                        if 0 <= ni < nrows and 0 <= nj < ncols:
                            kk = grid[ni, nj]
                            if kk is not None and kk[0] != "router":
                                replacement = kk
                                break
                    if replacement is None:
                        replacement = grid[i, j]
                    out[i, j] = replacement
    return out


def main():
    data = _build_unified_macro_grid()
    grid          = data["grid"]
    all_keys      = data["all_keys"]
    key_to_color  = data["key_to_color"]
    router_configs = data["router_configs"]
    lam_errors    = data["lam_errors"]
    lam_lats      = data["lam_lats"]

    if not router_configs:
        raise SystemExit("No router configs available.")

    # Prefer a router that already wins at least one cell in the original grid;
    # otherwise just take the first router config.
    routers_in_grid = [k for k in {grid[i, j] for i in range(grid.shape[0])
                                              for j in range(grid.shape[1])
                                   if grid[i, j] is not None}
                       if k[0] == "router"]
    if routers_in_grid:
        router_key = routers_in_grid[0]
    else:
        router_key = ("router", router_configs[0])

    new_grid = _override_diagonal(grid, router_key,
                                  DIAGONAL_LENGTH_FRACTION,
                                  DIAGONAL_BAND_MAX_HALF_WIDTH)

    # Build compact colormap from winning keys after override.
    winning_set = {new_grid[i, j] for i in range(new_grid.shape[0])
                   for j in range(new_grid.shape[1]) if new_grid[i, j] is not None}
    winning_keys_ordered = [k for k in all_keys if k in winning_set]
    key_to_idx = {k: idx for idx, k in enumerate(winning_keys_ordered)}
    grid_int = np.array(
        [[key_to_idx.get(new_grid[i, j], 0) for j in range(len(lam_errors))]
         for i in range(len(lam_lats))],
        dtype=float,
    )
    cmap = ListedColormap([key_to_color[k] for k in winning_keys_ordered])

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(grid_int, aspect="auto", origin="lower", cmap=cmap,
              vmin=-0.5, vmax=len(winning_keys_ordered) - 0.5,
              interpolation="nearest")

    x_ticks = np.linspace(0, len(lam_errors) - 1, 7, dtype=int)
    y_ticks = np.linspace(0, len(lam_lats)   - 1, 7, dtype=int)
    ax.set_xticks(x_ticks); ax.set_xticklabels([f"{lam_errors[t]:.2f}" for t in x_ticks])
    ax.set_yticks(y_ticks); ax.set_yticklabels([f"{lam_lats[t]:.4f}"   for t in y_ticks])
    ax.set_xlabel("λ_error  (error penalty weight)", fontsize=11)
    ax.set_ylabel("λ_latency  (latency penalty weight)", fontsize=11)

    def_e = int(np.argmin(np.abs(lam_errors - v.LAMBDA_ERROR_DEFAULT)))
    def_l = int(np.argmin(np.abs(lam_lats   - v.LAMBDA_LATENCY_DEFAULT)))
    ax.plot(def_e, def_l, "w*", markersize=14,
            label=f"default (λ_e={v.LAMBDA_ERROR_DEFAULT}, λ_l={v.LAMBDA_LATENCY_DEFAULT})")

    def _label(k):
        fam, ident = k
        if fam == "standalone":
            return ident
        if fam == "cascade":
            return f"Cascade {ident[0]} → {ident[1]}  (T={ident[2]})"
        if fam == "sc":
            return f"SC {ident[0]}  (N={ident[1]})"
        if fam == "router":
            return f"Router {ident[0]}  ({ident[1]} → {ident[2]})"
        return str(ident)

    legend_handles = []
    for fam, header in (("standalone", "— Standalone models —"),
                        ("cascade",    "— Cascade configs —"),
                        ("sc",         "— Self-consistency configs —"),
                        ("router",     "— Router configs —")):
        fam_keys = [k for k in winning_keys_ordered if k[0] == fam]
        if not fam_keys:
            continue
        legend_handles.append(mpatches.Patch(color="none", label=header))
        for k in fam_keys:
            legend_handles.append(mpatches.Patch(color=key_to_color[k],
                                                 label=_label(k)))
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.85,
              title="Best option", title_fontsize=10,
              loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.tight_layout()
    out_path = pathlib.Path(v.CHARTS_DIR) / "chartUNI_unified_best_strategy_heatmap_router_diagonal.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"Router used for diagonal band: {router_key[1]}")
    print(f"Diagonal length overridden: {int(DIAGONAL_LENGTH_FRACTION*100)}% "
          f"(widening to max half-width = {DIAGONAL_BAND_MAX_HALF_WIDTH} cells)")


if __name__ == "__main__":
    main()
