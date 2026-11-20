"""Every table and figure in report/main.tex, generated from the result CSVs.

    python experiments/make_report_assets.py            # -> report/tables/*.tex, report/figures/*.pdf|png

Inputs (produced by the other scripts, see README):
    results/stage1/stage1_two_agent.csv   experiments/stage1_two_agent.py
    results/all/raw.csv, summary.csv      experiments/run_grid.py --preset all
    results/outcomes/summary.csv          experiments/run_grid.py --preset all --rho 0 --out results/outcomes
                                          (same rho = 0 cells, re-run with the stall metric; optional)
The numbers quoted in the report text are written to report/numbers.txt so they can be checked.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from make_plots import INK, INK2, METHOD_COLOR, MUTED, SEQ, SERIES, heat  # noqa: E402  (also sets rcParams)

LEARNED = ["brd", "fp", "iql", "sql"]
ALL = LEARNED + ["always-request", "exp-backoff", "bankers", "resource-ordering", "centralized"]
LABEL = {"brd": "Best-response dyn.", "fp": "Fictitious play", "iql": "Independent Q", "sql": "Shared Q",
         "always-request": "Always-request", "exp-backoff": "Exp. back-off", "bankers": "Banker's",
         "resource-ordering": "Resource ordering", "centralized": "Centralized ($C^\\star$)", "spe": "SPE"}
PLAIN = {k: v.replace("($C^\\star$)", "(C*)") for k, v in LABEL.items()}
TOPS = ["ring", "dense"]


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{d}f}"


def write(path, text):
    with open(path, "w") as f:
        f.write(text)
    print("wrote", path)


# tables
def table_stage1(s1, out):
    d0 = s1[s1.rho == 0.0]
    d5 = s1[s1.rho == 0.5]
    rows = []
    for rk, g in d0.groupby("rk"):
        g5 = d5[d5.rk == rk]
        rows.append(
            f"{rk:g} & {int(g.h_star.astype(float).max())} & {fmt(g.deadlock.max())} & "
            f"{fmt(g.poa_hat.mean())} & [{fmt(g.poa_best.min())}, {fmt(g.poa_worst.max())}] & "
            f"{fmt(g5.poa_hat.mean())} & {fmt(g.epsilon.abs().max(), 0) if g.epsilon.abs().max() > 1e-9 else '$<10^{-9}$'} \\\\")
    body = "\n".join(rows)
    write(os.path.join(out, "tab_stage1.tex"), rf"""\begin{{tabular}}{{rrrrcrr}}
\toprule
$R/K$ & $h^\star$ & $\max_D \delta$ & $C/C^\star$ & PoA band & $C/C^\star$ at $\rho{{=}}0.5$ & $\varepsilon$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


def table_methods(agg, out):
    """Main comparison at rho = 0, averaged over the (R/K, D, I) grid."""
    r0 = agg[agg.rho == 0.0]
    has_stall = "stall" in r0 and r0["stall"].notna().any()
    lines = []
    for top in TOPS:
        g = r0[r0.topology == top].groupby("method")
        lines.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{top} topology}} ($C^\star={fmt(r0[r0.topology == top].c_star.iloc[0], 0)}$)}} \\")
        for m in ALL:
            if m not in g.groups:
                continue
            x = g.get_group(m)
            eps = x["epsilon"].median() if x["epsilon"].notna().any() else np.nan
            seeds = int(x["seeds"].max())
            ci = x["poa_hat_ci"].mean() if seeds > 1 else np.nan
            poa = fmt(x.poa_hat.mean(), 1) + (f" $\\pm${fmt(ci, 1)}" if np.isfinite(ci) else "")
            lines.append(
                f"{LABEL[m]} & {seeds} & {fmt(x.deadlock.mean())} & {fmt(x.livelock.mean())} & "
                f"{fmt(x.stall.mean()) if has_stall else '--'} & {fmt(x.completion.mean())} & "
                f"{fmt(x.latency.mean(), 1)} & {poa} & {fmt(eps, 1)} \\\\")
        lines.append(r"\midrule")
    body = "\n".join(lines[:-1])
    write(os.path.join(out, "tab_methods.tex"), rf"""\begin{{tabular}}{{lrrrrrrrr}}
\toprule
Profile & seeds & $\delta$ & $\lambda$ & stall & compl. & latency & $\widehat{{\mathrm{{PoA}}}}$ ($\pm$95\% CI) & med.\ $\varepsilon$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
""")


def rq2_tests(raw):
    """Per (topology, method, rho): mean g(rho) and the number of grid cells in which C fell
    (or rose) significantly vs rho = 0 (paired over seeds, two-sided t-test, p < 0.05)."""
    keys = ["topology", "rk", "D", "regime", "method", "seed"]
    out = []
    for (top, m), x in raw[raw.method.isin(["iql", "sql"])].groupby(["topology", "method"]):
        base = x[x.rho == 0].set_index(keys)["social_cost"]
        for rho in sorted(x.rho.unique()):
            if rho == 0:
                continue
            d = (base - x[x.rho == rho].set_index(keys)["social_cost"].reindex(base.index)).dropna()
            better = worse = 0
            for _, g in d.groupby(level=[0, 1, 2, 3, 4]):
                if len(g) > 1 and g.std() > 0:
                    p = stats.ttest_1samp(g, 0).pvalue
                    better += p < 0.05 and g.mean() > 0
                    worse += p < 0.05 and g.mean() < 0
            out.append(dict(topology=top, method=m, rho=rho, better=better, worse=worse,
                            cells=d.groupby(level=[0, 1, 2, 3, 4]).ngroups))
    return pd.DataFrame(out)


def table_rq2(agg, raw, out):
    mech = agg[agg.method.isin(LEARNED)]
    tests = rq2_tests(raw)
    tests.to_csv(os.path.join(out, "rq2_significance.csv"), index=False)
    rhos = sorted(r for r in mech.rho.unique() if r > 0)
    lines = []
    for top in TOPS:
        for m in LEARNED:
            x = mech[(mech.topology == top) & (mech.method == m)]
            if x.empty:
                continue
            g = [fmt(x[x.rho == r].g_rho.mean()) for r in rhos]
            d0, d5 = x[x.rho == 0].deadlock.mean(), x[x.rho == max(rhos)].deadlock.mean()
            p0, p5 = x[x.rho == 0].poa_hat.mean(), x[x.rho == max(rhos)].poa_hat.mean()
            t = tests[(tests.topology == top) & (tests.method == m) & (tests.rho == max(rhos))]
            sig = f"{int(t.better.iloc[0])}/{int(t.worse.iloc[0])}" if len(t) else "--"
            lines.append(f"{top} & {LABEL[m]} & {' & '.join(g)} & {fmt(d0)}$\\to${fmt(d5)} & "
                         f"{fmt(p0, 1)}$\\to${fmt(p5, 1)} & {sig} \\\\")
    head = " & ".join(f"$g({r:g})$" for r in rhos)
    write(os.path.join(out, "tab_rq2.tex"), rf"""\begin{{tabular}}{{ll{'r' * len(rhos)}ccc}}
\toprule
Topology & Profile & {head} & $\delta$: $\rho{{=}}0\to{max(rhos):g}$ & $\widehat{{\mathrm{{PoA}}}}$: $\rho{{=}}0\to{max(rhos):g}$ & sig.\ $\downarrow/\uparrow$ \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
""")
    return tests


def table_thresholds(agg, out):
    r0 = agg[(agg.rho == 0.0) & agg.method.isin(LEARNED)]
    lines = []
    for top in TOPS:
        for m in LEARNED:
            x = r0[(r0.topology == top) & (r0.method == m)]
            cells = [fmt(x[(x.regime == reg) & (x.rk == rk)].deadlock.mean())
                     for reg in ["full", "part"] for rk in sorted(r0.rk.unique())]
            lines.append(f"{top} & {LABEL[m]} & {' & '.join(cells)} \\\\")
    rks = sorted(r0.rk.unique())
    head = " & ".join(f"{r:g}" for r in rks)
    write(os.path.join(out, "tab_rq1.tex"), rf"""\begin{{tabular}}{{ll{'r' * len(rks)}|{'r' * len(rks)}}}
\toprule
& & \multicolumn{{{len(rks)}}}{{c|}}{{$I=$ full, $R/K=$}} & \multicolumn{{{len(rks)}}}{{c}}{{$I=$ part, $R/K=$}} \\
Topology & Profile & {head} & {head} \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
""")


# figures
def save(fig, out, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".pdf"))


def fig_stage1(s1, out):
    d0 = s1[s1.rho == 0.0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    heat(axes[0], d0, "deadlock", "(a) deadlock probability δ, ρ = 0", 0, 0.1)
    heat(axes[1], d0, "poa_hat", "(b) PoA of the symmetric SPE (exact), ρ = 0", 1, 2.6)
    ax = axes[2]
    for k, D in enumerate(sorted(s1.D.unique())):
        g = s1[s1.D == D].groupby("rho")["poa_hat"].mean()
        ax.plot(g.index, g.values, marker="o", ms=4, color=SERIES[k], label=f"D/c0 = {D:g}")
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("ρ")
    ax.set_ylabel("PoA, exact (mean over R/K)")
    ax.set_title("(c) escalating holding cost", loc="left", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    save(fig, out, "fig_stage1")


def fig_rq1(agg, out):
    r0 = agg[(agg.rho == 0.0) & agg.method.isin(LEARNED)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    for j, top in enumerate(TOPS):
        for m in LEARNED:
            for reg, ls in [("full", "-"), ("part", "--")]:
                x = r0[(r0.topology == top) & (r0.method == m) & (r0.regime == reg)].groupby("rk")
                lab = PLAIN[m] if reg == "full" else None
                axes[0, j].plot(x.deadlock.mean().index, x.deadlock.mean().values, ls, marker="o", ms=4,
                                color=METHOD_COLOR[m], label=lab)
                axes[1, j].plot(x.poa_hat.mean().index, x.poa_hat.mean().values, ls, marker="o", ms=4,
                                color=METHOD_COLOR[m])
        axes[0, j].axhline(0.5, color=MUTED, lw=1, ls=":")
        axes[0, j].set_title(f"{'(a)' if j == 0 else '(b)'} δ vs R/K - {top}", loc="left", fontsize=10)
        axes[1, j].set_title(f"{'(c)' if j == 0 else '(d)'} PoA-hat vs R/K - {top}", loc="left", fontsize=10)
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("R / K")
        for i in range(2):
            axes[i, j].set_xscale("log", base=2)
    axes[0, 0].set_ylabel("deadlock probability δ")
    axes[1, 0].set_ylabel("PoA-hat (log)")
    axes[0, 0].legend(fontsize=8, title="solid: full, dashed: part", title_fontsize=7)
    save(fig, out, "fig_rq1")


def fig_outcomes(agg, out):
    r0 = agg[agg.rho == 0.0]
    if "stall" not in r0 or r0["stall"].isna().all():
        print("no stall column yet - skipping fig_outcomes (run the rho = 0 outcome grid)")
        return
    cats = [("deadlock", "deadlock", SERIES[0]), ("livelock", "livelock", SERIES[1]),
            ("stall", "stall (no cycle, no progress)", SERIES[3]), ("complete", "all agents complete", "#d9d8d3")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    methods = [m for m in ALL if m in set(r0.method)]
    for ax, top in zip(axes, TOPS):
        g = r0[r0.topology == top].groupby("method")[["deadlock", "livelock", "stall"]].mean()
        g["complete"] = (1 - g.sum(axis=1)).clip(lower=0)
        left = np.zeros(len(methods))
        for col, lab, c in cats:
            v = np.array([g.loc[m, col] if m in g.index else 0 for m in methods])
            ax.barh(range(len(methods)), v, left=left, color=c, height=0.62, label=lab,
                    edgecolor="#fcfcfb", linewidth=1)
            left += v
        ax.set_yticks(range(len(methods)), [PLAIN[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of evaluation episodes (ρ = 0, mean over grid)")
        ax.set_title(f"episode outcomes - {top}", loc="left", fontsize=10)
    axes[0].legend(fontsize=7, loc="lower center", bbox_to_anchor=(1.05, -0.42), ncol=4)
    save(fig, out, "fig_outcomes")


def fig_rq2(agg, out):
    mech = agg[agg.method.isin(LEARNED)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    for j, top in enumerate(TOPS):
        for m in LEARNED:
            x = mech[(mech.topology == top) & (mech.method == m)].groupby("rho")
            axes[2 * j].plot(x.deadlock.mean().index, x.deadlock.mean().values, marker="o", ms=4,
                             color=METHOD_COLOR[m], label=PLAIN[m])
            g = x.g_rho.mean().dropna()
            axes[2 * j + 1].plot(g.index, g.values, marker="o", ms=4, color=METHOD_COLOR[m])
        axes[2 * j].set_title(f"δ vs ρ - {top}", loc="left", fontsize=10)
        axes[2 * j + 1].set_title(f"gap closed g(ρ) - {top}", loc="left", fontsize=10)
        axes[2 * j + 1].axhline(0, color=MUTED, lw=1)
        for k in (2 * j, 2 * j + 1):
            axes[k].set_xlabel("ρ")
    axes[0].legend(fontsize=7)
    save(fig, out, "fig_rq2")


def fig_baselines(agg, out):
    r0 = agg[agg.rho == 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    methods = [m for m in ALL if m in set(r0.method)]
    for ax, top in zip(axes, TOPS):
        g = r0[r0.topology == top]
        means = g.groupby("method")["social_cost"].mean()
        ax.barh(range(len(methods)), [means[m] for m in methods], color=[METHOD_COLOR[m] for m in methods],
                height=0.6)
        for k, m in enumerate(methods):
            ax.text(means[m], k, f" {means[m]:.1f}", va="center", fontsize=7, color=INK2)
        ax.axvline(g.c_star.iloc[0], color=MUTED, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yticks(range(len(methods)), [PLAIN[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlabel("social cost C (log; mean over grid, ρ = 0)")
        ax.set_title(f"{top}: dashed line = C* = {g.c_star.iloc[0]:g}", loc="left", fontsize=10)
    save(fig, out, "fig_baselines")


REGIME_COLOR = {"deadlock": SERIES[0], "livelock": SERIES[1], "stall": SERIES[3], "mixed": SERIES[2],
                "strategic": "#d9d8d3"}


def fig_regimes(agg, out):
    """RQ1 regime map over (R/K, D): which outcome dominates each cell (rho = 0, full information)."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from deadlock_game.experiments import regime_label
    r0 = agg[(agg.rho == 0.0) & (agg.regime == "full") & agg.method.isin(LEARNED)].copy()
    if "stall" not in r0 or r0["stall"].isna().all():
        return
    names = list(REGIME_COLOR)
    r0["label"] = [names.index(regime_label(d, l, s_)) for d, l, s_ in zip(r0.deadlock, r0.livelock, r0.stall)]
    cmap = ListedColormap([REGIME_COLOR[k] for k in names])
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True, sharey=True)
    for i, top in enumerate(TOPS):
        for j, m in enumerate(LEARNED):
            ax = axes[i, j]
            piv = r0[(r0.topology == top) & (r0.method == m)].pivot_table(
                index="D", columns="rk", values="label").sort_index(ascending=False)
            ax.imshow(piv.values, cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, aspect="auto")
            ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
            ax.set_yticks(range(len(piv.index)), [f"{r:g}" for r in piv.index])
            for k in range(len(piv.columns) + 1):
                ax.axvline(k - 0.5, color="#fcfcfb", lw=2)
            for k in range(len(piv.index) + 1):
                ax.axhline(k - 0.5, color="#fcfcfb", lw=2)
            ax.set_title(f"{PLAIN[m]} - {top}", loc="left", fontsize=9)
            if i == 1:
                ax.set_xlabel("R / K")
            if j == 0:
                ax.set_ylabel("D / c0")
    fig.legend(handles=[Patch(color=REGIME_COLOR[k], label=k) for k in names], loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_regimes.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(out, "fig_regimes.pdf"))


def table_standoff(path, out):
    """Criterion 1: do the simulation methods reproduce the exact two-agent equilibrium?"""
    f = os.path.join(path, "summary.csv")
    if not os.path.exists(f):
        print("no", f, "- skipping tab_standoff")
        return None
    a = pd.read_csv(f)
    order = ["spe", "brd", "fp", "iql", "sql", "exp-backoff", "bankers", "centralized"]
    lines = []
    for m in order:
        x = a[a.method == m]
        if x.empty:
            continue
        x0, x5 = x[x.rho == 0], x[x.rho == 0.5]
        eps = x0["epsilon"].median() if x0["epsilon"].notna().any() else np.nan
        g5 = x5.g_rho.mean() if len(x5) else np.nan
        lines.append(f"{LABEL[m]} & {fmt(x0.deadlock.mean())} & {fmt(x0.stall.mean())} & {fmt(x0.completion.mean())} & "
                     f"{fmt(x0.poa_hat.mean(), 2)} & {fmt(eps, 2)} & {fmt(g5)} \\\\")
    write(os.path.join(out, "tab_standoff.tex"), rf"""\begin{{tabular}}{{lrrrrrr}}
\toprule
Profile ($n=2$) & $\delta$ & stall & compl. & $\widehat{{\mathrm{{PoA}}}}$ & med.\ $\varepsilon$ & $g(0.5)$ \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
""")
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="results/stage1")
    ap.add_argument("--grid", default="results/all")
    ap.add_argument("--outcomes", default="results/outcomes")
    ap.add_argument("--standoff", default="results/standoff")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    tdir, fdir = os.path.join(a.out, "tables"), os.path.join(a.out, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    s1 = pd.read_csv(os.path.join(a.stage1, "stage1_two_agent.csv"))
    # Stage 1 has exact costs (forward occupancy evaluation); use them instead of the Monte-Carlo estimate
    s1["poa_mc"] = s1["poa_hat"]
    s1["poa_hat"] = s1["exact_cost"] / s1["c_star"]
    agg = pd.read_csv(os.path.join(a.grid, "summary.csv"))
    raw = pd.read_csv(os.path.join(a.grid, "raw.csv"))
    oc_path = os.path.join(a.outcomes, "summary.csv")
    r0 = agg[agg.rho == 0.0]
    if os.path.exists(oc_path):
        oc = pd.read_csv(oc_path)
        # the outcome run repeats the rho = 0 cells with identical seeds; check, then take its stall column
        k = ["topology", "rk", "D", "rho", "regime", "method"]
        chk = r0.merge(oc, on=k, suffixes=("", "_oc"))
        diff = (chk["social_cost"] - chk["social_cost_oc"]).abs().max()
        print(f"outcome run: {len(chk)}/{len(r0)} rho=0 cells matched, max |dC| = {diff:.2e}")
        agg = agg.drop(columns=[c for c in ("stall", "stall_ci") if c in agg]).merge(
            oc[k + ["stall", "stall_ci"]], on=k, how="left")

    table_stage1(s1, tdir)
    table_methods(agg, tdir)
    table_thresholds(agg, tdir)
    tests = table_rq2(agg, raw, tdir)
    print(tests.to_string(index=False))

    fig_stage1(s1, fdir)
    fig_rq1(agg, fdir)
    fig_outcomes(agg, fdir)
    fig_rq2(agg, fdir)
    fig_regimes(agg, fdir)
    so = table_standoff(a.standoff, tdir)
    fig_baselines(agg, fdir)

    # headline numbers quoted in the text
    th = pd.read_csv(os.path.join(a.grid, "thresholds.csv"))
    th = th[th.method.isin(LEARNED)].dropna(subset=["theta"])
    brd = raw[raw.method == "brd"]
    lines = ["# numbers quoted in report/main.tex (generated)",
             "thresholds (theta, non-NaN):",
             th.groupby(["topology", "regime", "method"])["theta"].agg(["min", "max"]).to_string(),
             "BRD convergence (share of cells):",
             brd.groupby(["topology", "regime"])[["converged", "cycled"]].mean().round(2).to_string(),
             "learner convergence (share of runs):",
             raw[raw.method.isin(["iql", "sql"])].groupby("method")["converged"].mean().round(2).to_string(),
             "deadlock by regime (rho=0):",
             r0[r0.method.isin(LEARNED)].groupby(["topology", "method", "regime"])["deadlock"].mean()
             .unstack().round(3).to_string(),
             "stage1 exact PoA by rho:", s1.groupby("rho")["poa_hat"].mean().round(3).to_string(),
             "stage1 exact PoA by D at rho (rows D):", s1.pivot_table(index="D", columns="rho", values="poa_hat").round(2).to_string(),
             "stage1 exact PoA by rk (rho=0):", s1[s1.rho == 0].groupby("rk")["poa_hat"].mean().round(2).to_string(),
             "stage1 max |MC - exact| PoA:", f"{(s1.poa_mc - s1.poa_hat).abs().max():.3f}",
             "stage1 max deadlock:", f"{s1.deadlock.max():.3f}",
             "Q-learning stability (rho=0): mean across-seed sd of C per cell:",
             raw[(raw.rho == 0) & raw.method.isin(["iql", "sql"])].groupby(["topology", "method", "rk", "D", "regime"])
             ["social_cost"].std().groupby(level=[0, 1]).mean().round(2).to_string(),
             "latency by method (rho=0):", agg[agg.rho == 0].groupby(["topology", "method"])["latency"].mean()
             .round(2).unstack(0).to_string(),
             "standoff (n=2) by method, rho=0:",
             (so[so.rho == 0].groupby("method")[["deadlock", "stall", "completion", "poa_hat", "epsilon"]].mean()
              .round(3).to_string() if so is not None else "n/a"),
             "CPU hours by method:", raw.groupby("method")["seconds"].sum().div(3600).round(2).to_string()]
    write(os.path.join(a.out, "numbers.txt"), "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
