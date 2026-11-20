"""Figures for the report.

    python experiments/make_plots.py --stage1 results/stage1 --grid results/all_quick --out results/figures

Stage 1  : heatmaps of h* and delta over (R/K, D) for rho = 0; PoA bounds vs R/K
Grid     : RQ1 regime maps (delta, lambda, PoA-hat) per method/topology/regime at rho = 0;
           RQ2 delta and g(rho) vs rho per method (mean with 95% CI); baseline social cost.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# colours shared by all plots
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
METHOD_ORDER = ["brd", "iql", "fp", "sql", "spe", "always-request", "bankers", "resource-ordering",
                "exp-backoff", "centralized"]
# selfish methods get their own colour, the classical baselines are grey
METHOD_COLOR = {"brd": SERIES[0], "iql": SERIES[1], "fp": SERIES[2], "sql": SERIES[3], "spe": SERIES[4],
                "always-request": "#b4b2ab", "bankers": "#898781", "resource-ordering": "#898781",
                "exp-backoff": "#b4b2ab", "centralized": "#52514e"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "lines.linewidth": 2, "font.size": 10, "legend.frameon": False,
})


def heat(ax, df, value, title, vmin=None, vmax=None, fmt="{:.2f}"):
    piv = df.pivot_table(index="D", columns="rk", values=value, aggfunc="mean").sort_index(ascending=False)
    arr = piv.values.astype(float)
    im = ax.imshow(arr, cmap=SEQ, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{r:g}" for r in piv.index])
    ax.set_xlabel("R / K")
    ax.set_ylabel("D / c0")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    lo, hi = np.nanmin(arr) if vmin is None else vmin, np.nanmax(arr) if vmax is None else vmax
    for (r, c), v in np.ndenumerate(arr):
        if np.isfinite(v):
            dark = (v - lo) / (hi - lo + 1e-12) > 0.55
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="#ffffff" if dark else INK)
    return im


def stage1_plots(path, out):
    df = pd.read_csv(os.path.join(path, "stage1_two_agent.csv"))
    d0 = df[df["rho"] == 0.0].copy()
    d0["h_star_num"] = pd.to_numeric(d0["h_star"], errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    heat(axes[0], d0, "h_star_num", "h*: first holding time with Yield in support", fmt="{:.0f}")
    heat(axes[1], d0, "deadlock", "deadlock-episode probability δ", 0, 1)
    heat(axes[2], d0, "poa_hat", "empirical PoA (symmetric SPE)")
    fig.suptitle("Stage 1 - exact two-agent equilibria (ρ = 0)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stage1_heatmaps.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k, D in enumerate(sorted(d0["D"].unique())):
        g = d0[d0["D"] == D].sort_values("rk")
        c = SERIES[k % len(SERIES)]
        ax.fill_between(g["rk"], g["poa_best"], g["poa_worst"], color=c, alpha=0.15, lw=0)
        ax.plot(g["rk"], g["poa_worst"], color=c, label=f"D/c0={D:g}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("R / K")
    ax.set_ylabel("PoA (band: best..worst SPE)")
    ax.set_title("Stage 1 - Price of Anarchy bounds", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stage1_poa.png"), dpi=150)
    plt.close(fig)

    if df["rho"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, D in enumerate(sorted(df["D"].unique())):
            g = df[df["D"] == D].groupby("rho")["social_cost"].mean()
            ax.plot(g.index, g.values, marker="o", ms=5, color=SERIES[k % len(SERIES)], label=f"D/c0={D:g}")
        ax.set_xlabel("ρ")
        ax.set_ylabel("social cost C (mean over R/K)")
        ax.set_title("Stage 1 - effect of the escalating holding cost", loc="left", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "stage1_rho.png"), dpi=150)
        plt.close(fig)


def grid_plots(path, out):
    agg = pd.read_csv(os.path.join(path, "summary.csv"))
    learned = agg[agg["method"].isin(["brd", "iql", "fp", "sql"]) & (agg["rho"] == 0.0)]
    for (top, reg), g in learned.groupby(["topology", "regime"]):
        methods = [m for m in METHOD_ORDER if m in set(g["method"])]
        fig, axes = plt.subplots(3, len(methods), figsize=(3.6 * len(methods), 9.5), squeeze=False)
        for j, m in enumerate(methods):
            gm = g[g["method"] == m]
            heat(axes[0, j], gm, "deadlock", f"{m}: δ (deadlock)", 0, 1)
            heat(axes[1, j], gm, "livelock", f"{m}: λ (livelock)", 0, 1)
            heat(axes[2, j], gm, "poa_hat", f"{m}: PoA-hat")
        fig.suptitle(f"RQ1 regime map - topology={top}, information={reg}, ρ=0", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"rq1_regime_{top}_{reg}.png"), dpi=150)
        plt.close(fig)

    mech = agg[agg["method"].isin(["brd", "iql", "fp", "sql"])]
    if mech["rho"].nunique() > 1:
        for top, g in mech.groupby("topology"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for m in [x for x in METHOD_ORDER if x in set(g["method"])]:
                gm = g[g["method"] == m].groupby("rho")
                c = METHOD_COLOR[m]
                for ax, col in zip(axes, ["deadlock", "social_cost", "g_rho"]):
                    mu = gm[col].mean()
                    ci = gm[col + "_ci"].mean() if col + "_ci" in g else None
                    ax.plot(mu.index, mu.values, marker="o", ms=5, color=c, label=m)
                    if ci is not None:
                        ax.fill_between(mu.index, mu - ci, mu + ci, color=c, alpha=0.12, lw=0)
            for ax, t in zip(axes, ["deadlock probability δ", "social cost C", "gap closed g(ρ)"]):
                ax.set_xlabel("ρ")
                ax.set_title(t, loc="left", fontsize=10)
            axes[0].legend(fontsize=8)
            fig.suptitle(f"RQ2 - coordinator-free escalating holding cost (topology={top}; averaged over grid)",
                         x=0.01, ha="left")
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"rq2_mechanism_{top}.png"), dpi=150)
            plt.close(fig)

    comp = agg[agg["rho"] == 0.0]
    if len(comp):
        for top, g in comp.groupby("topology"):
            means = g.groupby("method")["social_cost"].mean()
            order = [m for m in METHOD_ORDER if m in means.index]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(range(len(order)), [means[m] for m in order], color=[METHOD_COLOR[m] for m in order],
                    height=0.6)
            ax.set_yticks(range(len(order)), order)
            ax.invert_yaxis()
            for k, m in enumerate(order):
                ax.text(means[m], k, f" {means[m]:.1f}", va="center", fontsize=8, color=INK2)
            ax.axvline(g["c_star"].iloc[0], color=MUTED, ls="--", lw=1)
            ax.text(g["c_star"].iloc[0], -0.6, " C*", color=INK2, fontsize=8)
            ax.set_xscale("log")
            ax.set_xlabel("social cost C, log scale (mean over R/K, D grid; ρ=0)")
            ax.set_title(f"Selfish profiles vs classical baselines - {top}", loc="left", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"baselines_{top}.png"), dpi=150)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=None)
    ap.add_argument("--grid", nargs="*", default=[])
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.stage1:
        stage1_plots(args.stage1, args.out)
    for g in args.grid:
        grid_plots(g, args.out)
    print("figures in", args.out, ":", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
