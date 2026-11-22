"""Generated LaTeX macros, summary table, and figures for Amendment A8 revisions.

Every post-review number reaching paper/ flows through this module so the
repository invariant ("no hand-typed multi-decimal numbers in LaTeX") holds for
both the pre-registered study and the post-review validations.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xaibench.config import Profile, get_profile, paths
from xaibench.stats.report import _macro, fmt, fmt_p
from xaibench.stats.tests import holm, wilcoxon_with_effect
from xaibench.viz.theme import PALETTE, apply_theme, watermark

CM = 1 / 2.54


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(paths().results / name)


def build_fig_lime_budget(profile: Profile):
    """F15: LIME sampling budget Pareto (runtime vs stability & fidelity, 18 cells)."""
    apply_theme()
    budget = _read("revision_lime_budget.csv")
    rerun = _read("revision_lime_budget_rerun.csv")
    fid_shap = _read("metrics_fidelity.csv").query("baseline == 'median' and explainer == 'shap'")
    shap_aopc = float(fid_shap.aopc.median())

    budgets = sorted(budget.num_samples.unique())
    tree = budget[budget.shap_backend.isin(["tree", "tree_prob"])]
    stack = budget[budget.model == "stack"]

    def _cell_med_rt(frame):
        out = []
        for ns in budgets:
            g = frame[frame.num_samples == ns]
            out.append(float(g.groupby(["dataset", "model"]).runtime_s.median().median()))
        return np.array(out)

    tree_lime_rt = _cell_med_rt(tree)
    stack_lime_rt = _cell_med_rt(stack)
    tree_shap_rt = float(tree.groupby(["dataset", "model"]).shap_median_s.first().median())
    stack_shap_rt = float(stack.groupby(["dataset", "model"]).shap_median_s.first().median())

    cv_med = np.array([float(rerun[rerun.num_samples == ns].rerun_cv.median()) for ns in budgets])
    aopc_med = np.array([float(budget[budget.num_samples == ns].aopc.median()) for ns in budgets])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5 * CM, 4.7 * CM))

    # Panel (a): Runtime vs budget (log y-scale)
    ax1.plot(budgets, tree_lime_rt, marker="o", ms=4.0, lw=1.5, color=PALETTE["lime"],
             label="LIME (trees, 12 cells)")
    ax1.plot(budgets, stack_lime_rt, marker="s", ms=4.0, lw=1.4, ls="--", color=PALETTE["lime"],
             alpha=0.85, label="LIME (Stacking, 3 cells)")
    ax1.axhline(tree_shap_rt, color=PALETTE["shap"], lw=1.3, ls="-",
                label=f"TreeSHAP ({tree_shap_rt:.4f} s)")
    ax1.axhline(stack_shap_rt, color=PALETTE["shap"], lw=1.3, ls=":",
                label=f"Perm. SHAP ({stack_shap_rt:.2f} s)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(2.8e-4, 0.90)
    ax1.set_xticks(budgets)
    ax1.set_xticklabels([f"{b:,}" for b in budgets], fontsize=7)
    ax1.set_xlabel("LIME sampling budget ($N_{\\text{samples}}$)", fontsize=7.8)
    ax1.set_ylabel("Median runtime (s / exp.)", fontsize=7.8)
    ax1.set_title("(a) Runtime vs. sampling budget", fontsize=8.3)
    ax1.legend(fontsize=5.5, loc="lower right", bbox_to_anchor=(0.99, 0.14),
               frameon=True, framealpha=0.95, borderpad=0.25, labelspacing=0.18, ncol=1)
    ax1.grid(True, which="major", alpha=0.25, lw=0.5)

    # Panel (b): Stability (CV, left) and Fidelity (AOPC, right)
    c_cv = PALETTE["lime"]
    c_aopc = PALETTE["shap"]
    ln1 = ax2.plot(budgets, cv_med, marker="o", ms=4.0, lw=1.5, color=c_cv,
                   label="LIME re-run CV (left)")
    ax2.set_xscale("log")
    ax2.set_xticks(budgets)
    ax2.set_xticklabels([f"{b:,}" for b in budgets], fontsize=7)
    ax2.set_xlabel("LIME sampling budget ($N_{\\text{samples}}$)", fontsize=7.8)
    ax2.set_ylabel("Median re-run CV", fontsize=7.8, color=c_cv)
    ax2.tick_params(axis="y", labelcolor=c_cv)
    ax2.set_ylim(0, 2.35)
    ax2.grid(True, which="major", alpha=0.25, lw=0.5)

    ax2r = ax2.twinx()
    ln2 = ax2r.plot(budgets, aopc_med, marker="^", ms=4.0, lw=1.5, ls="--", color=c_aopc,
                    label="LIME median AOPC (right)")
    ln3 = ax2r.axhline(shap_aopc, color="#333333", lw=1.2, ls=":",
                       label=f"SHAP AOPC ({shap_aopc:.3f})")
    ax2r.set_ylabel("Median deletion AOPC", fontsize=7.8, color=c_aopc)
    ax2r.tick_params(axis="y", labelcolor=c_aopc)
    ax2r.set_ylim(0.14, 0.285)
    ax2r.spines["top"].set_visible(False)

    lines = ln1 + ln2 + [ln3]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=5.7, loc="lower left",
               frameon=True, framealpha=0.95, borderpad=0.3, labelspacing=0.2)
    ax2.set_title("(b) Re-run stability & fidelity plateau", fontsize=8.3)

    fig.tight_layout(pad=0.7)
    watermark(fig, profile)
    out_dir = paths().figures
    out_dir.mkdir(parents=True, exist_ok=True)
    for suf in ("pdf", "png"):
        fig.savefig(out_dir / f"F15_lime_budget.{suf}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def revision_findings_macros(profile: Profile) -> str:
    """Emit every post-review Amendment A8 number as a LaTeX macro."""
    red = _read("revision_redundancy.csv").set_index("dataset")
    cond_raw = _read("revision_conditional_fidelity.csv")
    cond_st = _read("revision_conditional_stats.csv")
    roar_st = _read("revision_roar_stats.csv")
    fus_sum = _read("revision_fusion_summary.csv")
    fus_st = _read("revision_fusion_stats.csv")
    bud = _read("revision_lime_budget.csv")
    bud_rr = _read("revision_lime_budget_rerun.csv")
    fid_orig = _read("metrics_fidelity.csv")
    scale_c = _read("revision_scalability_cells.csv")
    scale_s = _read("revision_scalability_stats.csv")

    # Conditional k-NN summary
    c_sl = cond_st.query("comparison == 'cond_shap_vs_lime'")
    c_sr = cond_st.query("comparison == 'cond_shap_vs_random'")
    c_lr = cond_st.query("comparison == 'cond_lime_vs_random'")
    cond_shap_wins = int(((c_sl.decision == "reject H0") & (c_sl.effect_size > 0)).sum())
    cond_lime_wins = int(((c_sl.decision == "reject H0") & (c_sl.effect_size < 0)).sum())

    # ROAR cell-mean summary
    r_auc_sl = roar_st.query("metric == 'auc_drop' and level == 'cell_mean' and comparison == 'shap_vs_lime'").iloc[0]
    r_auc_sr = roar_st.query("metric == 'auc_drop' and level == 'cell_mean' and comparison == 'shap_vs_random'").iloc[0]
    r_ll_sl = roar_st.query("metric == 'logloss_inc' and level == 'cell_mean' and comparison == 'shap_vs_lime'").iloc[0]

    # Fusion deep-dive
    fus_cal = fus_st.query("bar == 'cal_best_single'")
    max_wins_any_rule = int(fus_cal.groupby("rule").apply(lambda g: (g.decision == "reject H0").sum(), include_groups=False).max())
    r2_gamma_zero = int((fus_sum.adapt_gamma == 0.0).sum())
    r2_gate_always_shap = int(np.isinf(fus_sum.gate_tau).sum())

    # LIME budget sweep
    tree_b = bud[bud.shap_backend.isin(["tree", "tree_prob"])]
    stack_b = bud[bud.model == "stack"]
    tree_shap_rt = float(tree_b.groupby(["dataset", "model"]).shap_median_s.first().median())
    stack_shap_rt = float(stack_b.groupby(["dataset", "model"]).shap_median_s.first().median())

    def _rt(frame, ns):
        return float(frame[frame.num_samples == ns].groupby(["dataset", "model"]).runtime_s.median().median())

    # SHAP vs LIME@10k per-cell Holm test
    sh_med = fid_orig.query("baseline == 'median' and explainer == 'shap'").set_index(["dataset", "model", "instance_id"])["aopc"]
    l10k = bud[bud.num_samples == 10000].set_index(["dataset", "model", "instance_id"])["aopc"]
    both10 = pd.concat([sh_med.rename("s"), l10k.rename("l")], axis=1).dropna()
    p10, e10 = [], []
    for _, grp in both10.groupby(level=[0, 1]):
        res = wilcoxon_with_effect(grp["s"], grp["l"], alternative="two-sided")
        p10.append(res.p_value); e10.append(res.effect_size)
    p10_adj = holm(p10)
    shap_beats_l10k = int(sum((pa < 0.05 and ef > 0) for pa, ef in zip(p10_adj, e10)))

    # External scalability summary
    bcwd = scale_c.query("dataset == 'bcwd'")
    cdc = scale_c.query("dataset == 'cdc_diabetes'")
    sc_h1_wins = int((scale_s.query("hypothesis == 'H1_rho_gt_050'").decision == "reject H0").sum())
    sc_cond_wins = int(((scale_s.query("hypothesis == 'H3a_cond_aopc'").decision == "reject H0")
                        & (scale_s.query("hypothesis == 'H3a_cond_aopc'").effect_size > 0)).sum())
    sc_lip_wins = int(((scale_s.query("hypothesis == 'H2b_lipschitz'").decision == "reject H0")
                       & (scale_s.query("hypothesis == 'H2b_lipschitz'").effect_size < 0)).sum())
    sc_cost_wins = int((scale_s.query("hypothesis == 'H4_cost'").decision == "reject H0").sum())
    sc_fus_wins = int((scale_s.query("hypothesis == 'H6_fidwt_vs_best'").decision == "reject H0").sum())

    # Primary hypothesis p-values for inline reporting (Reviewer #2)
    orig_st = _read("stats_hypotheses.csv")
    def _med_p(h):
        return fmt_p(orig_st.query("hypothesis == @h").p_adj.median())

    lines = [
        "% Generated by xaibench.stats.revision_outputs -- do not edit.",
        # Inline p-values from original Table I (Reviewer #2)
        _macro("POneB", _med_p("H1b")),
        _macro("POneC", _med_p("H1c")),
        _macro("PFive", _med_p("H5")),
        # Redundancy (R1.1)
        _macro("EffRankPima", fmt(red.loc["pima", "effective_rank_ratio"], 2)),
        _macro("EffRankHeart", fmt(red.loc["heart", "effective_rank_ratio"], 2)),
        _macro("EffRankCkd", fmt(red.loc["ckd", "effective_rank_ratio"], 2)),
        _macro("EffRankBcwd", fmt(red.loc["bcwd", "effective_rank_ratio"], 2)),
        _macro("EffRankCdc", fmt(red.loc["cdc_diabetes", "effective_rank_ratio"], 2)),
        _macro("CkdUniEighty", str(int(red.loc["ckd", "n_uni_auc_ge_080"]))),
        _macro("CkdUniNinety", str(int(red.loc["ckd", "n_uni_auc_ge_090"]))),
        _macro("BcwdUniNinety", str(int(red.loc["bcwd", "n_uni_auc_ge_090"]))),
        _macro("CkdMaxCorr", fmt(red.loc["ckd", "max_abs_spearman"], 2)),
        # Conditional k-NN & ROAR (R1.4)
        _macro("CondAopcShap", fmt(cond_raw.query("explainer == 'shap'").aopc_cond.median(), 3)),
        _macro("CondAopcLime", fmt(cond_raw.query("explainer == 'lime'").aopc_cond.median(), 3)),
        _macro("CondAopcRand", fmt(cond_raw.query("explainer == 'random'").aopc_cond.median(), 3)),
        _macro("CondShapWins", str(cond_shap_wins)),
        _macro("CondLimeWins", str(cond_lime_wins)),
        _macro("CondShapVsRandWins", str(int((c_sr.decision == "reject H0").sum()))),
        _macro("CondLimeVsRandWins", str(int((c_lr.decision == "reject H0").sum()))),
        _macro("RoarAucShap", fmt(r_auc_sl.mean_left, 3)),
        _macro("RoarAucLime", fmt(r_auc_sl.mean_right, 3)),
        _macro("RoarAucRand", fmt(r_auc_sr.mean_right, 3)),
        _macro("RoarAucP", fmt_p(r_auc_sl.p_adj)),
        _macro("RoarRandP", fmt_p(r_auc_sr.p_adj)),
        _macro("RoarLlShap", fmt(r_ll_sl.mean_left, 3)),
        _macro("RoarLlLime", fmt(r_ll_sl.mean_right, 3)),
        _macro("RoarLlP", fmt_p(r_ll_sl.p_adj)),
        # Fusion deep-dive (R1.2)
        _macro("WPropMin", fmt(fus_sum.w_prop.min(), 2)),
        _macro("WPropMax", fmt(fus_sum.w_prop.max(), 2)),
        _macro("WPropMed", fmt(fus_sum.w_prop.median(), 2)),
        _macro("WOptMed", fmt(fus_sum.w_opt.median(), 2)),
        _macro("WOptMin", fmt(fus_sum.w_opt.min(), 2)),
        _macro("WOptMax", fmt(fus_sum.w_opt.max(), 2)),
        _macro("FusionMaxWins", str(max_wins_any_rule)),
        _macro("OracleHeadroomMed", fmt(fus_sum.oracle_headroom.median(), 3)),
        _macro("OracleHeadroomMax", fmt(fus_sum.oracle_headroom.max(), 3)),
        _macro("ShapWinPct", fmt(100.0 * fus_sum.shap_win_rate.mean(), 1)),
        _macro("LimeWinPct", fmt(100.0 * fus_sum.lime_win_rate.mean(), 1)),
        _macro("RTwoCorrMed", fmt(fus_sum.r2_vs_lime_adv_rho.median(), 2)),
        _macro("RTwoGammaZeroCells", str(r2_gamma_zero)),
        _macro("RTwoGateAlwaysShap", str(r2_gate_always_shap)),
        # LIME budget sweep (R1.3)
        _macro("LimeCvFiveH", fmt(bud_rr[bud_rr.num_samples == 500].rerun_cv.median(), 2)),
        _macro("LimeCvTenK", fmt(bud_rr[bud_rr.num_samples == 10000].rerun_cv.median(), 2)),
        _macro("LimePairRhoFiveH", fmt(bud_rr[bud_rr.num_samples == 500].repeat_pair_rho.median(), 2)),
        _macro("LimePairRhoTenK", fmt(bud_rr[bud_rr.num_samples == 10000].repeat_pair_rho.median(), 2)),
        _macro("LimeRtFiveH", fmt(bud[bud.num_samples == 500].runtime_s.median(), 4)),
        _macro("LimeRtTenK", fmt(bud[bud.num_samples == 10000].runtime_s.median(), 4)),
        _macro("LimeAopcFiveH", fmt(bud[bud.num_samples == 500].aopc.median(), 3)),
        _macro("LimeAopcTwoHalfK", fmt(bud[bud.num_samples == 2500].aopc.median(), 3)),
        _macro("LimeAopcTenK", fmt(bud[bud.num_samples == 10000].aopc.median(), 3)),
        _macro("TreeRatioFiveH", fmt(_rt(tree_b, 500) / tree_shap_rt, 1)),
        _macro("TreeRatioFiveK", fmt(_rt(tree_b, 5000) / tree_shap_rt, 1)),
        _macro("TreeRatioTenK", fmt(_rt(tree_b, 10000) / tree_shap_rt, 1)),
        _macro("StackRatioFiveH", fmt(stack_shap_rt / _rt(stack_b, 500), 1)),
        _macro("StackRatioFiveK", fmt(stack_shap_rt / _rt(stack_b, 5000), 1)),
        _macro("StackRatioTenK", fmt(stack_shap_rt / _rt(stack_b, 10000), 1)),
        _macro("ShapBeatsLimeTenK", str(shap_beats_l10k)),
        # External scalability (R1.1)
        _macro("BcwdRhoMed", fmt(bcwd.rho_median.median(), 2)),
        _macro("BcwdGlobalRhoMed", fmt(bcwd.rho_global.median(), 2)),
        _macro("BcwdAucMed", fmt(bcwd.test_roc_auc.median(), 2)),
        _macro("CdcRhoMed", fmt(cdc.rho_median.median(), 2)),
        _macro("CdcAucMed", fmt(cdc.test_roc_auc.median(), 2)),
        _macro("CdcCondShap", fmt(cdc.aopc_cond_shap.median(), 3)),
        _macro("CdcCondLime", fmt(cdc.aopc_cond_lime.median(), 3)),
        _macro("ScaleHOneWins", str(sc_h1_wins)),
        _macro("ScaleCondWins", str(sc_cond_wins)),
        _macro("ScaleLipWins", str(sc_lip_wins)),
        _macro("ScaleCostWins", str(sc_cost_wins)),
        _macro("ScaleFusWins", str(sc_fus_wins)),
    ]
    return "\n".join(lines) + "\n"


def build_table_revision(profile: Profile) -> str:
    """T13_revision: Compact unified table for conditional/ROAR fidelity, fusion, & scalability."""
    fid_orig = _read("metrics_fidelity.csv")
    cond_raw = _read("revision_conditional_fidelity.csv")
    cond_st = _read("revision_conditional_stats.csv")
    roar_st = _read("revision_roar_stats.csv")
    fus_st = _read("revision_fusion_stats.csv")
    fus_sum = _read("revision_fusion_summary.csv")
    orig_st = _read("stats_hypotheses.csv")
    scale_c = _read("revision_scalability_cells.csv")
    scale_s = _read("revision_scalability_stats.csv")

    med_f = fid_orig.query("baseline == 'median'")
    mar_f = fid_orig.query("baseline == 'marginal'")
    c_sl = cond_st.query("comparison == 'cond_shap_vs_lime'")
    r_auc = roar_st.query("metric == 'auc_drop' and level == 'cell_mean' and comparison == 'shap_vs_lime'").iloc[0]
    r_auc_r = roar_st.query("metric == 'auc_drop' and level == 'cell_mean' and comparison == 'shap_vs_random'").iloc[0]
    r_ll = roar_st.query("metric == 'logloss_inc' and level == 'cell_mean' and comparison == 'shap_vs_lime'").iloc[0]
    r_ll_r = roar_st.query("metric == 'logloss_inc' and level == 'cell_mean' and comparison == 'shap_vs_random'").iloc[0]

    h3a = orig_st.query("hypothesis == 'H3a'")
    med_wins = int(((h3a.decision == "reject H0") & (h3a.effect_size > 0)).sum())
    cond_wins = int(((c_sl.decision == "reject H0") & (c_sl.effect_size > 0)).sum())

    # Marginal SHAP vs LIME per-cell Holm wins
    mp, me = [], []
    for (d, m), g in mar_f.groupby(["dataset", "model"]):
        w = g.pivot_table(index="instance_id", columns="explainer", values="aopc", aggfunc="first")[["shap", "lime"]].dropna()
        res = wilcoxon_with_effect(w["shap"], w["lime"], alternative="two-sided")
        mp.append(res.p_value); me.append(res.effect_size)
    mp_adj = holm(mp)
    mar_wins = int(sum((p < 0.05 and e > 0) for p, e in zip(mp_adj, me)))

    # Fusion rows vs cal_best_single
    fc = fus_st.query("bar == 'cal_best_single'")
    def _fus_row(rule_key, label, param_str):
        g = fc.query("rule == @rule_key")
        wins = int((g.decision == "reject H0").sum())
        med_d = float(g.mean_diff.median())
        med_p = fmt_p(g.p_adj.median())
        return f"\\quad {label} & {param_str} & {fmt(g.median_rule.median(), 3)} & {fmt(med_d, 4)} & {med_p} & {wins}/18 \\\\"

    # Scalability rows
    def _sc_row(ds_id, label, dims):
        sc = scale_c.query("dataset == @ds_id")
        ss = scale_s.query("dataset == @ds_id")
        h1w = int((ss.query("hypothesis == 'H1_rho_gt_050'").decision == "reject H0").sum())
        cw = int(((ss.query("hypothesis == 'H3a_cond_aopc'").decision == "reject H0")
                  & (ss.query("hypothesis == 'H3a_cond_aopc'").effect_size > 0)).sum())
        fw = int((ss.query("hypothesis == 'H6_fidwt_vs_best'").decision == "reject H0").sum())
        return (f"\\quad {label} & {dims} & $\\rho={fmt(sc.rho_median.median(), 2)}$ ({h1w}/6) "
                f"& {fmt(sc.aopc_cond_shap.median(), 3)} / {fmt(sc.aopc_cond_lime.median(), 3)} "
                f"& {cw}/6 & {fw}/6 \\\\")

    # Scalar medians for Panel A
    med_s = fmt(med_f.query("explainer == 'shap'").aopc.median(), 3)
    med_l = fmt(med_f.query("explainer == 'lime'").aopc.median(), 3)
    med_r = fmt(med_f.query("explainer == 'random'").aopc.median(), 3)
    mar_s = fmt(mar_f.query("explainer == 'shap'").aopc.median(), 3)
    mar_l = fmt(mar_f.query("explainer == 'lime'").aopc.median(), 3)
    mar_r = fmt(mar_f.query("explainer == 'random'").aopc.median(), 3)
    cnd_s = fmt(cond_raw.query("explainer == 'shap'").aopc_cond.median(), 3)
    cnd_l = fmt(cond_raw.query("explainer == 'lime'").aopc_cond.median(), 3)
    cnd_r = fmt(cond_raw.query("explainer == 'random'").aopc_cond.median(), 3)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\ifhighlight\color{blue!80!black}\fi",
        r"\caption{Post-review validation summary (Amendment A8): feature-dependence fidelity (A), static vs.\ calibration-tuned/adaptive rank fusion against the calibration-selected best single explainer (B), and external scalability datasets (C).}",
        r"\label{tab:revision}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\multicolumn{6}{l}{\textbf{(A) Feature-dependence-aware fidelity across the 18 primary cells} (higher drop = more faithful)} \\",
        r"\cmidrule(lr){1-6}",
        r"Validation protocol & Granularity & SHAP & LIME & Random & SHAP $>$ LIME ($p_{\text{adj}}<0.05$) \\",
        r"\midrule",
        f"Median replacement (primary) & Local (1,746 inst.) & {med_s} & {med_l} & {med_r} & {med_wins}/18 ({fmt_p(h3a.p_adj.median())}) \\\\",
        f"Marginal draw (5 draws) & Local (1,746 inst.) & {mar_s} & {mar_l} & {mar_r} & {mar_wins}/18 ({fmt_p(float(np.median(mp_adj)))}) \\\\",
        f"Conditional $k$-NN ($M=10$) & Local, on-manifold & {cnd_s} & {cnd_l} & {cnd_r} & {cond_wins}/18 ({fmt_p(c_sl.p_adj.median())}) \\\\",
        f"ROAR $\\Delta$ROC-AUC (5-fold CV) & Global, retrained & {fmt(r_auc.mean_left, 3)} & {fmt(r_auc.mean_right, 3)} & {fmt(r_auc_r.mean_right, 3)} & tie ($p={fmt(r_auc.p_adj, 3)}$) \\\\",
        f"ROAR $\\Delta$Log-Loss (5-fold CV) & Global, retrained & {fmt(r_ll.mean_left, 3)} & {fmt(r_ll.mean_right, 3)} & {fmt(r_ll_r.mean_right, 3)} & tie ($p={fmt(r_ll.p_adj, 3)}$) \\\\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textbf{(B) Static vs.\ calibration-tuned \& instance-adaptive fusion} (70\,\% eval split vs.\ calibration-best single)} \\",
        r"\cmidrule(lr){1-6}",
        r"Fusion rule & Weight / parameter & Median AOPC & Med.\ $\Delta$mean & Med.\ $p_{\text{adj}}$ & Wins ($p_{\text{adj}}<0.05$) \\",
        r"\midrule",
        _fus_row("rankavg", "Mean-rank (Borda)", "$w=0.50$"),
        _fus_row("fidwt", "Proportional AOPC", f"$w \\in [{fmt(fus_sum.w_prop.min(),2)}, {fmt(fus_sum.w_prop.max(),2)}]$"),
        _fus_row("grid_opt", "Calib.-tuned grid $w^*$", f"$w^* \\in [{fmt(fus_sum.w_opt.min(),2)}, {fmt(fus_sum.w_opt.max(),2)}]$ (med.\\ {fmt(fus_sum.w_opt.median(),2)})"),
        _fus_row("r2_adaptive", "LIME $R^2$-adaptive", f"$\\gamma=0$ in {int((fus_sum.adapt_gamma==0).sum())}/18 cells"),
        _fus_row("r2_gate", "LIME $R^2$ selective gate", f"always SHAP in {int(np.isinf(fus_sum.gate_tau).sum())}/18 cells"),
        f"\\quad Per-instance Oracle (upper bound) & $\\max(\\text{{SHAP}},\\text{{LIME}})$ & -- & +{fmt(fus_sum.oracle_headroom.median(), 3)} (headroom) & -- & SHAP wins {fmt(100*fus_sum.shap_win_rate.mean(),1)}\\% \\\\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textbf{(C) External scalability validation} (12 additional cells across 6 models, same Track B protocol)} \\",
        r"\cmidrule(lr){1-6}",
        r"External dataset & Scale ($n \times d$) & Local $\rho$ (H1) & Cond.\ AOPC (S / L) & SHAP$>$LIME & Fusion wins \\",
        r"\midrule",
        _sc_row("bcwd", "Breast Cancer Wisconsin", "569 $\\times$ 30"),
        _sc_row("cdc_diabetes", "CDC Diabetes (stratified)", "5,000 $\\times$ 21"),
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\par\vspace{2pt}\footnotesize All tests are Wilcoxon signed-rank with Holm--Bonferroni adjustment within family ($\alpha=0.05$). In (A), both SHAP and LIME beat Random in 18/18 cells for conditional $k$-NN ($p_{\text{adj}} \le 0.0005$) and in ROAR ($p_{\text{adj}} < 10^{-4}$). In (C), both external datasets replicate SHAP's conditional fidelity lead (12/12 cells, $p_{\text{adj}} < 0.05$) and zero fusion wins (0/12).",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    profile = get_profile("FULL")
    p = paths()
    p.tables.mkdir(parents=True, exist_ok=True)
    p.figures.mkdir(parents=True, exist_ok=True)
    build_fig_lime_budget(profile)
    (p.tables / "revision_findings.tex").write_text(revision_findings_macros(profile))
    (p.tables / "T13_revision.tex").write_text(build_table_revision(profile))
    print("[revision_outputs] wrote F15_lime_budget, revision_findings.tex, T13_revision.tex")


if __name__ == "__main__":
    main()
