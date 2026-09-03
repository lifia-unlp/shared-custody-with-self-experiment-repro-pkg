#!/usr/bin/env python3
"""Reproduce every statistic and figure in the Results section of the paper.

Input:   data/shared-custody-sus-experiment.csv (pseudonymized, n = 67)
Output:  output/report.md            all statistics, in paper order
         output/tables/*.tex         LaTeX tables
         output/figures/*.pdf|png    all Results figures

Run with `uv run analysis` (or `uv run python scripts/analysis.py`).
All paths are resolved relative to this file; the script is deterministic
(fixed random seed for jitter in box plots).

Statistical procedures (scipy 1.17 / statsmodels 0.14 defaults unless noted):

  Normality       scipy.stats.shapiro; skewness scipy.stats.skew(bias=False);
                  excess kurtosis scipy.stats.kurtosis(fisher=True, bias=False)
                  with scipy.stats.kurtosistest for its p-value; outliers by
                  Tukey fences (1.5 x IQR, numpy linear percentiles), reported
                  only, never removed.
  RQ1             one-sample Wilcoxon signed-rank on (SUS - 68),
                  alternative="greater", zero_method="wilcox", method="exact"
                  (the exact null distribution, which is what scipy < 1.18
                  selected automatically for n <= 50; SUS scores are tied,
                  so the asymptotic p-value is also reported);
                  Mann-Whitney U two-sided between single- and multi-signature
                  groups; TOST arms (Wilcoxon on SUS - 63 "greater" and
                  SUS - 73 "less") reported for reference.
  RQ1 per treat.  post-hoc: one-sample t vs 68 (alternative="greater") with
                  the one-sided 95% lower confidence bound, exact Wilcoxon,
                  sign test (binomtest), and Holm correction over the two
                  multi-signature treatments.
  RQ2             Mann-Whitney U two-sided, T1 vs T4 and T2 vs T3.
  RQ3             security score = ((sum odd - 3) + (15 - sum even)) x 2.5,
                  defined only for complete six-item responses; participants
                  with any missing security item excluded (7 blank + 1
                  partial = 8, leaving n = 59); Cronbach's alpha on the six
                  items (negatively worded items reverse-coded); attrition
                  checks (Mann-Whitney on SUS answered vs not; chi-square
                  treatment x answered with its minimum expected count;
                  Fisher-Freeman-Halton exact test by treatment and by day,
                  Monte Carlo with 20000 resamples, seed 0); Shapiro per
                  group; Welch t-test, Welch 95% CI, Hedges' g, Mann-Whitney;
                  TOST equivalence with margins of d = 0.3, 0.5, 0.8 pooled
                  SDs (two one-sided Welch tests, p = max of the arms).
  Correlation     Spearman rho with two-sided p, overall and per treatment,
                  on the n = 59 participants with a security score.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "shared-custody-sus-experiment.csv")
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")

PDF_META = {"CreationDate": None, "ModDate": None}  # byte-reproducible PDFs
BENCHMARK = 68.0
MARGIN = 5.0
SINGLE = (1, 4)
MULTI = (2, 3)
TREAT_NAMES = {1: "T1 (single-sig, mobile)", 2: "T2 (2-of-2, desktop init)",
               3: "T3 (2-of-2, mobile init)", 4: "T4 (single-sig, desktop)"}
SHORT = {1: "T1", 2: "T2", 3: "T3", 4: "T4"}
SEC_LONG = {1: "Treatment 1 - Mobile Device Only",
            2: "Treatment 2 - Two Devices - Starting on PC",
            3: "Treatment 3 - Two Devices - Starting on Mobile",
            4: "Treatment 4 - PC Only"}
SUS_ITEMS = [
    "I think that I would like to use this system frequently",
    "I found the system unnecessarily complex",
    "I thought the system was easy to use",
    "I think that I would need the support of a technical person",
    "I found the various functions well integrated",
    "I thought there was too much inconsistency in this system",
    "I would imagine that most people would learn to use this system very quickly",
    "I found the system very cumbersome to use",
    "I felt very confident using the system",
    "I needed to learn a lot of things before I could get going",
]

# ------------------------------------------------------------------ helpers


class Report:
    def __init__(self):
        self.lines = []
        self.values = {}

    def add(self, s=""):
        self.lines.append(s)

    def h(self, level, s):
        self.add()
        self.add("#" * level + " " + s)
        self.add()

    def table(self, header, rows):
        self.add("| " + " | ".join(header) + " |")
        self.add("|" + "---|" * len(header))
        for r in rows:
            self.add("| " + " | ".join(str(c) for c in r) + " |")
        self.add()

    def val(self, key, v):
        """Record a named numeric value for verify.py."""
        self.values[key] = float(v)
        return v

    def write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        with open(os.path.join(OUT, "values.json"), "w") as f:
            json.dump(self.values, f, indent=1, sort_keys=True)


def f(x, d=4):
    return f"{x:.{d}f}"


def tukey_outliers(data):
    q1, q3 = np.percentile(data, [25, 75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return sorted(x for x in data if x < lo or x > hi)


def grade(score):
    """Sauro-Lewis curved grading scale."""
    for cut, g in [(84.1, "A+"), (80.8, "A"), (78.9, "A-"), (77.2, "B+"),
                   (74.1, "B"), (72.6, "B-"), (71.1, "C+"), (65.0, "C"),
                   (62.7, "C-"), (51.7, "D")]:
        if score >= cut:
            return g
    return "F"


def quartile(score):
    """Sauro-Lewis quartile scale."""
    if score < 62.7:
        return "1st"
    if score < 72.6:
        return "2nd"
    if score < 78.9:
        return "3rd"
    return "4th"


def acceptability(score):
    if score < 51.7:
        return "Not acceptable"
    return "Acceptable" if score >= 72.6 else "Marginal"


def nps(score):
    if score >= 78.9:
        return "Promoter"
    return "Detractor" if score < 62.7 else "Passive"


def industry(score):
    return "Above Average" if score >= BENCHMARK else "Below Average"


def cronbach_alpha(items):
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum()
                          / items.sum(axis=1).var(ddof=1))


def hedges_g(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                 / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    return d * j, d


def welch(a, b):
    res = stats.ttest_ind(a, b, equal_var=False)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    ci = stats.t.interval(0.95, res.df, loc=diff, scale=se)
    return res.statistic, res.df, res.pvalue, diff, ci


def pooled_sd(a, b):
    return np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                   / (len(a) + len(b) - 2))


def tost_welch(a, b, margin):
    """TOST equivalence via two one-sided Welch tests; returns max arm p."""
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dof = stats.ttest_ind(a, b, equal_var=False).df
    diff = a.mean() - b.mean()
    p_lower = stats.t.sf((diff + margin) / se, dof)   # H1: diff > -margin
    p_upper = stats.t.cdf((diff - margin) / se, dof)  # H1: diff < +margin
    return max(p_lower, p_upper)


def fisher_freeman_halton_mc(labels, groups, n_resamples=20000, seed=0):
    """Monte Carlo Fisher-Freeman-Halton test on an RxC table, chi-square
    statistic, permutation of group labels; seeded, so deterministic."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    obs = stats.chi2_contingency(pd.crosstab(groups, labels)).statistic
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_resamples):
        perm = rng.permutation(labels)
        s = stats.chi2_contingency(pd.crosstab(groups, perm)).statistic
        hits += s >= obs - 1e-12
    return (hits + 1) / (n_resamples + 1)


def holm_adjust(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def latex_table(path, header, rows, caption, label):
    cols = "l" + "c" * (len(header) - 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\\begin{table}[tbp]\n\\centering\n")
        fh.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
        fh.write(" & ".join(header) + " \\\\ \\midrule\n")
        for r in rows:
            fh.write(" & ".join(str(c) for c in r) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
        fh.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


# ------------------------------------------------------------------ figures


def boxplot(groups, labels, ref, ylabel, title, name, ylim):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 2, 5))
    ax.boxplot(groups, tick_labels=labels, widths=0.5, showfliers=True)
    for i, g in enumerate(groups, 1):
        ax.scatter(i + rng.normal(0, 0.05, len(g)), g, s=14, alpha=0.5,
                   color="tab:blue", zorder=3)
        ax.scatter([i], [np.mean(g)], marker="D", color="black", s=28, zorder=4)
        ax.annotate(f"{np.mean(g):.2f}", (i, np.mean(g)),
                    textcoords="offset points", xytext=(14, -4), fontsize=8)
    ax.axhline(ref, color="red", linestyle="--", linewidth=1,
               label=f"Reference = {ref:g}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"), metadata=PDF_META)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150)
    plt.close(fig)


def scatter(x, y, rho, p, title, name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, alpha=0.7)
    if len(x) > 1:
        m, c = np.polyfit(x, y, 1)
        xs = np.array([min(x), max(x)])
        ax.plot(xs, m * xs + c, color="tab:red", linewidth=1)
    ax.set_xlabel("SUS score")
    ax.set_ylabel("Perceived-security score")
    ax.set_title(f"{title}\nSpearman rho = {rho:.2f}, p = {p:.3g}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"), metadata=PDF_META)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- analysis


def main():
    for d in (OUT, FIG, TAB):
        os.makedirs(d, exist_ok=True)
    df = pd.read_csv(DATA)
    R = Report()
    R.add("# Reproduction report")
    R.add()
    R.add("Generated by `scripts/analysis.py` from "
          "`data/shared-custody-sus-experiment.csv`.")
    sus = df["sus_score"].to_numpy(float)
    by_t = {t: df.loc[df.treatment == t, "sus_score"].to_numpy(float)
            for t in (1, 2, 3, 4)}
    single = np.concatenate([by_t[t] for t in SINGLE])
    multi = np.concatenate([by_t[t] for t in MULTI])

    # ---------------------------------------------- normality of SUS scores
    R.h(2, "Normality of the SUS scores (Table: Shapiro-Wilk of the sample)")
    W, p = stats.shapiro(sus)
    sk = stats.skew(sus, bias=False)
    ku = stats.kurtosis(sus, fisher=True, bias=False)
    kp = stats.kurtosistest(sus).pvalue
    outl = tukey_outliers(sus)
    rows = [("P-value", f(R.val("sw_p", p), 8)), ("W", f(R.val("sw_W", W))),
            ("n", len(sus)), ("Mean", f(R.val("sus_mean", sus.mean()), 3)),
            ("Median", f(R.val("sus_median", np.median(sus)), 1)),
            ("SD", f(R.val("sus_sd", sus.std(ddof=1)))),
            ("Skewness", f(R.val("sus_skew", sk))),
            ("Excess kurtosis", f"{f(R.val('sus_kurt', ku))} (p = {f(kp, 3)})"),
            ("Outliers (Tukey)", ", ".join(map(str, outl)))]
    R.table(["Parameter", "Value"], rows)
    R.values["sus_outliers"] = outl
    latex_table(os.path.join(TAB, "shapiro_sample.tex"), ["Parameter", "Value"],
                rows, "Shapiro-Wilk test of the SUS score sample.",
                "tab:sus-shapiro-wilk")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sus, bins=10, edgecolor="black")
    ax.set_xlabel("SUS score")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of SUS scores (n = 67)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sus_ux_histogram.pdf"), metadata=PDF_META)
    fig.savefig(os.path.join(FIG, "sus_ux_histogram.png"), dpi=150)
    plt.close(fig)

    R.h(2, "Shapiro-Wilk per treatment")
    rows = []
    for t in (1, 2, 3, 4):
        W, p = stats.shapiro(by_t[t])
        R.val(f"sw_T{t}_W", W)
        R.val(f"sw_T{t}_p", p)
        rows.append((SHORT[t], len(by_t[t]), f(W), f(p, 3),
                     "Rejected" if p < 0.05 else "Not rejected"))
    R.table(["Treatment", "N", "W", "p-value", "Normality"], rows)
    latex_table(os.path.join(TAB, "shapiro_treatments.tex"),
                ["Treatment", "N", "W", "p-value", "Normality"], rows,
                "Shapiro-Wilk normality test results per treatment.",
                "tab:shapiro-wilk")

    # ------------------------------------------------- SUS per treatment
    R.h(2, "SUS scores per treatment")
    rows, rows2, rows_tex = [], [], []
    for t in (1, 2, 3, 4):
        d = by_t[t]
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        R.val(f"T{t}_mean", d.mean()); R.val(f"T{t}_sd", d.std(ddof=1))
        R.val(f"T{t}_median", med); R.val(f"T{t}_q1", q1); R.val(f"T{t}_q3", q3)
        R.val(f"T{t}_min", d.min()); R.val(f"T{t}_max", d.max())
        rows.append((SHORT[t], len(d), f(d.mean(), 2), f(d.std(ddof=1), 2),
                     f(d.min(), 1), f(d.max(), 1), f(q1, 2), f(med, 2), f(q3, 2)))
        rows2.append((SHORT[t], grade(d.mean()), quartile(d.mean()),
                      acceptability(d.mean()), nps(d.mean()), industry(d.mean())))
    R.table(["Treatment", "N", "Mean", "SD", "Min", "Max", "Q1", "Median", "Q3"], rows)
    R.add("Interpretation (Sauro-Lewis grade, quartile, acceptability, NPS, "
          "industry benchmark of 68), based on the treatment mean:")
    R.add()
    R.table(["Treatment", "Grade", "Quartile", "Acceptability", "NPS", "Industry"], rows2)
    latex_table(os.path.join(TAB, "sus_descriptives.tex"),
                ["Treatment", "N", "SUS (mean)", "SD", "Min", "Max", "1st Q", "Median", "3rd Q"],
                rows, "Descriptive statistics of the SUS scores per treatment.",
                "tab:sus-score-stats")
    latex_table(os.path.join(TAB, "sus_interpretation.tex"),
                ["Treatment", "Grade Scale", "Quartile Scale", "Acceptability", "NPS", "Industry Benchmark"],
                rows2, "Adjective and acceptability interpretation of the SUS scores per treatment.",
                "tab:sus-score-stats-1")
    boxplot([by_t[t] for t in (1, 2, 3, 4)], [SHORT[t] for t in (1, 2, 3, 4)],
            BENCHMARK, "SUS score", "SUS scores per treatment",
            "boxplot_SUS_treatments", (0, 105))

    # per-question means (unrecoded 1..5) and bar plot
    R.h(2, "SUS items per treatment (raw 1-5 responses, mean and SD)")
    long = df.melt(id_vars=["treatment"], value_vars=[f"sus_q{i}" for i in range(1, 11)],
                   var_name="item", value_name="response")
    long["item"] = long["item"].str.replace("sus_q", "Q").astype(str)
    long["Treatment"] = long["treatment"].map(SHORT)
    g = long.groupby(["item", "Treatment"], observed=True)["response"].agg(["mean", "std", "count"])
    g = g.reindex([f"Q{i}" for i in range(1, 11)], level=0)
    rows = [(i, t, f(m, 2), f(s, 2), int(c)) for (i, t), (m, s, c) in g.iterrows()]
    R.table(["Item", "Treatment", "Mean", "SD", "n"], rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    order = [f"Q{i}" for i in range(1, 11)]
    sns.barplot(data=long, x="item", y="response", hue="Treatment", order=order,
                hue_order=["T1", "T2", "T3", "T4"], errorbar="sd", capsize=0.05, ax=ax)
    ax.axhline(5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("SUS item")
    ax.set_ylabel("Response (1-5)")
    ax.set_title("SUS responses per item, grouped by treatment (mean +/- SD)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sus_questions_barplot.pdf"), metadata=PDF_META)
    fig.savefig(os.path.join(FIG, "sus_questions_barplot.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------- RQ1
    R.h(2, "RQ1: usability against the SUS benchmark of 68")
    rows = []
    for name, d in (("Single-signature (T1+T4)", single), ("Multi-signature (T2+T3)", multi)):
        W, p = stats.shapiro(d)
        rows.append((name, len(d), f(W), f(p, 5)))
    R.val("sw_single_W", stats.shapiro(single).statistic)
    R.val("sw_single_p", stats.shapiro(single).pvalue)
    R.val("sw_multi_W", stats.shapiro(multi).statistic)
    R.val("sw_multi_p", stats.shapiro(multi).pvalue)
    R.table(["Group", "N", "Shapiro-Wilk W", "p-value"], rows)
    latex_table(os.path.join(TAB, "shapiro_groups.tex"),
                ["Group", "N", "Shapiro-Wilk $W$", "P-value"], rows,
                "Shapiro-Wilk normality test results per group.", "tab:shapiro_group_results")

    rows = []
    for key, name, d in (("single", "Single-signature (T1+T4)", single),
                         ("multi", "Multi-signature (T2+T3)", multi)):
        res = stats.wilcoxon(d - BENCHMARK, alternative="greater", method="exact")
        apx = stats.wilcoxon(d - BENCHMARK, alternative="greater", method="approx")
        R.val(f"rq1_{key}_W", res.statistic); R.val(f"rq1_{key}_p", res.pvalue)
        R.val(f"rq1_{key}_p_approx", apx.pvalue)
        R.val(f"{key}_mean", d.mean()); R.val(f"{key}_median", np.median(d))
        rows.append((name, len(d), f(d.mean(), 2), f(np.median(d), 2),
                     f(res.statistic, 3), f(res.pvalue, 5), f(apx.pvalue, 5)))
    R.add("One-sample Wilcoxon signed-rank test, H1: median > 68 "
          "(`scipy.stats.wilcoxon(x - 68, alternative='greater', method='exact')`; "
          "the last column is the asymptotic p-value, `method='approx'`):")
    R.add()
    R.table(["Group", "N", "Mean", "Median", "Wilcoxon W", "p-value (exact)", "p-value (approx)"], rows)
    latex_table(os.path.join(TAB, "wilcoxon_groups.tex"),
                ["Group", "N", "Mean", "Median", "Wilcoxon $W$", "P-value (exact)", "P-value (approx.)"], rows,
                "Wilcoxon signed-rank test results per group against the reference value (68); exact and asymptotic p-values.",
                "tab:wilcoxon_group_results")

    res = stats.mannwhitneyu(single, multi, alternative="two-sided")
    R.val("rq1_mw_U", res.statistic); R.val("rq1_mw_p", res.pvalue)
    R.add("Mann-Whitney U, single-signature vs multi-signature, two-sided "
          "(`scipy.stats.mannwhitneyu(single, multi, alternative='two-sided')`):")
    R.add()
    R.table(["Comparison", "U", "p-value (two-sided)"],
            [("Single-sign vs Multi-sign", f(res.statistic, 2), f(res.pvalue, 5))])
    latex_table(os.path.join(TAB, "mannwhitney_groups.tex"),
                ["Comparison", "Mann-Whitney $U$", "P-value (two-sided)"],
                [("Single-sign vs. Multi-sign", f(res.statistic, 2), f(res.pvalue, 5))],
                "Mann-Whitney U test comparing single-signature and multi-signature groups' SUS scores.",
                "tab:mannwhitney_group_comparison")

    lo = stats.wilcoxon(multi - (BENCHMARK - MARGIN), alternative="greater", method="exact")
    hi = stats.wilcoxon(multi - (BENCHMARK + MARGIN), alternative="less", method="exact")
    R.val("tost_multi_lower_W", lo.statistic); R.val("tost_multi_lower_p", lo.pvalue)
    R.val("tost_multi_upper_W", hi.statistic); R.val("tost_multi_upper_p", hi.pvalue)
    R.add(f"For reference, TOST arms for the multi-signature group with margin +/- {MARGIN:g} "
          "(Wilcoxon signed-rank):")
    R.add()
    R.table(["Arm", "W", "p-value"],
            [(f"median > {BENCHMARK - MARGIN:g} (non-inferiority)", f(lo.statistic, 3), f(lo.pvalue, 5)),
             (f"median < {BENCHMARK + MARGIN:g}", f(hi.statistic, 3), f(hi.pvalue, 5))])
    R.add("Equivalence within the margin requires BOTH arms to be significant; "
          "non-inferiority requires only the first.")
    boxplot([single, multi], ["Single-signature (T1+T4)", "Multi-signature (T2+T3)"],
            BENCHMARK, "SUS score", "SUS scores by signature scheme", "boxplot_RQ1", (0, 105))

    R.h(3, "Post-hoc: each treatment against the benchmark")
    R.add("One-sample t-test vs 68 (H1: mean > 68) with the one-sided 95% lower "
          "confidence bound, exact Wilcoxon, and sign test. Holm correction is "
          "applied over the two multi-signature treatments (T2, T3), matching "
          "the caveat discussed in the paper.")
    R.add()
    rows, tp = [], {}
    for t in (1, 2, 3, 4):
        d = by_t[t]
        res = stats.ttest_1samp(d, BENCHMARK, alternative="greater")
        se = d.std(ddof=1) / np.sqrt(len(d))
        lcb = d.mean() - stats.t.ppf(0.95, len(d) - 1) * se
        wp = stats.wilcoxon(d - BENCHMARK, alternative="greater", method="exact").pvalue
        pos = int((d > BENCHMARK).sum())
        nz = int((d != BENCHMARK).sum())
        sp_ = stats.binomtest(pos, nz, 0.5, alternative="greater").pvalue
        tp[t] = res.pvalue
        R.val(f"rq1_T{t}_t", res.statistic); R.val(f"rq1_T{t}_t_p", res.pvalue)
        R.val(f"rq1_T{t}_lcb", lcb); R.val(f"rq1_T{t}_wilcoxon_p", wp)
        R.val(f"rq1_T{t}_sign_p", sp_)
        rows.append((SHORT[t], len(d), f(d.mean(), 2), f(res.statistic, 3),
                     f(res.pvalue, 4), f(lcb, 2), f(wp, 4), f(sp_, 4)))
    holm = holm_adjust([tp[2], tp[3]])
    R.val("rq1_T2_holm_p", holm[0]); R.val("rq1_T3_holm_p", holm[1])
    R.table(["Treatment", "N", "Mean", "t", "p (t, one-sided)", "95% LCB",
             "p (Wilcoxon exact)", "p (sign)"], rows)
    R.add(f"Holm-adjusted t-test p over T2/T3: T2 = {f(holm[0], 4)}, T3 = {f(holm[1], 4)}.")
    R.add()

    # ------------------------------------------------------------- RQ2
    R.h(2, "RQ2: effect of the initiating device (Mann-Whitney U, two-sided)")
    rows = []
    for key, name, a, b in (("single", "Single-signature (T1 vs T4)", by_t[1], by_t[4]),
                            ("multi", "Multi-signature (T2 vs T3)", by_t[2], by_t[3])):
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
        R.val(f"rq2_{key}_U", res.statistic); R.val(f"rq2_{key}_p", res.pvalue)
        rows.append((name, len(a), len(b), f(res.statistic, 3), f(res.pvalue, 5)))
    R.table(["Comparison", "N1", "N2", "U", "p-value (two-sided)"], rows)
    latex_table(os.path.join(TAB, "mannwhitney_device.tex"),
                ["Comparison", "$N_1$", "$N_2$", "Mann-Whitney $U$", "P-value (two-sided)"], rows,
                "Mann-Whitney U test results for the effect of the initiating device.",
                "tab:mannwhitney_device_results")

    # ------------------------------------------------------------- RQ3
    R.h(2, "RQ3: security perception")
    sec_cols = [f"security_q{i}" for i in range(1, 7)]
    answered = df["security_items_answered"] == 6
    n_blank = int((df.security_items_answered == 0).sum())
    n_part = int(((df.security_items_answered > 0) & ~answered).sum())
    R.add(f"Analysis set: complete six-item responses only, n = {int(answered.sum())} "
          f"of {len(df)}. Excluded: {n_blank} participants who answered no item and "
          f"{n_part} ({df.loc[(df.security_items_answered > 0) & ~answered, 'participant_id'].item()}) "
          "who answered 5 of 6, per the exclusion rule stated in the paper "
          "(see data/CODEBOOK.md and the sensitivity block below).")
    R.add()
    items = df.loc[answered, sec_cols].to_numpy(float)
    items[:, 1::2] = 6 - items[:, 1::2]
    alpha = cronbach_alpha(items)
    R.val("cronbach_alpha", alpha)
    R.add(f"Cronbach's alpha (six items, negatively worded items reverse-coded, "
          f"complete cases n = {len(items)}): {f(alpha, 3)}")
    R.add()
    ct = pd.crosstab(df.treatment, answered)
    chi = stats.chi2_contingency(ct)
    min_expected = chi.expected_freq.min()
    R.val("attrition_chi2_p", chi.pvalue)
    mw = stats.mannwhitneyu(sus[answered.to_numpy()], sus[~answered.to_numpy()], alternative="two-sided")
    R.val("attrition_mw_p", mw.pvalue)
    ffh_t = fisher_freeman_halton_mc(answered.to_numpy(), df.treatment.to_numpy())
    ffh_d = fisher_freeman_halton_mc(answered.to_numpy(), df.day.to_numpy())
    R.val("attrition_ffh_treatment_p", ffh_t)
    R.val("attrition_ffh_day_p", ffh_d)
    R.add(f"Attrition checks: Mann-Whitney SUS answered vs not, U = {f(mw.statistic, 1)}, "
          f"p = {f(mw.pvalue, 3)}. Chi-square treatment x answered, chi2 = {f(chi.statistic, 3)}, "
          f"df = {chi.dof}, p = {f(chi.pvalue, 3)}, but its minimum expected count is "
          f"{f(min_expected, 2)}, so the approximation is not valid; the "
          f"Fisher-Freeman-Halton exact test (Monte Carlo, 20000 resamples, seed 0) gives "
          f"p = {f(ffh_t, 3)} by treatment and p = {f(ffh_d, 3)} by experiment day. "
          "Excluded participants per treatment: "
          + ", ".join(f"{SHORT[t]}={int(v)}" for t, v in ct[False].items())
          + "; per day: "
          + ", ".join(f"day {d}={int(v)}" for d, v in
                      pd.crosstab(df.day, answered)[False].items()))
    R.add()

    sdf = df[answered].copy()
    sec_t = {t: sdf.loc[sdf.treatment == t, "security_score"].to_numpy(float) for t in (1, 2, 3, 4)}
    sec_single = np.concatenate([sec_t[t] for t in SINGLE])
    sec_multi = np.concatenate([sec_t[t] for t in MULTI])
    rows = []
    for name, key, d in [(SHORT[t], f"sec_T{t}", sec_t[t]) for t in (1, 2, 3, 4)] + [
            ("T1+T4 (single)", "sec_single", sec_single), ("T2+T3 (multi)", "sec_multi", sec_multi)]:
        R.val(f"{key}_mean", d.mean()); R.val(f"{key}_median", np.median(d))
        R.val(f"{key}_sd", d.std(ddof=1)); R.val(f"{key}_min", d.min())
        rows.append((name, len(d), f(d.mean(), 2), f(np.median(d), 2), f(d.std(ddof=1), 2),
                     f(d.min(), 1), f(d.max(), 1)))
    R.table(["Group", "N", "Mean", "Median", "SD", "Min", "Max"], rows)
    latex_table(os.path.join(TAB, "security_descriptives.tex"),
                ["Group", "N", "Mean", "Median", "SD", "Min", "Max"], rows,
                "Descriptive statistics of the perceived-security scores.", "tab:security-stats")
    boxplot([sec_t[t] for t in (1, 2, 3, 4)] + [sec_single, sec_multi],
            ["T1", "T2", "T3", "T4", "T1+T4", "T2+T3"], 36.0,
            "Perceived-security score (0-60)", "Perceived-security scores per treatment and group",
            "boxplot_security", (0, 65))

    sw1, sw2 = stats.shapiro(sec_single), stats.shapiro(sec_multi)
    R.val("sec_sw_single_p", sw1.pvalue); R.val("sec_sw_multi_p", sw2.pvalue)
    t, dof, p, diff, ci = welch(sec_single, sec_multi)
    R.val("sec_t", t); R.val("sec_t_df", dof); R.val("sec_t_p", p)
    R.val("sec_diff", diff); R.val("sec_ci_lo", ci[0]); R.val("sec_ci_hi", ci[1])
    g_, d_ = hedges_g(sec_single, sec_multi)
    R.val("sec_hedges_g", g_)
    mw = stats.mannwhitneyu(sec_single, sec_multi, alternative="two-sided")
    R.val("sec_mw_U", mw.statistic); R.val("sec_mw_p", mw.pvalue)
    sp = pooled_sd(sec_single, sec_multi)
    tost_rows = []
    for dd in (0.3, 0.5, 0.8):
        pv = tost_welch(sec_single, sec_multi, dd * sp)
        R.val(f"sec_tost_d{int(dd*10):02d}_p", pv)
        tost_rows.append((f"d = {dd:g} (+/- {dd * sp:.2f} points)", f(pv, 4)))
    R.table(["Statistic", "Value"], [
        ("Shapiro-Wilk p, single / multi", f"{f(sw1.pvalue, 3)} / {f(sw2.pvalue, 3)}"),
        ("Welch t", f"t({f(dof, 2)}) = {f(t, 3)}, p = {f(p, 5)}"),
        ("Mean difference (single - multi)", f"{f(diff, 2)}, 95% CI [{f(ci[0], 2)}, {f(ci[1], 2)}]"),
        ("Hedges' g (Cohen's d)", f"{f(g_, 3)} ({f(d_, 3)})"),
        ("Mann-Whitney U", f"U = {f(mw.statistic, 1)}, p = {f(mw.pvalue, 5)}"),
    ])
    R.add("TOST equivalence (two one-sided Welch tests, margin expressed as a "
          "multiple of the pooled SD; equivalence requires p < 0.05):")
    R.add()
    R.table(["Margin", "p (max of both arms)"], tost_rows)

    # sensitivity: the two n = 60 scoring variants that circulated earlier
    R.h(3, "Sensitivity: alternative treatments of the partial responder P53")
    R.add("P53 answered five of six items. The analysis above drops the "
          "participant (the paper's rule). Two other rules appeared in earlier "
          "versions of the analysis: the co-authors' spreadsheet scored the "
          "blank as 0 (score 7.5), and an earlier version of this package "
          "scored it as neutral (score 15.0). Neither changes any conclusion:")
    R.add()
    alt_rows = []
    keep = sdf["security_score"].to_numpy(float)
    keep_t = sdf["treatment"].to_numpy()
    p53 = df[df.participant_id == "P53"].iloc[0]
    for label, key, val in (("blank as 0 (spreadsheet)", "alt_stored", 7.5),
                            ("blank as neutral (score 15.0)", "alt_neutral", 15.0)):
        av = np.append(keep, val)
        at = np.append(keep_t, p53.treatment)
        a_s = av[np.isin(at, SINGLE)]
        am = av[np.isin(at, MULTI)]
        t2, dof2, p2, diff2, ci2 = welch(a_s, am)
        R.val(f"{key}_multi_mean", am.mean()); R.val(f"{key}_t_p", p2)
        alt_rows.append((label, len(av), f(am.mean(), 2), f(am.std(ddof=1), 2),
                         f"t({f(dof2, 1)}) = {f(t2, 2)}", f(p2, 3)))
    R.table(["Rule", "n", "Multi mean", "Multi SD", "Welch t", "p"], alt_rows)

    # ------------------------------------------------------ correlations
    R.h(2, "Correlation between SUS and perceived security (Spearman)")
    rows = []
    x, y = sdf["sus_score"].to_numpy(float), sdf["security_score"].to_numpy(float)
    rho, p = stats.spearmanr(x, y)
    R.val("rho_overall", rho); R.val("rho_overall_p", p)
    rows.append(("Overall", len(x), f(rho, 3), f(p, 5)))
    scatter(x, y, rho, p, "All participants", "spearman_general")
    for t in (1, 2, 3, 4):
        s = sdf[sdf.treatment == t]
        x, y = s["sus_score"].to_numpy(float), s["security_score"].to_numpy(float)
        rho, p = stats.spearmanr(x, y)
        R.val(f"rho_T{t}", rho); R.val(f"rho_T{t}_p", p)
        rows.append((SHORT[t], len(x), f(rho, 3), f(p, 5)))
        scatter(x, y, rho, p, SEC_LONG[t], f"spearman_T{t}")
    R.table(["Group", "n", "rho", "p-value"], rows)
    latex_table(os.path.join(TAB, "spearman.tex"), ["Group", "$n$", "$\\rho$", "P-value"], rows,
                "Spearman correlation between SUS and perceived-security scores.", "tab:spearman")

    R.write(os.path.join(OUT, "report.md"))
    print(f"wrote {os.path.relpath(OUT, ROOT)}/report.md, values.json, "
          f"{len(os.listdir(TAB))} tables, {len(os.listdir(FIG))} figure files")


if __name__ == "__main__":
    main()
