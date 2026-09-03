# Reproduction package: usability of multi-signature self-custody

Data, scripts, and notebooks that reproduce every statistic, table, and figure
in the Results section of:

> Gindre, F., Riva, M. F., Clavería, M. G., Urbieta, M., Rossi, G.
> *Measuring the impact of multi-signature self-custody on crypto-wallet
> usability and security perception.* (under review, 2026)

The experiment is a between-subjects usability study (n = 67) of the
*Shared-custody with Self* pattern: participants completed a cryptocurrency
spending task with the Safe wallet on the Ethereum Sepolia test network under
one of four treatments, then answered the System Usability Scale (SUS) and a
six-item security-perception questionnaire.

| Treatment | Scheme | Devices |
|---|---|---|
| T1 | single-signature (1-of-1) | mobile only |
| T2 | multi-signature (2-of-2) | initiated on desktop browser, authorized on mobile |
| T3 | multi-signature (2-of-2) | initiated on mobile, authorized on desktop browser |
| T4 | single-signature (1-of-1) | desktop browser only |

## Contents

| Path | What it is |
|---|---|
| `data/shared-custody-sus-experiment.csv` | Pseudonymized dataset, one row per participant (n = 67). |
| `data/CODEBOOK.md` | Variable definitions, scoring formulas, and provenance notes. |
| `scripts/analysis.py` | Recomputes every number in the Results section and regenerates every figure and table into `output/`. |
| `scripts/verify.py` | Compares `output/values.json` with the values printed in the manuscript (`paper_values.json`). |
| `scripts/build_dataset.py` | Builds the CSV from the raw Google Forms export. The raw export is private (see Data provenance) and is not distributed. |
| `notebooks/` | The co-authors' analysis notebooks, one per research question, adapted to read the public CSV. |
| `output/` | Committed results: `report.md`, `verification.md`, `values.json`, `tables/*.tex`, `figures/*.pdf|png`, and `notebooks/` (files written by the notebooks). You can inspect these without running anything. |
| `paper_values.json` | The values as printed in the manuscript, with the tolerance implied by the decimals shown. |

## Quick start

The package is managed with [uv](https://docs.astral.sh/uv/), which pins
the Python interpreter (`.python-version`) and every dependency (`uv.lock`)
so the same numbers are obtained on macOS, Linux, and Windows.

```sh
# 1. install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
# or: brew install uv
# or, on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. clone and enter the repository
git clone https://github.com/lifia-unlp/shared-custody-with-self-experiment-repro-pkg
cd shared-custody-with-self-experiment-repro-pkg

# 3. create the environment (downloads Python 3.11 if needed) and run
uv sync
uv run scripts/analysis.py        # -> output/report.md, tables, figures, values.json
uv run scripts/verify.py          # -> output/verification.md, exit code 0 when all match
```

To execute the notebooks (optional, requires the `dev` dependency group,
which `uv sync` installs by default):

```sh
cd notebooks
uv run jupyter nbconvert --to notebook --execute --inplace *.ipynb
# or open them interactively:
uv run jupyter lab
```

The notebooks write their own tables and figures to `output/notebooks/`.

## What is reproduced

`scripts/analysis.py` computes, in the order of the paper's Results section:

1. **Normality of the SUS scores.** Shapiro-Wilk W and p for the 67 scores,
   mean, median, SD, skewness, excess kurtosis (with its test p-value), Tukey
   outliers, and the histogram (`figures/sus_ux_histogram.pdf`). Shapiro-Wilk
   per treatment.
2. **SUS scores per treatment.** Mean, SD, min, max, quartiles, median;
   Sauro-Lewis grade, quartile, acceptability, NPS category, and industry
   benchmark interpretation; the per-treatment box plot
   (`figures/boxplot_SUS_treatments.pdf`); per-item means and SDs and the
   grouped bar plot (`figures/sus_questions_barplot.pdf`).
3. **RQ1, usability against the SUS benchmark of 68.** Shapiro-Wilk per
   signature scheme; one-sample Wilcoxon signed-rank test of each scheme
   against 68 (H1: median > 68); Mann-Whitney U between schemes; both arms of
   a TOST equivalence test with a +/- 5 margin, for reference; box plot by
   scheme (`figures/boxplot_RQ1.pdf`); post-hoc per-treatment tests against
   the benchmark (one-sample t with the one-sided 95% lower confidence bound,
   exact Wilcoxon, sign test, Holm correction over the two multi-signature
   treatments).
4. **RQ2, effect of the initiating device.** Mann-Whitney U, T1 vs T4 and T2
   vs T3.
5. **RQ3, security perception.** Security score per participant, defined only
   for complete six-item responses; exclusion of the seven participants with
   no security answers and of the one partial responder (n = 59, see Known
   data notes); Cronbach's alpha; attrition checks (Mann-Whitney on SUS,
   chi-square with its minimum expected count, and the Fisher-Freeman-Halton
   exact test by treatment and by experiment day); descriptives per treatment
   and per scheme; Shapiro-Wilk, Welch's t with 95% CI, Hedges' g,
   Mann-Whitney U; TOST equivalence at margins of 0.3, 0.5, and 0.8 pooled
   SDs; box plot (`figures/boxplot_security.pdf`). A sensitivity block
   recomputes RQ3 under the two other P53 rules that circulated earlier.
6. **Correlation analysis.** Spearman's rho between SUS and security scores,
   overall and per treatment (n = 59), with scatter plots
   (`figures/spearman_*.pdf`).

`output/verification.md` lists 104 values from the manuscript next to the
reproduced value. All reproduce within the printed precision except one
documented misprint (the treatment 2 correlation, computed 0.817, printed as
0.81 in the Results section while the Discussion says 0.82).

## Statistical procedures, exactly as run

All tests use `scipy.stats` (version pinned in `uv.lock`).

| Step | Call |
|---|---|
| Shapiro-Wilk | `shapiro(x)` |
| Skewness, kurtosis | `skew(x, bias=False)`, `kurtosis(x, fisher=True, bias=False)`, `kurtosistest(x)` |
| Outliers | Tukey fences, 1.5 x IQR, `numpy.percentile` linear interpolation; reported, never removed |
| RQ1 one-sample | `wilcoxon(x - 68, alternative="greater", method="exact")` |
| RQ1 between schemes | `mannwhitneyu(single, multi, alternative="two-sided")` |
| TOST arms | `wilcoxon(x - 63, alternative="greater", method="exact")`, `wilcoxon(x - 73, alternative="less", method="exact")` |
| RQ2 | `mannwhitneyu(a, b, alternative="two-sided")` |
| RQ1 per treatment | `ttest_1samp(x, 68, alternative="greater")` with the one-sided 95% lower confidence bound; exact Wilcoxon; sign test via `binomtest`; Holm over T2/T3 |
| RQ3 | `ttest_ind(a, b, equal_var=False)`; Welch-Satterthwaite CI; Hedges' g with the small-sample correction; `mannwhitneyu(a, b, alternative="two-sided")` |
| RQ3 equivalence | TOST via two one-sided Welch tests, margins of 0.3 / 0.5 / 0.8 pooled SDs, p = max of the two arms |
| Cronbach's alpha | standard formula on the six items after reverse-coding the even (negatively worded) items, complete cases only |
| Attrition | `mannwhitneyu(SUS answered, SUS not answered)`; `chi2_contingency(treatment x answered)` reported with its minimum expected count; Fisher-Freeman-Halton exact test (Monte Carlo permutation of the chi-square statistic, 20000 resamples, seed 0) by treatment and by day |
| Correlation | `spearmanr(sus, security)` |

Two choices deserve a note:

- **Wilcoxon p-values are from the exact null distribution.** SUS scores are
  multiples of 2.5, so the differences from 68 contain ties. `scipy` versions
  before 1.18 chose the exact method automatically for n <= 50 regardless of
  ties; version 1.18 switches to the asymptotic method when ties are present.
  The manuscript reports the exact values (p = 0.00049 and 0.19273); the
  asymptotic ones (0.00073 and 0.18810) are printed alongside in
  `output/report.md`. The conclusions are the same under either method.
- **No outliers are removed and no responses are capped.** Every analysis
  uses the full 1 to 5 Likert responses of all 67 participants.

## Data provenance and privacy

Responses were collected with Google Forms in Spanish on 6, 7, and 8 November
2024 at the Faculty of Informatics, National University of La Plata. The raw
export contains exact timestamps and a free-text comment field and is not
distributed. `scripts/build_dataset.py` turns that export into the public CSV:

- participants are sorted by submission time and renamed P01 to P67;
- timestamps are reduced to the experiment day (1, 2, 3);
- headers and categorical values are translated to English;
- Likert labels such as `5 (Totalmente de acuerdo)` are parsed to integers;
- the SUS score and the security score are computed from the items;
- the free-text comment column is dropped.

The transformation is deterministic. Researchers who need the raw export or
the comments can request them from the corresponding author.

## Known data notes

- **P53 (treatment 2) answered five of the six security items.** Following
  the exclusion rule stated in the paper, the participant is excluded from
  RQ3 and the correlation analysis, and `security_score` is empty in the CSV
  (it is defined only for complete six-item responses). Two other rules
  circulated in earlier versions of the analysis (the co-authors' spreadsheet
  summed the blank as 0, giving 7.5; an earlier version of this package
  scored it as neutral, giving 15.0); `output/report.md` includes a
  sensitivity block showing that neither changes any conclusion. The column
  `security_items_answered` lets analysts apply another rule.
- **Seven participants answered no security item.** Together with P53 this
  leaves n = 59 for RQ3 and the correlation analysis. The exclusions are
  unrelated to SUS score and treatment but concentrate in the first
  experiment day (Fisher-Freeman-Halton p = 0.010), which the paper notes as
  a threat.
- **Row order.** The co-authors' spreadsheet keeps the form's export order,
  in which two submissions of 7 November appear out of chronological order;
  the CSV is in chronological order, so P25 and P26 are swapped relative to
  the spreadsheet. No statistic depends on row order.

## Notebooks

`notebooks/` contains the six analysis notebooks written by the co-authors
(originally run on Windows with Python 3.11 against the private spreadsheet,
September 2026 revision).
The only edits are: a first cell that imports `repro_data.py`, which rebuilds
the spreadsheet's `Datos cuanti` sheet from the public CSV with the original
Spanish column names and redirects file output to `output/notebooks/`; the
`pd.read_excel(...)` calls replaced by `repro_data.read_datos_cuanti()`; a
file-existence check disabled; and `method="exact"` added to the Wilcoxon
calls (see above). Comments and printed labels remain in Spanish.

| Notebook | Paper section |
|---|---|
| `normality_shapiro_wilk.ipynb` | Normality of the SUS scores |
| `histograms_and_items.ipynb` | SUS histogram, per-item bar plot |
| `rq1_sus_benchmark.ipynb` | RQ1 descriptives, Wilcoxon, Mann-Whitney, box plots |
| `rq2_initiating_device.ipynb` | RQ2 Mann-Whitney |
| `security_perception.ipynb` | RQ3 tests: alpha, Welch, Mann-Whitney, TOST, attrition |
| `security_boxplot.ipynb` | RQ3 descriptives and box plot |
| `correlation_spearman.ipynb` | Spearman correlations and scatter plots |

The paper repository also contains `reproduction/v2.1/`, a scripted rework of
the notebooks by one of the co-authors; its numbers agree with this package
for every value the manuscript reports.

## Citation and license

This package is released under the Creative Commons Attribution 4.0
International license (CC BY 4.0), see `LICENSE`. Please cite the article
above and the package (`CITATION.cff`; a Zenodo DOI will be added on deposit).
