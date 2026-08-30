# Submission checklist — MDPI (JRFM / Risks)

## Before you submit

- [ ] **Corresponding author email** — `main.tex` currently has no email
- [ ] **ORCIDs** for all three authors (MDPI requires at least the corresponding author's)
- [ ] **Confirm author contributions** — the CRediT statement in `main.tex` was carried
      over from the original draft and may not reflect who did what
- [ ] **Funding statement** — MDPI requires one; write "This research received no
      external funding" if that is the case
- [ ] **Institutional Review Board / Informed Consent** — MDPI requires both fields;
      "Not applicable" for a study using only public market data
- [ ] **Zenodo release** — tag the repository, mint a DOI, and put it in the Data
      Availability Statement alongside the GitHub link
- [ ] **Co-author read-through** — nobody but the first author has read the
      rewritten Methodology, Results and Conclusion

## Journal choice

| | JRFM | Risks |
|---|---|---|
| Scope fit | applied finance, forecasting | risk measurement, calibration |
| Why it fits | density forecasting of equity returns | interval calibration is the core result |
| Impact factor | ~2.0 | ~1.9 |
| APC | ~CHF 1600 | ~CHF 2400 |

**Risks** is arguably the better fit: the paper's central contribution is the
calibration of predictive distributions, which is a risk-measurement question.
**JRFM** is cheaper and broader.

## Converting to the MDPI template

MDPI supplies `mdpi.cls` via their LaTeX template pack. The conversion is
mechanical because the content is already in separate `\input` files:

1. Download the template from the journal's Instructions for Authors.
2. Replace the preamble in `main.tex` with the MDPI preamble; keep the
   `\input{...}` lines unchanged.
3. MDPI uses its own bibliography style (`mdpi.bst`) — `references.bib` needs no
   edits, only the `\bibliographystyle` line changes.
4. MDPI wants figures as separate files with captions in the text; ours are
   already separate PDFs in `stock_project/reports/figures_v2/`.
5. Tables use `booktabs`, which MDPI supports.

## Cover letter — the three points worth making

1. The paper evaluates density forecasts rather than point forecasts, which is
   where the practical value of a return forecast lies, and reports calibration
   alongside sharpness.
2. Every comparison is tested by Diebold--Mariano with a HAC variance, and
   per-asset results carry a multiplicity correction. The sampler is validated
   against synthetic data with known parameters.
3. The full pipeline is released with pinned dependencies and two audit scripts
   that gate the failure mode documented in Section 4.7.

## Anticipated referee questions, and where they are answered

| Question | Answer |
|---|---|
| Why not roll the GARCH parameters? | Limitations, stated as the obvious extension |
| Is the calibration actually good? | Section 4.1: improved, not achieved; PIT rejects at p = 0.012 |
| Did you tune the neural networks? | Section 4.6: six configurations selected on validation, as for ridge |
| Only eight assets? | Limitations: the between-ticker variance is identified by eight groups |
| Is the squared return an adequate proxy? | Methodology: QLIKE is proxy-robust (Patton 2011) |
| Does the model beat a baseline on RMSE? | No, and the paper says so in Section 4.6 |
