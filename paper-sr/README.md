# FoldFlow -- Scientific Reports Submission Package

This directory contains the Scientific Reports (Nature) journal
version of the FoldFlow paper. The original TMLR / NeurIPS version
lives under `/Users/kmagdy-ma-eg/Workspace/Research/FoldFlow/` and
must not be modified.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Complete Scientific Reports manuscript (Introduction -> Results -> Discussion -> Methods, single-column, `fleqn,10pt`) |
| `references.bib` | Bibliography -- copy of the FoldFlow references, augmented with biophysics citations (Hartl, Onuchic, Bryngelson, Wolynes, Hopfield, Greydanus) appropriate for SR's multidisciplinary audience |
| `SUBMISSION_NOTES.md` | Subject-area recommendations, draft 300-word cover letter, suggested reviewers (4-7), data/code/COI/author-contributions statements, full number audit, pre-submission checklist, open questions for Karim |
| `README.md` | This file |

Figures are referenced from the original paper via
`\includegraphics{../../FoldFlow/paper/figures/FILENAME.pdf}` so the
FoldFlow originals are never touched.

## Key differences vs. the NeurIPS / TMLR versions

| Area | NeurIPS / TMLR | Scientific Reports |
|------|----------------|----------------------|
| Structure | Abstract -> Introduction -> Related Work -> Method -> Experiments -> Analysis -> Discussion -> Conclusion | Abstract -> Introduction -> Results -> Discussion -> Methods -> References -> Acknowledgements -> Author Contributions -> Competing Interests -> Data/Code Availability (Nature order) |
| Title | "FoldFlow: Which Principles of Protein Folding Transfer to Neural Architecture Design?" | "FoldFlow: A Systematic Study of Bio-Inspired Inductive Biases for Neural Architecture Design" (advisor-revised 2026-04-27 to remove the "principles transfer" overclaim and reframe as a systematic empirical study) |
| Abstract | 200 words, ML framing first | 200 words, bio-inspiration framing first, empirical result second, no references (SR style) |
| Introduction | Starts at neural-network inductive biases | Starts at "how living matter computes" and narrows to protein folding, then to neural modules (broader scope for multidisciplinary audience) |
| Audience | ML researchers | Cross-disciplinary (ML + computational biophysics); biophysics intuition is used to motivate each module |
| "Dynamic topology" (C2) | Called "dynamic topology" in some prose | Explicitly corrected: C2 is applied ONCE at initialisation, not iteratively. Text and Algorithm both clarify this. |
| FiLM attribution (C5) | Cited in the original | Cited + differences explicitly discussed in both Methods and Discussion (Softplus-shifted scale, same-encoder conditioning) |
| Table 3 bolding | Full-model row bolded in NeurIPS version | Bolded on the w/o-Langevin row (the numerically best result, tied with the full model) -- corrects the REVISION_PLAN concern |
| Canonical numbers | CIFAR-10 85.7\%; WT2 512.5 vs 598.4 (14.4\%) | CIFAR-10 85.72\%; WT2 515.9 vs 594.3 (13.2\%) -- per Karim's brief. Reconciliation noted in SUBMISSION_NOTES.md |
| Author block | Anonymous | Karim Magdy, Ghada Khoriba, Hala Abbas (Arab Open University, Egypt) |
| Broader impact | Short paragraph at end (NeurIPS) | Merged into Discussion as "Broader impact" subsection |
| Reproducibility | Single line in NeurIPS; brief section in TMLR | Dedicated Data Availability + Code Availability sections (SR requires them); explicitly names Apple M4 + Google Colab Pro |
| Page limit | 9 pages (NeurIPS) / flexible (TMLR) | No hard page limit at SR, but this draft is similar length to TMLR |

## SPRO / REVISION_PLAN items addressed

See `SUBMISSION_NOTES.md` Section 6 for the full checklist. In brief:

- F2 number audit -- one canonical set locked in; discrepancies
  documented for Karim.
- F4 -- no `\todo` macros.
- F5 -- Broader Impact paragraph included.
- Sec 3 "dynamic topology" -- corrected to "data-dependent, statically
  computed."
- Sec 3 FiLM citation + difference -- added in Methods and Discussion.
- Table 3 bolding -- best (w/o Langevin) row bolded.
- Separability limitation -- explicitly acknowledged in Discussion >
  Limitations, pairwise-energy parameterisations flagged as future
  work.

## How to compile

```bash
cd /Users/kmagdy-ma-eg/Workspace/Research/ScientificReports_Submissions/FoldFlow_SR
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If the `naturemag.bst` bibliography style file is not installed, fall
back to `unsrt` or `plain` (SR accepts either at submission; final
typesetting is handled by the journal).

## Unresolved questions for Karim

See `SUBMISSION_NOTES.md` Section 12 -- primary item is confirming
which canonical number set (515.9 / 594.3 vs 512.5 / 598.4) is the
correct, reproducible one from the latest experimental runs.
