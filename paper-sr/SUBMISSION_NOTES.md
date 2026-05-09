# Scientific Reports Submission Notes -- FoldFlow

**Manuscript:** FoldFlow: A Systematic Study of Bio-Inspired Inductive Biases for Neural Architecture Design
**Authors:** Karim Magdy, Ghada Khoriba, Hala Abbas (Arab Open University, Egypt)
**Target journal:** *Nature Scientific Reports*
**Corresponding author:** Karim Magdy (karimagdy22@gmail.com)
**Last updated:** 2026-05-04 (Tier A acceptance-push; Tier B in progress on Milano-Bicocca server)

---

## 0. 2026-05-04 reviewer-response pass — Tier A complete

External peer reviewer returned a "major revisions" verdict with 12 weaknesses across four categories. Their Overall Assessment named **four explicit acceptance gates**:

> *"If the authors can (i) rigorously ensure and demonstrate causal safety of LM gating, (ii) add the missing control comparisons for both LM and vision, (iii) clarify parameter matching and 'topology' persistence, and (iv) correct reporting inconsistencies, the paper would make a meaningful and practically useful contribution."*

### Reviewer's positive verbatim quotes (use in cover letter)

The reviewer's own positive language strongly supports our reframe of the contribution:

- *"thoughtful, empirically grounded exploration"*
- *"compelling contribution"* (re: modality-decomposition + energy-gated attention as "a strong, simple mechanism")
- *"careful, honest decomposition"*
- *"commendably honest, uses multiple seeds, and provides ablations and parameter-matched controls"*
- *"modest but statistically significant gains"* on vision; *"larger improvement on language perplexity"*
- *"would make a meaningful and practically useful contribution"* once revisions are addressed
- on the broader meta-point: *"a valuable meta-point: bio-inspired priors should be evaluated component-wise and modality-wise; not all inspirations generalize. This measured stance can positively influence how inductive biases from biological systems are ported into ML."*

### Acceptance-gate mapping

| Reviewer gate | Plan item(s) | Status |
|---|---|---|
| (i) Demonstrate causal safety of LM gating (W1) | **Tier B1** — code fix at `lm.py:155` (replace `x.mean(dim=1)` with causal cumulative mean per position) + Methods subsection + future-token-shuffle cheat-detection pilot + 25-run LM re-run | In progress (Tier B) |
| (ii) Add missing control comparisons for LM and vision (W4 + W5) | **Tier B2** — `PerTokenCausalGate` + `TalkingHeadsAttention` baselines (10 runs); **Tier B3** — vision MHA-only control (3 runs) | In progress (Tier B) |
| (iii) Clarify parameter matching and topology persistence (W9 + W2) | **Tier A1** — explicit topology-persistence wording with audit-trail file:line ref (already true in code; clarified in main.tex). **Tier A3** — exhaustive parameter-budget breakdown footnote on Table 1 (encoder fixed at 1.35 M; modules add up to ~1.48 M; no rebalancing) | ✓ Done (Tier A) |
| (iv) Correct reporting inconsistencies (W8) | **Tier A2** — Table 5 "no throughput penalty" reframed as platform-bound 2.26× MPS-only artefact, expected to vanish on CUDA | ✓ Done (Tier A) |

### Tier A items applied (all text-only, 2026-05-04)

| Item | Reviewer issue | Where | Status |
|---|---|---|---|
| A1 | W2 + W10 — topology persistence ambiguity | New sentence after the existing "C2 is applied once" passage citing the audit-trail file:line | ✓ Done |
| A2 | W8 — throughput vs text inconsistency | Replaced "no throughput penalty" framing with explicit MPS-only 2.26× caveat + CUDA expectation | ✓ Done |
| A3 | W9 — vision parameter constancy ambiguity | Extended Table 1 caption with exhaustive module-size breakdown + "no hidden capacity rebalancing" disclaimer | ✓ Done |
| A4 | W11 — LM head-gating literature gap | New "Relation to head-level gating literature" Discussion paragraph + 4 new BibTeX entries (Talking-Heads, SE-Net, Voita head-pruning, Michel sixteen-heads) | ✓ Done |
| A5 | W12 — vision topology citation gap | New "Relation to topology-aware vision transformers" Discussion paragraph + 3 new BibTeX entries (HGFormer, Focal Transformer, IBiT) | ✓ Done |
| A6 | Cover-letter framing | This document — quote bank + acceptance-gate mapping | ✓ Done |

Total added: 7 BibTeX entries, 2 new Discussion paragraphs, 3 surgical text revisions. No new claims added; all changes either clarify, contextualise, or honestly caveat existing claims.

### Tier B in progress (server-side, ~25–40 hrs wall-clock)

- B1: causality fix in `lm.py:155` + 25-run LM re-run on Milano-Bicocca GPUs 0+1
- B2: `PerTokenCausalGate` + `TalkingHeadsAttention` baselines, 10 runs
- B3: vision MHA-only control, 3 runs CIFAR-10
- B4: d=560 LR sweep, 9 runs

**Honest pre-emption**: the Tier B re-runs may shrink the headline 14.4 % perplexity gap if the original gate was leaking future tokens. We will report the post-fix numbers transparently and, citing the reviewer's "honest decomposition" framing, position the contribution around modality-aware ablation evidence rather than a single-number claim.

---

## (legacy notes follow)



> Title was revised on 2026-04-27 (from "FoldFlow: Which Principles
> of Protein Folding Transfer to Neural Architecture Design?") to
> de-emphasise a "principles-transfer" overclaim and to frame the
> contribution as a systematic empirical study, in line with
> advisor revisions. The abstract, intro contributions, CIFAR-10
> Results, and WikiText-2 Results were also rewritten on the same
> date to: (a) state explicitly that the +1.4 pp CIFAR-10 gain is
> small in absolute terms; (b) acknowledge upfront that Langevin
> refinement has *no measurable effect* on WikiText-2 perplexity;
> (c) cite the parameter-matched 50.9 M Transformer++ control
> ($593.3 \pm 1.1$) every time the 14.4% headline appears, and
> additionally report the 13.6% same-budget reduction.

---

## 1. Why Scientific Reports is the right venue

FoldFlow is the strongest Scientific Reports fit among Karim's four active
papers because its scientific contribution sits genuinely at the interface
of computational biophysics (protein folding as a computational principle)
and machine learning (neural architecture design). The Nature family,
especially SR, explicitly welcomes *methodologically-cross-disciplinary*
manuscripts that would be penalised at a single-field venue for being
"out of scope" of one or the other. Our framing---"which physical
principles of folding transfer to differentiable neural modules, and in
which modalities do they dominate?"---is a genuinely cross-disciplinary
question with a clean empirical answer.

---

## 2. Suggested Subject Areas

Scientific Reports requires at least one primary subject area from its
taxonomy. We recommend the following (in order of priority):

1. **Machine learning** (primary, under "Computer science")
2. **Computational biophysics** (under "Biological physics" / "Physics")
3. **Biological physics** (under "Physics")
4. (Secondary) **Statistical physics, thermodynamics and nonlinear dynamics**

The first two drive the submission: Machine learning signals the
immediate technical audience; Computational biophysics signals the
cross-disciplinary framing. Biological physics is a useful tertiary
tag because the folding-landscape metaphor is explicitly grounded in
biophysical theory (Bryngelson, Wolynes, Onuchic, Dill, Hartl).

---

## 3. Draft Cover Letter (~300 words)

> Dear Editor,
>
> We are pleased to submit our manuscript, "FoldFlow: A
> protein-folding-inspired neural architecture reveals
> modality-dependent inductive biases," for consideration by
> *Scientific Reports*.
>
> Proteins fold reliably from disordered chains into precise
> three-dimensional structures using a compact set of physical
> principles---energy minimisation, dynamic contact formation,
> cooperativity, chaperone-assisted repair and environmental
> sensitivity. While these principles have driven spectacular
> advances in structural biology and protein-structure prediction
> (AlphaFold, ESMFold, RoseTTAFold), their potential as inductive
> biases for neural architectures outside structural biology
> remains essentially unexplored. Our manuscript asks a
> principled cross-disciplinary question: which of these
> folding principles, when instantiated as differentiable neural
> modules, genuinely improve standard machine-learning tasks?
>
> We introduce FoldFlow, a neural architecture that realises five
> hallmarks of folding as neural operations, and evaluate each
> module through systematic ablations across two modalities. We
> uncover a striking modality dependence: on CIFAR-10 image
> classification, adaptive topology (the neural analogue of
> native contact formation) dominates and drives the full model
> to 85.72\% top-1 accuracy; on WikiText-2 language modelling,
> energy-gated attention (the analogue of energy-dependent contact
> strength) drives a 13.2\% perplexity reduction over a
> parameter-matched Transformer baseline. One folding principle
> (Langevin refinement) transfers to vision but measurably not to
> language---a null result we report transparently.
>
> We believe this work is a good fit for *Scientific Reports*
> because it bridges computational biophysics and machine learning,
> reports both positive and null findings with statistical rigor
> (5 seeds, paired t-tests, parameter-matched controls), and
> produces directly reusable architectural primitives. The work
> was performed entirely on modest computational resources (a
> single Apple M4 chip plus Google Colab Pro), making it
> reproducible for a broad cross-disciplinary readership.
>
> We confirm that this manuscript is not under consideration
> elsewhere, and we declare no competing interests.
>
> Sincerely,
> Karim Magdy, on behalf of all authors.

---

## 4. Suggested Reviewers (4--6)

A deliberately mixed slate of ML researchers (especially in energy-based
models, score-based dynamics, and attention/architecture design) and
computational biophysicists who work on folding.

1. **Yilun Du** (MIT / Harvard) -- energy-based models, Langevin sampling,
   iterative refinement; first author of Du \& Mordatch (2019) *Implicit
   Generation and Modeling with Energy Based Models*. Strong match on the
   Langevin dynamics and energy-landscape framing.

2. **Yang Song** (OpenAI) -- score-based generative modelling via Langevin
   dynamics; author of Song \& Ermon (2019). Directly relevant on the
   energy-based inference side.

3. **Avishek Joey Bose** (Mila / Oxford) -- flow matching on SE(3) for
   protein backbone generation (Bose et al., 2024). Sits at the ML /
   protein-structure intersection and can judge the biological framing.

4. **Ken A. Dill** (Stony Brook) -- foundational work on the
   protein-folding problem, energy funnels, and theoretical folding
   kinetics; co-author of *The Protein-Folding Problem, 50 Years On*
   (Dill \& MacCallum, 2012). Biophysics credibility on the folding
   metaphor.

5. **Jos\'e N. Onuchic** (Rice University) -- theory of folding energy
   landscapes (Onuchic, Luthey-Schulten \& Wolynes, 1997). Expert
   reviewer on whether our "folding-funnel" framing is physically
   coherent.

6. **Hubert Ramsauer** (JKU Linz) -- modern Hopfield networks
   (Ramsauer et al., 2021), which directly connect energy-based
   dynamics and attention. Ideal for judging the energy-gated
   attention contribution.

7. *(Optional seventh)* **Ethan Perez** (Anthropic) -- first author of
   FiLM (Perez et al., 2018); our C5 module is FiLM-style. Useful if
   the handling editor wants a reviewer on the feature-wise modulation
   angle.

**Editors to avoid / conflicts:** Ghada Khoriba and Hala Abbas are
the co-authors; no other conflicts are known.

---

## 5. Number audit and reconciliation (IMPORTANT FOR KARIM)

The source LaTeX files report slightly different canonical numbers than
the user's brief specified. The SR manuscript uses the brief's
numbers throughout the text, abstract, and Table 3. This note records
the discrepancy so Karim can verify and reconcile before final
submission.

**Canonical numbers locked into `main.tex` (per Karim's brief):**

| Quantity                                     | Value used                         |
|---------------------------------------------|------------------------------------|
| CIFAR-10 extended-training accuracy          | **85.72%** top-1                   |
| CIFAR-10 full model (50 epochs, 3 seeds)    | 83.0 $\pm$ 0.4%                   |
| CIFAR-10 baseline                            | 81.6 $\pm$ 0.2%                   |
| WikiText-2 FoldFlow-LM (5 seeds)             | **515.9 $\pm$ 3.6** PPL            |
| WikiText-2 Transformer++ baseline (5 seeds)  | **594.3 $\pm$ 2.6** PPL            |
| WikiText-2 FoldFlow-LM w/o Langevin          | 515.9 $\pm$ 3.5 PPL (identical)   |
| WikiText-2 relative improvement              | **13.2%**                          |
| Parameter-matched T++ ($d{=}560$)           | 593.3 $\pm$ 1.1 PPL                |
| $\Delta$PPL from removing energy gate       | $+57.5$ PPL                        |

**Conflicting numbers found in source files (for Karim to reconcile):**

- `main_tmlr.tex` and `main.tex` (NeurIPS version) report
  FoldFlow-LM at **512.5 $\pm$ 3.6** and Transformer++ at
  **598.4 $\pm$ 3.9**, yielding a 14.4\% relative improvement.
- The same files report the CIFAR-10 extended-training number as
  **85.7\%** (two significant figures), matching 85.72\% but
  without the trailing precision.
- The ``w/o energy gate'' baseline (573.4 $\pm$ 1.9) is unchanged
  across sources, so the energy-gate $\Delta$ was recomputed as
  $573.4 - 515.9 = 57.5$ PPL (using the brief's canonical
  FoldFlow-LM value). The NeurIPS/TMLR reports $573.4 - 512.5 =
  60.9$ PPL.
- SPRO critique / revision plan references Table 3 bolding with
  numbers **521.0** vs **525.2**. These appear nowhere in the
  NeurIPS/TMLR main.tex (which shows 512.5 for both full and
  w/o-Langevin variants); this is residue from an earlier
  checkpoint and has been superseded. Table 3 in the SR version
  bolds the w/o-Langevin row (515.9 $\pm$ 3.5) as the best result,
  consistent with the REVISION_PLAN instruction "bold best result
  521.0 (not 525.2)."

**Recommended action for Karim:** Before sending to SR, re-run
the final WikiText-2 evaluation on 5 seeds and confirm whether the
canonical numbers are 515.9 / 594.3 / 13.2\% (this manuscript) or
512.5 / 598.4 / 14.4\% (the TMLR draft). If the TMLR draft is the
correct set, a single search-and-replace in `main.tex` will restore
consistency. Either way, the qualitative finding (energy gate
dominant, Langevin null in LM, topology dominant in vision) is
robust across both number sets.

---

## 6. SPRO / REVISION_PLAN items addressed in this SR draft

| # | Item (source) | Status in SR draft |
|----|----------------|----------------------|
| F2 | Full number audit across abstract/body/supplementary | Done: single canonical set used throughout; discrepancies logged above for Karim to reconcile |
| F4 | Remove ALL `\todo` items | Done: no `\todo` macros in SR source |
| F5 | Add Broader Impact / Ethics statement | Done: Discussion > "Broader impact" paragraph |
| Sec 3 | Fix "dynamic topology" misnomer (C2 applied once, not iteratively) | Done: Methods > C2 explicitly says "applied **once** after initialisation ... fixed for the duration of the Langevin dynamics"; also noted in Results > "Mapping folding principles" |
| Sec 3 | Cite FiLM (Perez et al., 2018) for C5 and discuss differences | Done: Methods > C5 cites FiLM; Discussion > "Relation to feature-wise modulation" paragraph discusses Softplus-shifted scale and single-encoder conditioning |
| Table 3 | Bold best result | Done: Table 3 bolds the w/o-Langevin row as the numerically best result (tied with the full model); both are reported as 515.9 |
| -- | Acknowledge separability limitation OR discuss pairwise interactions | Done: Discussion > "Limitations" notes that the learned energy function is mean-field and does not capture pairwise interactions, and flags pairwise-energy parameterisations as future work |
| -- | Reproducibility: seeds, hyperparameters, dataset versions, Mac M4 + Colab Pro | Done: Methods > Experimental protocol lists 3--5 seeds, identical hyperparameters, AdamW, cosine decay; Code Availability lists Apple M4 + Google Colab Pro |

Items **not** addressed (scope of journal conversion, not number
audit):

- W1 scale (CIFAR-10, WikiText-2) -- acknowledged honestly in
  Discussion > Limitations; ImageNet / billion-token evaluation
  remains future work.
- W2 individual components are known mechanisms -- explicit
  honest framing is retained; novelty claim rests on the
  cross-modality decomposition, as in the original draft.
- W3 large absolute perplexity (from limited epochs, no
  pretraining) -- explained honestly in Results > WikiText-2.
- W4 Langevin null in LM -- reported prominently and mechanistically
  discussed.

---

## 7. Data Availability statement

CIFAR-10 and CIFAR-100 are available from the University of Toronto
image-recognition database
(https://www.cs.toronto.edu/~kriz/cifar.html). WikiText-2 is available
from the Salesforce Research pointer-sentinel release
(https://huggingface.co/datasets/wikitext). All datasets were used
via their standard Hugging Face distributions and no new datasets
were generated. Per-seed numerical results (CIFAR-10 ablations on
3 seeds, WikiText-2 on 5 seeds) are tabulated in the Supplementary
Material.

---

## 8. Code Availability statement

All code for training, evaluation, figure generation, and per-seed
logs will be released as a public GitHub repository at the time of
acceptance. Training configurations (exact seeds, hyperparameters,
augmentation schedules), model checkpoints, and the scripts used to
generate every figure and table in this paper will be included. The
primary training environment was an Apple M4 chip with the Metal
Performance Shaders backend of PyTorch; the codebase is also
compatible with CUDA and has been verified on Google Colab Pro
A100/H100 instances. Complete computational budget (~70 hours
total) is documented in the Supplementary Material.

---

## 9. Competing Interests

The authors declare no competing interests. No external funding
was received for this work.

---

## 10. Author Contributions

- **K.M.** (Karim Magdy) -- Conceptualisation, Methodology, Software,
  Formal Analysis, Investigation, Data Curation, Writing (Original
  Draft), Visualisation.
- **G.K.** (Ghada Khoriba) -- Conceptualisation, Supervision,
  Writing (Review \& Editing).
- **H.A.** (Hala Abbas) -- Methodology (statistical advice), Writing
  (Review \& Editing).

All authors read and approved the final manuscript.

---

## 11. Pre-submission checklist

- [ ] Karim: verify canonical numbers (515.9 vs 512.5 question in Section 5 above)
- [ ] Karim: compile `main.tex` with `naturemag` bibliography style (or switch to `unsrt` if naturemag not available locally) and verify references render
- [ ] Karim: supply ORCID IDs for all three authors on submission portal
- [ ] Karim: confirm ORCID / affiliation formatting on AOU letterhead for cover letter
- [ ] Karim: verify `\includegraphics` figure paths resolve (figures live in `../../FoldFlow/paper/figures/`) -- otherwise copy figures into a local `figures/` subdirectory
- [ ] Karim: package supplementary material (hyperparameters, per-seed tables, CIFAR-100 extension) into a Supplementary PDF
- [ ] Karim: request at least 3 and ideally 4 reviewers from Section 4 above
- [ ] Karim: upload code tarball or private GitHub link at submission

---

## 12. Unresolved questions / requests for Karim

1. **Which canonical number set is correct?** 515.9 / 594.3 (brief)
   or 512.5 / 598.4 (TMLR draft)? The current SR manuscript uses
   the brief's numbers; if the TMLR numbers are correct, a single
   search-and-replace will restore consistency.
2. **Has the 85.72\% CIFAR-10 number been reproduced in a final
   200-epoch run?** The NeurIPS/TMLR drafts report 85.7\%; the
   brief reports 85.72\%. Confirm the two-decimal-point precision.
3. **Should Ghada and Hala be listed on the submission portal with
   their institutional email addresses?** The current `authblk`
   block lists them as AOU-affiliated without email; SR requires
   an email per author at submission.
4. **Do you want a highlights box / graphical abstract?** SR
   supports these; the cross-domain decomposition figure
   (Fig.~4) would make a natural graphical abstract with minor
   restyling.
5. **Is there any reason to hold back code until after acceptance?**
   SR's Data/Code Availability statements are evaluated; submitting
   code in a private repository that the handling editor and
   reviewers can access avoids later negotiation.

---

## 2026-05-05 — Tier B sweep complete; headline number revised

**The Tier B causally-safe re-run completed.** GPUs 0+1 of Milano-Bicocca, 5 seeds × 17 epochs × 4 attention variants, ~9 wall-clock hours. Numbers locked in `tab:lm_attn_variants` and propagated through Abstract, Introduction, Results, Table 1 (`tab:lm`), Figure 3 caption, and Discussion. The headline ratio is now **4.1%** (post-fix, causal cumulative-mean gate) versus the pre-fix **14.4%** (non-causal global mean, leakage-confounded).

**Final numbers (mean ± std across 5 seeds, 17 epochs each, CUDA / GTX 1080 Ti):**

| Variant | Parameters | PPL | Δ vs Tpp |
|---------|-----------|----:|---------:|
| Transformer++ ($d{=}512$) | 44.8 M | 602.3 ± 4.2 | — |
| FoldFlow-LM, Causal Energy (post-fix) | 50.2 M | 577.6 ± 1.9 | **−4.10%** |
| FoldFlow-LM, Per-token causal gate | 50.2 M | **575.2 ± 1.8** | **−4.50%** |
| FoldFlow-LM, Talking-Heads attention | 49.8 M | 606.1 ± 6.1 | +0.62% |
| FoldFlow-LM, original Energy gate (pre-fix) | 50.2 M | 512.5 ± 3.6 | −14.4% (leakage) |

**Key findings (for the reviewer-response cover letter):**

1. **The W1 fix shrinks the gap from 14.4% to 4.1%, but the gap survives.** Paired t-test across 5 seeds confirms the post-fix Causal Energy improvement over Transformer++ at p < 10⁻³. The structural narrative ("head-level gating helps a small but reproducible amount; modality-dependent magnitude") survives the correction; only the magnitude shifts.
2. **Per-token causal gate is the simplest variant and is marginally better than Causal Energy.** This says the cumulative-mean structure adds nothing over the simplest causal alternative; the contribution is genuinely a head-level gating effect rather than anything specific to the energy framing. We update the abstract and Discussion to reflect this.
3. **Talking-Heads attention does not improve over Transformer++ on this benchmark.** A small +0.6% regression. The reviewer's most-cited literature comparison therefore does NOT close the gap; the per-token causal gate does.
4. **Original component ablations are retained as a pre-fix qualitative ranking** (energy ≫ chaperone ≫ Langevin) but the absolute magnitudes are flagged as inflated; the post-fix sweep confirms Langevin contributes nothing.

**Honest-reporting language baked into the paper text:**
- Abstract revised to lead with 4.1% (matching new tab:lm_attn_variants), with one sentence describing the trace-and-correct of the original 14.4% claim. Word count: 197 (under 200 cap).
- Introduction (line ~144) revised in parallel.
- Results §"WikiText-2" subsection title changed from "energy gating drives a 14.4% perplexity reduction" to "causally safe energy gating yields a 4.1% perplexity reduction", with explicit pre-fix / post-fix block in Table 1.
- Figure 3 caption rewritten with the new ratio, the leakage explanation, and a pointer to Methods §"Causal energy gating (W1 fix)".
- Discussion §"Folding principles are useful inductive biases" rewritten to honestly state that the small post-fix gain survives but the magnitude is small enough that the choice of head-gating mechanism matters.

**Cover letter quote bank** (reviewer's positive verbatims for re-use): *"thoughtful, empirically grounded", "compelling contribution", "honest decomposition", "modest but statistically significant", "commendably honest"*. The revised paper amplifies the honest-decomposition framing rather than competing on a single-number claim.

**Tier B3 + B4 status**: launched on freed server GPUs at 09:08 CEST (B3: vision topology_only, ~1 hr; B4: d=560 LR sweep at LRs {1e-4, 3e-4, 5e-4} × 3 seeds × 17 epochs, ~6.6 hr).

**B3 result (2026-05-05 09:58 CEST, 50 min total):** topology_only (encoder + 1 MHA layer applied once, no folding dynamics) attains $82.86\% \pm 0.22\%$ on CIFAR-10 across 3 seeds × 50 epochs, versus a parameter-matched encoder-only baseline at $81.82\% \pm 0.21\%$ (+1.04 pp). Compared to the full FoldFlow at $83.0\% \pm 0.4\%$, the topology MHA layer alone closes **~74% of the +1.4 pp full-model gap**; the remaining four folding modules (Langevin, cooperativity, chaperone, environmental modulation) together contribute the residual ~0.14 pp. **Honest reporting required**: this validates reviewer W5's hypothesis that the C2 gain is largely attributable to "one extra MHA layer" rather than the folding interpretation. The structural narrative survives ("encoder + one attention call is itself a folding-inspired construction"), but the multi-module folding programme contributes far less than the original Table 1 rows could suggest in isolation. Updated:
- **Abstract** (199 words): one sentence added quantifying the 74% closure and reframing "the dominant CIFAR-10 contributor is the addition of one attention call rather than the folding interpretation".
- **Table 1** (`tab:ablation`): new row "Topology MHA only (C2, no C1/C3/C4/C5)" 1.48 M @ 82.86 ± 0.22%, in a dedicated "Reviewer W5 control" sub-block.
- **Table 1 caption** extended with W5 control description.
- **Results §"Component ablation"** (line 266+): new paragraph quantifying the 74% closure and explicitly reporting the trade-off honestly.

**B4 progress (in flight)**: lr=1e-4 done at 11:21 CEST = $602.07 \pm 3.67$ (3 seeds × 17 epochs, d=560). lr=3e-4 in flight, lr=5e-4 queued. Expected B4 complete by ~15:47 CEST.

*Last updated: 2026-05-05.*
