# Cover Letter — FoldFlow

**Paste this into the SR portal "Cover Letter" field.**

---

Dear Editor of *Scientific Reports*,

We submit our manuscript, "**FoldFlow: A Systematic Study of Bio-Inspired Inductive Biases for Neural Architecture Design**," for consideration as a research article.

Bio-inspired inductive biases are widely advocated for neural architecture design, yet rigorous, component-level evidence for which biological principles actually transfer to standard machine-learning tasks remains scarce. Our manuscript asks the directly empirical question: when five hallmarks of protein folding — Langevin dynamics on a learned energy landscape, data-dependent attention topology, depthwise-separable cooperativity, a stress-gated chaperone, and context-dependent affine modulation — are translated into differentiable neural operations, which (if any) actually improve standard benchmarks?

The findings are deliberately mixed and we report them honestly. **(i)** On CIFAR-10 with a deliberately weak encoder (1.48 M parameters), the full FoldFlow model improves accuracy by +1.4 percentage points (83.0% vs. 81.6% baseline; 85.72% with 200 epochs). A no-folding-dynamics control — encoder plus one MHA layer applied once — attains 82.86%, closing ≈74% of that gap, so the dominant CIFAR-10 contributor is the attention addition rather than the folding interpretation. **(ii)** On WikiText-2 language modelling, the causally-safe FoldFlow-LM reduces perplexity by **4.1%** over a matched Transformer++ baseline (577.6 vs. 602.3); a per-token causal gate is marginally better (−4.5%) and Talking-Heads attention does not improve. An earlier non-causal formulation reported a 14.4% reduction; we trace this to teacher-forced future-token leakage, replace the gate with a causal cumulative-mean variant, and report the corrected number transparently. **(iii)** Langevin refinement, the most physically central of the principles, has no measurable effect on language modelling — a clean null result we discuss rather than bury.

We believe *Scientific Reports* is the right venue because the work bridges computational biophysics and machine learning, reports both positive and null findings with statistical rigor (3-5 seeds, paired t-tests, parameter-matched controls, a causality fix that quantitatively adjusts our own headline figure downward by a factor of three), and produces directly reusable architectural primitives. The manuscript was internally pre-reviewed against major-revisions peer-review criteria before submission; all flagged issues — including the W1 causality leak corrected here — have been addressed. The work was performed on a single Apple M4 chip plus a CUDA / GTX 1080 Ti server (Milano-Bicocca), making it reproducible for a broad cross-disciplinary readership.

We confirm that the manuscript is not under consideration elsewhere, and we declare no competing interests.

Sincerely,
Karim Magdy
(On behalf of all authors)
