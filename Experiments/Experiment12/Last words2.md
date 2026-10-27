Experiment 12 gets the main conclusion right, but `report_v2.md`, `corrections.md` and "Last words" contain errors that should be fixed before anyone reads them.

## What holds up

- **The algebra is exact.** Norm composition, alternativity and Moufang all pass to about 1e-14. The unit associator has mean 1.08 and max 1.86, under the bound of 2. The earlier ~22 came from unnormalised vectors (22.88 measured, about 2.83³).
- **The standalone `phase1.json` reproduces my ablation:**

| Variant | `phase1.json` | My run |
|---|---|---|
| HRU full | 2.303 | 2.2985 |
| Recurrence removed | 2.294 | 2.2896 |
| `W_ctx2` removed | 2.505 | 2.499 |

- **The pre-registered rule fails for HRU**, correctly.
- **Pure-rotor isometry is exact** (spectral norm 1.0 up to T=1024).
- **HRU is not the better model.** At about 9.5k parameters it scores 13.70 PPL against 11.99 for the SO(8) control (a 3.5% parameter gap) and 12.71 for no recurrence (which has 44% more parameters).

## Errors in the deliverables

1. **The report uses the broken Phase 1 numbers.**
   - Section 3.2 and `results_v2.json` still show 3.376 / 3.383 / 3.388 / 3.418. That is worse than the corpus's unigram entropy (3.313), so those models never trained.
   - The good numbers sit in `phase1.json`. C2's "evidence" cites the broken ones.
2. **The sedenion verdict (C6) is wrong or at least mislabelled.**
   - `phase0.json` says the paper's example "holds in the standard convention: false". In my independent recursive Cayley-Dickson implementation (float64), `(e1+e10)(e4−e15)` has norm 0.0000 in both conventions.
   - Their own `results_v2.json` shows `e1+e10` has a null vector (product norm 4.5e-16), so it is a zero divisor in their indexing too. The partner differs, so their basis labelling isn't standard recursive Cayley-Dickson.
   - So the paper's pair is a real zero divisor under standard indexing, and the verdict should be "convention-dependent, general fact true".
   - The report's own sentence for this row ("2.00 under A, but fails under B (2.00)") is self-contradictory.
3. **The sedenion "gradient collapse" claim was never tested.** No k=16 model was trained.
4. **Everything ran on CPU, not GPU.**
   - The metadata says `device: cpu`, and the run took 517 s.
   - The report header says "Kaggle GPU" and the speed table says "cuDNN GRU/LSTM". Both are false.
   - The 13× slowdown is a CPU eager-mode number.
5. **The gradient table hides its key information.**
   - It prints 1e-9, 1e-36 and 1e-152 all as "0.0000". The SO(8) control collapsing to 1e-152 at T=1024 means it isn't really orthogonal as built.
   - The gated HRU gives 5.7, 21.8 and 0.0008 (std 5.5 and 30.7), so it **explodes and vanishes**. C3 should be PARTLY, not SUPPORTED.
   - The report never says whether these are init or trained values. Trained language models are supposed to forget the initial state, so a tiny value isn't a defect.
6. **C5's "REFUTED" is a hardcoded string.**
   - `convergence_speed_claim.audit_verdict` is literal text. No epochs-to-90% were measured, and no model reached 90% on any task.
   - The paper's 3.2× claim (rotational memory, delayed copy) was never run. It should be UNTESTABLE.
   - Parity is at chance for both models even at L=64, so it's INCONCLUSIVE by the pre-registered ≥95% rule.
7. **The bracket result is circular.**
   - The label is defined as the sign of the associator's e1 component, so "100% separability" only shows that a classifier given the associator can compute the associator. No neural model was trained, and the quaternion "0.0" is an argument rather than an experiment.
   - I ran the actual test on 100k samples. A linear model on the raw inputs gets 0.501, a linear model on the label's own ingredients (`(ab)c`, `a(bc)`) gets 0.999, and a generic 256-256 MLP with no octonion structure gets **0.926** on raw inputs.
   - So the "octonions shine on trees" takeaway in "Last words" is unsupported.
8. **The line numbers in `corrections.md` are wrong.** It cites 173, 209–210, 218, 263, 337 and 349. The real ones are 223, 256–257, 266, 408, 426 and 468. The README from Experiment 11 had them right.
9. **"Last words" over-explains.**
   - It says isometries "can't forget" and so the gate was needed, but the pure rotor scores 13.63 PPL and the gated version 13.70, so the gate made no measurable difference.
   - The comparison to GRU isn't equal-parameter (13.8k vs 9.5k).
   - Section 3.5 attributes one result to the wrong test.

## Verdicts I'd defend

| Claim | Verdict |
|---|---|
| LM advantage (C1) | **Refuted**, via the ablation, the count-trigram baseline and the matched controls |
| 8:1 compression (C2) | **Refuted for the code**: `W_hurwitz` is unused and its gradient is zeroed. The expressivity of compression remains untested |
| Isometry prevents vanishing gradients (C3) | **Holds only for the pure rotor.** The gated HRU both explodes and vanishes |
| σ ≈ 1.000 during training, 3.2× faster (C4) | **Untested**, and false for the gated step |
| +14.8% bracket accuracy | **Untested**. The task is circular, and a generic MLP gets 92.6% |
| k=16 zero divisors (C6) | **True in general. The paper's pair is a zero divisor under standard indexing and not in the repo's.** Gradient collapse untested |
| Speed / Triton | The kernel is dead code. HRU is 13× slower than GRU on CPU |

## On wording
Your Experiment 11 README uses "benchmark manipulation". The code shows an asymmetry (`W_ctx2` only in HRU, uncounted parameters, zeroed baseline gradients), but not intent. If any of this goes public, "unreported asymmetry" is the claim the evidence supports.
