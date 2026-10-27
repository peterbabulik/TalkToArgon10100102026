ROLE
You are continuing an adversarial audit of the Hurwitz Recurrent Unit (HRU).
Your previous report (report.md, results.json, integrity_report.md,
run_experiments.py) has been independently reviewed. Several statements in it
were wrong or unsupported. Fix them by RE-RUNNING code, not by editing text.
Every number must come from executed code. Where something cannot be
established, write UNTESTABLE or INCONCLUSIVE; do not guess. Do not defend
earlier conclusions. If a re-run reverses a verdict, say so.

ENVIRONMENT
Kaggle GPU, PyTorch autograd, Internet on. Time budget <= 3 h: expose
TIME_BUDGET, and if short, cut seeds BEFORE cutting ablations. Checkpoint
results.json after every run. Write outputs to /kaggle/working:
results_v2.json, report_v2.md, corrections.md.

STEP 0 - CORRECTIONS LOG (corrections.md)
Open the ORIGINAL hru_pytorch_lm.py and verify against the source, quoting
line numbers:
 a) Real names: HRU_Octonion_LM, EuclideanGatedBaselineLM, num_parameters,
    fano_octonion_mul_kernel. Your earlier report used names that do not
    exist (BaseLineRNN, HRULanguageModel, count_parameters,
    _hru_octonion_scan_kernel). Fix every occurrence.
 b) Which models have W_ctx1 and which have W_ctx2. (HRU has both; the
    baselines have only W_ctx1 with a 0.82 factor. Verify, do not assume.)
 c) The ORIGINAL parameter counts: reported 13,137 (HRU), 24,133 (LSTM),
    23,249 (GRU). Compute counts WITH all arrays (incl. W_ctx1/W_ctx2).
    13,649 and 22,099 are the counts of YOUR rewrite, not the original;
    do not mix them up.
 d) The perplexities in the original JSON are 9.85 / 11.13 / 11.31.
    Remove any "2.94 vs 3.12 / 3.08" figures; they appear in no repo file.
 e) What fano_octonion_mul_kernel actually does (one pointwise octonion
    product, no scan) and that main() never calls it or any torch class.
Write each correction as: old statement -> verified statement -> evidence.

STEP 1 - FIX THE GRADIENT DIAGNOSTICS (Test 2)
 1.1 Use the SPECTRAL norm (largest singular value),
     torch.linalg.matrix_norm(J, ord=2), NOT the Frobenius norm.
     2.8283 = sqrt(8) is the Frobenius norm of an 8x8 orthogonal matrix;
     the correct value for an isometry is 1.0. Recompute all rows.
 1.2 Also report the full singular-value spectrum (min/median/max) of
     the one-step Jacobian restricted to the tangent space of S^7, for:
     (z=1, alpha=0), (z=0.634, alpha=0), (z=0.634, alpha=0.22), and the
     TRAINED HRU_Full gate values (read z and alpha from the trained model).
 1.3 Your LSTM result (3e-17 at T=64, exactly 0.0 later) is not credible.
     Debug it: differentiate through BOTH h and c, use float64, check
     gradient flow with torch.autograd.grad on a small T first, test an
     LSTM with forget-gate bias +1, and compare with a finite-difference
     estimate. Report the corrected numbers, or the bug you found.
 1.4 Compute the same quantity for GRU and for the SO8 orthogonal control.
 1.5 For all models report values at T=64, 256, 1024 at init AND after
     training, 3 seeds, mean +/- std.

STEP 2 - MATCHED-PARAMETER LM BENCHMARK (Test 1 redo)
 2.1 Parameter budgets must match WITHIN 5%: build every model at two
     budgets, ~14k and ~25k (counted with sum(p.numel() for p in
     model.parameters())). Print the exact count next to every result.
     Your earlier table compared 13.6k (HRU) with 41.7k (LSTM) and
     33.3k (GRU); that comparison is invalid. Models: LSTM, GRU,
     HRU_Full, HRU_Alpha0, HRU_PureRotor, SO8_OrthogonalRNN,
     quaternion-block (k=4), complex-block (k=2), NoRecurrence,
     plus count-based bigram and trigram (no parameters, as references).
 2.2 Training sanity BEFORE concluding anything about HRU:
     your HRU_Full (12.51) equals NoRecurrence (12.54). Establish whether
     this is an architecture result or a training failure. For each HRU
     variant log: training loss curve, gradient norm per parameter
     tensor every 100 steps, gate z statistics (mean/std per fiber),
     alpha trajectory, and the fraction of steps with exactly-zero grads
     on any tensor. Run a tiny-overfit test (train on 1 batch, must
     reach loss < 0.5) for HRU and for LSTM. If HRU cannot overfit one
     batch, report that as a training bug, fix it (init, LR, normalisation
     eps, gradient clipping), and re-run.
 2.3 LR: grid {3e-4, 1e-3, 3e-3, 1e-2, 3e-2} per model on a TUNE split
     (first 90% of train, last 10% as tune-val), then retrain with the best
     LR and report on a separate TEST split (>= 500k chars, fixed
     sequences, same for all models). Log the full grid.
 2.4 Seeds: >= 3 (5 if time). Report mean +/- std of val loss, BPC and PPL.
     Run a paired comparison (same seeds, same batches) HRU vs best
     baseline, with a Welch t-test or paired bootstrap, 95% CI.
 2.5 PRE-REGISTERED RULE, stated in report_v2.md BEFORE results:
     "HRU wins only if its mean test loss is lower than the best baseline
     at equal parameters by more than 2 std AND beats NoRecurrence."
     Apply the rule mechanically in the verdict.
 2.6 Include the original-architecture control: HRU with the original
     extra tables (W_ctx1 + W_ctx2), recurrence removed, vs with recurrence,
     under autograd. Does the recurrence add anything once the tables exist?

STEP 3 - STATE-TRACKING TASKS THAT ARE ACTUALLY SOLVABLE (Test 4 redo)
Your parity result had GRU at 59.5% at L=500, i.e. near chance; at that
accuracy nothing can be concluded. Fix:
 3.1 First make sure the task is learnable: curriculum (train L=16 -> 64 ->
     256), report each model's accuracy at train length. A model counts as
     "solving" a task only at >= 95% accuracy at train length.
 3.2 Tasks: parity (test L = 64, 256, 1024), Dyck-2 and Dyck-3 closure
     (train L<=64, test L = 64/256/512), group word problem on Q8 (the
     quaternion group) and S5, and the bracket-disambiguation task.
     For the bracket task: if the paper's description is underspecified,
     say so, implement the closest well-defined version (distinguish
     (a*b)*c from a*(b*c) using octonion units), and label it
     "NOT the paper's task".
 3.3 >= 2000 test samples per task, class-balanced, >= 3 seeds, matched
     parameters from STEP 2, mean +/- std. Report the binomial standard
     error and do not interpret differences smaller than 2 SE.
 3.4 Accuracy stratified by depth must show the sample count per bucket.
     Add a random-label control (accuracy must be ~50%).
 3.5 3.2x-fewer-epochs claim: report epochs to reach 90% accuracy per
     model, or "never".

STEP 4 - ASSOCIATOR AND ALGEBRA CHECKS
 4.1 Associator magnitude: your earlier mean of 27.78 (and the repo's 20.49)
     is impossible for UNIT octonions (bound ||[a,b,c]|| <= 2). Recompute
     on explicitly normalised inputs, report mean/max, and state what the
     earlier number was actually measuring (non-unit vectors? sum over
     fibers?).
 4.2 Zero divisors (C6): print the exact 16D multiplication convention you
     used (Cayley-Dickson formula and indexing). Test (e1+e10)*(e4-e15)
     under (a) your convention, (b) the other standard Cayley-Dickson
     indexing. Also run a numerical search (random sparse pairs or
     nullspace of the left-multiplication matrix L(x) for random x) that
     finds zero-divisor pairs, and verify at least one by direct product
     to <= 1e-12 in float64. State clearly: does the paper's specific
     example hold in the convention it implies, yes/no? Remove "SUPPORTED"
     if the example fails and only the general fact (zero divisors exist at
     k=16) is established.
 4.3 Use float64 for all algebra checks (your float32 4.8e-7 is fine for
     models, not for identity verification).
 4.4 Empirical k=16 variant vs k=8 under the matched protocol.

STEP 5 - SPEED
Same batch/L/hardware for all models, eager and torch.compile, report
tokens/sec and the slowdown factor of each HRU variant vs fused GRU.

STEP 6 - FINAL REPORT (report_v2.md)
 1. Pre-registered rule at the top.
 2. Verdict table C1-C7 with labels SUPPORTED / PARTLY / REFUTED /
    INCONCLUSIVE / UNTESTABLE and the deciding number each. Use
    REFUTED only when a properly powered test contradicts the claim;
    use INCONCLUSIVE when the test is underpowered or the task unsolved.
 3. Full tables (mean +/- std, exact parameter counts).
 4. Ablation table (what each component adds).
 5. List of everything not run or scaled down, and why.
 6. A "what changed vs report.md" section based on corrections.md.
No marketing language. Do not state a conclusion that the numbers do not
support. If HRU turns out to work in some regime, say exactly where.
