ROLE
You are an adversarial reviewer. Your job is to FALSIFY the claims of the
Hurwitz Recurrent Unit (HRU) work, not to confirm them. Report negative
results plainly. Every number must come from code you executed; never
estimate or recall numbers. If something is untestable, say UNTESTABLE.

ENVIRONMENT
Kaggle notebook, GPU (T4/P100), PyTorch with real autograd. Time budget
<= 3 h total: expose a TIME_BUDGET variable, reduce steps/seeds if needed,
and report exactly what was scaled down. Checkpoint results to JSON after
every run so a session cut loses nothing. Run long jobs in background /
chunks (watchdog-safe). Write everything to /kaggle/working:
results.json, per-seed logs, report.md.

INPUT FILES
hurwitz_fano_core.py, hru_pytorch_lm.py, hru_pytorch_lm_results.json,
Paper_1_Hurwitz_Recurrent_Unit_HRU.md (and optionally hru_benchmark_results.json).

CLAIMS UNDER TEST
C1 HRU reaches 9.85 val PPL vs LSTM 11.13 / GRU 11.31 with fewer params
   (13,137 vs 24,133 / 23,249) on Tiny Shakespeare, L=256.
C2 8:1 Hurwitz compression costs no representational power.
C3 Exact S^7 isometry prevents vanishing/exploding gradients.
C4 Jacobian singular values stay ~1.000 +/- 1e-4 during training; 3.2x
   fewer epochs on long-horizon tasks.
C5 Octonion associator gives +14.8% hierarchical bracket accuracy over
   associative algebras (k in {1,2,4}).
C6 Sedenions (k=16) fail because of zero divisors, e.g.
   (e1+e10)*(e4-e15)=0.
C7 HRU: Dyck test acc 0.70 vs 0.44/0.50 baselines; parity 1.0 vs 0.63/0.43.

TEST 0 - Integrity of the original benchmark
- Assert corpus length == 1,115,394 AND that it is the real Karpathy text
  (not the fallback): vocab 65, text contains "First Citizen:" and the
  first 1000 chars are not repeated later. Abort otherwise.
- Re-run the original script; compare to the published JSON.
- Count parameters honestly: every trainable array, including W_ctx1/W_ctx2
  and any unused weights. Report which parameters receive zero or heuristic
  gradients (W_hurwitz, gate_w, W_ih, W_hh, w_rotor, alpha).
- Check whether the Triton kernel has a backward pass and whether main()
  ever calls it.

TEST 1 - Fair language-model comparison (answers C1, C2)
Rewrite all models in PyTorch with TRUE autograd through all timesteps and
ALL recurrent weights trainable. Identical embedding dim, identical linear
readout, no extra n-gram skip tables for any model (a second table-matched
variant is allowed). Models:
 a) LSTM  b) GRU
 c) HRU-full (octonion product, learnable gate z, learnable alpha)
 d) HRU alpha=0   e) HRU z=1, alpha=0 (pure rotor)
 f) quaternion-block k=4 and complex-block k=2 versions at equal width
 g) plain orthogonal-block RNN (SO(8) per fiber, same normalisation) as a
    "no octonion structure" control
 h) no-recurrence control (embedding + readout only)
 i) count-based bigram and trigram on the same split (reference)
Parameter budgets: ~25k and ~100k, counted with sum(p.numel()). Report
the exact count of each model. Same optimizer, steps, batch, schedule.
Tune LR per model over a small grid on a held-out TUNE split; report on a
separate TEST split (>= 500k chars, fixed sequences). >= 3 seeds (5 if time).
Report mean +/- std val loss, BPC and PPL, plus paired bootstrap or Welch
test for HRU vs best baseline. Pre-registered rule: HRU "wins" only if it
beats the best baseline by > 2 std at equal parameters and beats (h).

TEST 2 - Gradient / isometry diagnostics (answers C3, C4)
For untrained and trained models of each variant: measure
||dh_T/dh_0||_2 via autograd at T = 64, 256, 1024; the tangent-space
singular value spectrum of one step on S^7; and the same quantity for
LSTM/GRU. State separately what is guaranteed by construction (post-hoc
renormalisation) and what is measured. Check whether the isometry holds
in the configuration that was actually trained (z != 1, alpha != 0).

TEST 3 - Algebra (C6)
Verify numerically: norm composition, alternativity, Moufang identity,
non-associativity. Build 16D Cayley-Dickson (standard convention) and
test whether (e1+e10)*(e4-e15) = 0. If not, report the convention under
which a zero divisor exists and give one verified example. Show whether
zero divisors affect the model empirically (k=16 variant vs k=8).

TEST 4 - State-tracking tasks (C4, C5, C7)
Use >= 2000 test samples per task, balanced classes, >= 3 seeds, all
baselines from Test 1 with matched params:
 - Parity, L = 500 and 1000.
 - Dyck-2 and Dyck-3 closure, train L<=64, test L = 64/256/512,
   accuracy stratified by nesting depth with counts per bucket.
 - Group word problem (quaternion group Q8 and S5) to test non-commutative
   tracking.
 - Bracket disambiguation per C5: implement the task exactly as the paper
   describes. If underspecified, say so, build the closest well-defined
   version, and label it as NOT the paper's task.
Report accuracy +/- std, epochs to reach 90% (for the 3.2x claim), and
whether results change with a random-label sanity check.

TEST 5 - Speed
Tokens/sec for each model at equal batch/L, with and without
torch.compile. Report HRU's slowdown honestly.

REPORT (report.md)
1. Verdict table: each of C1-C7 -> SUPPORTED / PARTLY / REFUTED /
   UNTESTABLE, with the number that decides it.
2. All tables with mean +/- std and parameter counts.
3. Ablation table from Test 1 (what each component contributes).
4. List of anything you could not run and why.
No marketing language. No conclusions that the numbers do not support.
