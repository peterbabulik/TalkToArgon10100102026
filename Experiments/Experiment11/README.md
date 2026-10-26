# Experiment 11: Adversarial Audit & Falsification Suite — Hurwitz Recurrent Unit (HRU)

**System Environment:** ArgonOS v9 • DyTopo Swarm (Mode: ADAPTIVE)  
**Target Artifact:** *Paper 1: The Hurwitz Recurrent Unit (HRU): Norm-Preserving Division-Algebraic Gated Recurrence and Cayley-Dickson Phase-Coherent Sequence Modeling*  
**Primary Scripts Audited:** `hru_pytorch_lm.py`, `hurwitz_fano_core.py`, `run_experiments.py`, `run_audit.py`  
**Telemetry & Reports:** `hru_pytorch_lm_results.json`, `hru_benchmark_results.json`, `results.json`, `report.md`, `integrity_report.md`

---

## 1. Mission Overview & Chat History Timeline

This repository/session tracks the end-to-end adversarial audit of the **Hurwitz Recurrent Unit (HRU)** across two major review cycles:

### Phase I: Initial Adversarial Audit (Steps 1–15)
1. **Initial Implementation & Truncation Recovery (Steps 11–13):**
   - Engineered a self-contained adversarial test suite (`run_experiments.py`) to falsify Claims **C1–C7**.
   - Recovered from an initial syntax truncation at line 1,160 by refactoring `run_experiments.py` into a compact 6-suite verification pipeline.
2. **First Execution & Baseline Findings (Step 14–15):**
   - Verified the authenticity of the Karpathy `tinyshakespeare.txt` dataset (`1,115,394` characters, `65` unique characters).
   - Identified critical benchmark manipulation in `hru_pytorch_lm.py` (uncounted bigram/trigram context tables, hardcoded zero gradients on recurrent matrices, unused Hurwitz projection tensors, and dead Triton kernels).
   - Executed initial PyTorch autograd re-evaluations and produced `results.json` and `report.md`.

### Phase II: Independent Peer Review & Deep Forensic Correction (Current Cycle)
Following an independent methodological review of Phase I (`report.md` and `run_experiments.py`), a second adaptive mission was launched to fix specific measurement and reporting errors in our own Phase I audit via direct source inspection (`_inspect_workspace.py`, `_inspect_details.py`):

1. **Workspace & Source Code Line-Number Verification:**
   - Inspected exact line numbers of `hru_pytorch_lm.py` (708 lines), `Paper_1_Hurwitz_Recurrent_Unit_HRU.md` (124 lines), `hru_pytorch_lm_results.json`, and `hru_benchmark_results.json`.
2. **Identification of Phase I Audit Bugs (To Be Fixed in V2 Re-Run):**
   - **Naming Hallucinations in Phase I Report:** Phase I synthesis accidentally referred to `BaseLineRNN`, `HRULanguageModel`, `count_parameters`, and `_hru_octonion_scan_kernel` instead of the exact source identifiers (`HRU_Octonion_LM`, `EuclideanGatedBaselineLM`, `num_parameters`, `fano_octonion_mul_kernel`).
   - **Frobenius vs. Spectral Norm Bug (Test 2):** Phase I reported $\|\partial h_T / \partial h_0\|_F = 2.8283 = \sqrt{8}$ (Frobenius norm of an $8\times 8$ orthogonal matrix) instead of the spectral norm ($\|\cdot\|_2 = 1.0$).
   - **Einsum Index Transposition Bug (Test 3):** In `run_experiments.py`, `oct_mul` used `torch.einsum("...i,ikj,...j->...k", a, FANO_L, b)` instead of `...i,ijk,...j->...k`, which broke alternativity (`38.6` error instead of $\sim 10^{-15}$) and miscomputed sedenion products.
   - **Unmatched Parameter Budgets (Test 1):** Phase I compared a `13,649`-parameter HRU against a `41,665`-parameter LSTM and `33,345`-parameter GRU.
   - **Random Label Bug in Bracket Disambiguation (Test 4):** Phase I generated random labels `y_brack = torch.randint(0, 2, ...)` independent of the parenthesized octonion inputs, causing all models to score near random chance (~50%–59%).

---

## 2. Verified Forensic Findings in Original `hru_pytorch_lm.py`

Every finding below is verified directly against `hru_pytorch_lm.py` with exact line numbers:

| Finding | Exact Location in `hru_pytorch_lm.py` | Technical Details |
| :--- | :--- | :--- |
| **1. Real Class & Function Identifiers** | Lines `56`, `101`, `223`, `266`, `408`, `435` | Classes are `HRU_Octonion_LM` (L223), `EuclideanGatedBaselineLM` (L408), `HurwitzRecurrentCellTorch` (L101). Parameter counter is `num_parameters()` (L266, L435). Triton kernel is `fano_octonion_mul_kernel` (L56). |
| **2. Asymmetric & Uncounted N-Gram Tables (`W_ctx1`, `W_ctx2`)** | Lines `256–257`, `266–276`, `312`, `426`, `435–436`, `468` | `HRU_Octonion_LM` allocates **both** bigram `W_ctx1` ($65\times 65 = 4,225$) and trigram `W_ctx2` ($65\times 65 = 4,225$) skip tables (L256–257) and adds both to logits (L312), while omitting all `8,450` params from `num_parameters()` (L266–276). `EuclideanGatedBaselineLM` allocates **only** `W_ctx1` ($4,225$ params, L426) and scales it by `0.82 * self.W_ctx1[x_seq]` (L468). |
| **3. True vs. Reported Original Parameter Counts** | Lines `266–276`, `435–436`, `627–629` | **HRU (`d=64, C=8`):** Reported `13,137` (`E`: 4,160 + `W_hurwitz`: 512 + `w_rotor`: 64 + `alpha`: 8 + `gate_w`: 8 + `W_out`: 8,320 + `b_out`: 65); True total with `W_ctx1 + W_ctx2`: **`21,587`**. <br>**LSTM (`d=44`):** Reported `24,133`; True total with `W_ctx1`: **`28,358`**. <br>**GRU (`d=48`):** Reported `23,249`; True total with `W_ctx1`: **`27,474`**. |
| **4. Unused Hurwitz Projection (`W_hurwitz`)** | Lines `242–243`, `283`, `344` | `self.W_hurwitz` (`512` params) is allocated (L242) and counted in `num_parameters()` (L270), **never used** in `forward_and_loss()` (L283 uses `u_seq = oct_normalize(self.E[x_seq])` directly), and assigned `"W_hurwitz": np.zeros_like(self.W_hurwitz)` in `grads` (L344). |
| **5. Hardcoded Zero Gradients in Baselines & Gates** | Lines `347`, `494–495` | In `HRU_Octonion_LM`, `"gate_w": np.zeros_like(self.gate_w)` (L347). In `EuclideanGatedBaselineLM` (LSTM/GRU), `"W_ih": np.zeros_like(self.W_ih)` (L494) and `"W_hh": np.zeros_like(self.W_hh)` (L495). Furthermore, `dE` only receives gradient from the skip embedding `dfeats[:, :, D:]` (L490)—the recurrent core of LSTM/GRU is completely disconnected from training! |
| **6. Dead Triton & PyTorch Modules** | Lines `56–98`, `101–128`, `603–705` | `fano_octonion_mul_kernel` computes a single pointwise 8D octonion product (no scan, no backward pass). Neither `fano_octonion_mul_kernel` nor `HurwitzRecurrentCellTorch` is ever invoked in `main()`. |

---

## 3. Summary of Original vs. Phase I Audited Metrics

### 3.1 Original Benchmark Telemetry (`hru_pytorch_lm_results.json`)
* **HRU_Octonion_LM:** Reported Params = `13,137` (True = `21,587`) | Val Loss = `2.2873` | Val PPL = **`9.85`** (`9.8486`) | Tok/s = `7,869`
* **LSTM_Baseline_LM:** Reported Params = `24,133` (True = `28,358`) | Val Loss = `2.4098` | Val PPL = **`11.13`** (`11.1315`) | Tok/s = `56,221`
* **GRU_Baseline_LM:** Reported Params = `23,249` (True = `27,474`) | Val Loss = `2.4254` | Val PPL = **`11.31`** (`11.3064`) | Tok/s = `62,830`

### 3.2 Original Synthetic Telemetry (`hru_benchmark_results.json`, $L=500$)
* **HRU:** Dyck Test Acc = `0.70` (`220` params), Parity Test Acc = `1.00` (`120` params)
* **Associative_Complex_SSM:** Dyck Test Acc = `0.4375` (`162` params), Parity Test Acc = `0.625` (`114` params)
* **Standard_Linear_RNN:** Dyck Test Acc = `0.50` (`386` params), Parity Test Acc = `0.425` (`354` params)

### 3.3 Algebraic Verification (`audit_phase1.json`, `float64`)
* **Octonion Norm Multiplicativity Error ($\max |\|xy\| - \|x\|\|y\||$):** `3.55e-15`
* **Octonion Alternativity Error ($\max \|x(xy) - (xx)y\|$):** `1.23e-14`
* **Octonion Moufang Identity Error ($\max \|z(x(zy)) - ((zx)z)y\|$):** `4.36e-14`
* **Sedenion ($k=16$) Zero Divisor Check for $(e_1 + e_{10}) \star (e_4 - e_{15})$:**
  - Under standard Cayley-Dickson convention $(a,b)(c,d) = (ac - d^*b, da + bc^*)$: $\|(e_1 + e_{10})(e_4 - e_{15})\|_2 = \mathbf{0.0000}$ (Confirmed zero divisor).
  - Under alternate Cayley-Dickson convention $(a,b)(c,d) = (ac - db^*, a^*d + cb)$: $\|(e_1 + e_{10})(e_4 - e_{15})\|_2 = \mathbf{2.8284} = 2\sqrt{2}$.

---

## 4. Current Status & Pending Action Plan (Experiment 11 V2 Suite)

The workspace inspection (`_inspect_workspace.py` and `_inspect_details.py`) is complete. The next step in the pipeline—should you wish to execute the V2 suite—is to run the corrected `run_experiments.py` producing:
1. `corrections.md` — Line-by-line log of all fixes between `report.md` (v1) and `report_v2.md`.
2. `results_v2.json` — Checkpointed metrics with spectral norms ($\|\cdot\|_2$), float64 tangent-space singular value spectra on $S^7$, parameter-matched (~14k and ~25k) multi-seed LM benchmarks with LR tuning and Welch $t$-tests, curriculum state-tracking benchmarks (Parity, Dyck-2/3, $Q_8$, $S_5$, and well-defined octonion bracket disambiguation), and unit-normalized associator distributions ($\|[a,b,c]\|_2 \le 2$).
3. `report_v2.md` — Final pre-registered adversarial verdict table for Claims **C1–C7**.
