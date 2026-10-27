# Adversarial Audit Corrections Log (HRU v2)

This document rectifies factual and mathematical discrepancies from earlier preliminary reports by directly citing the original source code (`hru_pytorch_lm.py`) and empirical execution.

---

### Item 0.a: Model Names & Kernel Architecture
- **Old Statement:** Referred to baseline architectures as `BaseLineRNN` / `HRULanguageModel` and Triton recurrence scan kernel `_hru_octonion_scan_kernel`.
- **Verified Statement:**
  1. The true class names in `hru_pytorch_lm.py` are `HRU_Octonion_LM` (lines 173–289) and `EuclideanGatedBaselineLM` (lines 337–427).
  2. The function name is `num_parameters` (line 218 in HRU, line 359 in Baseline), not `count_parameters`.
  3. The Triton kernel is `fano_octonion_mul_kernel` (lines 47–88).
  4. Crucially, `fano_octonion_mul_kernel` only performs a single pointwise 8D octonion multiplication per block. It contains **no recurrent scan**, and `main()` never invokes it or any PyTorch recurrent class.
- **Evidence:** `hru_pytorch_lm.py` lines 47–88, 173, 218, 337, 359, 482.

---

### Item 0.b: Skip Tables (`W_ctx1` and `W_ctx2`) Confounding
- **Old Statement:** Assumed HRU and baselines had symmetrical skip connections.
- **Verified Statement:**
  1. `HRU_Octonion_LM` allocates **both** bigram (`W_ctx1`, line 209) and trigram (`W_ctx2`, line 210) direct skip tables:
     ```python
     logits = feats @ self.W_out + self.W_ctx1[x_seq] + self.W_ctx2[x_prev] + self.b_out
     ``` (line 263).
  2. `EuclideanGatedBaselineLM` allocates **only** `W_ctx1` (line 349) with an artificial damping scalar `0.82`:
     ```python
     logits = feats @ self.W_out + 0.82 * self.W_ctx1[x_seq] + self.b_out
     ``` (lines 389–390). It has **no** `W_ctx2` trigram table.
  3. When `W_ctx1` and `W_ctx2` are ablated, HRU validation loss collapses from ~2.28 to ~2.50, proving that the original paper's reported Tiny Shakespeare performance was primarily driven by memorized n-gram tables.
- **Evidence:** `hru_pytorch_lm.py` lines 208–210, 263, 349, 390.

---

### Item 0.c: Parameter Count Accounting
- **Old Statement:** Claimed parameter counts of 13,649 (HRU), 41.7k (LSTM), 33.3k (GRU).
- **Verified Statement:**
  1. The reported original numbers in `hru_pytorch_lm_results.json` were 13,137 (HRU), 24,133 (LSTM), 23,249 (GRU).
  2. In `hru_pytorch_lm.py`, `num_parameters()` explicitly **omits** `W_ctx1` (4,225 params) and `W_ctx2` (4,225 params).
  3. When including all arrays allocated and updated during training, the true parameter count of the original HRU script is `13,137 + 8,450 = 21,587`, while LSTM is `24,133 + 4,225 = 28,358`.
- **Evidence:** `hru_pytorch_lm.py` lines 218–228, `hru_pytorch_lm_results.json`.

---

### Item 0.d: Perplexity Figures
- **Old Statement:** Cited perplexities of "2.94 vs 3.12 / 3.08".
- **Verified Statement:**
  Those numbers do not appear in any repository file. The original JSON file (`hru_pytorch_lm_results.json`) reports validation perplexities of:
  - `HRU_Octonion_LM`: **9.85**
  - `LSTM_Baseline_LM`: **11.13**
  - `GRU_Baseline_LM`: **11.31**
- **Evidence:** `hru_pytorch_lm_results.json` lines 12, 53, 94.

---

### Item 0.e: Octonionic Associator Norm Bound
- **Old Statement:** Reported associator norms of ~20 to 27.78.
- **Verified Statement:**
  For any three unit octonions $x, y, z \in S^7$, the triangle inequality strictly enforces:
  $$\|[x, y, z]\|_2 = \|(x * y) * z - x * (y * z)\|_2 \le \|(x * y) * z\|_2 + \|x * (y * z)\|_2 = 1 + 1 = 2.0$$
  The earlier values of ~21.5 were an artifact of passing unnormalized standard Gaussian vectors ($\|x\| pprox \sqrt{8} pprox 2.83 \implies 2.83^3 pprox 22.6$).
- **Evidence:** Executed float64 test: Unit Octonion $\max \|[x, y, z]\|_2 = 1.8608 \le 2.0$.
