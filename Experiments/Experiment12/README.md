# Experiment 12: Adversarial Audit & Re-Evaluation of the Hurwitz Recurrent Unit (HRU)

**System:** ArgonOS v9  
**Experiment ID:** Experiment 12  
**Status:** Completed (All Phases 0–5 Executed & Verified)  
**Primary Artifacts:** `phase0.py`, `run_experiments.py`, `results_v2.json`, `corrections.md`, `report_v2.md`

---

## 1. Overview & Objective

**Experiment 12** is a rigorous, clean-slate adversarial audit and empirical re-evaluation of the **Hurwitz Recurrent Unit (HRU)**—an $8\text{D}$ octonionic recurrent neural architecture based on Hurwitz's Normed Division Algebra Theorem ($k \in \{1, 2, 4, 8\}$) and Fano-plane multiplication tables.

The objective of this experiment was to:
1. **Verify or refute Claims C1–C7** from the HRU specification using exact `float64` algebraic tests and controlled PyTorch autograd benchmarks.
2. **Audit the original repository codebase** (`hurwitz_fano_core.py`, `hru_pytorch_lm.py`, and `hru_pytorch_lm_results.json`) for architectural confounders, hidden parameter tables, and uncalled GPU kernels.
3. **Rectify methodological flaws** from earlier preliminary evaluations (such as Frobenius vs. spectral norm conflation, unnormalized Gaussian associator inflation, and mismatched parameter counts).
4. **Enforce a strict pre-registered decision rule** on fair, parameter-matched language modeling benchmarks on the real Tiny Shakespeare corpus ($1,115,394$ characters).

---

## 2. Pre-Registered Decision Rule & Outcome

> **Pre-Registered Rule:**  
> *"HRU wins only if its mean test loss is lower than the best matched baseline at equal parameters by more than 2 std AND it beats NoRecurrence."*

### Mechanical Verdict: **FAIL (REFUTED)**
Under fair PyTorch autograd Backpropagation Through Time (BPTT) with all uncounted n-gram skip tables removed:
* **HRU_Full Test Loss:** $2.6174 \pm 0.0051$ (Perplexity: $13.70$)
* **Best Matched Baseline (GRU) Test Loss:** $2.3797 \pm 0.0047$ (Perplexity: $10.80$)
* **Matched LSTM Baseline Test Loss:** $2.4414 \pm 0.0137$ (Perplexity: $11.49$)
* **NoRecurrence (Feedforward) Baseline Test Loss:** $2.5424 \pm 0.0031$ (Perplexity: $12.71$)

Not only did `HRU_Full` fail to outperform the best matched baseline by $>2\sigma$, it underperformed both Euclidean gated baselines (`GRU` and `LSTM`), the `SO(8)` orthogonal control ($2.4838$), and even the `NoRecurrence` feedforward control ($2.5424$).

---

## 3. Summary of Audit Phases & Key Findings

### Step 0: Source Code Discrepancy Audit (`corrections.md`)
Direct inspection of `hru_pytorch_lm.py` revealed critical discrepancies between the paper's narrative and the actual code:
* **Exact Identifiers:** The actual classes are `HRU_Octonion_LM` and `EuclideanGatedBaselineLM` (with method `num_parameters()`), not `HRULanguageModel` or `BaseLineRNN`.
* **Asymmetric N-Gram Skip Tables ($W_{\text{ctx1}}, W_{\text{ctx2}}$):** `HRU_Octonion_LM` adds **both** an uncounted unigram table `W_ctx1` ($65 \times 65$) and a bigram skip table `W_ctx2` ($65 \times 65$) directly to the output logits (`logits = feats @ W_out + W_ctx1[x_seq] + W_ctx2[x_prev] + b_out`). In contrast, `EuclideanGatedBaselineLM` only includes `W_ctx1` scaled by $0.82$ and **completely lacks `W_ctx2`**.
* **Parameter Count Omission:** `HRU_Octonion_LM.num_parameters()` reported $13,137$ parameters by manually excluding `W_ctx1` ($4,225$) and `W_ctx2` ($4,225$), hiding $8,450$ active parameters ($39.1\%$ of the true $21,587$ total).
* **Uncalled Triton Kernel:** `fano_octonion_mul_kernel` is a pointwise $8\text{D}$ multiplication kernel—not a parallel recurrent scan—and is never called during training or evaluation in `main()`, which runs entirely in CPU NumPy.

---

### Phase 0: Float64 Algebraic Verification Gate
Executed in strict double precision (`torch.float64` / `np.float64`):
* **Norm Composition ($\|x \star y\|_2 = \|x\|_2 \|y\|_2$):** Verified to machine precision; max error $= 5.33 \times 10^{-15} < 10^{-12}$.
* **Alternativity & Moufang Identities:** Verified to $< 10^{-14}$ max error across $N=1,000$ random octonion triples.
* **Unit Octonion Associator Bound ($\|[u, v, w]\|_2 \le 2.0$):** For unit octonions $u, v, w \in S^7$, the associator norm achieved a mean of $1.0819$ and a max of $1.8608 \le 2.0$. *(Resolved earlier buggy scripts that passed unnormalized $\mathcal{N}(0, I_8)$ vectors of norm $\approx \sqrt{8}$, inflating the cubic product by $(\sqrt{8})^3 \approx 22.6$).*
* **Sedenion ($16\text{D}$) Zero Divisors:** Under Cayley–Dickson doubling, the specific basis pair $(e_1 + e_{10}) \star (e_4 - e_{15})$ yields norm $2.0$ under our Fano sign table in both Conventions A and B, showing that zero-divisor coordinates depend on the specific Fano plane orientation. However, SVD nullspace analysis of the left-multiplication operator $L(x)$ confirmed exact zero divisors in $16\text{D}$ Sedenions with product norm $< 10^{-15}$.

---

### Phase 1: Mechanism Verification on Tiny Shakespeare
Evaluated on $200$ fixed validation sequences ($L=256$) from the held-out last $10\%$ of Tiny Shakespeare ($1,115,394$ chars, $V=65$):

| Model / Variant | Validation Loss | Validation Perplexity | Mechanism Finding |
| :--- | :---: | :---: | :--- |
| **Count-Based Bigram** (Laplace $\alpha=0.05$) | $2.4860$ | $12.01$ | Zero neural parameters |
| **Count-Based Trigram** (Interpolated) | $2.0540$ | $7.80$ | Pure 2-step lookup beats all neural models |
| **`HRU_Full`** (12 epochs, original protocol) | $2.3030$ | $10.00$ | Includes $W_{\text{ctx1}} + W_{\text{ctx2}}$ |
| **`HRU_NoRecurrence`** ($h_t \equiv 0$, tables kept) | $2.2940$ | $9.91$ | **Zeroing recurrence slightly improves loss** |
| **`HRU_NoWctx2`** (Recurrence kept, $W_{\text{ctx2}} \equiv 0$) | $2.5050$ | $12.24$ | **Removing $W_{\text{ctx2}}$ collapses performance** |
| **`LSTM_Baseline_LM`** (Original protocol) | $2.4100$ | $11.13$ | Lacked $W_{\text{ctx2}}$ in original repo |
| **`GRU_Baseline_LM`** (Original protocol) | $2.4250$ | $11.30$ | Lacked $W_{\text{ctx2}}$ in original repo |

**Finding:** In the original codebase, HRU's lower validation loss was $100\%$ an artifact of the uncounted bigram skip table `W_ctx2`. Removing the octonionic recurrence ($h_t \equiv 0$) did not hurt performance at all ($2.303 \to 2.294$), whereas removing `W_ctx2` degraded validation loss to $2.505$ (worse than both LSTM and GRU).

---

### Phase 2: Fair Parameter-Matched Autograd LM Benchmark
All models were implemented in PyTorch with full BPTT, no skip tables, verified on a single-batch overfit sanity check (all reached loss $< 0.5$), and evaluated across $3$ random seeds (`42, 101, 202`):

| Model Architecture | Parameters | Test Loss (Mean $\pm$ Std) | Test Perplexity | Bits Per Char (BPC) |
| :--- | :---: | :---: | :---: | :---: |
| **`GRU` Baseline** | $13,835$ | **$2.3797 \pm 0.0047$** | **$10.80$** | **$3.433$** |
| **`LSTM` Baseline** | $13,355$ | $2.4414 \pm 0.0137$ | $11.49$ | $3.522$ |
| **`SO8_OrthogonalRNN` Control** | $9,815$ | $2.4838 \pm 0.0052$ | $11.99$ | $3.583$ |
| **`NoRecurrence` Control** | $13,715$ | $2.5424 \pm 0.0031$ | $12.71$ | $3.668$ |
| **`HRU_PureRotor`** ($z=1, \alpha=0$) | $9,473$ | $2.6121 \pm 0.0041$ | $13.63$ | $3.768$ |
| **`HRU_Full`** ($k=8$, Fano Associator) | $9,485$ | $2.6174 \pm 0.0051$ | $13.70$ | $3.776$ |
| **`HRU_Alpha0`** ($\alpha=0$) | $9,479$ | $2.6188 \pm 0.0047$ | $13.72$ | $3.778$ |

---

### Phase 3: Spectral Norm (`ord=2`) & $S^7$ Tangent Space Diagnostics
Computed in `float64` using the exact operator 2-norm (`torch.linalg.matrix_norm(J, ord=2)`):
* **Pure Rotor Isometry:** For `HRU_PureRotor` ($z=1, \alpha=0$), the $7\text{D}$ tangent-space singular values are identically $[1.0000, 1.0000, 1.0000]$, and the end-to-end Jacobian spectral norm $\|\partial h_T / \partial h_0\|_2 = 1.0000$ across $T \in \{64, 256, 1024\}$. *(Correction: Earlier reports cited $2.8284 = \sqrt{8}$ because they computed the Frobenius norm $\|I_8\|_F = \sqrt{8}$ rather than the spectral operator 2-norm).*
* **Gated HRU Spectrum:** Introducing the radial spherical projection gate ($z=0.634, \alpha=0.22$) breaks exact isometry on $S^7$, spreading the single-step tangent-space singular values to $[\sigma_{\min} = 0.6161, \sigma_{\text{med}} = 1.4498, \sigma_{\max} = 1.5098]$.

---

### Phase 4 & Phase 5: State Tracking & Hardware Throughput
* **Non-Associative Bracket Disambiguation:** Octonionic features separated left- vs. right-bracketed products $((a \star b) \star c)$ vs. $(a \star (b \star c))$ with $100.0\%$ accuracy, whereas associative quaternion controls remain at $50.0\%$ chance.
* **Parity Length Generalization:** When trained on $L=16$, neither `HRU` ($49.3\%$) nor `LSTM` ($50.7\%$) generalized zero-shot to $L=256$.
* **GPU Throughput (`Phase 5`):** Due to sequential Python-level loop unrolling of `einsum` contractions, `HRU_Full` processed **$51,207\text{ tokens/sec}$** compared to **$670,287\text{ tokens/sec}$** for fused cuDNN `GRU`—a **$13.09\times$ slowdown**.

---

## 4. Executive Verdict Table (Claims C1–C7)

| Claim | Description | Verdict | Empirical Evidence from Experiment 12 |
| :---: | :--- | :---: | :--- |
| **C1** | HRU outperforms LSTM & GRU on character-level LM | **REFUTED** | Fair autograd BPTT: HRU PPL ($13.70$) is worse than GRU ($10.80$), LSTM ($11.49$), and SO(8) ($11.99$). |
| **C2** | $8:1$ Hurwitz block compression yields superior efficiency | **REFUTED** | Original repo hid $8,450$ skip-table parameters; when ablated (`NoWctx2`), HRU loss collapses from $2.303 \to 2.505$. |
| **C3** | Exact $S^7$ norm conservation & unit spectral norm | **SUPPORTED** | Pure rotor maintains $\|\partial h_T / \partial h_0\|_2 = 1.0000$ at $T=1024$ and unit norm drift $< 10^{-14}$. |
| **C4** | Radial gating preserves isometric gradient flow | **PARTLY** | Gating + projection warps $S^7$ tangent singular values to $[0.6161, 1.5098]$, destroying exact isometry. |
| **C5** | $3.2\times$ faster convergence on state-tracking tasks | **REFUTED** | HRU converges at parity with or slower than GRU/LSTM and fails length generalization at $L=256$ ($49.3\%$). |
| **C6** | Sedenion ($k=16$) zero-divisor boundary | **PARTLY** | Nullspace SVD confirms exact $16\text{D}$ zero divisors ($<10^{-15}$), though basis pair coordinates depend on Fano sign convention. |
| **C7** | Triton GPU kernel accelerates recurrent scan | **REFUTED** | `fano_octonion_mul_kernel` is pointwise only, never called in `main()`, and PyTorch HRU is $13.09\times$ slower than cuDNN GRU. |

---

## 5. Repository Structure & Reproduction Instructions

### Generated Files
* `phase0.py` — Standalone double-precision (`float64`) algebra gate and original mechanism ablation script; generates `phase0.json`, `phase1.json`, and `corrections.md`.
* `run_experiments.py` — Full end-to-end PyTorch GPU benchmark suite executing Phases 0 through 5 and generating all final reports.
* `results_v2.json` — Structured JSON containing all raw metrics, singular value spectra, parameter counts, and multi-seed losses.
* `corrections.md` — Line-by-line audit log documenting code discrepancies in `hru_pytorch_lm.py`.
* `report_v2.md` — Comprehensive adversarial audit report.

### How to Run
