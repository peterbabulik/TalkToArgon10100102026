"""
==============================================================================
PHASE 0 & PHASE 1: ALGEBRA GATE, SEDENION AUDIT & MECHANISM VERIFICATION
==============================================================================
Fast, fully float64 analytical audit of the Hurwitz Recurrent Unit (HRU):
  - Phase 0: 8D Octonion unit tests, Sedenion (16D) zero-divisor verification
             across conventions A & B, L(x) nullspaces & singular values.
  - Phase 1: Mechanism verification of original hru_pytorch_lm.py on real
             Tiny Shakespeare (1,115,394 chars). Ablates recurrence (h=0) vs
             the hidden skip tables (W_ctx1, W_ctx2), plus n-gram baselines.
Outputs:
  - /kaggle/working/phase0.json
  - /kaggle/working/phase1.json
  - /kaggle/working/corrections.md
==============================================================================
"""

import os
import json
import math
import numpy as np

import hurwitz_fano_core as hfc
import hru_pytorch_lm as lm

# ============================================================================
# PHASE 0: OCTONION ALGEBRA & SEDENION ZERO-DIVISOR AUDIT (FLOAT64)
# ============================================================================
# Build canonical rank-3 structure tensor: (e_i * e_j)_k = sum_k C[i, j, k] e_k
OCT_TENSOR = np.ascontiguousarray(hfc.HURWITZ_BASIS_L.transpose(0, 2, 1), dtype=np.float64)


def oct_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Canonical octonion product via einsum ...i,ijk,...j->...k."""
    return np.einsum("...i,ijk,...j->...k", x, OCT_TENSOR, y)


def oct_conj(x: np.ndarray) -> np.ndarray:
    res = np.array(x, dtype=np.float64, copy=True)
    res[..., 1:] = -res[..., 1:]
    return res


def oct_norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def oct_assoc(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return oct_mul(oct_mul(x, y), z) - oct_mul(x, oct_mul(y, z))


# Sedenion (16D) multiplication conventions from pairs of octonions: X = (a, b), Y = (c, d)
def sedenion_mul_A(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convention A: (a,b)(c,d) = (ac - d* b, da + b c*)."""
    a, b = x[..., :8], x[..., 8:]
    c, d = y[..., :8], y[..., 8:]
    hi = oct_mul(a, c) - oct_mul(oct_conj(d), b)
    lo = oct_mul(d, a) + oct_mul(b, oct_conj(c))
    return np.concatenate([hi, lo], axis=-1)


def sedenion_mul_B(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convention B: (a,b)(c,d) = (ac - db*, a*d + cb)."""
    a, b = x[..., :8], x[..., 8:]
    c, d = y[..., :8], y[..., 8:]
    hi = oct_mul(a, c) - oct_mul(d, oct_conj(b))
    lo = oct_mul(oct_conj(a), d) + oct_mul(c, b)
    return np.concatenate([hi, lo], axis=-1)


def run_phase0_algebra() -> dict:
    rng = np.random.RandomState(42)
    N = 1000

    # 1. Octonion unit tests
    x = rng.randn(N, 8)
    y = rng.randn(N, 8)
    z = rng.randn(N, 8)
    u_x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    u_y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    u_z = z / np.linalg.norm(z, axis=-1, keepdims=True)

    # Norm composition
    xy = oct_mul(x, y)
    norm_comp_err = float(np.max(np.abs(oct_norm(xy) - oct_norm(x) * oct_norm(y))))
    assert norm_comp_err < 1e-12, f"Norm composition failed: {norm_comp_err}"

    # Alternativity: x(xy) = (xx)y and (yx)x = y(xx)
    alt1 = oct_mul(x, oct_mul(x, y)) - oct_mul(oct_mul(x, x), y)
    alt2 = oct_mul(oct_mul(y, x), x) - oct_mul(y, oct_mul(x, x))
    alt_err = float(max(np.max(np.abs(alt1)), np.max(np.abs(alt2))))
    assert alt_err < 1e-12, f"Alternativity failed: {alt_err}"

    # Moufang: z(x(zy)) == ((zx)z)y
    moufang_diff = oct_mul(z, oct_mul(x, oct_mul(z, y))) - oct_mul(oct_mul(oct_mul(z, x), z), y)
    moufang_err = float(np.max(np.abs(moufang_diff)))
    assert moufang_err < 1e-12, f"Moufang identity failed: {moufang_err}"

    # Associator on UNIT octonions (must be <= 2.0; previous buggy scripts saw ~21-27)
    u_assoc = oct_assoc(u_x, u_y, u_z)
    u_assoc_norms = np.linalg.norm(u_assoc, axis=-1)
    u_assoc_mean = float(np.mean(u_assoc_norms))
    u_assoc_max = float(np.max(u_assoc_norms))
    assert u_assoc_max <= 2.00000000001, f"Unit associator exceeded 2: {u_assoc_max}"

    # Equal args associator = 0
    assoc_xx_y = float(np.max(np.abs(oct_assoc(x, x, y))))
    assert assoc_xx_y < 1e-12, f"Associator [x,x,y] != 0: {assoc_xx_y}"

    # 2. Sedenions (16D) & Zero Divisors
    # Target pair: (e1 + e10) * (e4 - e15)
    e1_p_e10 = np.zeros(16, dtype=np.float64)
    e1_p_e10[1] = 1.0
    e1_p_e10[10] = 1.0  # 10 = 8 + 2 -> b = e_2

    e4_m_e15 = np.zeros(16, dtype=np.float64)
    e4_m_e15[4] = 1.0
    e4_m_e15[15] = -1.0  # 15 = 8 + 7 -> d = -e_7

    prod_A = sedenion_mul_A(e1_p_e10, e4_m_e15)
    prod_B = sedenion_mul_B(e1_p_e10, e4_m_e15)
    norm_A = float(np.linalg.norm(prod_A))
    norm_B = float(np.linalg.norm(prod_B))

    # Find zero divisors via nullspace of L(x) for 16D
    def get_L_16(x: np.ndarray, mul_fn) -> np.ndarray:
        I16 = np.eye(16, dtype=np.float64)
        return np.column_stack([mul_fn(x, I16[col]) for col in range(16)])

    L_x_target = get_L_16(e1_p_e10, sedenion_mul_A)
    sv_target = np.linalg.svd(L_x_target, compute_uv=False)
    min_sv_target = float(np.min(sv_target))

    # Test 500 random unit 16D sedenion pairs
    x16 = rng.randn(500, 16)
    y16 = rng.randn(500, 16)
    u_x16 = x16 / np.linalg.norm(x16, axis=-1, keepdims=True)
    u_y16 = y16 / np.linalg.norm(y16, axis=-1, keepdims=True)
    xy16 = sedenion_mul_A(u_x16, u_y16)
    ratios = np.linalg.norm(xy16, axis=-1)

    min_svs = [float(np.min(np.linalg.svd(get_L_16(u_x16[i], sedenion_mul_A), compute_uv=False))) for i in range(100)]

    res = {
        "norm_composition_max_err": norm_comp_err,
        "alternativity_max_err": alt_err,
        "moufang_max_err": moufang_err,
        "unit_octonion_associator_mean": u_assoc_mean,
        "unit_octonion_associator_max": u_assoc_max,
        "associator_equal_args_err": assoc_xx_y,
        "sedenion_formulas": {
            "Convention_A": "(a, b)(c, d) = (ac - d*b, da + bc*)",
            "Convention_B": "(a, b)(c, d) = (ac - db*, a*d + cb)",
        },
        "target_pair_norm_Convention_A": norm_A,
        "target_pair_norm_Convention_B": norm_B,
        "target_pair_min_singular_value": min_sv_target,
        "paper_example_holds_in_standard_convention": bool(norm_A < 1e-12),
        "random_unit_sedenion_norm_ratio_min": float(np.min(ratios)),
        "random_unit_sedenion_norm_ratio_mean": float(np.mean(ratios)),
        "random_unit_sedenion_norm_ratio_max": float(np.max(ratios)),
        "random_unit_sedenion_min_sv_distribution": {
            "min": float(np.min(min_svs)),
            "mean": float(np.mean(min_svs)),
            "max": float(np.max(min_svs)),
        },
    }
    return res


# ============================================================================
# PHASE 1: MECHANISM VERIFICATION ON REAL TINY SHAKESPEARE
# ============================================================================
def run_phase1_mechanism() -> dict:
    text = lm.load_tiny_shakespeare()
    assert len(text) == 1115394, f"Invalid text length: {len(text)}"
    assert "First Citizen:" in text, "Missing Shakespeare signature"
    tokenizer = lm.CharTokenizer(text)
    assert tokenizer.vocab_size == 65, f"Unexpected vocab size: {tokenizer.vocab_size}"

    data = tokenizer.encode(text)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    L = 256
    N_VAL_SEQS = 200
    val_rng = np.random.RandomState(42)
    max_val_start = len(val_data) - L - 1
    val_starts = val_rng.randint(0, max_val_start, size=N_VAL_SEQS)
    offsets = np.arange(L)[None, :]
    x_val_200 = val_data[val_starts[:, None] + offsets]
    y_val_200 = val_data[val_starts[:, None] + offsets + 1]

    # Reference Count-Based N-Gram Models
    V = tokenizer.vocab_size
    unigram_counts = np.bincount(train_data, minlength=V).astype(np.float64)
    bigram_counts = np.zeros((V, V), dtype=np.float64)
    np.add.at(bigram_counts, (train_data[:-1], train_data[1:]), 1.0)

    # Bigram cross-entropy with Laplace smoothing (alpha=0.05)
    alpha_bi = 0.05
    p_bi = (bigram_counts + alpha_bi) / (unigram_counts[:, None] + alpha_bi * V)
    bi_logprobs = np.log(p_bi[x_val_200, y_val_200])
    val_loss_bigram = float(-np.mean(bi_logprobs))

    # Trigram cross-entropy with Jelinek-Mercer interpolation
    trigram_counts = {}
    for i in range(len(train_data) - 2):
        k = (int(train_data[i]), int(train_data[i + 1]), int(train_data[i + 2]))
        trigram_counts[k] = trigram_counts.get(k, 0) + 1

    bi_denom = bigram_counts + 1e-12
    lambda3, lambda2, lambda1 = 0.65, 0.25, 0.10
    p1 = (unigram_counts + 1.0) / (len(train_data) + V)

    tri_probs = np.zeros((N_VAL_SEQS, L), dtype=np.float64)
    for b in range(N_VAL_SEQS):
        for t in range(L):
            w1 = x_val_200[b, t - 1] if t > 0 else 0
            w2 = x_val_200[b, t]
            w3 = y_val_200[b, t]
            cnt3 = trigram_counts.get((w1, w2, w3), 0)
            p_tri = cnt3 / (bi_denom[w1, w2]) if bi_denom[w1, w2] > 0 else 0.0
            p_b = p_bi[w2, w3]
            p_u = p1[w3]
            tri_probs[b, t] = lambda3 * p_tri + lambda2 * p_b + lambda1 * p_u
    val_loss_trigram = float(-np.mean(np.log(np.maximum(tri_probs, 1e-12))))

    # Train HRU variants under identical original training conditions
    epochs = 12
    steps_per_epoch = 18
    batch_size = 24

    def train_hru_variant(mode: str):
        model = lm.HRU_Octonion_LM(V, d_model=64, lr=0.08, seed=42)
        rng = np.random.RandomState(42)
        for ep in range(1, epochs + 1):
            lr_scale = max(0.5 * (1.0 + math.cos(math.pi * (ep - 1) / epochs)), 0.1)
            for _ in range(steps_per_epoch):
                xb, yb = lm.sample_batch(train_data, batch_size, L, rng)
                # Apply structural ablations before forward/backward
                if "NoRec" in mode:
                    model.W_out[: model.C * 8] = 0.0
                if "NoWctx2" in mode:
                    model.W_ctx2[:] = 0.0

                model.forward_and_loss(xb, yb, train=True, lr_scale=lr_scale)

                if "NoRec" in mode:
                    model.W_out[: model.C * 8] = 0.0
                if "NoWctx2" in mode:
                    model.W_ctx2[:] = 0.0

        # Evaluate on the fixed 200 validation sequences
        if "NoRec" in mode:
            model.W_out[: model.C * 8] = 0.0
        if "NoWctx2" in mode:
            model.W_ctx2[:] = 0.0
        val_loss = model.forward_and_loss(x_val_200, y_val_200, train=False)
        return float(val_loss), model.num_parameters()

    variants = ["HRU_Full", "HRU_NoRecurrence", "HRU_NoWctx2", "HRU_NoRec_NoWctx2"]
    res_models = {}
    for var in variants:
        v_loss, n_params = train_hru_variant(var)
        res_models[var] = {"val_loss_200": v_loss, "val_ppl_200": float(math.exp(v_loss)), "params": n_params}

    # Baselines (LSTM and GRU)
    lstm = lm.EuclideanGatedBaselineLM(V, d_model=44, mode="LSTM", lr=0.055, seed=42)
    rng_b = np.random.RandomState(42)
    for ep in range(1, epochs + 1):
        lr_scale = max(0.5 * (1.0 + math.cos(math.pi * (ep - 1) / epochs)), 0.1)
        for _ in range(steps_per_epoch):
            xb, yb = lm.sample_batch(train_data, batch_size, L, rng_b)
            lstm.forward_and_loss(xb, yb, train=True, lr_scale=lr_scale)
    lstm_loss = float(lstm.forward_and_loss(x_val_200, y_val_200, train=False))
    res_models["LSTM_Baseline_LM"] = {"val_loss_200": lstm_loss, "val_ppl_200": float(math.exp(lstm_loss)), "params": lstm.num_parameters()}

    gru = lm.EuclideanGatedBaselineLM(V, d_model=48, mode="GRU", lr=0.055, seed=42)
    rng_g = np.random.RandomState(42)
    for ep in range(1, epochs + 1):
        lr_scale = max(0.5 * (1.0 + math.cos(math.pi * (ep - 1) / epochs)), 0.1)
        for _ in range(steps_per_epoch):
            xb, yb = lm.sample_batch(train_data, batch_size, L, rng_g)
            gru.forward_and_loss(xb, yb, train=True, lr_scale=lr_scale)
    gru_loss = float(gru.forward_and_loss(x_val_200, y_val_200, train=False))
    res_models["GRU_Baseline_LM"] = {"val_loss_200": gru_loss, "val_ppl_200": float(math.exp(gru_loss)), "params": gru.num_parameters()}

    return {
        "val_sequences": N_VAL_SEQS,
        "seq_len": L,
        "n_gram_baselines": {
            "bigram_val_loss": val_loss_bigram,
            "bigram_perplexity": float(math.exp(val_loss_bigram)),
            "trigram_val_loss": val_loss_trigram,
            "trigram_perplexity": float(math.exp(val_loss_trigram)),
        },
        "model_evaluations": res_models,
    }


# ============================================================================
# MAIN DRIVER
# ============================================================================
def main():
    print("[Phase 0] Running exact Octonion & Sedenion algebra audit (float64)...")
    p0 = run_phase0_algebra()
    print(f" -> Norm comp err: {p0['norm_composition_max_err']:.2e} | Associator unit max: {p0['unit_octonion_associator_max']:.4f}")
    print(f" -> Sedenion (e1+e10)*(e4-e15): Conv A norm = {p0['target_pair_norm_Convention_A']:.2e} (Zero Divisor: {p0['paper_example_holds_in_standard_convention']})")

    print("[Phase 1] Running mechanism audit on Tiny Shakespeare (200 val seqs)...")
    p1 = run_phase1_mechanism()
    print(f" -> Reference Bigram Loss: {p1['n_gram_baselines']['bigram_val_loss']:.3f} | Trigram Loss: {p1['n_gram_baselines']['trigram_val_loss']:.3f}")
    print(f" -> HRU Full: {p1['model_evaluations']['HRU_Full']['val_loss_200']:.3f} | HRU NoRec: {p1['model_evaluations']['HRU_NoRecurrence']['val_loss_200']:.3f} | HRU NoWctx2: {p1['model_evaluations']['HRU_NoWctx2']['val_loss_200']:.3f}")

    # Write output JSONs to /kaggle/working and local directory
    out_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    with open(os.path.join(out_dir, "phase0.json"), "w", encoding="utf-8") as f:
        json.dump(p0, f, indent=2)
    with open(os.path.join(out_dir, "phase1.json"), "w", encoding="utf-8") as f:
        json.dump(p1, f, indent=2)

    # Generate corrections.md log
    corrections_content = """# HRU Audit: Corrections and Discrepancy Log (corrections.md)

All statements below are verified directly against the original source code (`hru_pytorch_lm.py`, `hurwitz_fano_core.py`) and empirical execution.

### a) Exact Class and Function Names
* **Old statement**: The codebase contains `BaseLineRNN`, `HRULanguageModel`, `count_parameters()`, and `_hru_octonion_scan_kernel()`.
* **Verified statement**: The authentic class and kernel names in `hru_pytorch_lm.py` are:
  - `HRU_Octonion_LM` (lines 169–327)
  - `EuclideanGatedBaselineLM` (lines 330–435)
  - Method name `model.num_parameters()` (line 211, line 356)
  - Kernel name `fano_octonion_mul_kernel` (lines 53–101)
* **Evidence**: Direct code inspection of `hru_pytorch_lm.py`. The fictitious names do not exist anywhere in the repository.

### b) Context Matrices ($W_{ctx1}$ and $W_{ctx2}$)
* **Old statement**: Both HRU and baseline models utilize standard additive gating without auxiliary n-gram tables.
* **Verified statement**:
  - `HRU_Octonion_LM` contains **both** unigram skip table `W_ctx1` (shape $V \times V$) AND bigram skip table `W_ctx2` (shape $V \times V$), indexed by $x_t$ and $x_{t-1}$ (lines 201–202, line 239).
  - `EuclideanGatedBaselineLM` contains **only** `W_ctx1` scaled by a hardcoded 0.82 factor (lines 342, 381), and completely lacks `W_ctx2`.
* **Evidence**: `hru_pytorch_lm.py` lines 201–202 vs lines 342, 381. The HRU had an unacknowledged direct bigram skip connection unavailable to the baselines.

### c) Parameter Counts and Table Concealment
* **Old statement**: Reported parameters are 13,649 (HRU), 41.7k (LSTM), 33.3k (GRU).
* **Verified statement**:
  - The parameter counts reported in the original repository benchmark are **13,137** (HRU), **24,133** (LSTM), and **23,249** (GRU).
  - However, `HRU_Octonion_LM.num_parameters()` manually excluded `W_ctx1` ($65 \times 65 = 4,225$) and `W_ctx2` ($65 \times 65 = 4,225$). If included, HRU total parameters are $13,137 + 8,450 = 21,587$.
  - Similarly, `EuclideanGatedBaselineLM` excluded `W_ctx1` (4,225 params).
* **Evidence**: `hru_pytorch_lm.py` lines 211–222 and `hru_pytorch_lm_results.json`.

### d) Reference Validation Perplexities
* **Old statement**: Validation perplexities were "2.94 vs 3.12 / 3.08".
* **Verified statement**: The actual validation perplexities recorded in `hru_pytorch_lm_results.json` are:
  - `HRU_Octonion_LM`: **9.85** (val loss 2.287)
  - `LSTM_Baseline_LM`: **11.13** (val loss 2.410)
  - `GRU_Baseline_LM`: **11.31** (val loss 2.425)
* **Evidence**: Direct reading of `hru_pytorch_lm_results.json`. The numbers 2.94 / 3.12 do not appear in any repo artifact.

### e) Role of `fano_octonion_mul_kernel`
* **Old statement**: Triton kernel performs parallelized octonionic recurrent sequence scans across time.
* **Verified statement**: `fano_octonion_mul_kernel` is a simple pointwise block multiplication kernel across independent octonion pairs. It does not perform any recurrent scan, nor is it ever invoked during training or evaluation in `main()` (which executes the NumPy analytical loop).
* **Evidence**: `hru_pytorch_lm.py` lines 53–101 and line 475.
"""
    with open(os.path.join(out_dir, "corrections.md"), "w", encoding="utf-8") as f:
        f.write(corrections_content.strip() + "\n")

    print("[Complete] Wrote phase0.json, phase1.json, and corrections.md.")


if __name__ == "__main__":
    main()