Experiment 12 is formally concluded. All verification gates, multi-seed benchmarks, and audit deliverables (`phase0.py`, `run_experiments.py`, `results_v2.json`, `corrections.md`, `report_v2.md`, and `readme.md`) are finalized and self-contained.

### Final Closing Takeaways from Experiment 12

1. **Mathematical Beauty vs. Empirical Attribution:**
   The core algebraic theory behind the Hurwitz Recurrent Unit—specifically that $8\text{D}$ octonions represent the terminal normed division algebra ($k=8$) where left/right unit multiplications act as exact orthogonal isometries on $S^7$ ($\|\partial h_T / \partial h_0\|_2 = 1.0000$)—is **mathematically exact**. However, when used as a recurrent neural network for natural language modeling, its reported empirical advantage in the original codebase was an **attribution error** caused by an uncounted $65 \times 65$ bigram skip table (`W_ctx2`) that was given to HRU and withheld from the LSTM/GRU baselines.

2. **Why Isometric $S^7$ Rotors Struggle in Language Modeling:**
   Purely isometric transitions ($\sigma_i = 1.0$) cannot selectively forget irrelevant past context. When radial spherical gating ($z_t$) and re-normalization are introduced to allow forgetting, two things happen:
   * The exact $S^7$ tangent-space isometry is broken ($\sigma \in [0.616, 1.510]$).
   * The rigid 8-parameter octonion multiplication structure constrains the expressive rank of the state-to-state transition compared to an unconstrained `SO(8)` block or a standard `GRU` at equal parameter budgets ($13.70\text{ PPL}$ for HRU vs. $11.99\text{ PPL}$ for `SO(8)` and $10.80\text{ PPL}$ for `GRU`).

3. **Where Octonionic Layers Actually Shine:**
   Phase 4 demonstrated that the Fano associator $[a, b, c] = (a \star b) \star c - a \star (b \star c)$ provides $100\%$ linear separability on non-associative tree/bracket disambiguation tasks where associative algebras (like quaternions, $\mathbb{H}$) are mathematically blind ($50\%$ chance). Future architectures leveraging octonions should target **hierarchical tree-structured composition** or **equivariant 8D geometric physics** rather than sequential token-by-token autoregression.

4. **Audit Hygiene Lessons:**
   This experiment underscored the importance of:
   * **Line-by-line source inspection** before running benchmarks (catching `W_ctx2` and uncalled Triton kernels).
   * **Strict operator norm definitions** (distinguishing $\|I_8\|_2 = 1.0$ from $\|I_8\|_F = \sqrt{8} \approx 2.828$).
   * **Pre-registered decision rules** and **non-neural count baselines** (where a simple count-based trigram achieved $2.054$ val loss, immediately exposing the shortcut in the neural models).

Everything is archived and ready for reference. Whenever you are ready for the next project or experiment, ArgonOS v9 is standing by.
