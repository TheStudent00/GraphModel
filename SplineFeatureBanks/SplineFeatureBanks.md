# Title

**Function-Space Message Passing via Monotone Spline Feature Banks and Sparse Learned Permutations**

---

## Abstract

We introduce a neural computation paradigm in which inter-module communication occurs not through discrete parameter vectors but through projections onto a continuous *rank-domain function space*.
Each module exposes a fixed, globally ordered bank of monotone spline features that forms a universal communication interface.
Messages from other modules are transformed by learned **sparse permutation matrices** that align their output ordering to this canonical feature basis.
Interactions between modules are performed analytically through precomputed spline overlap integrals, enabling *a priori* sparsification of computation based on functional orthogonality.
Cross-resolution mixing is achieved through analytic **least-common-refinement (LCR)** operators that preserve differentiability across different spline resolutions.
The framework allows plug-compatible transfer learning, where a frozen “subject-matter-expert (SME)” module can be reused by training only a permutation adapter, and provides a foundation for continuous, resolution-aware neural architectures.

---

## 1. Background and Motivation

Modern neural networks represent all internal states as discrete vectors or tensors of fixed size.
While efficient, this representation discards information about continuity and ordering, and hinders interoperability between modules of differing dimensionality or resolution.
Graph neural networks, convolutional networks, and transformers all rely on vector-space message passing; alignment between heterogeneous layers requires bespoke adapters.

This work explores a **function-space alternative**: representing activations as continuous monotone functions over rank space and communicating via analytic projections.
This provides a common “language” for all nodes or sub-graphs, independent of their native vector dimensionality.

---

## 2. Core Components

### 2.1 Monotone Spline Feature Bank

Each module defines a bank
[\mathcal{F}={f_c(r)\mid c=1,\dots,C,\ r\in[0,1]},]
where each (f_c) is a monotone cubic (or higher-order) spline on a fixed knot vector.
The bank is globally ordered by monotonic increase and remains sorted during learning.
Parameters of (f_c) include control coefficients and optional knot offsets; monotonicity is enforced by non-negative slope constraints.

### 2.2 Learned Sparse Permutation

A sender module produces an output vector (m\in\mathbb{R}^n).
A learned permutation matrix (\Pi\in\mathbb{R}^{n\times n}) reorders the elements of (m) to align with the receiver’s feature order:
[\tilde m = \Pi m .]
(\Pi) is trained via a differentiable relaxation such as a band-masked **Gumbel–Sinkhorn** operator, constrained to be sparse and near-one-hot.
During inference, (\Pi) collapses to a hard gather/scatter index with (O(n)) cost.

### 2.3 Analytic Message Projection

Given precomputed rank-grid weights
[w_{c,k}=\int_{(k-1)/n}^{k/n}\phi_k(r)f_c(r),dr,]
the interaction (or “message score”) between a permuted message (\tilde m) and feature (f_c) is
[s_c = \sum_{k=1}^{n} w_{c,k},\tilde m_k .]
This analytic inner product replaces conventional learned dot products.
Because the weights depend only on (f_c), they are reusable across all senders.

### 2.4 Overlap-Based Gating

Functional overlaps
[S_{cd}=\int f_c(r)f_d(r),dr]
approximate expected correlation between features.
Interactions with (|S_{cd}|<\tau) are pruned or softly gated:
[g_{cd}=\sigma(\alpha(S_{cd}-\tau)).]
This provides *analytic attention sparsity*—a static prediction of negligible interactions that reduces computation without loss of accuracy.

### 2.5 Least-Common-Refinement (LCR) Mixing

Modules operating at different spline resolutions are connected through fixed linear maps:

* **Prolongation** (P:V_h!\to!V_{h\wedge h'}) by knot insertion.
* **Restriction** (R:V_{h'}!\to!V_{h\wedge h'}) by (L^2)-projection.
  Mixing always occurs at the LCR space (V_{h\wedge h'}), ensuring continuous, differentiable cross-resolution interaction.

---

## 3. Learning and Optimization

1. **Permutation phase** – Freeze feature banks; train permutation parameters (\Pi) using differentiable relaxations and entropy/band regularization.
2. **Amplitude phase** – Unfreeze spline control coefficients; update with standard optimizers or Riemannian preconditioning using the spline Gram matrix (G).
3. **Optional knot tuning** – Adjust knot positions under ordering constraints and curvature regularization.

All operations are differentiable; custom optimizers can precondition gradients by (G^{-1}) to account for functional geometry.

---

## 4. Applications

* **Graph Neural Networks:** Nodes communicate via spline projections instead of vector messages.  Heterogeneous sub-graphs can interoperate through learned permutations only.
* **Transformer-style Attention:** Keys and queries represented as monotone splines; attention weights predicted by analytic overlaps.
* **Convolutional Networks:** Local patches converted to rank-domain signals; convolution replaced by spline projection kernels.
* **Transfer and Modular Learning:** Pretrained SME modules (feature banks + overlaps) reused across tasks by retraining only permutation adapters.

---

## 5. Advantages

| Property                   | Effect                                      |
| -------------------------- | ------------------------------------------- |
| Universal monotone basis   | Shared functional “language” across modules |
| Learned sparse permutation | Plug-compatible alignment, (O(n)) inference |
| Analytic overlap gating    | Predictive sparsity, reduced compute        |
| LCR mixing                 | Resolution-stable interaction               |
| Functional preconditioning | Better gradient conditioning                |
| Plug-compatible transfer   | Reuse without SME retraining                |

---

## 6. Comparative Novelty

| Prior art                    | Limitation                                  | Present contribution                                       |
| ---------------------------- | ------------------------------------------- | ---------------------------------------------------------- |
| NeuralSort / Gumbel–Sinkhorn | Permutations for sorting/equivariance only  | Permutations as cross-module alignment in functional space |
| DeepSets / Set Transformer   | Vector aggregation, no functional interface | Rank-domain spline interface, analytic projection          |
| SplineCNN                    | Spatial convolution with B-spline kernels   | Universal monotone feature bank + analytic gating          |
| Neural Spline Flows          | 1-D invertible mappings                     | Message passing and SME transfer                           |
| Attention sparsity methods   | Heuristic/data-driven                       | Analytic sparsity from functional overlaps                 |

---

## 7. Experimental Program (Outline)

1. **Synthetic function matching:** Train permutation adapters between random spline modules; evaluate alignment accuracy and energy preservation.
2. **Graph classification:** Replace vector messages in a GNN with spline projections; measure accuracy/FLOP trade-off.
3. **Transfer learning:** Freeze a trained spline bank; retrain permutations on new graph domains; compare to full finetuning.
4. **Ablation:** analytic gating on/off, LCR vs. ad-hoc sampling, monotone vs. unconstrained splines.

---

## 8. Claims and Enabling Disclosure (for IP)

1. **A neural architecture** wherein all inter-module communications are expressed as projections of signals onto a bank of monotone spline functions defined over a standardized rank domain.
2. **A method** of aligning heterogeneous modules by learning sparse permutation matrices that reorder sender outputs to match the receiver’s spline ordering, trainable via differentiable relaxation and evaluable as an (O(n)) gather.
3. **A computational gating scheme** based on precomputed spline overlap integrals that statically suppresses low-signal interactions.
4. **A cross-resolution mixing mechanism** using analytic prolongation and restriction (least-common-refinement) operators between spline spaces.
5. **A training strategy** employing Riemannian or Gram-preconditioned optimization within the spline function space.
6. **Applications** to graph, attention, and convolutional networks where SME modules can be frozen and reused by training only permutation adapters.

---

## 9. Discussion and Outlook

This approach shifts neural computation from discrete parameter spaces to continuous functional manifolds.
It provides built-in alignment, sparsity, and transfer properties that naturally emerge from the geometry of monotone splines.
Beyond immediate architectural gains, it suggests a broader paradigm in which neural systems communicate through standardized continuous representations—potentially bridging symbolic modularity and continuous learning.

---

*(End of disclosure draft)*
