# Graph Model Intellectual Property (IP) Description

This document captures the conceptual, architectural, and theoretical foundations of the Graph Model Oracle, defining the novel components, structural primitives, and meta-learning dynamics protected under the OTU Green License.

The Graph Model is a general-purpose, modular, hierarchical learning system built from spline-standardized features, attention-based routing primitives, self-modifying architecture, and multi-regime optimization dynamics. It is designed for sparse, flexible, expandable intelligence.

---

# 1. Purpose and Scope

This description asserts IP protection over:

- the monotone-spline feature standardization process,
- the Core/Module/MindsEye structural hierarchy,
- attention-as-atomic-primitive routing embedded into modular computation,
- the 33-level nascent abstraction hierarchy,
- dynamic architecture search (NAS) with cloning, merging, and complexity gradients,
- overlap management and symmetry-breaking mechanisms,
- internal memory compression/expansion logic,
- module-to-module logistics (request/response),
- learnable permutation structures per contact,
- meta-learning strategies for bootstrapping MindsEye,
- and the general organizational logic of the Graph Model Oracle.

General ML concepts are not claimed—only the novel structural organization and interplay.

---

# 2. High-Level Summary

The Graph Model is a modular learning substrate composed of interacting Modules, each containing Cores responsible for state, context, service, and contact behaviors. It uses:

- spline-standardized features,
- atomic Q/K/V attention within each Core,
- multi-level abstraction hierarchy,
- dynamic module cloning and specialization,
- multi-regime optimization,
- and meta-learning governing self-modification.

The system aims to unify:

- sparse modularity,
- feature universality,
- evolutionary architectural dynamics,
- differentiable learning,
- and agency-like request/response structures.

---

# 3. Core Architectural Components

## 3.1 Core

A Core is the atomic computational operator. It includes:

- **Monotone-spline feature extraction** with sorted-input canonicalization.
- **Learned-permutation projection**, enabling mapping between feature-space orders.
- **Parallel Q/K/V feature layers** for attention-based routing and transformation.
- **Contextual mixing** via softmax(QK)·V.
- **Expandable parallel and downstream feature layers** through NAS.

A Core is an embedded expert capable of both computation and communication.

---

## 3.2 Module

A Module is a container for multiple Cores:

- **State Core**  
  Long-term, slow-changing representation used to generate outputs.

- **Context Core**  
  Short-lived state modified by ongoing interactions. A fast-timescale modulator.

- **Service Core**  
  A callable function providing transformations for other Modules on request.

- **Contact Core**  
  Decides whether to respond immediately, consult experts, or route the request.

- **Memory subsystem**  
  Recent (uncompressed) and older (compressed) memories in embedded space.

- **Logistics subsystem**  
  Handles nested requests, message passing, identity, and scheduling.

- **Complexity subsystem**  
  Tracks space/time tokens and drives reduction or expansion under pressure.

Modules behave as adaptive, semi-autonomous computational agents.

---

## 3.3 MindsEye (Meta-Learner)

MindsEye monitors the entire Graph:

- adjusts optimization regimes,
- triggers architectural exploration,
- allocates complexity tokens,
- evaluates module utility,
- manages cloning and merging,
- routes learning signals,
- and coordinates global structural updates.

It is not a controller; it is an evolving meta-optimizer embedded within the system.

---

# 4. Monotone-Spline Feature Standardization

A key innovation:

1. Input or internal feature vectors are **sorted** (descending or ascending).
2. The sorted vector is treated as a **monotone curve**.
3. The curve is encoded via **monotone splines**.
4. A learned-permutation matrix maps between raw feature ordering and canonicalized forms.

Benefits:

- universal geometric representation,
- feature comparability across Modules,
- reduced entanglement,
- stable dynamics for Q/K/V mixing,
- scalable expandability.

This canonicalization is central to Graph Model’s universality.

---

## 4.1 Spectral Permutation Families as Core-Level Geometry

Permutation logic is no longer stored inside Contact structures, nor is it modeled by simple curves. Instead, each Core owns one or more **Spectral Permutation Families**—continuous, learnable fields that generate discrete permutations for any input length via a hybrid Fourier basis.

A Spectral Permutation Family `F` is defined as a learnable function $f(t)$ composed of two distinct frequency regimes:

1.  **Absolute Frequencies (Global Geometry):**
    Standard Fourier terms $\sin(k \pi t)$ that capture macroscopic reordering (e.g., rotation, reversal, pivot-sorting). These are invariant to input size $N$.

2.  **Relative Frequencies (Microstructure):**
    $N$-dependent terms $\sin(k \cdot N \cdot \pi t)$ that capture relative structural patterns (e.g., interleaving, local neighbor swaps). These scale their frequency to match the resolution of the input vector.

**The Nyquist Guardrail:**
To prevent aliasing—where high-frequency logic creates chaotic artifacts on small vectors—the system applies a dynamic **Nyquist Mask**. For an input of size $N$, any frequency component $f_{comp} > N/2$ is zeroed out. This ensures universality across variable-length inputs without aliasing.

Each Core may maintain multiple families (e.g., "canonical" → F0, "alt1" → F1), allowing it to project the same spline-standardized features into different local geometric dialects depending on state or context.

---

## 4.2 Receiver-Centric Feature-Space Interpretation

Permutation Families belong strictly to the **receiving module**, not the sender.

This ensures:
- The receiver controls its own feature-space geometry (Subjective Interpretation).
- Cores maintain a stable internal canonical form regardless of input source.
- The number of permutations scales as $O(\text{modules} \times \text{families})$ rather than $O(\text{modules}^2)$.

**Differentiable Rank & Pipeline:**
Incoming messages are processed through a differentiable pipeline:

    raw_in → monotone-spline sorting → Spectral f(t) scores → SoftSort/Relaxation → Q/K/V projection

Unlike traditional discrete methods (Gumbel-Sinkhorn), this architecture learns the **Law of the Permutation** via **Spectral Regularization**. By penalizing high-frequency coefficients, the system enforces a "Complexity Gradient" on the structure itself, learning simple geometric moves (low cost) before evolving complex, high-frequency shuffles.

This enforces a clean separation between:
- **Universal Reality:** Spline geometry (shared by all modules),
- **Local Perspective:** Spectral coordinate frames (defined per-Core),
- **Communication:** Metadata (handled by Contact).

---

## 4.3 Contact Structure Simplification

`class Contact` no longer stores permutation matrices.

Its role is now minimal and strictly logistical:

    - identify target module
    - store communication metadata
    - track request/response channels
    - optionally store sparse routing preferences

The Contact object is no longer responsible for index geometry.

This prevents:
- combinatorial explosion of pairwise per-module permutations,
- degradation of expert-local identity,
- cross-module entanglement of feature geometry.

The geometry is handled solely in Cores.

---

## 4.4 Integration With Attention and Feature Canonicalization

Cores now follow the consistent pipeline:

1. **Sort** incoming vector into monotone canonical curve.  
2. **Spline-encode** canonicalized curvature.  
3. **Select** a permutation family based on internal logic.  
4. **Generate** a discrete permutation for the vector length.  
5. **Reorder** the spline embedding into the local coordinate frame.  
6. **Apply Q/K/V attention** within that frame.

This preserves:
- universality of splines,
- diversity of expert geometries,
- compatibility of heterogeneous modules,
- and stability of all attention operations.

Permutation Families act as **local geometric dialects** spoken by each expert module.


# 5. Attention as an Atomic Routing Primitive

Attention is not a layer—it is a primitive inside every Core:

- **Q** = what a Core seeks (query).
- **K** = how a Core exposes its structure (key).
- **V** = what a Core communicates (value).

Softmax(QK) determines routing strengths; V mixes messages.

Attention serves as:

- routing,
- contextual integration,
- specialization scaffold,
- message passing,
- and internal transformation.

NAS can expand Cores, but attention remains stable as the central operator.

---

# 6. Multi-Regime Optimization

The system switches training regimes:

- **Large mini-batches** for early stability,
- **Small mini-batches** for structural differentiation,
- **Online SGD** for fine causal learning,
- **Batch fallback** when gradients destabilize.

MindsEye monitors graph-wide signals (gradient variance, module drift, complexity usage) to choose regimes.

---

# 7. Nascent 33-Level Hierarchy

The system initializes with a set of **nascent abstract levels**:

- Only ground-state Modules (input/output handlers) exist initially.
- Higher levels are **identity-mapped virtual Modules**.
- They cost almost nothing until activated.
- NAS can instantiate real Modules at any level.
- 33 levels provide log₂-scale coverage for extremely large input structures without overcommitting architecture.

This is a flexible namespace for structural growth, not a rigid stack.

---

# 8. Interface Structure

## 8.1 Input Interface → Input Module  
The input-module receives raw samples, applies spline-standardized feature extraction, and begins upward routing.

## 8.2 Output Module → Output Interface  
The output-module synthesizes final embeddings and decodes them into environment outputs.

## 8.3 Input→Output Path  
Default execution:

input interface
→ input-module
→ hierarchical module graph
→ output-module
→ output interface


---

# 9. Overlap and Cloning (NAS Exploration)

NAS may clone a Module when its workload or representational pressure grows too high.

### 9.1 Cloning  
`M → M0, M1`

### 9.2 Overlap (before replacement)  
Original approach: a wrapper that averages outputs:

\[
\frac{1}{2} M0 + \frac{1}{2} M1
\]

Preserves downstream invariants.

### 9.3 Improved Approach: Learnable Scalars  
Each Module output has a learnable scalar α:

- Before cloning: α = 1
- After cloning:  
  - α₀ = r  
  - α₁ = 1 − r  
  with r = 0.5 for identity-preservation.

These scalars drift over training, breaking symmetry smoothly without bottlenecking.

### 9.4 Routing Upward to Abstract Module  
When needed:

- wrap clones,
- route merged output upward,
- allow abstract-level Modules to mediate divergence.

This prevents suppression of specialization.

### 9.5 Evolutionary Backtracking (Safety Mechanism)

MindsEye maintains a **phylogenetic tree** of architectural states. If a newly explored branch (after cloning or mutation) fails to surpass the parent's utility-to-complexity ratio within a set window, the system triggers a **Revert**.

* The system rolls back to the previous stable architectural checkpoint.
* The failed branch is logged as a "negative exploration," pruning that specific direction from immediate re-exploration.
* This prevents "optimization drift" where the model chases local minima into a dead end.

---

# 10. Memory System

Features:

- uncompressed recent memory,
- compressed older memory with exponential compression curvature,
- embedded traces for internal error checking,
- learnable compression/decompression Cores,
- complexity-driven space limits.

Memory acts as a continuous, lossy, self-structuring store.

---

# 11. Complexity Gradient

A regulatory mechanism analogous to special relativity:

- As module complexity (space/time) approaches its limit, the cost of extra expansion → ∞.
- Encourages pruning, compression, merging.
- Controls NAS growth.
- Ensures efficient, sparse structural development.

This is an architectural energy model guiding structural evolution.

---

# 12. Contact Structure and Multi-Permutation Support

Each Module maintains a **Contact** mapping:

contact[target_module_id] → { permutation structures }


Why?

- Different Modules require feature detectors in different canonical orders.
- A single learned permutation is insufficient.
- Modules need *module-specific permutations* for stable interaction.
- Contact-specific permutation sets allow:

  - contextual feature alignment,
  - multiple valid permutations per feature,
  - expandable communication channels,
  - robust multi-agent modularity.

This preserves spline standardization while enabling diverse remappings.

---

# 13. Symmetry Breaking in Cloned Modules

### 13.1 Overwork-Driven Divergence  
Cloning reduces workload per Module, altering gradient landscapes.  
Even without noise, clones diverge due to different optimization trajectories.

### 13.2 Perturbation-Accelerated Divergence  
Small perturbations—spline jitter, permutation noise, unshared dropout—ensure reliable divergence.

### 13.3 When Clones Do Not Diverge  
If no advantage emerges, complexity-gradient penalizes clone maintenance → prune redundant clone.

### 13.4 Summary  
The system supports safe, controlled bifurcation:

1. Clone  
2. Learnable scalar split  
3. Symmetry-breaking perturbation  
4. Divergence under pressure  
5. Prune or specialize

This forms the system’s evolutionary architecture.

---

# 14. Bootstrapping MindsEye

To avoid erratic early meta-learning, MindsEye can be bootstrapped with supervised mappings:

- For tasks with known optimal architectures:
  - convert optimal architectures to GraphModel form,
  - perturb them,
  - train MindsEye to map perturbed → optimal.

This teaches:

- structural correction,
- optimization regime selection,
- architectural priors,
- proper routing patterns.

This replaces blind meta-learning with grounded, stable initialization.

---

# 15. Internal Rhythm and Temporal Error

The system possesses an **Internal Clock** that generates a "sense of rhythm" for signal propagation. This introduces a second, internal source of differentiable error:

* **ETA Prediction:** When a Module issues a request, it predicts an **Expected Time of Arrival (ETA)** (in internal ticks) for the response.
* **Temporal Loss:** The difference between `Actual_Arrival_Time` and `Predicted_ETA` generates a **Temporal Error** gradient.
    * *Late Response:* Penalizes inefficiency and computational bloat.
    * *Infinite Loops:* Generate rapidly expanding temporal error, forcing the optimizer to "break" the loop logic to minimize pain.
    * *Rhythm:* Encourages the graph to settle into efficient, predictable signal cycles (isochronal processing).

This creates a **Dual-Objective Optimization**:

1.  **External:** Minimize prediction error (Accuracy).
2.  **Internal:** Minimize temporal error (Efficiency/Rhythm).

---

# 16. IP Coverage Summary

Protected elements include:

- spline-standardized features,
- multi-permutation Contact logic,
- attention-as-atomic-Core,
- multi-regime optimization logic,
- hierarchical virtual identity levels,
- learnable-scalar Overlap replacement,
- NAS cloning + divergence + pruning,
- memory compression curvature,
- MindsEye meta-optimization mechanics,
- the complete logistical request/response,
- and internal rhythm-temporal error system.

This document unifies all architectural components under a single protected conceptual framework.
