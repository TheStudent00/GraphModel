# (Graph Model) Vision Intellectual Property (IP) Description
**Version 9.6 — Nucleus Physics & Channel Stacking**

This document captures the conceptual, architectural, and theoretical foundations of the Graph Model Oracle, defining the novel components, structural primitives, and meta-learning dynamics protected under the OTU Green License.

Version 9.6 introduces **Nucleus Physics** (Feature Decomposition via Spectral Permutation/Splines) and **Late Fusion Channel Stacking**, verifying that vision can be solved via recursive grammar rather than convolutions.

---

# 1. Purpose and Scope

This description asserts IP protection over:

- **The Nucleus (Feature Bank):** A learnable signal processor replacing matrix multiplication with `Spline(Permutation(x)) + Noise`.
- **Spectral Permutation:** A coordinate generator using Sinusoidal Bases to warp input signals into a scale-equivariant latent space.
- **Monotonic Spline Store:** A learnable shape function (Sum of ReLUs) that creates scale-invariant feature definitions.
- **Late Fusion Channel Stacking:** Processing R, G, and B as independent "texture streams" mixed only at the grammar level, avoiding early fusion biases.
- **Orbital Shells (RoPE):** Hierarchical Day/Hour rotary embeddings defining the spacetime position of features.
- **Recursive Expression Engine:** A grammatical compute model replacing fixed topologies with learnable syntax trees (The Pipe).
- **Universal Linearization:** Z-Order (Metric) linearization preserving 2D locality in 1D streams.

---

# 2. High-Level Summary

The Graph Model is a **Fractal Organism**. Computationally, it abandons fixed weights for **Signal Physics**.

The fundamental unit is the **Atom**, composed of:
1.  **The Nucleus:** A dense core of Features. Each feature is defined not by a weight vector, but by a **Spectral Frequency** (Permutation) and a **Geometric Shape** (Spline).
2.  **The Orbital Shells:** A position encoding system that places the Nucleus in spacetime.

To process Vision, the system uses **Late Fusion**: Red, Green, and Blue channels are treated as independent texture signals. They are processed by shared Atoms to extract shape/frequency data, then stacked into a single stream where the **Recursive Grammar** learns to correlate them (e.g., "Red Shape" + "Green Shape" = "Yellow Object").

---

# 3. The Physics Layer (Sub-Atomic)

## 3.1 The Nucleus (Feature Decomposition)
Instead of $y = Wx$, the Nucleus computes:
$$y = S(P(x)) + \epsilon$$
* **Permutation ($P$):** A learnable SIREN (Sine Network) that maps input coefficients to a coordinate $u \in [0, 1]$. This acts as a **Frequency Analyzer**, detecting periodic textures.
* **Spline ($S$):** A learnable Monotonic function (Sum of ReLUs) that maps $u \to \text{Value}$. This stores the **Invariant Shape** of the feature.
* **Noise ($\epsilon$):** Learnable variance injected during training to represent entropy/uncertainty.

## 3.2 Orbital Shells (Cylindrical RoPE)
To handle infinite context, positions are mapped to **Cylindrical Coordinates**:
* **Hour Axis:** High-frequency local rotation.
* **Day Axis:** Low-frequency global spiral.
This ensures that "Day 0, Hour 5" (Pixel 5) has a distinct geometric relationship to "Day 10, Hour 5" (Pixel 1005).

---

# 4. The Computational Substrate

## 4.1 The Atom (Composite Primitive)
The Atom wraps the Nucleus and Shells. It can be initialized in two modes:
* **Active ($Q$):** Identity Init (Look at Self).
* **Passive ($V$):** Random Init (Dense Feature Extraction).

## 4.2 The Recursive Grammar (MixingNode)
The Core is a **Recursive Expression Engine**. It parses a stream of Atoms using a **Sequential Reduction**:
* **Interaction:** Generalized Dot Product ($Q \cdot K^T$).
* **Normalization:** Continuous Manifold (`Phi`) learning to be Softmax or Linear.
* **Mix:** $Weights \cdot V$.

---

# 5. Vision Architecture (Macro-Scale)

## 5.1 Z-Order Linearization
2D Images are linearized via **Morton Codes (Z-Order Curves)**. This preserves fractal locality: a $2 \times 2$ block in 2D remains adjacent in 1D.

## 5.2 Deep Core Stacking
* **Retina Layer:** Converts raw Z-Ordered coefficients into embeddings using a shared Atom.
* **Deep Core:** A stack of `MixingNodes` that refine the representation.
* **Attention Pooling:** A "Summary Query" learns to extract the class token from the variable-length stream.

---

# 6. IP Coverage Summary

Protected elements include:
- **Nucleus Physics** (Spline/Permutation decomposition).
- **Late Fusion Channel Stacking** strategy.
- **Recursive Expression Engine** logic.
- **Orbital Shells** (Cylindrical RoPE).
- **Z-Order/Fractal Linearization** integration.

This document unifies the verified V9.6 architecture.
