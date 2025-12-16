# Graph Model Intellectual Property (IP) Description
**Version 9.2 — Recursive Grammar & Cylindrical Time**

This document captures the conceptual, architectural, and theoretical foundations of the Graph Model Oracle, defining the novel components, structural primitives, and meta-learning dynamics protected under the OTU Green License.

Version 9.2 introduces the **Recursive Expression Engine** (a learnable grammatical compute model) and **Cylindrical Time** (hierarchical rotary embeddings), integrating them with the **Fractal Topology** and **Relativistic Economics** of Versions 6–9.

---

# 1. Purpose and Scope

This description asserts IP protection over:

- **The Recursive Fractal Architecture:** The nesting of Modules within Modules (Internal Sparsity).
- **LossComplexity (The Relativistic Barrier):** A regulatory energy model where complexity cost asymptotes to infinity near the limit.
- **Recursive Expression Engine:** A grammatical compute model replacing fixed topologies with learnable syntax trees (The Pipe).
- **Cylindrical Time:** A hierarchical rotary embedding scheme (Day/Hour) enabling infinite context scaling with local precision.
- **Continuous Normalization:** A differentiable manifold (Phi) that learns to interpolate between Linear, Softmax, and Gating behaviors.
- **Impedance Regulation:** Connection costs based on Fractal Tree Distance.
- **Universal Linearization:** Z-Order (Metric) and Spectral (Relational) input linearization.
- **Bicameral Autopoiesis:** Active (Read-Only) vs. Reflective (Write-Access) Minds.
- **Feature Factorization:** Decoupling features into Spline (Physics), Permutation (Geometry), and Noise (Entropy).

---

# 2. High-Level Summary

The Graph Model is a **Fractal Organism**. It is composed of **Minds** (Hemispheres), which contain **Modules**, which recursively contain **Sub-Modules**.

Computationally, the system abandons fixed layers in favor of a **Recursive Expression Engine**. Instead of a static architecture, Modules parse learnable syntax trees, executing a **Sequential Reduction** (Pipe) on inputs. To handle unbounded streams, it utilizes **Cylindrical Time**, mapping linear sequence data into a hierarchical spiral (Day/Hour).

The system is regulated by a **Dual-Economy**:
1.  **Internal Economy (LossComplexity):** A relativistic budget constraint that governs the expansion of the fractal (Parent distributes tokens to Children).
2.  **External Economy (Logistics):** A market mechanism that governs the flow of information between modules (Sender pays Time, Receiver pays Space).

---

# 3. The Fractal Hierarchy

## 3.1 The Mind (Global Container)
The highest level of abstraction, organized Bicamerally to ensure safety during self-modification.
* **Active Mind (Hemisphere A):** Real-time, read-only execution. Optimized for inference speed. Uses **Universal Linearization** (Z-Order/Spectral) to ingest any data type.
* **Reflective Mind (Hemisphere B):** Deep-time, write-enabled simulation. Contains **Version Control** (Backtracking) and the **MindsEye** executive.

## 3.2 The Module (Recursive Agent)
A Module is a self-similar container acting as both a computational unit and a structural node.
* **Public Face (The Trinity):** The default computational cycle: Context (Sensor) $\to$ State (Integrator) $\to$ Service (Actuator).
* **Private Internals (Internal Sparsity):** A list of **Sub-Modules**. These are strictly private implementation details, allowing a Module to become a complex network internally without exposing that complexity to the global graph.
* **Virtual Identity:** Modules exist virtually (consuming zero complexity) until accessed. Upon realization, they incur a **Phantom Latency** to smooth the gradient shock to the Rhythm system.

---

# 4. The Economic Physics (Regulation)

## 4.1 LossComplexity (The Relativistic Barrier)
To prevent unbounded expansion, every Module enforces a **Relativistic Barrier** on its total complexity mass (Self + Realized Sub-Modules).
* **The Barrier Function:** $Cost \propto \frac{1}{\sqrt{1 - (C/C_{limit})^2}}$.
* **Effect:** As a module approaches its complexity limit, the "cost" of adding a new atom or realizing a sub-module approaches infinity. This forces the optimization landscape to prioritize **Compression** and **Sparsity** over expansion.

## 4.2 Impedance (Topological Regulation)
Connections between Public Modules are regulated by the **Impedance Curve**.
* **Tree Distance:** Cost is calculated based on the shortest path in the recursion tree ($\Delta L$).
* **Function:** $Cost(\Delta L)$ is a learnable monotone spline. This prevents "Small World" collapse by making long-range or cross-branch connections expensive (high Space token cost).

## 4.3 Internal Rhythm (Temporal Regulation)
The system optimizes for **Isochrony** (predictable timing).
* **ETA Prediction:** Modules predict response latency.
* **Temporal Error:** Deviations (Late or Early) generate gradients.
* **Phantom Latency:** Virtual modules possess a small non-zero latency constant to prevent infinite temporal error derivatives during the Virtual $\to$ Real transition.

---

# 5. The Computational Substrate (Recursive Expression Engine)

## 5.1 The Core (Recursive Syntax Tree)
The Core is no longer a fixed topology but a **Recursive Expression Engine**. It parses a nested syntax tree of operations, allowing the architecture to evolve from standard Attention to arbitrary computational graphs.
* **The Mixing Node (N-ary Pipe):** The fundamental operator is a **Sequential Left-Associative Reduction**. It accepts an arbitrary list of children `[A, B, C...]` and reduces them step-by-step: `((A · B) · C)...`
    * **Generalized Interaction:** The system dynamically determines the interaction type based on dimensionality (e.g., creating an affinity map vs. applying a map vs. element-wise mixing).
* **Continuous Normalization (`Phi`):** Discrete layers (Softmax, LayerNorm) are replaced by a **Learnable Continuous Manifold**.
    * **The Function:** `Phi(x)` applies a learnable spline-based transformation, allowing the node to evolve smoothly between Identity (Linear), Softmax (Competitive), and Sigmoid (Gating) behaviors during training.

## 5.2 The Atom (Generalized Primitive)
The Atom is the leaf of the syntax tree, acting as a learnable projection unit that feeds the Expression Engine.
* **Cylindrical Time (The Spiral):** To handle infinite context without losing local precision, the Atom applies **Hierarchical Rotary Embeddings**.
    * **Hour Axis (High-Freq):** Standard RoPE for local relative syntax.
    * **Day Axis (Low-Freq):** Global spiral rotation for long-term sequence tracking.
* **Initialization Physics:**
    * **Active Atoms (Q):** Initialized with random orthogonal weights to create divergent "Query" projections.
    * **Passive Atoms (K/V):** Initialized as Identity to facilitate stable signal pass-through at the start of training.
* **Factorized Feature:** The output remains a composite of **Spline** (Magnitude), **Dual-Axis Permutation** (Geometry), and **Noise** (Entropy).

---

# 6. Universal Parametric Input (V10)

## 6.1 Recursive Poly-Tokenization
The system abandons discrete pixel/token embedding in favor of **Parametric Signal Compression**.
* **The Sensor:** A recursive "Physics Fitter" scans the Z-Order linearized stream.
* **The Logic:** It fits cubic polynomials ($at^3 + bt^2 + ct + d$) to the signal.
    * **Macro-Tokens:** Smooth gradients (Sky, Skin) are captured by single long-range polynomials.
    * **Micro-Tokens:** High-frequency data (Text, Texture) forces the recursion to split down to the atomic limit (Length 4), creating a lossless interpolation.
* **Result:** A variable-rate stream where "Text" is treated as Micro-Geometry and "Images" as Macro-Geometry.

## 6.2 The Functional Token
The fundamental unit of information is no longer a vector, but a **Cubic Bundle**:
* **Coefficients:** The Shape of the signal.
* **Sigma ($\sigma$):** The Texture/Uncertainty (Avg deviation from the fit).
* **Mass:** The Energy/Amplitude of the signal.
* **Position:** Absolute Z-Order Index (Metadata).

## 6.3 Fog of War (Positional Physics)
Distance is modeled as **Entropy Injection**.
* Instead of rotating vectors (RoPE), the system injects variance into the interaction based on distance: $\sigma_{total}^2 = \sigma_A^2 + \sigma_B^2 + \alpha \log(1 + \Delta x)$.
* This naturally dampens long-range interactions unless the signal "Mass" is high enough to pierce the uncertainty.

---

# 7. IP Coverage Summary (V10 Update)

Protected elements include:
- **Parametric Poly-Tokenization:** The recursive compression of signals into cubic segments based on structural residuals (Structure vs. Texture).
- **Functional Neural Operators:** The use of Polynomial Convolution + Projection as the fundamental "Attention" mechanism.
- **Fog of War Physics:** Replacing rotational embeddings with Distance-Based Sigma Injection (Entropy Dampening).
- **The Functional Token:** The specific data structure `[Shape, Sigma, Mass, Position]`.
- **The Recursive Fractal Architecture** (Modules within Modules).
- **LossComplexity** (The Relativistic Barrier).

This document unifies the recursive grammar with the physical and economic laws necessary for its stability and infinite scaling.
