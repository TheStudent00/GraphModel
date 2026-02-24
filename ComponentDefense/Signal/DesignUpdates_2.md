```yaml
title: "Intellectual Property Specification: Fractal-Aperture Spectral Transformer (FAST) & Hybrid Routing"
project: "GraphModel"
author: "Student"
date: "2026-02-23"
status: "Architecture Specification / Theoretical Core - V2"
tags: 
  - machine intelligence
  - spectral algebra
  - complex neural physics
  - hybrid compression
  - symmetric padding
  - abstract dimension reduction
```

# 1. Executive Summary

This document specifies the architecture and fundamental intellectual property (IP) of the **Fractal-Aperture Spectral Transformer (FAST)**, the computational core of the GraphModel framework. FAST introduces a paradigm shift in generalized machine intelligence by establishing the Attention Block as the universal axiom for abstract dimension reduction. It leverages complex-valued neural physics, spectral algebra, and a hybrid multi-scale routing pipeline to process unbounded sequences dynamically, circumventing both floating-point coordinate collapse and catastrophic $O(N^2)$ scaling limits.

# 2. The Universal Axiom: Abstract Dimension Reduction

Standard deep learning relies on static routers (e.g., CNNs or MLPs) that perform arbitrary geometric dimension reduction blindly based on physical proximity. GraphModel establishes the **Attention Block** as its atomic primitive because it functions as a **Dynamic Semantic Router**. 

* **Mechanism:** Rather than mechanically truncating data, attention performs a differentiable search for the shared semantic essence of physical spaces. 
* **Effect:** It reduces *information entropy* rather than just spatial dimensions. It allows the network to collapse redundant signals (e.g., a flat sky) while preserving high-frequency complexities (e.g., a bird), executing true "Abstract Dimension Reduction" at every scale. 

# 3. The `Atom` Pipeline (Hierarchical Information Flow)

The structural hierarchy is managed by the `Atom` pipeline, which isolates mathematical routing into three distinct phases to ensure perfect translation invariance locally and $O(N \log N)$ scalability globally.

## 3.1 BottomUpCompressor (Hybrid Local-Global Reduction)
The ascending path constructs the hierarchical context by compressing the sequence. To balance local boundary preservation with global computational limits, it utilizes a dual-mode hybrid architecture.

### 3.1.1 Local Phase: Fixed-Depth Overlapping Compression
* **Mechanism:** For a fixed depth $d$, sections of tokens (e.g., size 32) are merged with a stride of 1 (overlapping). 
* **Purpose:** This overlap guarantees that high-frequency details falling on section boundaries (e.g., the pupil of an eye) are processed holistically in at least one attention block, eliminating grid artifacts and blind spots. 
* **Complexity:** Scales at exactly $O(d \cdot N)$. 

### 3.1.2 Transition Phase: Symmetric Null Padding
* **Mechanism:** Upon reaching depth $d$, the sequence length is unlikely to be a perfect power of two. The network symmetrically appends `NULL` sections (tokens initialized with zero mass and zero density) to both ends of the sequence until reaching $2^x$.
* **Purpose:** Symmetric padding maintains the absolute balance of the spatial Centroids, ensuring the "Center of Mass" of the sequence is not artificially skewed, accurately modeling the boundary edges of the data manifold.

### 3.1.3 Global Phase: Binary Tree Reduction
* **Mechanism:** The symmetrically padded sequence switches to non-overlapping compression (stride 2), halving the sequence length at each subsequent level.
* **Complexity:** Scales at $O(N \log N)$. Density-Weighted Attention mathematically ignores the `NULL` tokens during this phase because they carry no physical mass. 

## 3.2 TopDownInfuser (Descending Context Injection)
If processing halts at the global root, local resolution is lost to abstraction. 
* **Mechanism:** A descending cross-attention pass where high-resolution Level $n$ sections (Queries) attend to their overlapping parent summaries at Level $n+1$ (Keys/Values). 
* **Effect:** Injects global spatial and directional awareness back into the microscopic geometric details.

## 3.3 SparseFractalAttention (The Light Cone Aperture)
The final processing phase for any localized token stack relies on a dynamically constructed "Stack of Stacks," geometrically forming a Light Cone of attention.
* **Mechanism:** For any base Level 0 section, the aperture constructs a bounded sub-stack containing its local neighbors, its Level 1 parents, and its expanding Level 2+ ancestry. 
* **Implicit Fog:** The Light Cone explicitly removes the need for fabricated mathematical distance penalties. The "distance" to far-away tokens is physically enforced by the polynomial smoothness of the upper-level spectral summaries it queries.

# 4. Ontology & Complex Neural Physics

The architecture redefines tokens and weights as physical objects interacting in a complex-valued field ($M = m_{real} + i \cdot m_{virt}$ and $Z = z_{real} + i \cdot z_{virt}$).

* **Real Axis (Physical Geometry):** $m_{real}$ represents data inertia. $z_{real}$ tracks the Local Spatial Phase within the 32-token sequence block, keeping coordinates safely bounded within $[0, 1]$ to prevent precision loss.
* **Virtual Axis (Semantic Depth):** $m_{virt}$ represents functional weight. $z_{virt}$ tracks abstraction level.
* **Hypernetworks (Metamorphic Projectors):** Weights (Projectors) are instantiated as zero-real-mass `SpectralTensor` objects. Data modifies Data (Attention), and Projectors modify Projectors (Meta-Learning), turning optimization into gravitational settling. The Center of Mass interaction denominator ($M_A + M_B$) is protected from $0/0$ division errors because $|M| = \sqrt{m_{real}^2 + m_{virt}^2} \neq 0$.

# 5. Spectral Tokenization & Algebra

## 5.1 Atomic Unit: The Spectral Token
* **Structure:** A single segment, Order 3 Chebyshev Spline.
* **Heteroscedastic Uncertainty:** Modeled via a Dual-Curve Interval ($[L, U]$) to capture skew.
* **Disentangled Normalization:** Signal Mean ($\mu$) and Uncertainty Width ($\sigma$) are normalized independently to prevent high-uncertainty regions from mathematically crushing precise signals.

## 5.2 Spectral Algebra Engine
* **Sigmoid Gate Projectors:** Uncertainty activation is executed via `SpectralAlgebra.composition`. A learnable "Gate-Projector" curve maps the uncertainty interval to a transmission coefficient, structurally differentiating noise-tolerant textures from noise-intolerant structures.
