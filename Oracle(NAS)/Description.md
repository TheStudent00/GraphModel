# Graph Model Intellectual Property (IP) Description

This document captures the conceptual, architectural, and theoretical foundations of the Graph Model Oracle, defining the novel components, structural primitives, and meta-learning dynamics protected under the OTU Green License.

The Graph Model is a general-purpose, modular, hierarchical learning system built from space-filling curve linearization, spline-standardized features, attention-based routing primitives, and fractal scale-equivariant dynamics. It is designed for sparse, flexible, expandable intelligence capable of processing any metric or relational data structure.

---

# 1. Purpose and Scope

This description asserts IP protection over:

- the **Universal Linearization** process (Metric Z-Order vs. Relational Spectral),
- the **Sub-Feature Factorization** into Spline (Physics), Permutation (Geometry), and Noise (Entropy),
- the **Fractal Scale Equivariance** mechanism (Renormalization flow),
- the Core/Module/MindsEye structural hierarchy (The Trinity Architecture),
- the Bicameral (Hemispheric) Autopoietic Architecture,
- attention-as-atomic-primitive routing embedded into modular computation,
- the 33-level nascent abstraction hierarchy,
- dynamic architecture search (NAS) with cloning, merging, and complexity gradients,
- overlap management and symmetry-breaking mechanisms,
- internal memory compression/expansion logic,
- module-to-module logistics (Sender-Time/Receiver-Space economy),
- learnable permutation structures per contact (Dual-Axis),
- meta-learning strategies for bootstrapping MindsEye,
- and the general organizational logic of the Graph Model Oracle.

General ML concepts are not claimed—only the novel structural organization and interplay.

---

# 2. High-Level Summary

The Graph Model is a modular learning substrate composed of interacting Modules, each containing specialized Cores responsible for state, context, service, and contact behaviors. It uses:

- **Space-Filling Curve (SFC) Linearization** for universal input handling,
- **Fractal Permutation Heads** for scale-equivariant upsampling,
- atomic Q/K/V attention within each Core,
- multi-level abstraction hierarchy,
- dynamic module cloning and specialization,
- multi-regime optimization,
- and meta-learning governing self-modification.

The system aims to unify:

- sparse modularity,
- feature universality (Resolution/Dimension Invariance),
- evolutionary architectural dynamics,
- differentiable learning,
- and agency-like request/response structures.

---

# 3. Core Architectural Components

## 3.1 Core (The Universal Operator)

A Core is the atomic computational operator. To ensure total universality, **Cores are agnostic to raw input dimension**. They rely on Universal Linearization to project variable-dimensional inputs into a fixed embedding space via "Worm" segments.

Components:
- **Factorized Feature Architecture:** Decoupling magnitude distributions (Splines), topological orientations (Fractal Permutations), and residual entropy (Noise).
- **Parallel Q/K/V feature layers** for attention-based routing and transformation.
- **Differentiable Gaussian Aperture** for smooth global-to-local evolution.
- **Fractal Permutation Head:** Capabilities for stochastic upsampling and texture generation via learned jitter prediction.

## 3.2 Module (The Trinity Architecture)

To support robust agentic behavior, privacy, and temporal integration, every Module is composed of three specialized Cores forming a cognitive cycle:

1.  **Context Core (The Grunt/Sensor):**
    * *Role:* Perception and Filtering.
    * *Function:* Aggregates inputs from all Connectors. It processes sequences of embeddings (Time) to handle super-resolution inputs.
    * *Output:* A compressed `Context_Vector`.
2.  **State Core (The Operator/Nucleus):**
    * *Role:* Identity, Memory, and Integration.
    * *Function:* Takes the `Context_Vector` and updates the internal, recurrent `State_Vector`.
    * *Privacy:* The State is **strictly private**. It allows the module to maintain a "Self" independent of its public output.
3.  **Service Core (The Agent/API):**
    * *Role:* Action and Response.
    * *Function:* Takes the `State` and `Context` to generate specific outputs ("Services") for other modules. It acts as the "Diplomat," deciding what information to reveal.

## 3.3 MindsEye (The Executive Module)

MindsEye is a **High-Connectivity Module** located within the **Reflective Hemisphere** (see Section 18). It interacts with the graph via standard Connectors but controls the hyperparameters (Aperture $\sigma$, Connection $\lambda$) of other modules, embedding the "Governor" inside the "Government."

---

# 4. Universal Linearization and Feature Factorization

A key innovation is the treatment of all data (Images, Audio, Graphs) as **Linearized Streams** with associated **Topology Tokens**.

## 4.1 The Dual-Path Interface (Metric vs. Relational)
The Interface linearizes data based on its fundamental nature, ensuring $O(N)$ efficiency where possible:
1.  **Metric Path (The Fast Path):** For data with absolute coordinates (Images, Video, Voxel Grids). Uses **Z-Order (Morton) Curves** to linearize $N$-dimensions into 1D while preserving locality.
2.  **Relational Path (The Graph Path):** For data defined by connectivity (Social Nets, Molecules). Uses **Spectral Ordering** (Fiedler Vector) to linearize nodes based on graph topology.

## 4.2 The Topology Token
Every linearized stream is accompanied by a **Topology Token** (e.g., `shape=[3, 16, 16, 16]` or `adjacency_hash`). This allows the Receiver to interpret the 1D stream correctly (e.g., selecting 2D vs 3D permutation banks).

## 4.3 Feature Factorization ("The Worm")
The linearized stream is segmented into overlapping "Worms" (Macro-Segments, e.g., $L=64$). Each Worm is factorized:
* **Spline (Physics):** The sorted magnitude distribution (Scale Invariant).
* **Permutation (Geometry):** The rank order of the Z-curve indices (Fractal).
* **Noise (Entropy):** The high-frequency residual lost during binning.

---

# 5. Attention as an Atomic Routing Primitive

Attention ($Softmax(QK^T)V$) is the primitive operator inside every Core, used for routing, mixing, and specialization.

---

# 11. Complexity Gradient and Logistics Economy

### 11.1 The Sender-Time / Receiver-Space Economy
To prevent spam:
1.  **Sender Pays Time:** The cost of transport/latency and Linearization.
2.  **Receiver Pays Space:** The cost of maintaining the Connector structure.

---

# 12. Connector and Dual-Axis Permutation (Routing)

Connections are managed by a **Connector** object owned by the *Receiver*.

### 12.1 Dual-Axis Spectral Permutation
Connectors align Sender/Receiver languages by permuting **Rows** (Topology) and **Columns** (Semantics) using spectral functions. The Basis functions for the permutation are selected dynamically based on the **Topology Token**.

### 12.4 Temporal Aperture (Resolution as Sequence)
Connectors maintain a **Sliding Buffer** of the last $T$ messages.
* **Super-Resolution Handling:** If an input object exceeds the Native Resolution Limit ($L_{max}$), it is transmitted as a *Sequence* of vectors.
* The Context Core's **Aperture** acts as a learnable convolution/downsampling filter over this sequence, allowing the module to trade Temporal Resolution for Spatial Fidelity.

---

# 16. Fractal Scale Equivariance and Renormalization

The system explicitly models Scale Equivariance as a flow between **Geometry** and **Entropy**.

### 16.1 The Renormalization Flow
* **Downstream (Encoding):** High-Resolution Geometry is smoothed (Rank Smoothing/Binning). The lost detail becomes **Entropy (Noise)**.
* **Upstream (Decoding):** Low-Resolution Geometry is expanded. The **Fractal Permutation Head** uses the Entropy parameter to predict and generate the missing High-Frequency Jitter (Texture).

### 16.2 Native Resolution Limit
The system defines a hard upper bound ($L_{max}$) for topological fidelity per vector. Information denser than this limit is handled either by **Binning** (increasing Entropy) or **Streaming** (increasing Sequence Length).

---

# 18. Autopoiesis and Bicameral Topology

The graph is organized into **Hemisphere A (Active)** and **Hemisphere B (Reflective)**.
* **Active:** Real-time execution (Body).
* **Reflective:** Simulation and Architecture (Mind).
* **Hot-Swap:** Updates are simulated in B and propagated to A only upon verification.

---

# 19. IP Coverage Summary

Protected elements include:
- Universal Linearization (Z-Order/Spectral Dual Path),
- The Trinity Module Architecture (State/Context/Service),
- Factorized Features (Spline/Fractal Permutation/Noise),
- Fractal Scale Equivariance (Renormalization Flow),
- Dual-Axis Connectors with Temporal Aperture,
- Multi-Regime Optimization,
- Hierarchical Virtual Identity,
- Sender-Time/Receiver-Space Logistics,
- Differentiable Aperture & Fractal Heads,
- and the Autopoietic Bicameral Topology.

This document unifies all architectural components under a single protected conceptual framework.
