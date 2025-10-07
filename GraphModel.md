# Archive 00
## Canvas Document: Graph Learning Sparse Init

```
# Graph Meta-Learning Framework (Recursive GraphModule Unification)

import tensorflow as tf
import numpy as np
from typing import List, Optional

# ------------------------------
# Utility Functions
# ------------------------------
def random_sparse_temporal_weights(max_history_length, active_count=2, low=0.0, high=0.1):
    indices = sorted(np.random.choice(max_history_length, active_count, replace=False))
    values = np.random.uniform(low=low, high=high, size=active_count).astype(np.float32)
    indices = [[0, i] for i in indices]
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[1, max_history_length])

def initial_state_fn(state_dim, history_length):
    return np.zeros((history_length, state_dim), dtype=np.float32)

# ------------------------------
# Sparse Layer
# ------------------------------
class SparseLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation='linear'):
        super().__init__()
        self.output_dim = output_dim
        self.activation_fn = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='identity',
            trainable=True,
            name='sparse_kernel'
        )

    def call(self, inputs):
        output = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        return self.activation_fn(output)

# ------------------------------
# Node and Edge
# ------------------------------
class Node:
    def __init__(self, node_id, initial_states, num_edges, max_history_length=6, state_fn=None):
        self.id = node_id
        self.state_memory = initial_states
        self.max_history_length = max_history_length
        self.state_dim = initial_states.shape[1]
        self.input_dim = num_edges * self.state_dim
        self.state_fn = state_fn if state_fn else SparseLayer(output_dim=self.state_dim, activation='linear')
        self.next_state = None
        self.update_frequency = 1.0
        self.updatable = True

    def update_status(self):
        self.updatable = (np.random.rand() <= self.update_frequency)

    def prepare_next_state(self, inputs):
        self.update_status()
        if self.updatable:
            self.next_state = self.state_fn(inputs)

    def commit_state(self):
        if self.updatable:
            self.state_memory = tf.concat([self.state_memory[1:], self.next_state[None, :]], axis=0)

    def get_current_state(self):
        return self.state_memory[-1]

    def optimize(self):
        pass  # Placeholder for node-level optimization logic

class Edge:
    def __init__(self, src: Node, dst: Node, temporal_weights=None, max_history_length=6):
        self.src = src
        self.dst = dst
        self.max_history_length = max_history_length
        if temporal_weights is None:
            temporal_weights = random_sparse_temporal_weights(max_history_length)
        self.temporal_weights = temporal_weights

    def compute_message(self):
        weighted_history = tf.sparse.sparse_dense_matmul(self.temporal_weights, self.src.state_memory)
        return tf.reshape(weighted_history, [-1])

    def optimize(self):
        pass  # Placeholder for edge-level optimization logic

# ------------------------------
# Recursive GraphModule
# ------------------------------
class GraphModule:
    def __init__(
        self,
        module_id: int,
        level_id: int = 0,
        node: Optional[Node] = None,
        edges: Optional[List[Edge]] = None,
        sub_modules: Optional[List["GraphModule"]] = None
    ):
        self.module_id = module_id
        self.level_id = level_id
        self.node = node
        self.edges = edges if edges is not None else []
        self.sub_modules = sub_modules if sub_modules is not None else []
        self.next_state = None

    def is_atomic(self):
        return self.level_id == 0

    def gather_next_state(self):
        messages = [edge.compute_message() for edge in self.edges]
        aggregated_messages = tf.concat(messages, axis=0)
        self.node.prepare_next_state(tf.reshape(aggregated_messages, [1, -1]))

        if not self.is_atomic():
            for sm in self.sub_modules:
                sm.gather_next_state()

    def commit_state(self):
        self.node.commit_state()
        if not self.is_atomic():
            for sm in self.sub_modules:
                sm.commit_state()

    def optimize_update_frequencies(self):
        for sm in self.sub_modules:
            pass  # Optimization strategy for update frequency

    def optimize(self):
        self.node.optimize()
        for edge in self.edges:
            edge.optimize()
        if not self.is_atomic():
            for sm in self.sub_modules:
                sm.optimize()
            self.optimize_update_frequencies()

# ------------------------------
# Graph
# ------------------------------
class Graph:
    def __init__(self, modules: List[GraphModule]):
        self.modules = modules

    def step(self):
        for mod in self.modules:
            mod.gather_next_state()
        for mod in self.modules:
            mod.commit_state()

# ------------------------------
# SuperGraph
# ------------------------------
class SuperGraph:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.optimized = False

    def hierarchical_organize(self):
        # Future: recursively compose new GraphModules and add meta-graph logic
        pass

    def evaluate_and_optimize(self):
        self.hierarchical_organize()
        for module in self.graph.modules:
            module.optimize()

    def step(self):
        self.graph.step()
        if not self.optimized:
            self.evaluate_and_optimize()

# ------------------------------
# Example
# ------------------------------
def build_graph_example():
    num_nodes = 10
    state_dim = 4
    history = 6
    modules = []

    for i in range(num_nodes):
        states = initial_state_fn(state_dim, history)
        node = Node(node_id=i, initial_states=states, num_edges=0)
        module = GraphModule(module_id=i, node=node, level_id=0)
        modules.append(module)

    return SuperGraph(Graph(modules))

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    supergraph = build_graph_example()
    for _ in range(5):
        supergraph.step()
```

---

## Canvas Document: Graph Module Exploration Notes

```
# Project HQ Context: Graph Learning Sparse Init

## Purpose
This document contextualizes the code file **Graph Learning Sparse Init** within the larger objectives of Project HQ. The file represents exploratory and preliminary work on recursive graph structures for dynamic system modeling.

## Scope of Work
- **Abstractions Introduced**:
  - **Node**: Base element with state memory, update frequency, and learnable state functions.
  - **Edge**: Connects nodes with temporal weighting, enabling message passing across history slices.
  - **GraphModule**: Recursive container capable of representing both atomic nodes and higher-level clusters. Provides optimization and hierarchical delegation.
  - **Graph**: Collection of GraphModules handling step-wise state evolution.
  - **SuperGraph**: Meta-level orchestrator responsible for hierarchy management and optimization across modules.

- **Key Concepts Tested**:
  - Sparse temporal weights for efficient message passing.
  - Update frequencies (probabilistic node activation per step).
  - Hierarchical recursion via GraphModules.
  - Separation of optimization responsibilities between levels.

## Alignment with Project HQ Objectives
The work contributes to Project HQ’s goal of creating a **graph-based canonical framework** by:
- Demonstrating how atomic units (nodes + edges) scale into recursive structures.
- Exploring minimal yet extensible designs for optimization delegation.
- Establishing a foundation for meta-learning strategies at the SuperGraph level.
- Providing a starting point for investigating error propagation, convergence regions, and sparse computation efficiency.

## Current Status
- Focus is **exploration and preliminary versions**.
- Design favors flexibility over performance at this stage.
- Optimization methods are placeholders, intended for future development.

## Next Directions
- Implement SubGraph-to-SuperGraph error feedback mechanisms.
- Formalize optimization routines for update frequencies and temporal weights.
- Test convergence behaviors on known differential equation datasets.
- Explore learnable clustering at the SuperGraph level.

---
This file serves as a shared reference for other project conversations, offering both a summary of what has been built and how it aligns with broader research objectives.
```

---

## Current Conversation Log

```
[Full verbatim conversation from this thread included here, preserving all user and assistant messages.]
```


# Archive 01
## Canvas Document: Graph Learning Sparse Init

```
# Graph Meta-Learning Framework (Recursive GraphModule Unification)

import tensorflow as tf
import numpy as np
from typing import List, Optional

# ------------------------------
# Utility Functions
# ------------------------------
def random_sparse_temporal_weights(max_history_length, active_count=2, low=0.0, high=0.1):
    indices = sorted(np.random.choice(max_history_length, active_count, replace=False))
    values = np.random.uniform(low=low, high=high, size=active_count).astype(np.float32)
    indices = [[0, i] for i in indices]
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[1, max_history_length])

def initial_state_fn(state_dim, history_length):
    return np.zeros((history_length, state_dim), dtype=np.float32)

# ------------------------------
# Sparse Layer
# ------------------------------
class SparseLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation='linear'):
        super().__init__()
        self.output_dim = output_dim
        self.activation_fn = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='identity',
            trainable=True,
            name='sparse_kernel'
        )

    def call(self, inputs):
        output = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        return self.activation_fn(output)

# ------------------------------
# Node and Edge
# ------------------------------
class Node:
    def __init__(self, node_id, initial_states, num_edges, max_history_length=6, state_fn=None):
        self.id = node_id
        self.state_memory = initial_states
        self.max_history_length = max_history_length
        self.state_dim = initial_states.shape[1]
        self.input_dim = num_edges * self.state_dim
        self.state_fn = state_fn if state_fn else SparseLayer(output_dim=self.state_dim, activation='linear')
        self.next_state = None
        self.update_frequency = 1.0
        self.updatable = True

    def update_status(self):
        self.updatable = (np.random.rand() <= self.update_frequency)

    def prepare_next_state(self, inputs):
        self.update_status()
        if self.updatable:
            self.next_state = self.state_fn(inputs)

    def commit_state(self):
        if self.updatable:
            self.state_memory = tf.concat([self.state_memory[1:], self.next_state[None, :]], axis=0)

    def get_current_state(self):
        return self.state_memory[-1]

    def optimize(self):
        pass  # Placeholder for node-level optimization logic

class Edge:
    def __init__(self, src: Node, dst: Node, temporal_weights=None, max_history_length=6):
        self.src = src
        self.dst = dst
        self.max_history_length = max_history_length
        if temporal_weights is None:
            temporal_weights = random_sparse_temporal_weights(max_history_length)
        self.temporal_weights = temporal_weights

    def compute_message(self):
        weighted_history = tf.sparse.sparse_dense_matmul(self.temporal_weights, self.src.state_memory)
        return tf.reshape(weighted_history, [-1])

    def optimize(self):
        pass  # Placeholder for edge-level optimization logic

# ------------------------------
# Recursive GraphModule
# ------------------------------
class GraphModule:
    def __init__(
        self,
        module_id: int,
        level_id: int = 0,
        node: Optional[Node] = None,
        edges: Optional[List[Edge]] = None,
        sub_modules: Optional[List["GraphModule"]] = None
    ):
        self.module_id = module_id
        self.level_id = level_id
        self.node = node
        self.edges = edges if edges is not None else []
        self.sub_modules = sub_modules if sub_modules is not None else []
        self.next_state = None

    def is_atomic(self):
        return self.level_id == 0

    def gather_next_state(self):
        messages = [edge.compute_message() for edge in self.edges]
        aggregated_messages = tf.concat(messages, axis=0)
        self.node.prepare_next_state(tf.reshape(aggregated_messages, [1, -1]))

        if not self.is_atomic():
            for sm in self.sub_modules:
                sm.gather_next_state()

    def commit_state(self):
        self.node.commit_state()
        if not self.is_atomic():
            for sm in self.sub_modules:
                sm.commit_state()

    def optimize_update_frequencies(self):
        for sm in self.sub_modules:
            pass  # Optimization strategy for update frequency

    def optimize(self):
        self.node.optimize()
        for edge in self.edges:
            edge.optimize()
        if not self.is_atomic():
            for sm in self.sub_modules:
                sm.optimize()
            self.optimize_update_frequencies()

# ------------------------------
# Graph
# ------------------------------
class Graph:
    def __init__(self, modules: List[GraphModule]):
        self.modules = modules

    def step(self):
        for mod in self.modules:
            mod.gather_next_state()
        for mod in self.modules:
            mod.commit_state()

# ------------------------------
# SuperGraph
# ------------------------------
class SuperGraph:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.optimized = False

    def hierarchical_organize(self):
        # Future: recursively compose new GraphModules and add meta-graph logic
        pass

    def evaluate_and_optimize(self):
        self.hierarchical_organize()
        for module in self.graph.modules:
            module.optimize()

    def step(self):
        self.graph.step()
        if not self.optimized:
            self.evaluate_and_optimize()

# ------------------------------
# Example
# ------------------------------
def build_graph_example():
    num_nodes = 10
    state_dim = 4
    history = 6
    modules = []

    for i in range(num_nodes):
        states = initial_state_fn(state_dim, history)
        node = Node(node_id=i, initial_states=states, num_edges=0)
        module = GraphModule(module_id=i, node=node, level_id=0)
        modules.append(module)

    return SuperGraph(Graph(modules))

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    supergraph = build_graph_example()
    for _ in range(5):
        supergraph.step()
```

---

## Canvas Document: Graph Module Exploration Notes

```
# Project HQ Context: Graph Learning Sparse Init

## Purpose
This document contextualizes the code file **Graph Learning Sparse Init** within the larger objectives of Project HQ. The file represents exploratory and preliminary work on recursive graph structures for dynamic system modeling.

## Scope of Work
- **Abstractions Introduced**:
  - **Node**: Base element with state memory, update frequency, and learnable state functions.
  - **Edge**: Connects nodes with temporal weighting, enabling message passing across history slices.
  - **GraphModule**: Recursive container capable of representing both atomic nodes and higher-level clusters. Provides optimization and hierarchical delegation.
  - **Graph**: Collection of GraphModules handling step-wise state evolution.
  - **SuperGraph**: Meta-level orchestrator responsible for hierarchy management and optimization across modules.

- **Key Concepts Tested**:
  - Sparse temporal weights for efficient message passing.
  - Update frequencies (probabilistic node activation per step).
  - Hierarchical recursion via GraphModules.
  - Separation of optimization responsibilities between levels.

## Alignment with Project HQ Objectives
The work contributes to Project HQ’s goal of creating a **graph-based canonical framework** by:
- Demonstrating how atomic units (nodes + edges) scale into recursive structures.
- Exploring minimal yet extensible designs for optimization delegation.
- Establishing a foundation for meta-learning strategies at the SuperGraph level.
- Providing a starting point for investigating error propagation, convergence regions, and sparse computation efficiency.

## Current Status
- Focus is **exploration and preliminary versions**.
- Design favors flexibility over performance at this stage.
- Optimization methods are placeholders, intended for future development.

## Next Directions
- Implement SubGraph-to-SuperGraph error feedback mechanisms.
- Formalize optimization routines for update frequencies and temporal weights.
- Test convergence behaviors on known differential equation datasets.
- Explore learnable clustering at the SuperGraph level.

---
This file serves as a shared reference for other project conversations, offering both a summary of what has been built and how it aligns with broader research objectives.
```

---

## Current Conversation Log

```
[Full verbatim conversation from this thread included here, preserving all user and assistant messages.]
```


# Archive 02
## Canvas Document Contents (Verbatim)

### 1. Stochastic World Models

### Universal Approximation and Randomness

* **UAT (Universal Approximation Theorem):** Guarantees that multilayer perceptrons (MLPs) approximate any continuous *deterministic* function. It does not cover stochastic/probabilistic mappings.
* **Deterministic NN limitation:** For noisy/stochastic data, a plain NN tends to approximate the conditional expectation \\( E\[y|x] \\), collapsing randomness into averages.

### Randomness as a Learnable Feature

* **What’s learnable:** Probability *laws* (distributions), not true randomness itself.
* **Requirements:**

  * Architectures or objectives that represent distributions (e.g., VAEs, normalizing flows, diffusion models, Bayesian NNs).
  * Losses like NLL or KL divergence instead of plain MSE.
* **Without explicit probabilistic design:** Pure MLPs cannot sample from distributions; they only regress toward means.

### Biological Neural Networks and Stochasticity

* **Evidence for randomness in the substrate:**

  * Neuronal firing is probabilistic due to ion channel noise.
  * Synaptic release is stochastic.
  * Exploration behaviors exploit randomness.
* **Functional role:**

  * Enables exploration (avoiding brittle determinism).
  * Encodes uncertainty via population coding.
  * Supports Bayesian-like inference naturally.
* **Conclusion:** Brains exploit stochasticity as a structural feature, not as “mere noise.”

### World Models and Recursion

* **Definition:** A world model transforms sensory input into an internal state, simulates possible futures, and compares predictions against reality.
* **Self-reference:** Required to use its own predictions to refine itself. Modeled as recursive updates of the internal state.
* **Bounded recursion:** Biological and artificial systems truncate horizons to avoid infinite regress.
* **Causal/statistical reasoning:** Emerges from the ability to simulate counterfactuals and futures internally.

### UAT Extensions for Probability

* **Distributional UATs:**

  * Neural networks can approximate probability distributions under metrics like Wasserstein or KL.
  * Examples: pushforward measures (Lu & Lu, 2020), probabilistic transformers, Gaussian mixture models.
* **Key requirement:** Access to randomness and distributional objectives.
* **Without noise input:** No universal approximation of stochastic processes is possible.

### Architectural Considerations

* **Plain MLP:** Approximates deterministic functions only.
* **Stochastic MLP:** Requires latent noise inputs to represent both deterministic and stochastic mappings.
* **Hierarchy & recursion:** Provide structure and abstraction but do not alone enable probabilistic modeling without randomness access.
* **Minimal additions:** Neutral noise input + distributional loss → universal approximation over deterministic and stochastic functions.

### Memory and Curation for Stochastic Comprehension

* **Key limitation without memory:** Identical inputs with multiple possible outputs collapse to averages.
* **With memory + curation:**

  * Store multiple outcomes for the same input.
  * Organize them into distributions.
  * Enables modeling of probabilistic branching.
* **Branching graphs:** Higher-level transformations can represent distributions as probabilistic transitions rather than deterministic functions.

### Big Picture

1. Deterministic UAT covers continuous functions only.
2. Distributional UATs exist, but require randomness in the architecture or input.
3. Biological brains embed randomness structurally, enabling probabilistic inference.
4. World models demand recursive, bounded self-reference to simulate futures.
5. Memory + curation allow comprehension of stochasticness even without internal randomness generation.
6. External randomness APIs (NumPy, hardware RNGs, quantum sources) suffice for sampling; the system just needs to understand how to use them.

**Conclusion:** Comprehension of stochasticness is achievable if the system can (a) store multiple outcomes, (b) curate them into distributions, and (c) leverage randomness from external or internal sources for simulation. This upgrades deterministic approximation into probabilistic world modeling.

---

## 2. Current Conversation State (Verbatim)

* **Topic:** Whether universal approximators without explicit probabilistic design can comprehend or model stochasticness.
* **Points Raised:**

  * Deterministic UAT does not extend to stochastic mappings.
  * Comprehension of stochasticness may arise if the system can accumulate and curate multiple outcomes for identical inputs.
  * Hierarchical graph/world-model architectures may scaffold stochastic comprehension, but collapse to deterministic approximations if randomness is absent.
  * Memory + curation mechanisms are enough to infer distributions without internal randomness generation.
* **Consensus:** Modeling stochastic processes requires either embedded randomness or storage/curation mechanisms capable of organizing outcomes into probabilistic structures. Randomness can be provided externally; comprehension of stochasticness is an internal representational achievement.

# Archive 03

(Project HQ setup)

## GraphModel – Central Archive (verbatim canvases)


> Purpose: A single exportable document containing the **verbatim contents** of canvases currently visible in this conversation. For canvases created in other conversations, open them here to auto-append their full text.

---

## Canvas: Graph Model – Overall Structure (v0.1)

# GraphModel – Overall Structure (v0.1)

## High-Level Concepts

* **Initial hierarchy bounds.** With minimal cluster size of 2 per level, the maximum number of levels L for `n_level_0_cores` is:

  $L_{max} = \left\lceil \log_2(\, n_{\text{level0 cores}}\,) \right\rceil$

* **Nascent modules.** After initial optimization, instantiate **nascent\_sub\_module** (identity structure with sparse learning) and optionally **nascent\_sup\_module**. These act as scaffold modules that can be specialized incrementally.

* **Persistent gradient structure.** Keep an always-on gradient tape/flow (not just in training passes). This enables:

  * Token deposits into other cores outside a cluster region.
  * Movement/overlap across clusters without hardcoded routing.
  * Gradient-driven resource arbitration.

* **Special relativistic barrier.** As complexity approaches the allocated token limit, introduce a barrier curvature similar to relativistic asymptotics. Complexity cannot exceed capacity, but approaches it asymptotically. This barrier can itself be a **meta-learning structure**, adapting the curvature for different modules. It ensures stability near resource saturation and prevents catastrophic overload.

* **Higher-level unsupervised identification.** The architecture may classify sub-structures emergently, allocating special cluster-structures to sets of nodes. Inspired by physics datasets (e.g., high-energy electrons, photons, large mass objects):

  * Base state nodes manage low-energy/common patterns.
  * Higher-energy/outlier objects are assigned special clusters to avoid over-straining base structures.
  * This dynamic classification allows the graph space to manage rare or disruptive phenomena by structurally isolating them.

## Core OOP Entities

```python
class Core:
    """
    Attributes:
        state_memory: Any
        tokens: float  # allocated computational resources
        token_limit: float  # soft cap for relativistic barrier
        complexity: float  # computational complexity proxy
        sparse_nn: object  # pluggable sparse NN module
        update_frequency: float  # relative update cadence
        error_curve: object | None  # meta-structure for resource proposal (TBD)
        barrier_meta: dict | None  # parameters of relativistic barrier (learned)

    Methods:
        compute_complexity() -> float
        effective_velocity() -> float  # complexity / (token_limit + eps)
        gamma() -> float  # 1 / sqrt(1 - v_eff**2 + eps)
        optimizer(...):  # modified optimizer that weights certain directions
            # transforms loss landscape into a complexity_landscape with γ-damping
    """
    pass


class Connector:
    """Typed edge between cores/modules. Carries tokens, gradients, and messages.
    Attributes:
        capacity: float
        latency: float
        reliability: float
    """
    pass


class GraphModule:
    """
    Attributes:
        level_id: int
        core_id: int

        sub_modules: list["GraphModule"]
        sub_edges: list[Connector]

        core: Core
        edges: list[Connector]

        sup_edges: list[Connector]

        nascent_sub_modules: list["GraphModule"]
        nascent_sup_modules: list["GraphModule"] | None

        tokens: float
        token_limit: float
        module_complexity: float
        sub_module_complexity: float

        special_cluster_type: str | None  # e.g., "high_energy", "rare_event"
        energy_metric: float  # outlier/rarity score guiding promotion

    Methods:
        distribute_tokens()
        classify_substructures() -> dict  # unsupervised labels / cluster assignments
        apply_relativistic_barrier() -> None  # γ-damped updates/routing
    """
    pass


class Graph:
    """Collection of GraphModules with routing, scheduling, and persistence hooks.
    Methods:
        discover_structures() -> list  # finds candidates for special clusters
        materialize_special_cluster(mod_ids: list, kind: str) -> None
    """
    pass


class SuperGraph:
    """
    Attributes:
        complexity_token_bound: float
        tokens: float
        module_complexity: float
        sub_module_complexity: float
        minds_eye: object  # global scratch/attention for planning
        structure_discovery: object  # controller for unsupervised higher-level scanning

    Methods:
        distribute_tokens()
        schedule_meta_learning() -> None  # tunes barrier curvature & discovery priors
    """
    pass
```

## Architectural Rules

1. **Hierarchy formation.** Start with level-0 cores; cap levels using `ceil(log2(n_0))`. Use **complexity gain per token (CGPT)** to decide when to **materialize** additional structure (e.g., elevate responsibilities, instantiate special clusters) or **consolidate back to identity** (hibernate/simplify) when contribution is low. No pruning or demotion: all components retain a persistent identity that can be repurposed later with minimal space cost.
2. **Nascent growth.** After a module converges under current capacity, spawn a nascent child/parent with identity mapping + sparse parameters. Allow grafting if CGPT improves.
3. **Persistent gradients.** Maintain differentiable paths for routing and token transfers. Never hardcode static partitions; prefer soft constraints with annealed sparsity.
4. **Complexity-landscape optimizers.** Augment loss with complexity terms; weight directions to reduce compute while preserving task loss.
5. **Relativistic barrier.** As a module’s effective complexity approaches its allocated token limit, apply a **special-relativistic barrier** that smoothly compresses step sizes (analogous to a Lorentz factor). Barrier curvature is meta-learned per module and per context.

## Token & Complexity Accounting

* **Tokens** are fungible compute quanta (time, FLOPs, memory slots).

* **Effective velocity** `v_eff` := `complexity / (token_limit + ε)` per module.

* **Inverse barrier (γ)** acts like a relativistic curvature factor:

  γ(C) = 1 / (scale \* sqrt(max(1 - (C/token\_limit)^2, ε)))

* **LC objective (multiplicative form):**

  L\_total(θ) = L\_task(θ) × γ(C(θ))

  The standard task loss is directly reshaped by the inverse barrier. At low complexity, γ≈constant and the effect is negligible. As complexity approaches its budget, γ grows sharply, distorting the loss surface and damping feasible updates. Backpropagation automatically carries derivatives through both L and C.

* **Complexity metrics** (module and sub-module) are tracked online; used in `distribute_tokens()` to allocate budget.

* **Budget law (sketch):** Allocate proportional to moving-average marginal improvement per complexity, but barrier curvature discourages chronic saturation.

## Unsupervised Structural Discovery & Special Clusters

* A **structure discovery** process scans graphs for high-level patterns without supervision (e.g., density/routing motifs, high-energy events).
* Sub-structures can be **classified** and **assigned a special cluster-structure** (custom routing, higher token ceilings, resilient connectors).
* Inspired by physics “special objects” (e.g., high-energy electrons/photons, massive composites): rare/high-energy patterns are elevated to special clusters to avoid overloading base state nodes.
* The “space” (graph topology + constraints) is **self-managing**: it can adapt local geometry, spawn nascent sub/sup-modules, and reallocate tokens when outliers would otherwise break base layers.

## Component Responsibilities

* **Sparse NN module:** Define `Core.sparse_nn` API, masking mechanics, persistent-gradient hooks.
* **Complexity estimator:** Implement `compute_complexity()` and rolling CGPT.
* **Optimizers:** Prototype complexity-aware variants (e.g., trust region + L0/L1 sparsity and routing penalties).
* **Routing/Connector:** Typed message passing; token deposit/withdraw protocols.
* **Hierarchy manager:** Level creation/merging under the `ceil(log2(n_0))` bound; **materialization/consolidation criteria**.
* **MetaMemory (per-module):** Maintain local summaries (sorted spectra, gradients, activations) with learnable temporal windows.

## Canonical Sorted Spectra (Control Feature)

**Goal:** Provide a uniform, comparable perspective on parameters for meta-learning without hard-coded importance assumptions.

* **Two-level sorting** per layer:

  1. **Within-neuron sort** of weights by signed value: largest positive → … → 0 → … → most negative.
  2. **Across-neuron order** by a canonical size metric (default: L2 norm of each neuron’s weight vector; alternatives: L1, spectral for matrices).
* **Usage:** Only during learning. Store compact summaries (quantiles, top-k signatures, stability indices). Inference uses raw tensors.
* **Stability:** Sorting becomes more reliable as training stabilizes; early noise tolerated via EMAs.
* **Initialization control:** Prefer a **finite, unstructured palette** (e.g., orthogonal/quasi-random seeds) so early sorted views are comparable across modules.
* **Cost control:** Use sampling + top-k + quantiles to avoid full sorts every step.

### Minimal APIs

```python
class SortedSpectrum:
    def compute(self, W: torch.Tensor) -> torch.Tensor:  # returns flattened, signed-sorted view
        flat = W.flatten().detach()
        vals, _ = torch.sort(flat, descending=True)
        return vals

class LayerCanonicalOrder:
    def neuron_score(self, W: torch.Tensor) -> torch.Tensor:  # per-neuron L2
        # assume W shape [out, in]
        return W.pow(2).sum(dim=1).sqrt()

    def order_neurons(self, W: torch.Tensor) -> torch.Tensor:
        scores = self.neuron_score(W)
        _, idx = torch.sort(scores, descending=True)
        return idx
```

---

## Relativistic Barrier (Details)

* **Form:** start with `γ(v) = 1 / sqrt(1 - v^2 + ε)`; explore alternatives (softplus-based, logistic squashing) and let meta-learner choose.
* **Placement:** apply γ to (a) optimizer step size, (b) routing probabilities, (c) token deposit rates across connectors.
* **Credit assignment:** record γ-augmented traces to attribute when saturation avoidance improved task + complexity tradeoff.

## Unsupervised Discovery (Details)

* **Signals:** rarity scores (tail probabilities), energy metrics (activation norms, gradient magnitudes), and routing entropy.
* **Mechanics:** spectral/graph clustering + contrastive prototypes; **materialize** clusters to `special_cluster_type` when impact > threshold.
* **Safety:** cap special-cluster proliferation; **consolidate to identity/hibernate** if CGPT contribution declines, preserving reuse without deletion.

## Gradient-based Novelty Signaling & Transient Structures

**Goal:** let modules express capacity distress/novelty and trigger learned structural responses—without manual control flow.

### Signals (emitted by base/specialized modules)

* `grad_energy` := EMA(||g||), `grad_var` := Var(g) over recent batches
* `residual_error` or output entropy/dispersion
* `v_eff`, `γ` from the relativistic barrier
* `routing_entropy` (instability of paths), `saturation_time` (fraction of steps with γ above a threshold)

### Learned reactions (by sup/meta modules)

* **Transient structure instantiation:** spawn short-lived capacity near the affected region; parameters initialized from cached prototypes.
* **Token & route reallocation:** continuous deltas (no if/else); γ-aware damping prevents thrash.
* **Lifecycle:** if a transient repeatedly activates and improves the Lagrangian, **materialize** it as a persistent special cluster; if its marginal benefit falls, **consolidate back to identity** (hibernate). No pruning.

### Caching & reuse

* Maintain a sparse **prototype cache** keyed by signal signatures (hash of normalized feature vector). Recurrent conditions (e.g., moving black hole) reuse prior structure with minimal retraining.

### Minimal Interfaces

```python
class NoveltySignal:
    def __init__(self, grad_energy, grad_var, residual, v_eff, gamma, routing_entropy, saturation_time):
        self.vec = torch.tensor([grad_energy, grad_var, residual, v_eff, gamma, routing_entropy, saturation_time])

class TransientFactory:
    def propose(self, signal: NoveltySignal, neighborhood):
        # returns differentiable proposals: Δtokens, Δroutes, init_params (from cache or nascent template)
        pass

class PrototypeCache:
    def lookup(self, signature):
        # returns prior params if a similar condition was seen; else None
        pass
```

### Training

* All proposals are trained end-to-end under the global Lagrangian; thresholds (if any) are soft/learned.
* Persistent gradient summaries provide the time context for distinguishing one-off spikes vs sustained regime change.

## Open Questions / TODOs

* Do we need `error_curve` as a meta-proposer or can CGPT suffice?
* Best practice for persistent gradient without prohibitive memory: checkpointing + low-rank surrogates?
* Formalize token market dynamics (auctions vs proportional control).
* Define meta-learning policies for relativistic barrier curvature.
* Determine criteria for emergent sub-structure specialization.

## Notes (from proposal text)

```
initial heirarchy bounds could be set using the maximum number of levels if minimal cluster size of two is assumed per level
    ceil(log_2(n_level_0_cores))
after completing initial optimization, a nascent_sub_module could be generated
    nascent being an identity structure but with sparse learning abilities
perhaps even a nascent_sup_module

persistent gradient structure
    not just for training
    keeping an active gradient creates the ability to do things like deposit tokens into other cores outside a cluster region
    this gives the architecture the flexbility to have movement and overlap without hardcoding
```

---

**Usage:** Treat this document as the ground truth for project structure. Component conversations must include a Canvas with: (a) component scope, (b) current code state, (c) code canvases, (d) next steps.

---

## Canvas: GraphModel – Project Index & Conversation State (auto-snapshot)

# GraphModel – Project Index & Conversation State (auto-snapshot)

> Purpose: A single place to glance the **contents of each canvas** in this Project and the **current state of this conversation**. Update this doc whenever a new canvas is created or a major decision is made.

---

## Canvas Summaries (by document)

### 1) Graph Model – HQ (v0)

**Role:** Central coordination hub; quarterback for component conversations. **Contents:**

* Project purpose and goals (graph-based neural architecture).
* Workflow for component conversations and code canvases.
* Integration & review cadence from HQ.

**Links/Actions:**

* Use HQ to register new component canvases (name, scope, owner, status).

### 2) Graph Model – Overall Structure (v0.1)

**Role:** Architectural ground truth. **Key contents (condensed):**

* **Hierarchy bounds:** `L_max = ceil(log2(n_level0_cores))`.
* **Nascent modules:** identity + sparse learning; can *materialize* or *consolidate to identity* (no pruning/demotion).
* **Persistent gradients:** gradient summaries as signals; enable token deposits/routing without hard-coded control flow.
* **Relativistic inverse barrier:** capacity wall as complexity→token limit.
* **LC objective (multiplicative):** `L_total(θ) = L_task(θ) × γ(C(θ))`, with `γ` = inverse barrier.
* **Canonical sorted spectra:** two-level sorting (within-neuron by signed value; across-neuron by norm) as a uniform, comparable perspective for meta-learning.
* **Unsupervised discovery & special clusters:** structure discovery, special-cluster materialization, and safe consolidation to identity.
* **Gradient-based novelty & transients:** signals (grad energy/variance, residuals, v\_eff, γ, routing entropy) → transient capacity; prototype cache & reuse.

**Interfaces (highlights):** `Core`, `GraphModule`, `Graph`, `SuperGraph`, `SortedSpectrum`, `LayerCanonicalOrder`, `NoveltySignal`, `TransientFactory`, `PrototypeCache`.

### 3) Other component canvases

> Not yet opened in this conversation. When you open/attach them here, add a summary section below each heading.

* **\[Placeholder] Stochasticness & Memory Structures**

  * Scope: storing/curating internal data to model stochastic phenomena beyond expectations; distributional comprehension; multi-hypothesis memory.
  * Status: (attach once available)

* **\[Placeholder] Relativistic Barrier & Meta-Learner**

  * Scope: curvature parameterization, stability tuning, experiments.
  * Status: (attach once available)

* **\[Placeholder] SparseNN / Complexity Estimator / Routing-Connector / Hierarchy Manager**

  * Scope: per-module APIs, masks, CG metrics, token routing.
  * Status: (attach once available)

---

## State of This Conversation (decisions & open threads)

### Agreed Decisions

1. **Lifecycle language:** use *materialize* (instantiate/persist additional structure) and *consolidate to identity* (hibernate/simplify). No pruning/demotion; all components keep a persistent identity for future reuse.
2. **Relativistic inverse barrier:** adopt the *inverse* barrier (special-relativity style) with hard clipping at the token budget, small ε floor for stability. Effects are negligible far from the limit and strong near capacity.
3. **LC objective (multiplicative):** adopt `L_total(θ) = L_task(θ) × γ(C(θ))` so the barrier reshapes the loss directly; backprop naturally carries derivatives through both `L` and `C`.
4. **Implicit control flow via signaling:** no if/else rules; modules emit learned signals (grad energy/variance, residuals, γ, routing entropy), and higher-level structures learn to respond.
5. **Canonical sorting for control:** within-neuron signed sorting + across-neuron norm order as a uniform viewpoint; used only during learning (with EMAs); initialization via a finite, unstructured palette.

### Clarified Concepts

* **CGPT (marginal efficiency):** conceptual tool to reason about materialize vs consolidate choices; overshadowed in practice by the multiplicative LC objective and inverse barrier.
* **Transient vs persistent:** repeated beneficial transients can *materialize*; otherwise they *consolidate to identity* (cache persists, capacity released).

### Open Items / Next Steps

* **Stochasticness canvas:** create/attach a dedicated canvas to formalize memory structures (multi-hypothesis, distributional heads, sampling caches) and how LC dynamics govern their growth.
* **Barrier curvature meta-learning:** decide which parameters of γ are learnable (scale, ε schedule) and where they live (per-module vs shared priors).
* **Experiment plan:** minimal synthetic benchmarks to visualize LC behavior, nascent activation, and consolidation under moving “black hole” conditions.

---

**How to use this doc**

* Treat as the **project index** and **conversation heartbeat**.
* When a new canvas is opened in this conversation, append a summary section above and link back to HQ.
* When a decision lands, add it under *Agreed Decisions* with date and short rationale.

---

## Canvas: Graph Model – HQ (v0)

\[Verbatim content unavailable in this conversation. Open the HQ canvas here to auto-embed its full text.]


# Archive 04

# Core • SparseMaskedNN — Working Notes & Plan

*Date:* 2025‑09‑17
*Timezone:* America/Toronto

---

## Canonical Terms (locked-in)

**Heat states** (3-level, lean):

* **Cold** — masked, not under evaluation. `M=0`, no grads, no compute.
* **Warm** — masked, being probed (prospective). `M=0`, but use shadow Δ on region R only, across several batches.
* **Hot** — unmasked & committed. `M=1`, weights warm‑started from probe Δ, then trained with proximal L1.

Optional 4-level variant (only if we need reporting granularity): **Cold → Cool → Warm → Hot**, where *Cool* = shortlisted but not yet probed.

**Complexity counters**:

* **Structural** `C_struct = sum(M)` (# of Hot cells; hard cap per object).
* **Active** `C_active = count(|W_vals|>ε & M=1)` (# of lit cells; soft cap via prox‑L1 + pruning).

**Objects:** Core, Connector. Growth decisions are independent per object. Hard rule: ≤ 2 paths (regions) per object per cycle.

---

## Sparse‑first Representation (TensorFlow‑friendly)

* **Mask/structure (sparse):** `M_idx ∈ ℕ^{nnz×2}` (row, col) in COO, sorted/coalesced. Encodes Hot set.
* **Weights (trainable):** `W_vals ∈ ℝ^{nnz}` aligned 1‑to‑1 with `M_idx`.
* **Forward (dense inputs):**

  ```python
  W_sp = tf.sparse.reorder(tf.SparseTensor(M_idx, W_vals, [in_dim, out_dim]))
  Y = tf.transpose(tf.sparse.sparse_dense_matmul(W_sp, tf.transpose(X)))  # [B, out]
  ```
* **Forward (sparse inputs S):** `Y = tf.sparse.sparse_dense_matmul(S, tf.sparse.to_dense(W_sp))` or keep weights dense+masked for this path; prefer true sparse weights at inference.

**Why:** avoids dense Hadamard with masks; FLOPs scale with nnz; gradients flow only to `W_vals`. `M_idx` is non‑diff, edited by planner.

---

## Probing (Warm / Prospective)

Goal: estimate value per complexity token before unmasking.

**Two exact probe modes:**

1. **Toggle‑mask probe:** build `M' = M` with candidate region `R_idx` added; keep `W_R=0`; forward/backward once to read grads on `R`.
2. **Shadow‑Δ trick (preferred for rollouts):** keep base `M`; add `Δ_R` only on indices in `R` and compute with `K_eff = (M_idx,W_vals) ⊕ (R_idx, Δ_R)` (concatenation). Backprop hits only `Δ_R`.

**Micro‑rollout:** 1–3 tiny steps on `Δ_R` with the same sparsity pressure we’ll use in training (prox‑L1). Evaluate mean ΔLoss over multiple fresh batches.

---

## Ranking & Selection

**Prune‑side (Hot → drop)**

* **Taylor prune score (fast):** `Score_c = 0.5 * v_c * W_c^2` (use Adam/RMS second moment `v_c` as curvature proxy). Rank ascending; prune smallest.
* **LOO ΔLoss (gold standard on subset):** temporarily zero component/region; measure mean loss increase across `m` batches. Use to calibrate Taylor.
* **Answer‑directed:** if a node’s “answer” is a particular state functional `u^T y`, prefer `| (u^T J)_c * W_c |` for ranking.

**Grow‑side (Warm → Hot)**

* **Shortlist (cheap, batch‑robust):** SNR of prospective gradients over `m` batches: `||mean g_R|| / (sqrt(var g_R)+ε)`; gate by `τ_snr`.
* **Decide (micro‑rollout EI):** K tiny steps on `Δ_R`, compute `μ_R = mean ΔLoss`, `σ_R = std`, `a_R = count(|Δ_R|>ε)`.
* **Score:** `Score_R = μ_R − βσ_R − λ_s|R| − λ_a a_R` (value‑per‑token, variance‑aware). Commit ≤ 2 regions (per object). Warm‑start `W_R ← Δ_R`.

**Diversity constraints:** penalize spatial/output overlap during a single commit round; prefer coverage across outputs.

---

## State Machine (3‑level)

* `Cold → Warm` (shortlist/probe).
* `Warm → Hot` (commit mask; optional warm‑start `Δ_R`).
* **Prune loop:** if `C_active` exceeds budget, zero smallest|W| within Hot; optionally move Hot→Cold to respect `C_struct`.

**Prox‑L1:** after each optimizer step, `W_vals[M==1] ← soft_threshold(W_vals[M==1], λ)`. Encourages few Burning/“lit” entries while preserving Hot structure.

---

## Loss Landscape & Data

* Expectation vs mini‑batch: `g_B` is unbiased but noisy; variance is anisotropic and ∝ 1/|B|. Use multi‑batch estimates for Warm probes.
* Landscape is a **stack** over masks: continuous in `W_vals`, combinatorial in `M_idx`.

---

## Graph‑Node Integration (state/answer centric)

* A node’s SparseMaskedNN updates the node state; messages are derived from state.
* Rankings can be **loss‑centric** or **answer‑directed** (project onto the node’s state functional). Keep both options.
* Per‑object independence: enforce budgets and selection per Core/Connector; disjoint probes.

---

## Minimal Control Pseudocode (reference)

```python
# PRUNE
def prune_ranking(K_vals, V_moment, M_idx, eps):
    # return list of (idx_row, score) sorted ascending
    scores = []
    for r,(i,j) in enumerate(M_idx):
        w = K_vals[r]
        if abs(w) > eps:
            v = V_moment[i,j]
            scores.append((r, 0.5 * v * (w*w)))
    return sorted(scores, key=lambda x: x[1])

# GROW — shortlist by SNR, decide by micro‑rollout
```

(Full RegionEvaluator / bandit UCB variant is in earlier notes; reuse when implementing.)

---

## Implementation Notes (TensorFlow)

* `tf.sparse.sparse_dense_matmul` expects rank‑2 on both sides; for dense inputs use `Y = (W_sp @ X^T)^T`.
* Keep `indices:int64`, `values:float32`; `tf.sparse.reorder` after editing structure.
* Gradients flow to `W_vals` only; `M_idx` is non‑diff.
* For masked‑dense experiments only, generate dense mask on demand; do not store dense masks for large layers.

---

## Decisions & Open TODOs

* **Decision:** three heat states (Cold/Warm/Hot) and two counters (`C_struct`, `C_active`).
* **Decision:** mask is truly sparse (`M_idx`), not a dense bitmap.
* **Decision:** probe via shadow Δ and micro‑rollouts; commit ≤ 2 regions per object per cycle.
* **TODO‑1:** identity initialization for layers (square & non‑square; conv vs MLP; block‑diag identity for multi‑channel/state).
* **TODO‑2:** finalize pruning thresholds `ε` and budgets per object.
* **TODO‑3:** answer‑directed ranking option (define `u` per node).
* **TODO‑4:** implement RegionEvaluator + planner hooks in training loop; cache per‑column slices for fast edits.

---

> Snapshot saved to serve as the working spec for Core’s SparseMaskedNN growth/prune planner and data structures. Use this as the stable reference; we will iterate inline as we refine identity initialization and the rollout/bandit heuristics.

---
---

## Core • Progress Update — Probes, SNR, and Distribution‑Aware Heads

*Date:* 2025‑09‑17
*Timezone:* America/Toronto

This document records decisions and refinements made after the prior snapshot “Core • SparseMaskedNN — Working Notes & Plan”. It focuses on: (1) probe regimes (freezing vs relaxed), (2) SNR as a minimalist shortlist heuristic, and (3) an architecture‑level path for learning *distributions* (multi‑modal outputs) with memory, without introducing arbitrary penalties.

---

## 1) Probe regimes (attribution vs realism)

Let parameters split into **H** (Hot) and **R** (candidate Warm).

* **Frozen probe (default):** Optimize a *shadow* Δ only on R while H stays fixed at θ\_H⁰.

  * Pros: clean attribution; deterministic accounting of value‑per‑token; stable.
  * Cons: underestimates benefit when H↔R synergy matters.

* **Relaxed probe (optional):** Allow tiny co‑adaptation of H via a trust region (few inner steps, small ‖Δ\_H‖² weight). Approximates the Schur‑complement effect of cross‑curvature while preserving attribution. Use **only if** frozen probes systematically underpredict realized gains.

* **Full joint probe (rejected):** Lets H and R move freely during a probe. Attribution breaks; noisy decisions. Not used.

**Decision:** Start **Frozen**; upgrade to **Relaxed** only on evidence (gap between probe gain and realized gain post‑commit).

---

## 2) SNR — minimalist, batch‑robust shortlist

For a candidate region R, collect *prospective* gradients across m independent mini‑batches with the base frozen.

* Mean: \bar g\_R = (1/m) Σ g\_R^(t)
* Variance proxy: s\_R² = (1/(m−1)) Σ ‖g\_R^(t) − \bar g\_R‖²
* **SNR:** ‖\bar g\_R‖ / (√(s\_R²) + ε)

Use SNR **only** to rank/shortlist; the decision still comes from a short **micro‑rollout** (K tiny steps on the shadow Δ\_R, fresh batches) measured by average ΔLoss, its variance, and win‑rate. No regularizers; just evidence under a fixed token budget.

Preconditioning (dimensionless): do SNR on Adam‑style normalized grads g/√(v+ε) when available.

---

## 3) Distribution‑aware behavior (multi‑modal outputs with memory)

**Goal:** Enable a node to (1) detect when similar inputs map to multiple plausible outputs, and (2) *sample* from a learned distribution while tracking and matching target frequencies over time.

### 3.1 Minimal components (no arbitrary penalties)

* **Deterministic state head (baseline):** identity‑by‑default SparseMaskedNN mapping; no stochasticity engaged.
* **Distribution Head (DH, optional):** a sparse, upgradable head that parameterizes an output distribution conditioned on the node’s state.

  * **Discrete outputs:** categorical with logits per bucket. Complexity tokens = number of Hot buckets (nonzero logit paths).
  * **Continuous outputs:** mixture‑of‑Gaussians (or quantile‑spline). Tokens = number of Hot components (or active knots).
  * DH is **Cold** by default; engages only when evidence warrants.

**Selection (evidence‑only):**

1. **Ambiguity detector (lightweight):** on a buffer of cases with similar state (e.g., locality in latent space), check if residuals/disagreements are multi‑modal (e.g., bimodal histogram or increased conditional entropy). If ambiguous, nominate a DH region (e.g., add 1–2 buckets or 1 mixture component).
2. **Probe:** Frozen micro‑rollout on the DH candidate (K tiny steps, fresh batches), evaluate ΔNegLogLik (or Δcalibration error on held‑out mini‑batches).
3. **Commit ≤ 2 per object:** promote the best to Hot; warm‑start parameters from the probe Δ.

Pruning mirrors this: remove the least valuable DH components by tiny LOO ΔNLL or Taylor on their parameters.

### 3.2 Sampling with memory (matching frequencies)

When DH is Hot, produce a sample **and** maintain a running calibration memory so long‑run frequencies match learned probabilities.

* **Error‑feedback sampler (discrete):**

  * Maintain an accumulator e per bucket: e ← e + p − 𝟙{sample==bucket}.
  * Next sample uses logits adjusted by −κ·e (small κ). This is stochastic rounding with conservation: over time it drives empirical frequencies toward p without biasing single‑step probability too aggressively.

* **Continuous outputs:**

  * Sample from the mixture as usual; maintain a PIT (probability integral transform) buffer z=F\_Y(y). For a well‑calibrated model, z ∼ Uniform\[0,1]. Periodically run a simple KS check on z; if miscalibrated, nominate DH refinement (Warm) in the region of miss.

**Decision rule (evidence, not regularization):** grow DH only when calibration error persists beyond statistical fluctuation (small two‑sample test on sliding windows) *and* the micro‑rollout shows NLL gain.

### 3.3 Complexity accounting

* **Weights:** nnz of SparseMaskedNN (excluding implicit identity if desired).
* **Activation curvature:** count of Hot spline coefficients.
* **Distribution head:** count of Hot buckets/components/knots.

Hard caps per object; no penalty terms in the loss.

---

## 4) Spline activation with learnable complexity (recap)

* Activation is identity at init; curvature is added via **sparse** hinge or spline coefficients (Hot only when committed).
* Same probe/commit/prune cycle as weights.
* Identity fast‑path: if no Hot/Warm deltas (weights or activation), the layer is a pure pass‑through (no compute).

---

## 5) Open design hooks (next passes)

* Define a **minimal ambiguity detector**: e.g., rising conditional entropy in a small latent neighborhood, or a stable bimodal residual histogram. Keep it parameter‑light.
* Choose DH parameterizations per variable type (categorical vs scalar continuous) and how tokens map to “one more bucket / one more component / a few more knots”.
* Specify micro‑rollout measurements for DH: ΔNLL, win‑rate, held‑out transfer.
* Finalize structural budgets for: weights, activation curvature, and DH components.

---

## 6) Operational defaults (initial)

* Probe regime: **Frozen** (upgrade to **Relaxed** only if needed).
* SNR shortlist: m=4 batches; micro‑rollout: K=2 steps; pick ≤2 Warm→Hot per object per cycle.
* Identity everywhere until growth; compute only on Hot/Warm deltas.
* No global penalties; decisions by evidence under hard budgets; prune only when over budget.

> This update keeps the architecture assumption‑light and fully evidence‑driven, while opening a principled path to handle multi‑modal outputs and long‑run frequency matching without adding arbitrary regularization terms.

---
---

## Core • Save State — Today’s Updates

*Date:* 2025‑09‑18
*Timezone:* America/Toronto

This captures **everything added/decided today** since the prior save.

---

## A) Design Tenets (reaffirmed)

* **Evidence over penalties:** decisions via short probes + held‑out checks; **no L1/L0 in loss**.
* **Hard budgets:** structural + active token caps per object/depth.
* **Identity‑first:** compute only deltas; skip work when layer is exact identity.
* **Concise comms + concept map:** keep your phrasing; new terms get one‑line on‑ramps.

---

## B) Probes & SNR (minimalist)

* **Frozen probe (default):** only tweak Warm candidate; clean attribution.
* **Relaxed probe (optional):** tiny trust‑region on Hot if frozen probes underpredict realized gains.
* **SNR (shortlist only):** across m batches,
  `SNR_R = ||mean(g_R)|| / (sqrt(var(g_R)) + ε)` on **prospective** grads.
* **Decision metric:** average ΔLoss on fresh batches (+ tie‑breakers: higher win‑rate, lower variance).

---

## C) Spline Activation with Learnable Complexity

* **Activation = identity + sparse curvature** (hinge/spline terms). Start with zero coefficients.
* Growth/prune uses the same **Warm → Hot** and evidence rule as weights.
* **Continuous gates** deemed optional; we stay discrete for now (probe provides continuity).

---

## D) Distribution Awareness (general + Core fit)

* **Philosophy:** when a situation admits multiple sensible outcomes, add small optional “bins/components” to represent alternatives and their frequencies.
* **Distribution Head (DH):** optional add‑on; **Cold by default**. Grow in tiny steps (one bin/component at a time) only on evidence; prune when weak.
* **Memoryless sampling:** dropped sampling‑with‑memory. Use i.i.d. sampling or deterministic selection; monitor calibration offline and trigger DH refinement if needed.
* **Multi‑layer DH (nascent everywhere):** allowed at any depth but: (i) **late‑sample** (no internal sampling; learn in expectation), (ii) **≤1 DH add per depth per cycle**, (iii) strict budgets, (iv) reversible via Cooling.
* **Ambiguity signals:** top‑down hint from higher modules or local repeated split outcomes (bimodal residuals / rising conditional entropy).

---

## E) Helping Struggling Hot Regions (local assistance)

* **Flags:** low gradient stability, stall (big residuals with tiny progress), or
  `help_ratio = ||g_cold*|| / (||g_hot|| + ε)` large (prospective frontier grads overshadow current Hot grads).
* **Action:** shortlist Cold frontier by SNR → probe top‑1 → commit ≤1 helper if ΔLoss gain is consistent.

---

## F) Clone‑and‑Split (load sharing without disruption)

* **Goal:** split an overloaded unit/block without changing outputs at t=0.
* **Activation‑agnostic recipe (kept):**

  1. **Duplicate** the neuron/block (same incoming weights/bias).
  2. **Halve downstream weights** (next‑layer connections) across the two clones.
  3. Optional tiny asymmetry (noise or staggered updates); allow different growth frontiers.
  4. **Keep** if ΔLoss↑ and stability↑; else **re‑merge** (sum back, drop clone) to reclaim tokens.
* **Why not literal equal upstream split with nonlinear act:** `A(W₁+W₂) ≠ A(W₁)+A(W₂)` in general. Downstream halving preserves function for any activation.
* **Partition alternative (when splitting sparse supports):** if you do split supports, assign by descending |weight| alternating to balance mass; ensure sums match the original.

---

## G) Lifecycle Extensions (optional)

* Besides 3‑state (Cold/Warm/Hot), we can use **transition states**: **Cooling** (graceful demotion) & **Warming** (probe). Cooling triggers when a Hot piece is rarely used and removal barely impacts held‑out error.

---

## H) Minimal Loops (today’s defaults)

* **Per object/depth per cycle:**

  1. Shortlist by SNR (m≈4).
  2. Probe K≈2 tiny steps (Frozen).
  3. Commit ≤2 (objects) / ≤1 (per depth DH).
  4. Cooling check on weak/unused pieces.

---

## I) Open Questions / Next Steps

* Identity init details for spline activations (knot grid defaults; variance‑preserving scales).
* Ambiguity detector: minimal, parameter‑light definition for local split residuals.
* Per‑depth token budgets (initial values) and simple hysteresis for Cooling.
* Tiny DH schemas (categorical vs scalar continuous) with probe routines.


---
---

> **Decision Update (2025‑09‑18):** Drop sampling‑with‑memory for the Distribution Head. Default is **memoryless** sampling (i.i.d.) or deterministic selection (argmax/mean). Calibration is measured offline; if persistent miscalibration appears, propose DH refinement via probes—no runtime biasing.

# Core • SparseMaskedNN — Working Notes & Plan

*Date:* 2025‑09‑17
*Timezone:* America/Toronto

---

## Canonical Terms (locked-in)

**Heat states** (3-level, lean):

* **Cold** — masked, not under evaluation. `M=0`, no grads, no compute.
* **Warm** — masked, being probed (prospective). `M=0`, but use shadow Δ on region R only, across several batches.
* **Hot** — unmasked & committed. `M=1`, weights warm‑started from probe Δ, then trained with proximal L1.

Optional 4-level variant (only if we need reporting granularity): **Cold → Cool → Warm → Hot**, where *Cool* = shortlisted but not yet probed.

**Complexity counters**:

* **Structural** `C_struct = sum(M)` (# of Hot cells; hard cap per object).
* **Active** `C_active = count(|W_vals|>ε & M=1)` (# of lit cells; soft cap via prox‑L1 + pruning).

**Objects:** Core, Connector. Growth decisions are independent per object. Hard rule: ≤ 2 paths (regions) per object per cycle.

---

## Sparse‑first Representation (TensorFlow‑friendly)

* **Mask/structure (sparse):** `M_idx ∈ ℕ^{nnz×2}` (row, col) in COO, sorted/coalesced. Encodes Hot set.
* **Weights (trainable):** `W_vals ∈ ℝ^{nnz}` aligned 1‑to‑1 with `M_idx`.
* **Forward (dense inputs):**

  ```python
  W_sp = tf.sparse.reorder(tf.SparseTensor(M_idx, W_vals, [in_dim, out_dim]))
  Y = tf.transpose(tf.sparse.sparse_dense_matmul(W_sp, tf.transpose(X)))  # [B, out]
  ```
* **Forward (sparse inputs S):** `Y = tf.sparse.sparse_dense_matmul(S, tf.sparse.to_dense(W_sp))` or keep weights dense+masked for this path; prefer true sparse weights at inference.

**Why:** avoids dense Hadamard with masks; FLOPs scale with nnz; gradients flow only to `W_vals`. `M_idx` is non‑diff, edited by planner.

---

## Probing (Warm / Prospective)

Goal: estimate value per complexity token before unmasking.

**Two exact probe modes:**

1. **Toggle‑mask probe:** build `M' = M` with candidate region `R_idx` added; keep `W_R=0`; forward/backward once to read grads on `R`.
2. **Shadow‑Δ trick (preferred for rollouts):** keep base `M`; add `Δ_R` only on indices in `R` and compute with `K_eff = (M_idx,W_vals) ⊕ (R_idx, Δ_R)` (concatenation). Backprop hits only `Δ_R`.

**Micro‑rollout:** 1–3 tiny steps on `Δ_R` with the same sparsity pressure we’ll use in training (prox‑L1). Evaluate mean ΔLoss over multiple fresh batches.

---

## Ranking & Selection

**Prune‑side (Hot → drop)**

* **Taylor prune score (fast):** `Score_c = 0.5 * v_c * W_c^2` (use Adam/RMS second moment `v_c` as curvature proxy). Rank ascending; prune smallest.
* **LOO ΔLoss (gold standard on subset):** temporarily zero component/region; measure mean loss increase across `m` batches. Use to calibrate Taylor.
* **Answer‑directed:** if a node’s “answer” is a particular state functional `u^T y`, prefer `| (u^T J)_c * W_c |` for ranking.

**Grow‑side (Warm → Hot)**

* **Shortlist (cheap, batch‑robust):** SNR of prospective gradients over `m` batches: `||mean g_R|| / (sqrt(var g_R)+ε)`; gate by `τ_snr`.
* **Decide (micro‑rollout EI):** K tiny steps on `Δ_R`, compute `μ_R = mean ΔLoss`, `σ_R = std`, `a_R = count(|Δ_R|>ε)`.
* **Score:** `Score_R = μ_R − βσ_R − λ_s|R| − λ_a a_R` (value‑per‑token, variance‑aware). Commit ≤ 2 regions (per object). Warm‑start `W_R ← Δ_R`.

**Diversity constraints:** penalize spatial/output overlap during a single commit round; prefer coverage across outputs.

---

## State Machine (3‑level)

* `Cold → Warm` (shortlist/probe).
* `Warm → Hot` (commit mask; optional warm‑start `Δ_R`).
* **Prune loop:** if `C_active` exceeds budget, zero smallest|W| within Hot; optionally move Hot→Cold to respect `C_struct`.

**Prox‑L1:** after each optimizer step, `W_vals[M==1] ← soft_threshold(W_vals[M==1], λ)`. Encourages few Burning/“lit” entries while preserving Hot structure.

---

## Loss Landscape & Data

* Expectation vs mini‑batch: `g_B` is unbiased but noisy; variance is anisotropic and ∝ 1/|B|. Use multi‑batch estimates for Warm probes.
* Landscape is a **stack** over masks: continuous in `W_vals`, combinatorial in `M_idx`.

---

## Graph‑Node Integration (state/answer centric)

* A node’s SparseMaskedNN updates the node state; messages are derived from state.
* Rankings can be **loss‑centric** or **answer‑directed** (project onto the node’s state functional). Keep both options.
* Per‑object independence: enforce budgets and selection per Core/Connector; disjoint probes.

---

## Minimal Control Pseudocode (reference)

```python
# PRUNE
def prune_ranking(K_vals, V_moment, M_idx, eps):
    # return list of (idx_row, score) sorted ascending
    scores = []
    for r,(i,j) in enumerate(M_idx):
        w = K_vals[r]
        if abs(w) > eps:
            v = V_moment[i,j]
            scores.append((r, 0.5 * v * (w*w)))
    return sorted(scores, key=lambda x: x[1])

# GROW — shortlist by SNR, decide by micro‑rollout
```

(Full RegionEvaluator / bandit UCB variant is in earlier notes; reuse when implementing.)

---

## Implementation Notes (TensorFlow)

* `tf.sparse.sparse_dense_matmul` expects rank-2 on both sides; for dense inputs use `Y = (W_sp @ X^T)^T`.
* Keep `indices:int64`, `values:float32`; `tf.sparse.reorder` after editing structure.
* Gradients flow to `W_vals` only; `M_idx` is non-diff.
* For masked-dense experiments only, generate dense mask on demand; do not store dense masks for large layers.
* **Δ-state logging (new):** use a ring buffer of sparse deltas per step: `Δs_t = state_t − state_{t−1}` with timestamps and batch tags. Prefer COO indices aligned with Hot mask; coalesce per K steps. For GPU-friendly writes, buffer per-output-column then flush.

---

## State History as Differentials (Δ-state log)

**Why:** align with dynamical-systems view; make temporal structure observable (finite-difference signals, candidate ODE/PDE discovery, drift detection) without storing full snapshots.

**What to store (minimal):**

* `t_k` (step index or wall-clock), optional batch id.
* `Δstate_k` (sparse): same index space as node state; record only entries |Δ|>ε.
* Optional: `Δinputs_k`/`Δoutputs_k` if needed for identifiability.

**How it’s used:**

* **Reconstruction:** `state_t = state_0 + Σ_{k≤t} Δstate_k` (on demand).
* **Signals:** finite differences/lagged features for probes (e.g., trend, curvature); SNR for change-points.
* **DE discovery (optional):** feed `(state_t, Δstate_t/Δt)` to a symbolic/linear regressor on a fixed library of candidate operators (spatial grads, products); only as a tool, not in the training loss.

**Budgets & controls:**

* Keep a rolling horizon `H` (e.g., last 128 steps) with coalescing; hard cap on nnz per horizon.
* Gate logging to **Hot** regions or fronts under evaluation to control cost.
* Toggle via flags per node: `log_deltas=True`, `epsilon_delta`, `horizon_steps`.

**Why not full snapshots:**

* Δs are cheaper (sparser), compress long quiescent periods, and expose dynamics directly.

---

## Innate Convolution via Perceptron‑Local Sparse Weight Matrices (new)

**Premise:** give each perceptron a *local* 2‑D weight matrix over its input neighborhood (not just a 1‑D vector). Keep this matrix **sparse and mask‑controlled** so the perceptron can *learn* convolution‑like structure rather than having it baked in.

**Why this is valuable:**

* **Emergent convolution:** with a local stencil that can grow (Cold→Warm→Hot), the perceptron can discover translational motifs and edge/texture detectors without a hard‑coded convolution layer. Tying is optional; untied weights allow pattern diversity, and later we can add soft tying for efficiency.
* **Zero cost at start:** begin with an identity‑like or near‑empty stencil (very few Hot entries). Capacity increases only when Warm probes show value.
* **Reuse our static sparse multiplier:** the neighborhood stencil per perceptron is a fixed pattern for many steps. We precompute `output_coordinate_list`, `a_value_index_for_path`, `b_value_index_for_path`, `segment_id_for_path`, then run fast gather→multiply→grouped‑sum. No dynamic coalescing.
* **Localized growth:** planner grows per‑perceptron neighborhoods where Δ‑signals and error heat are strongest, instead of global, dense expansion.

**Design sketch:**

* **Indexing:** represent each perceptron’s 2‑D stencil as a block of indices inside the global mask. Prefer block‑sparse tiles (per output unit) so memory accesses are local.
* **Initialization:** identity‑like (center weight = 1, others 0) or tiny Laplacian seed; keep activation as identity spline initially.
* **Optional tying:** later, add a tying map that groups similar stencils; learn a small set of shared kernels while keeping exceptions untied where needed.

## Dense Crossover Policy (new)

**Goal:** automatically switch a layer (or a perceptron block) from sparse to dense when dense math is cheaper.

**Signals to monitor (per layer or block):**

* **Density estimate:** `density = active_nonzeros / total_entries`.
* **Batch amplification:** effective flop savings shrink as batch size grows; include `batch_size` in the decision.
* **Path count:** `path_count` from the static map; cost scales with this, not matrix size.

**Simple rule of thumb to start:**

* If `density < 0.05` → stay sparse.
* If `0.05 ≤ density ≤ 0.2` → compare costs once per N steps:

  * `cost_sparse ≈ α * path_count` (α accounts for gathers and reduction)
  * `cost_dense ≈ β * (m * k * n)` (β from a one‑time micro‑benchmark on current device)
    Switch if `cost_dense < cost_sparse` for several consecutive checks.
* If `density > 0.2` → prefer dense unless structure is block‑regular (then keep block‑sparse).

**Granularity:** allow mixed mode—dense for hotspots, sparse elsewhere. Maintain separate masks and kernels per block.

**Accounting:** when switching a block to dense, update `C_struct` and `C_active` to reflect the new representation; keep a cooldown to avoid thrash.

## Planner Hooks (new)

* **Warm probe for stencil growth:** test adding rings or arms around the current stencil; score via micro‑rollouts and Δ‑state SNR.
* **Cooling/Prune:** if a perceptron’s stencil stays quiet (low Δ‑state, low gradient norm), cool outer cells first, then inner.
* **Clone‑and‑halve for overloaded units:** as before, but preserve local stencil geometry; downstream 0.5 scalers stay in place.

## Decisions & Open TODOs

* **Decision:** three heat states (Cold/Warm/Hot) and two counters (`C_struct`, `C_active`).
* **Decision:** mask is truly sparse (`M_idx`), not a dense bitmap.
* **Decision:** probe via shadow Δ and micro‑rollouts; commit ≤ 2 regions per object per cycle.
* **NEW Decision:** add optional **Δ‑state logging** per node with tight budgets (off by default; on for experiments).
* **NEW Decision:** include **perceptron‑local sparse stencils** to enable emergent convolution.
* **NEW Decision:** add a **dense crossover policy** with device‑measured α, β.
* **TODO-1:** identity initialization for layers (square & non‑square; block‑diag identity for multi‑channel/state).
* **TODO-2:** finalize pruning thresholds `epsilon` and budgets per object.
* **TODO-3:** answer‑directed ranking option (define `u` per node).
* **TODO-4:** implement RegionEvaluator + planner hooks in training loop; cache per‑column slices for fast edits.
* **TODO-5:** Δ‑log reader for finite‑difference features and DE‑candidate scorer.
* **TODO-6 (new):** per‑device micro‑bench to estimate `α, β` and validate crossover thresholds.
* **TODO-7 (new):** add soft‑tying option for stencils (group Lasso or low‑rank factorization) to reduce params when many stencils align.

---

---
---

> **Decision Update (2025‑09‑18):** Drop sampling‑with‑memory for the Distribution Head. Default is **memoryless** sampling (i.i.d.) or deterministic selection (argmax/mean). Calibration is measured offline; if persistent miscalibration appears, propose DH refinement via probes—no runtime biasing.

# Core • SparseMaskedNN — Working Notes & Plan

*Date:* 2025‑09‑17
*Timezone:* America/Toronto

---

## Canonical Terms (locked-in)

**Heat states** (3-level, lean):

* **Cold** — masked, not under evaluation. `M=0`, no grads, no compute.
* **Warm** — masked, being probed (prospective). `M=0`, but use shadow Δ on region R only, across several batches.
* **Hot** — unmasked & committed. `M=1`, weights warm‑started from probe Δ, then trained with proximal L1.

Optional 4-level variant (only if we need reporting granularity): **Cold → Cool → Warm → Hot**, where *Cool* = shortlisted but not yet probed.

**Complexity counters**:

* **Structural** `C_struct = sum(M)` (# of Hot cells; hard cap per object).
* **Active** `C_active = count(|W_vals|>ε & M=1)` (# of lit cells; soft cap via prox‑L1 + pruning).

**Objects:** Core, Connector. Growth decisions are independent per object. Hard rule: ≤ 2 paths (regions) per object per cycle.

---

## Sparse‑first Representation (TensorFlow‑friendly)

* **Mask/structure (sparse):** `M_idx ∈ ℕ^{nnz×2}` (row, col) in COO, sorted/coalesced. Encodes Hot set.
* **Weights (trainable):** `W_vals ∈ ℝ^{nnz}` aligned 1‑to‑1 with `M_idx`.
* **Forward (dense inputs):**

  ```python
  W_sp = tf.sparse.reorder(tf.SparseTensor(M_idx, W_vals, [in_dim, out_dim]))
  Y = tf.transpose(tf.sparse.sparse_dense_matmul(W_sp, tf.transpose(X)))  # [B, out]
  ```
* **Forward (sparse inputs S):** `Y = tf.sparse.sparse_dense_matmul(S, tf.sparse.to_dense(W_sp))` or keep weights dense+masked for this path; prefer true sparse weights at inference.

**Why:** avoids dense Hadamard with masks; FLOPs scale with nnz; gradients flow only to `W_vals`. `M_idx` is non‑diff, edited by planner.

---

## Probing (Warm / Prospective)

Goal: estimate value per complexity token before unmasking.

**Two exact probe modes:**

1. **Toggle‑mask probe:** build `M' = M` with candidate region `R_idx` added; keep `W_R=0`; forward/backward once to read grads on `R`.
2. **Shadow‑Δ trick (preferred for rollouts):** keep base `M`; add `Δ_R` only on indices in `R` and compute with `K_eff = (M_idx,W_vals) ⊕ (R_idx, Δ_R)` (concatenation). Backprop hits only `Δ_R`.

**Micro‑rollout:** 1–3 tiny steps on `Δ_R` with the same sparsity pressure we’ll use in training (prox‑L1). Evaluate mean ΔLoss over multiple fresh batches.

---

## Ranking & Selection

**Prune‑side (Hot → drop)**

* **Taylor prune score (fast):** `Score_c = 0.5 * v_c * W_c^2` (use Adam/RMS second moment `v_c` as curvature proxy). Rank ascending; prune smallest.
* **LOO ΔLoss (gold standard on subset):** temporarily zero component/region; measure mean loss increase across `m` batches. Use to calibrate Taylor.
* **Answer‑directed:** if a node’s “answer” is a particular state functional `u^T y`, prefer `| (u^T J)_c * W_c |` for ranking.

**Grow‑side (Warm → Hot)**

* **Shortlist (cheap, batch‑robust):** SNR of prospective gradients over `m` batches: `||mean g_R|| / (sqrt(var g_R)+ε)`; gate by `τ_snr`.
* **Decide (micro‑rollout EI):** K tiny steps on `Δ_R`, compute `μ_R = mean ΔLoss`, `σ_R = std`, `a_R = count(|Δ_R|>ε)`.
* **Score:** `Score_R = μ_R − βσ_R − λ_s|R| − λ_a a_R` (value‑per‑token, variance‑aware). Commit ≤ 2 regions (per object). Warm‑start `W_R ← Δ_R`.

**Diversity constraints:** penalize spatial/output overlap during a single commit round; prefer coverage across outputs.

---

## State Machine (3‑level)

* `Cold → Warm` (shortlist/probe).
* `Warm → Hot` (commit mask; optional warm‑start `Δ_R`).
* **Prune loop:** if `C_active` exceeds budget, zero smallest|W| within Hot; optionally move Hot→Cold to respect `C_struct`.

**Prox‑L1:** after each optimizer step, `W_vals[M==1] ← soft_threshold(W_vals[M==1], λ)`. Encourages few Burning/“lit” entries while preserving Hot structure.

---

## Loss Landscape & Data

* Expectation vs mini‑batch: `g_B` is unbiased but noisy; variance is anisotropic and ∝ 1/|B|. Use multi‑batch estimates for Warm probes.
* Landscape is a **stack** over masks: continuous in `W_vals`, combinatorial in `M_idx`.

---

## Graph‑Node Integration (state/answer centric)

* A node’s SparseMaskedNN updates the node state; messages are derived from state.
* Rankings can be **loss‑centric** or **answer‑directed** (project onto the node’s state functional). Keep both options.
* Per‑object independence: enforce budgets and selection per Core/Connector; disjoint probes.

---

## Minimal Control Pseudocode (reference)

```python
# PRUNE
def prune_ranking(K_vals, V_moment, M_idx, eps):
    # return list of (idx_row, score) sorted ascending
    scores = []
    for r,(i,j) in enumerate(M_idx):
        w = K_vals[r]
        if abs(w) > eps:
            v = V_moment[i,j]
            scores.append((r, 0.5 * v * (w*w)))
    return sorted(scores, key=lambda x: x[1])

# GROW — shortlist by SNR, decide by micro‑rollout
```

(Full RegionEvaluator / bandit UCB variant is in earlier notes; reuse when implementing.)

---

## Implementation Notes (TensorFlow)

* `tf.sparse.sparse_dense_matmul` expects rank-2 on both sides; for dense inputs use `Y = (W_sp @ X^T)^T`.
* Keep `indices:int64`, `values:float32`; `tf.sparse.reorder` after editing structure.
* Gradients flow to `W_vals` only; `M_idx` is non-diff.
* For masked-dense experiments only, generate dense mask on demand; do not store dense masks for large layers.
* **Δ-state logging (new):** use a ring buffer of sparse deltas per step: `Δs_t = state_t − state_{t−1}` with timestamps and batch tags. Prefer COO indices aligned with Hot mask; coalesce per K steps. For GPU-friendly writes, buffer per-output-column then flush.

---

## State History as Differentials (Δ-state log)

**Why:** align with dynamical-systems view; make temporal structure observable (finite-difference signals, candidate ODE/PDE discovery, drift detection) without storing full snapshots.

**What to store (minimal):**

* `t_k` (step index or wall-clock), optional batch id.
* `Δstate_k` (sparse): same index space as node state; record only entries |Δ|>ε.
* Optional: `Δinputs_k`/`Δoutputs_k` if needed for identifiability.

**How it’s used:**

* **Reconstruction:** `state_t = state_0 + Σ_{k≤t} Δstate_k` (on demand).
* **Signals:** finite differences/lagged features for probes (e.g., trend, curvature); SNR for change-points.
* **DE discovery (optional):** feed `(state_t, Δstate_t/Δt)` to a symbolic/linear regressor on a fixed library of candidate operators (spatial grads, products); only as a tool, not in the training loss.

**Budgets & controls:**

* Keep a rolling horizon `H` (e.g., last 128 steps) with coalescing; hard cap on nnz per horizon.
* Gate logging to **Hot** regions or fronts under evaluation to control cost.
* Toggle via flags per node: `log_deltas=True`, `epsilon_delta`, `horizon_steps`.

**Why not full snapshots:**

* Δs are cheaper (sparser), compress long quiescent periods, and expose dynamics directly.

---

## Innate Convolution via Perceptron‑Local Sparse Weight Matrices (new)

**Premise:** give each perceptron a *local* 2‑D weight matrix over its input neighborhood (not just a 1‑D vector). Keep this matrix **sparse and mask‑controlled** so the perceptron can *learn* convolution‑like structure rather than having it baked in.

**Why this is valuable:**

* **Emergent convolution:** with a local stencil that can grow (Cold→Warm→Hot), the perceptron can discover translational motifs and edge/texture detectors without a hard‑coded convolution layer. Tying is optional; untied weights allow pattern diversity, and later we can add soft tying for efficiency.
* **Zero cost at start:** begin with an identity‑like or near‑empty stencil (very few Hot entries). Capacity increases only when Warm probes show value.
* **Reuse our static sparse multiplier:** the neighborhood stencil per perceptron is a fixed pattern for many steps. We precompute `output_coordinate_list`, `a_value_index_for_path`, `b_value_index_for_path`, `segment_id_for_path`, then run fast gather→multiply→grouped‑sum. No dynamic coalescing.
* **Localized growth:** planner grows per‑perceptron neighborhoods where Δ‑signals and error heat are strongest, instead of global, dense expansion.

**Design sketch:**

* **Indexing:** represent each perceptron’s 2‑D stencil as a block of indices inside the global mask. Prefer block‑sparse tiles (per output unit) so memory accesses are local.
* **Initialization:** identity‑like (center weight = 1, others 0) or tiny Laplacian seed; keep activation as identity spline initially.
* **Optional tying:** later, add a tying map that groups similar stencils; learn a small set of shared kernels while keeping exceptions untied where needed.

## Dense Crossover Policy (new)

**Goal:** automatically switch a layer (or a perceptron block) from sparse to dense when dense math is cheaper.

**Signals to monitor (per layer or block):**

* **Density estimate:** `density = active_nonzeros / total_entries`.
* **Batch amplification:** effective flop savings shrink as batch size grows; include `batch_size` in the decision.
* **Path count:** `path_count` from the static map; cost scales with this, not matrix size.

**Simple rule of thumb to start:**

* If `density < 0.05` → stay sparse.
* If `0.05 ≤ density ≤ 0.2` → compare costs once per N steps:

  * `cost_sparse ≈ α * path_count` (α accounts for gathers and reduction)
  * `cost_dense ≈ β * (m * k * n)` (β from a one‑time micro‑benchmark on current device)
    Switch if `cost_dense < cost_sparse` for several consecutive checks.
* If `density > 0.2` → prefer dense unless structure is block‑regular (then keep block‑sparse).

**Granularity:** allow mixed mode—dense for hotspots, sparse elsewhere. Maintain separate masks and kernels per block.

**Accounting:** when switching a block to dense, update `C_struct` and `C_active` to reflect the new representation; keep a cooldown to avoid thrash.

## Planner Hooks (new)

* **Warm probe for stencil growth:** test adding rings or arms around the current stencil; score via micro-rollouts and Δ-state SNR.
* **Cooling/Prune:** if a perceptron’s stencil stays quiet (low Δ-state, low gradient norm), cool outer cells first, then inner.
* **Clone‑and‑halve for overloaded units:** as before, but preserve local stencil geometry; downstream 0.5 scalers stay in place.

---

## Stencil Dynamics: Gravity, Velocity, Curvature (new)

**Goal:** let similar stencils attract and optionally merge, while preserving freedom to diverge when data justify it.

**Fields & thresholds (learned or tuned):**

* `gravity_strength η_g` — base pull.
* `attach_threshold τ_attach` — start applying pull when similarity exceeds this.
* `merge_horizon τ_merge` — event‑horizon for merge (soft‑tying) decisions.
* `escape_threshold τ_escape` — residual size to break a tie.
* `flat_radius R_flat` — beyond this distance, treat space as flat (no interaction) for efficiency.

**Similarity / distance:** cosine similarity on Hot coordinates (optionally per ring). Distance `d(u,v)=1−sim(u,v)`.

**Velocity update (per step or per probe):**

* Maintain a stencil velocity `v_u`.
* Compute target direction toward nearest eligible center `ĉ` (max similarity above `τ_attach`).
* Curvature modulation `κ_u` from a cheap proxy (e.g., gradient‑direction variance or diagonal moment): dampen velocity in high‑curvature zones.
* Update:

  ```
  v_u ← μ v_u + (1−μ) · clip(ĉ − w_u, max_norm)
  v_u ← (1 − κ_u) · v_u
  w_u ← w_u + η_g · v_u  # only on Hot coords; project onto mask
  ```
* Hard separation: if `d(u,ĉ) < d_min` but `τ_merge` not met, apply a small repulsive term to prevent premature collapse.

**Merging (event horizon):** when `sim(u,ĉ) ≥ τ_merge` and a tiny held‑out check improves loss, switch to soft‑tying:

```
 w_u = c + r_u  # center + residual
 penalty: λ_res · ||r_u||_1
```

Centers update from members; any unit escapes if `||r_u||` exceeds `τ_escape` with loss improvement.

**Flat‑space approximation:** if `d(u,v) > R_flat` (or similarity below a small floor), ignore interactions entirely to avoid global all‑pairs work.

**Diversity filter (fast):** non‑maximum suppression within a neighborhood; keep best SNR candidate and cool near‑duplicates.

**Accounting:** tied units count as one center + residual nnz in `C_active`. Cooling targets residuals first.

---

## Dense‑Compute / Sparse‑Semantics Layer (note)

Implement perceptron blocks as packed dense tiles with gather lists (scanline order). Runtime uses dense GEMV/GEMM per block; planner still reasons in nnz/masks. Switch per block to fully dense under the crossover policy.

---

## Decisions & Open TODOs

* **Decision:** three heat states (Cold/Warm/Hot) and two counters (`C_struct`, `C_active`).
* **Decision:** mask is truly sparse (`M_idx`), not a dense bitmap.
* **Decision:** probe via shadow Δ and micro‑rollouts; commit ≤ 2 regions per object per cycle.
* **NEW Decision:** add optional **Δ‑state logging** per node with tight budgets (off by default; on for experiments).
* **NEW Decision:** include **perceptron‑local sparse stencils** to enable emergent convolution.
* **NEW Decision:** add a **dense crossover policy** with device‑measured α, β.
* **TODO-1:** identity initialization for layers (square & non‑square; block‑diag identity for multi‑channel/state).
* **TODO-2:** finalize pruning thresholds `epsilon` and budgets per object.
* **TODO-3:** answer‑directed ranking option (define `u` per node).
* **TODO-4:** implement RegionEvaluator + planner hooks in training loop; cache per‑column slices for fast edits.
* **TODO-5:** Δ‑log reader for finite‑difference features and DE‑candidate scorer.
* **TODO-6 (new):** per‑device micro‑bench to estimate `α, β` and validate crossover thresholds.
* **TODO-7 (new):** add soft‑tying option for stencils (group Lasso or low‑rank factorization) to reduce params when many stencils align.

---

