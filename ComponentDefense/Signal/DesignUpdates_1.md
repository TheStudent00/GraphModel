---
title: "Detailed Specification: Fractal Pyramid Compression and Light Cone Aperture"
project: "GraphModel"
author: "Student"
date: "2026-02-21"
status: "Architecture Specification / Theoretical Core - Expansion"
tags: 
  - hierarchical attention
  - information propagation
  - light cone aperture
  - structural fog
---

# 6. Detailed Specification: Fractal Pyramid Compression and Light Cone Aperture

The core mechanism for managing unbounded input sequences within finite computational limits is the structural generation of a Multi-Resolution Token Pyramid. Information routing occurs in three distinct phases: Bottom-Up Compression, Top-Down Infusion, and the Light Cone Forward Pass.

## 6.1 Bottom-Up Compression (The Ascending Path)
The ascending path constructs the hierarchical context by recursively compressing the token stack.

* **Overlapping Section Strategy:** To eliminate grid artifacts, the input stack is partitioned into sections of a fixed size (e.g., 32 tokens). These sections are merged with a stride of 1. For example, Level 0 `section_0` and `section_1` are combined to form the input for Level 1 `section_01`. `section_1` and `section_2` combine to form Level 1 `section_12`.
* **The Ascending Compressor:** The compression is executed via a specialized **Attention-Based Compressor** . This is not a static pooling operation.
    * It applies self-attention across the concatenated overlapping sections to determine semantic relevance before merging.
    * The output is a reduced section (halved in sequence length) representing a wider spatial domain.
    * **Semantic Depth Shift:** As tokens ascend the pyramid, their Virtual Centroid coordinate ($z_{virt}$) increments, signifying higher abstraction and physically smoother spectral curves.

## 6.2 Top-Down Infusion (The Descending Path)
If processing halts at the top of the pyramid, the network possesses global context but degraded local resolution. The descending path forces global awareness back into the high-resolution leaf nodes.

* **The Descending Compressor:** This requires a *different* attention-based compressor operating via **Cross-Attention** .
    * **Queries ($Q$):** Tokens from the lower, higher-resolution level (Level $n$, `section_i`).
    * **Keys/Values ($K, V$):** Tokens from the corresponding overlapping parent section at the higher, lower-resolution level (Level $n+1$, `section_ij`).
* **Output:** This process overwrites the Level $n$ sections. The updated tokens retain their exact high-resolution spectral structure but are infused with the global directional vectors of their macro-environment. This propagates downward until just above the base layer.

## 6.3 The Light Cone Aperture (The Forward Pass)
The final processing phase for any localized token stack relies on a dynamically constructed "Stack of Stacks," geometrically forming a **Light Cone** of attention.

* **Geometry of the Light Cone:** For any given base section (Level 0, `section_i`), the aperture does not look linearly across the $X$-axis of the unbounded sequence. Instead, it looks *diagonally upward and outward* through the pyramid.
    * **Center (High Resolution):** Level 0, `section_i`.
    * **Mid-Level (Medium Resolution):** Level 1, `section_{i-1, i}` and `section_{i, i+1}` (Left and Right parents).
    * **High-Level (Low Resolution):** Level 2+, expanding further left and right across the overlapping ancestry until reaching the root.
* **Aperture Execution:** An attention-like block processes this specific Light Cone.
    * **Query ($Q$):** The Level 0 base section.
    * **Keys/Values ($K, V$):** The concatenated `[Level 0, Level 1, Level 2... Level N]` Light Cone stack.
* **Implicit Fog Mechanism:** The Light Cone naturally enforces a "distance blur" . Because distant left/right information can only be accessed by querying the higher levels of the cone, the network can only extract smooth, low-frequency global information about distant events. The explicit mathematical distance penalty ($e^{-\Delta z}$) is structurally replaced by the polynomial simplicity of the upper-level spectral tokens.

### 6.3.1 Conceptual API: The Light Cone Builder
```python
class LightConeAperture:
    """
    Constructs and processes the custom stack-of-stacks for a given base section.
    """
    def __init__(self, attention_block):
        self.attention_block = attention_block

    def build_cone(self, pyramid: List[RaggedTensor], base_index: int) -> SpectralTensor:
        """
        pyramid: The fully compressed and top-down infused hierarchical stack.
        base_index: The index of the Level 0 section currently being processed.
        """
        cone_stack = []
        
        # Level 0: Pure Local Geometry
        cone_stack.append(pyramid[0].get_section(base_index))
        
        # Level 1 to N: Expanding Left/Right Ancestry
        for level in range(1, len(pyramid)):
            left_parent_idx = max(0, base_index - level)
            right_parent_idx = min(pyramid[level].num_sections - 1, base_index)
            
            # Fetch overlapping abstract representation
            cone_stack.append(pyramid[level].get_section_range(left_parent_idx, right_parent_idx))
            
        return SpectralTensor.concatenate(cone_stack)

    def forward(self, base_section: SpectralTensor, light_cone_stack: SpectralTensor):
        # The base section queries its customized hierarchical environment
        return self.attention_block(query=base_section, key_value=light_cone_stack)
