"""
High-level class-structured architecture for function-space message passing
via monotone spline feature banks and sparse learned permutations.

Notes:
- This is a conceptual blueprint to make the draft more tangible.
- Emphasizes explicit, readable OOP; avoids one-liners; uses descriptive names.
- No training loop or external deep learning framework is assumed.
- Replace stubs marked with `TODO` for production use.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable
import numpy as np


# -------------------------------
# Utilities
# -------------------------------

def safe_softplus(x: np.ndarray, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
    """
    Numerically stable softplus for enforcing nonnegativity in slope parameters.
    """
    result = np.empty_like(x)
    for idx in range(x.size):
        value = x.flat[idx]
        if beta * value > threshold:
            result.flat[idx] = value
        else:
            result.flat[idx] = np.log1p(np.exp(beta * value)) / beta
    return result


def trapezoid_integral(y_values: np.ndarray, x_values: np.ndarray) -> float:
    """
    Fallback numeric integral for diagnostics. Do not use in the main compute path.
    """
    if len(y_values) != len(x_values):
        raise ValueError("x and y must have same length")
    total = 0.0
    for idx in range(1, len(x_values)):
        width = x_values[idx] - x_values[idx - 1]
        area = 0.5 * (y_values[idx] + y_values[idx - 1]) * width
        total += area
    return float(total)


# -------------------------------
# Spline primitives
# -------------------------------

@dataclass
class CubicSplineSpan:
    """
    One cubic span in power form on interval [x0, x1]:
        s(x) = a + b*(x-x0) + c*(x-x0)^2 + d*(x-x0)^3
    """
    x0: float
    x1: float
    a: float
    b: float
    c: float
    d: float

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        values = np.empty_like(x, dtype=float)
        for idx in range(x.size):
            xv = x.flat[idx]
            if xv < self.x0:
                xv = self.x0
            if xv > self.x1:
                xv = self.x1
            t = xv - self.x0
            values.flat[idx] = self.a + self.b * t + self.c * t * t + self.d * t * t * t
        return values

    def integral_over(self, u: float, v: float) -> float:
        if u < self.x0:
            u = self.x0
        if v > self.x1:
            v = self.x1
        if v <= u:
            return 0.0
        # Integrate polynomial exactly
        du = u - self.x0
        dv = v - self.x0
        def F(t: float) -> float:
            return (
                self.a * t
                + 0.5 * self.b * t * t
                + (1.0 / 3.0) * self.c * t * t * t
                + 0.25 * self.d * t * t * t * t
            )
        return float(F(dv) - F(du))


@dataclass
class MonotoneCubicSpline:
    """
    Monotone cubic spline on r in [0, 1].
    Monotonicity is enforced via nonnegative slope basis (e.g., M-/I-spline logic).
    Here we expose a minimal power-form representation per span and rely on
    parameterization to keep slopes nonnegative in training code.
    """
    spans: List[CubicSplineSpan] = field(default_factory=list)

    def evaluate(self, r: np.ndarray) -> np.ndarray:
        values = np.zeros_like(r, dtype=float)
        for span in self.spans:
            mask = (r >= span.x0) & (r <= span.x1)
            if np.any(mask):
                values[mask] = span.evaluate(r[mask])
        # Handle out-of-domain by clamping at endpoints
        values[r < self.spans[0].x0] = self.spans[0].evaluate(np.array([self.spans[0].x0]))[0]
        values[r > self.spans[-1].x1] = self.spans[-1].evaluate(np.array([self.spans[-1].x1]))[0]
        return values

    def inner_product(self, other: "MonotoneCubicSpline") -> float:
        """
        Exact integral of product piecewise (assumes same span grid; if not,
        use LeastCommonRefinement to align first).
        """
        if len(self.spans) != len(other.spans):
            raise ValueError("Span mismatch; align via LCR before calling inner_product.")
        total = 0.0
        for idx in range(len(self.spans)):
            s1 = self.spans[idx]
            s2 = other.spans[idx]
            # Product of two cubics is degree-6 on each span. Integrate by expansion.
            # Expand (a + b t + c t^2 + d t^3)(A + B t + C t^2 + D t^3) with t = x - x0
            a,b,c,d = s1.a, s1.b, s1.c, s1.d
            A,B,C,D = s2.a, s2.b, s2.c, s2.d
            x0 = s1.x0
            if abs(s1.x0 - s2.x0) > 1e-12 or abs(s1.x1 - s2.x1) > 1e-12:
                raise ValueError("Span endpoints differ; run LCR alignment.")
            # Coefficients for degree-6 in t
            # k0..k6 such that prod(t) = sum_{k=0}^6 k_k * t^k
            k0 = a*A
            k1 = a*B + b*A
            k2 = a*C + b*B + c*A
            k3 = a*D + b*C + c*B + d*A
            k4 = b*D + c*C + d*B
            k5 = c*D + d*C
            k6 = d*D
            u = s1.x0; v = s1.x1
            du = u - x0; dv = v - x0
            def antiderivative(t: float) -> float:
                # Integral of sum k_i t^i dt = sum k_i * t^{i+1} / (i+1)
                acc = 0.0
                acc += k0 * t
                acc += k1 * (t**2) / 2.0
                acc += k2 * (t**3) / 3.0
                acc += k3 * (t**4) / 4.0
                acc += k4 * (t**5) / 5.0
                acc += k5 * (t**6) / 6.0
                acc += k6 * (t**7) / 7.0
                return acc
            total += antiderivative(dv) - antiderivative(du)
        return float(total)


# -------------------------------
# Feature bank and weights
# -------------------------------

@dataclass
class SplineFeatureBank:
    """
    A globally ordered, monotone spline feature bank (shared interface).
    """
    features: List[MonotoneCubicSpline] = field(default_factory=list)
    cached_rank_weights: Dict[int, np.ndarray] = field(default_factory=dict)

    def precompute_rank_grid_weights(self, rank_grid_size: int) -> np.ndarray:
        """
        Precompute weights w[c, k] = ∫_{(k-1)/n}^{k/n} phi_k(r) * f_c(r) dr
        using piecewise-linear hat basis phi_k on the rank grid.
        Result is cached per rank_grid_size.
        """
        if rank_grid_size in self.cached_rank_weights:
            return self.cached_rank_weights[rank_grid_size]

        n = rank_grid_size
        weights = np.zeros((len(self.features), n), dtype=float)
        # Rank grid nodes r_k = k / (n-1) for plotting; cell k uses [(k-1)/n, k/n]
        cell_edges = np.linspace(0.0, 1.0, n + 1)

        for feature_index, feature in enumerate(self.features):
            for k in range(n):
                left = cell_edges[k]
                right = cell_edges[k + 1]
                # phi_k is a hat centered near r_k; for a simple, robust rule on [0,1]
                # we approximate with two affine pieces per neighboring cells.
                # Here we integrate f(r) against a unit-mass piecewise-linear basis
                # centered at the cell; this produces stable quadrature weights.
                # Left affine: from left to mid
                midpoint = 0.5 * (left + right)
                w_left = self._integrate_feature_times_affine(feature, left, midpoint,
                                                              alpha=1.0/(midpoint-left),
                                                              beta=-left/(midpoint-left))
                # Right affine: from mid to right
                w_right = self._integrate_feature_times_affine(feature, midpoint, right,
                                                               alpha=-1.0/(right-midpoint),
                                                               beta=right/(right-midpoint))
                weights[feature_index, k] = w_left + w_right
        self.cached_rank_weights[rank_grid_size] = weights
        return weights

    def _integrate_feature_times_affine(self,
                                        feature: MonotoneCubicSpline,
                                        u: float,
                                        v: float,
                                        alpha: float,
                                        beta: float) -> float:
        """
        Compute ∫_{u}^{v} (alpha * r + beta) * f(r) dr exactly per span.
        """
        total = 0.0
        for span in feature.spans:
            seg_u = max(u, span.x0)
            seg_v = min(v, span.x1)
            if seg_u >= seg_v:
                continue
            # Affine * cubic -> quartic; integrate exactly.
            # Let r = x; t = r - x0
            a,b,c,d = span.a, span.b, span.c, span.d
            x0 = span.x0
            def integral_on_segment(p: float, q: float) -> float:
                # Integrate (alpha*r + beta)*(a + b*(r-x0) + c*(r-x0)^2 + d*(r-x0)^3)
                # Expand into powers of t = r - x0; r = t + x0
                # Then integrate term-by-term.
                def F(t: float) -> float:
                    r = t + x0
                    # construct polynomial coefficients explicitly for clarity
                    val_a = (alpha * r + beta) * a
                    val_b = (alpha * r + beta) * b * t
                    val_c = (alpha * r + beta) * c * t * t
                    val_d = (alpha * r + beta) * d * t * t * t
                    # integrate each as polynomial in t (linear in r = t + x0)
                    # For readability, use small incremental trapezoid as a diagnostic fallback
                    # In production, replace with closed-form antiderivatives.
                    return val_a + val_b + val_c + val_d
                # Use small Gauss-Legendre 5-point for robustness here (quartic exact)
                # nodes and weights on [-1,1]
                nodes = np.array([0.0,
                                  -0.5384693101056831,
                                  0.5384693101056831,
                                  -0.9061798459386640,
                                  0.9061798459386640])
                weights = np.array([0.5688888888888889,
                                    0.4786286704993665,
                                    0.4786286704993665,
                                    0.2369268850561891,
                                    0.2369268850561891])
                half = 0.5 * (q - p)
                mid = 0.5 * (q + p)
                acc = 0.0
                for j in range(nodes.size):
                    rj = mid + half * nodes[j]
                    tj = rj - x0
                    acc += weights[j] * F(tj)
                return float(half * acc)
            total += integral_on_segment(seg_u, seg_v)
        return total


# -------------------------------
# Least-common-refinement (LCR)
# -------------------------------

@dataclass
class LCRAligner:
    """
    Align two spline spaces by refining to their least common refinement (LCR)
    of knot intervals. In this simplified version, we only align span breakpoints
    and rebuild power-form spans; shape-preserving reparameterization is assumed.
    """
    def align(self,
              features_a: List[MonotoneCubicSpline],
              features_b: List[MonotoneCubicSpline]) -> Tuple[List[MonotoneCubicSpline], List[MonotoneCubicSpline]]:
        # Collect all breakpoints
        def breaks(feat: MonotoneCubicSpline) -> List[float]:
            return [s.x0 for s in feat.spans] + [feat.spans[-1].x1]
        all_points: List[float] = []
        for feat in features_a + features_b:
            all_points.extend(breaks(feat))
        all_points = sorted(set(all_points))
        # Rebuild each feature onto the unified grid by local polynomial fits
        def regrid(feat: MonotoneCubicSpline) -> MonotoneCubicSpline:
            new_spans: List[CubicSplineSpan] = []
            for i in range(len(all_points) - 1):
                u = all_points[i]
                v = all_points[i + 1]
                # Sample 4 points to fit a cubic in power form on [u, v]
                sample_rs = np.linspace(u, v, 4)
                sample_vals = feat.evaluate(sample_rs)
                # Fit power polynomial relative to u: s(t) = a + b t + c t^2 + d t^3
                T = np.zeros((4, 4), dtype=float)
                for r_idx in range(4):
                    t = sample_rs[r_idx] - u
                    T[r_idx, 0] = 1.0
                    T[r_idx, 1] = t
                    T[r_idx, 2] = t * t
                    T[r_idx, 3] = t * t * t
                coeffs = np.linalg.lstsq(T, sample_vals, rcond=None)[0]
                new_spans.append(CubicSplineSpan(x0=u, x1=v,
                                                 a=float(coeffs[0]),
                                                 b=float(coeffs[1]),
                                                 c=float(coeffs[2]),
                                                 d=float(coeffs[3])))
            return MonotoneCubicSpline(spans=new_spans)
        return [regrid(f) for f in features_a], [regrid(f) for f in features_b]


# -------------------------------
# Permutation adapter
# -------------------------------

@dataclass
class SparsePermutationAdapter:
    """
    Train-time: differentiable, band-masked near-permutation (e.g., Gumbel–Sinkhorn).
    Eval-time: hard gather indices for O(n) application.
    This class only stores the learned structure and exposes forward operators.
    """
    size: int
    band_width: int
    temperature: float
    logits: np.ndarray = field(init=False)
    hard_indices: Optional[np.ndarray] = field(default=None)

    def __post_init__(self) -> None:
        self.logits = np.full((self.size, self.size), -np.inf, dtype=float)
        for i in range(self.size):
            j_min = max(0, i - self.band_width)
            j_max = min(self.size, i + self.band_width + 1)
            for j in range(j_min, j_max):
                self.logits[i, j] = 0.0  # uniform init inside band

    def forward_soft(self, x: np.ndarray) -> np.ndarray:
        """
        Apply soft permutation: P_soft @ x. This is a placeholder using row-wise softmax.
        Replace with Gumbel–Sinkhorn for proper bistochastic constraints.
        """
        P = np.zeros_like(self.logits)
        for i in range(self.size):
            # masked softmax on row i
            row = self.logits[i].copy()
            row_max = np.max(row)
            numer = np.exp((row - row_max) / max(self.temperature, 1e-6))
            denom = np.sum(numer)
            if denom == 0.0:
                P[i, :] = 0.0
            else:
                P[i, :] = numer / denom
        return P @ x

    def set_hard_indices_from_logits(self) -> None:
        indices = np.zeros(self.size, dtype=int)
        for i in range(self.size):
            indices[i] = int(np.argmax(self.logits[i]))
        self.hard_indices = indices

    def forward_hard(self, x: np.ndarray) -> np.ndarray:
        if self.hard_indices is None:
            raise RuntimeError("Hard indices not set; call set_hard_indices_from_logits().")
        y = np.empty_like(x)
        for i in range(self.size):
            j = int(self.hard_indices[i])
            y[i] = x[j]
        return y


# -------------------------------
# Overlap-based gating
# -------------------------------

@dataclass
class OverlapGate:
    threshold: float
    alpha: float

    def soft_gate(self, overlap_matrix: np.ndarray) -> np.ndarray:
        """
        Sigmoid gate g = sigmoid(alpha * (S - threshold)).
        """
        g = np.empty_like(overlap_matrix)
        for i in range(overlap_matrix.shape[0]):
            for j in range(overlap_matrix.shape[1]):
                z = self.alpha * (overlap_matrix[i, j] - self.threshold)
                g[i, j] = 1.0 / (1.0 + np.exp(-z))
        return g

    def hard_mask(self, overlap_matrix: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(overlap_matrix, dtype=bool)
        for i in range(overlap_matrix.shape[0]):
            for j in range(overlap_matrix.shape[1]):
                mask[i, j] = bool(overlap_matrix[i, j] > self.threshold)
        return mask


# -------------------------------
# Message projection pipeline
# -------------------------------

@dataclass
class MessageProjector:
    feature_bank: SplineFeatureBank
    gate: Optional[OverlapGate] = None
    overlap_matrix: Optional[np.ndarray] = None  # precomputed S_cd

    def precompute_overlaps(self) -> None:
        C = len(self.feature_bank.features)
        self.overlap_matrix = np.zeros((C, C), dtype=float)
        for i in range(C):
            for j in range(C):
                self.overlap_matrix[i, j] = self.feature_bank.features[i].inner_product(
                    self.feature_bank.features[j]
                )

    def project(self, permuted_message: np.ndarray) -> np.ndarray:
        """
        Compute s_c = sum_k w_{c,k} * m'_k
        Returns an array of shape [C].
        """
        n = permuted_message.size
        weights = self.feature_bank.precompute_rank_grid_weights(n)
        C = weights.shape[0]
        scores = np.zeros(C, dtype=float)
        for c in range(C):
            total = 0.0
            for k in range(n):
                total += weights[c, k] * float(permuted_message[k])
            scores[c] = total
        return scores


# -------------------------------
# Graph layer skeleton
# -------------------------------

@dataclass
class SplineMessagePassingLayer:
    """
    One layer of message passing using the functional interface.
    For each edge u->v: align sender vector with receiver bank via permutation,
    then project onto receiver's spline features to produce channel scores.
    """
    receiver_bank: SplineFeatureBank
    permutation_adapter: SparsePermutationAdapter
    projector: MessageProjector

    def forward_edge(self, sender_vector: np.ndarray) -> np.ndarray:
        permuted = self.permutation_adapter.forward_soft(sender_vector)
        scores = self.projector.project(permuted)
        return scores


# -------------------------------
# Gram-based preconditioning (optional)
# -------------------------------

@dataclass
class GramPreconditioner:
    bank: SplineFeatureBank
    gram_matrix: Optional[np.ndarray] = None
    cholesky_L: Optional[np.ndarray] = None

    def build(self) -> None:
        C = len(self.bank.features)
        G = np.zeros((C, C), dtype=float)
        for i in range(C):
            for j in range(C):
                G[i, j] = self.bank.features[i].inner_product(self.bank.features[j])
        # Cholesky with small ridge for stability
        ridge = 1e-6
        for i in range(C):
            G[i, i] += ridge
        self.gram_matrix = G
        self.cholesky_L = np.linalg.cholesky(G)

    def apply_inverse(self, grad_coeffs: np.ndarray) -> np.ndarray:
        if self.cholesky_L is None:
            raise RuntimeError("Call build() before apply_inverse().")
        # Solve G x = grad via two triangular solves
        y = np.linalg.solve(self.cholesky_L, grad_coeffs)
        x = np.linalg.solve(self.cholesky_L.T, y)
        return x


# -------------------------------
# Example wiring function (pseudo-demo)
# -------------------------------

def build_demo_layer(num_features: int,
                     num_spans: int,
                     rank_size: int,
                     perm_band: int) -> SplineMessagePassingLayer:
    """
    Construct a toy layer with synthetic monotone splines and a permutation adapter.
    """
    # Build a simple bank: increasing linear-on-each-span cubics (for demo)
    features: List[MonotoneCubicSpline] = []
    knot_positions = np.linspace(0.0, 1.0, num_spans + 1)
    for c in range(num_features):
        spans: List[CubicSplineSpan] = []
        scale = 0.5 + 0.5 * (c + 1) / num_features
        for i in range(num_spans):
            u = float(knot_positions[i])
            v = float(knot_positions[i + 1])
            width = v - u
            # Simple monotone cubic: a + b t, with small convexity
            a = scale * (u)
            b = scale * (1.0)
            c2 = 0.1 * scale * width  # weak curvature
            d3 = 0.0
            spans.append(CubicSplineSpan(x0=u, x1=v, a=a, b=b, c=c2, d=d3))
        features.append(MonotoneCubicSpline(spans=spans))

    bank = SplineFeatureBank(features=features)
    projector = MessageProjector(feature_bank=bank)
    projector.precompute_overlaps()

    adapter = SparsePermutationAdapter(size=rank_size, band_width=perm_band, temperature=0.5)

    layer = SplineMessagePassingLayer(receiver_bank=bank,
                                      permutation_adapter=adapter,
                                      projector=projector)
    return layer


# -------------------------------
# End of module
# -------------------------------
