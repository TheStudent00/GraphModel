import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from tqdm import tqdm

# ============================================================
# PART 1: PHYSICS PRIMITIVES (The Trust Engine)
# ============================================================

class TrustGate(nn.Module):
    """
    [The Safety Valve]
    Bounded Precision Normalization.

    Physics:
    1. Splits Signal into Direction (Shape) and Amplitude (Loudness).
    2. Calculates Trust Score = Amplitude / (Sigma + epsilon).
    3. Bounds Gain using Tanh(Score).
    4. Output = Direction * Trusted_Gain.

    Result:
    - High Amp / Low Sigma -> Gain ~ 1.0 (Trusted)
    - High Amp / High Sigma -> Gain ~ 0.0 (Silenced)
    """
    def __init__(self, sensitivity: float = 1.0, epsilon: float = 1e-6):
        super().__init__()
        self.sensitivity = sensitivity
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 5) -> [c3, c2, c1, c0, sigma]

        # 1. Split Data
        coeffs = x[..., :4]
        sigma = x[..., 4:5]

        # 2. Measure Amplitude (L2 Norm of coeffs)
        # (..., 1)
        amplitude = torch.norm(coeffs, p=2, dim=-1, keepdim=True)

        # 3. Calculate Normalized Direction (Unit Sphere)
        # Avoid div by zero for silent signals
        direction = coeffs / (amplitude + self.epsilon)

        # 4. Calculate Trust Score (The SNR)
        # "How much louder is the signal than the noise?"
        trust_score = amplitude / (sigma + self.epsilon)

        # 5. Determine Bounded Gain (The Safety Tanh)
        # Maps Score [0, inf] -> Gain [0, 1]
        # If Score is high (e.g. 10), Tanh approaches 1.0.
        # If Score is low (e.g. 0.5), Tanh drops.
        trusted_gain = torch.tanh(trust_score * self.sensitivity)

        # 6. Apply Gain to Direction
        # The output magnitude is now strictly determined by Trust, not raw Amplitude.
        out_coeffs = direction * trusted_gain

        # Re-attach the sigma (it propagates to the next layer)
        return torch.cat([out_coeffs, sigma], dim=-1)

class Atom(nn.Module):
    """
    [The Phoneme]
    Resolution-Agnostic Warper with Trust Gating.

    Processing:
    1. Resamples Input (M) -> Internal (N).
    2. Warps Coefficients (Interaction).
    3. Resamples/Propagates Sigma.
    4. Applies TrustGate to silence noisy outputs.
    """
    def __init__(self, out_segments: int):
        super().__init__()
        self.out_segments = out_segments

        # The Basis: A fixed template of N segments
        self.basis = nn.Parameter(torch.randn(out_segments, 4))

        # The Gate: Enforces precision at the output
        self.gate = TrustGate(sensitivity=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M, 5) -> [Coeffs(4) | Sigma(1)]
        B, T, M, C = x.shape
        N = self.out_segments

        # 1. Resample (Project M -> N)
        # We process Coeffs and Sigma together
        if M != N:
            flat_x = x.view(B * T, M, C).transpose(1, 2) # (BT, 5, M)
            # Linear interp for coeffs is valid.
            # Linear interp for Sigma is a reasonable approximation for "Smearing" uncertainty.
            resampled = F.interpolate(flat_x, size=N, mode='linear', align_corners=True)
            current_stream = resampled.transpose(1, 2).view(B, T, N, C)
        else:
            current_stream = x

        # Split
        in_coeffs = current_stream[..., :4] # (B, T, N, 4)
        in_sigma = current_stream[..., 4:5] # (B, T, N, 1)

        # 2. Warp (Interaction)
        # Modulate internal basis by the input signal
        # This creates the new "Shape"
        warped_coeffs = in_coeffs * self.basis

        # 3. Propagate Uncertainty
        # Basic heuristic: Warping implies scaling.
        # If we scaled the coeffs, we technically scaled the error too.
        # But TrustGate will handle the renormalization.
        # We pass the resampled sigma through.

        # Recombine
        warped_stream = torch.cat([warped_coeffs, in_sigma], dim=-1)

        # 4. Trust Gating
        # If the Warp created a massive signal but the underlying Sigma was high,
        # TrustGate will crush it back down.
        return self.gate(warped_stream)

# ============================================================
# PART 2: GRAMMAR STRUCTURES
# ============================================================

class Compound(nn.Module):
    """
    [The Phrase]
    Recursive Container with Uncertainty Propagation.
    """
    def __init__(self, dna_config):
        super().__init__()
        self.dna = dna_config

        # Parse DNA
        node_type, op_mode, content = dna_config
        self.mode = node_type # "SERIES" or "PARALLEL"
        self.op = op_mode     # "SUM", "PRODUCT", or None

        self.branches = nn.ModuleList()

        # Build Branches
        for child_dna in content:
            self.branches.append(build_node(child_dna))

        # Global Gate for the Compound
        self.gate = TrustGate()

    def forward(self, x):
        # 1. Execution
        if self.mode == "SERIES":
            results = x
            for child in self.branches:
                results = child(results)

        elif self.mode == "PARALLEL":
            # Run all branches
            branch_outputs = [child(x) for child in self.branches]

            # 2. Interaction
            if self.op == "SUM":
                # Bundling: Add Coefficients, Add Variances
                # C_new = C1 + C2
                # Sigma_new = sqrt(s1^2 + s2^2)

                # Stack to sum easily
                stack = torch.stack(branch_outputs, dim=0) # (Branches, B, T, N, 5)
                coeffs = stack[..., :4].sum(dim=0)

                # Variance summation
                sigmas = stack[..., 4]
                new_sigma = torch.sqrt((sigmas ** 2).sum(dim=0)).unsqueeze(-1)

                results = torch.cat([coeffs, new_sigma], dim=-1)

            elif self.op == "PRODUCT":
                # Binding: Coeffs * Coeffs
                # For PoC, we propagate Max Sigma (Conservative estimate)
                # or we can do sqrt sum of squares for "Fog of War" approx

                # Base
                base = branch_outputs[0]
                c_acc = base[..., :4]
                s_acc_sq = base[..., 4] ** 2

                for b in branch_outputs[1:]:
                    c_next = b[..., :4]
                    s_next = b[..., 4]

                    c_acc = c_acc * c_next
                    s_acc_sq = s_acc_sq + (s_next ** 2)

                s_final = torch.sqrt(s_acc_sq).unsqueeze(-1)
                results = torch.cat([c_acc, s_final], dim=-1)
            else:
                results = branch_outputs[0]

        # 3. Regulation
        return self.gate(results)

def build_node(dna):
    """Factory: Converts DNA list into Objects"""
    node_type, op, content = dna

    if node_type == "LEAF":
        return Atom(out_segments=content)
    elif node_type in ["SERIES", "PARALLEL"]:
        return Compound(dna)
    else:
        raise ValueError(f"Unknown Node Type: {node_type}")

# ============================================================
# PART 3: THE EYE (Retina with Residuals)
# ============================================================

class VisualRetina(nn.Module):
    """
    [V10.9 Physics]
    Converts Pixels -> Trustworthy Spline Stream.
    Outputs: [c3, c2, c1, c0, RMSE]
    """
    def __init__(self, window_size: int = 32):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('z_indices', self._precompute_z(window_size))

    def _precompute_z(self, size):
        y = torch.arange(size).repeat_interleave(size)
        x = torch.arange(size).repeat(size)
        z = torch.zeros_like(y)
        for i in range(8):
            z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
        return torch.argsort(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 3, H, W)
        B, C, H, W = x.shape
        x_resized = F.interpolate(x, size=(self.window_size, self.window_size))

        # Z-Order Flatten
        x_flat = x_resized.view(B, C, -1)
        x_z = torch.gather(x_flat, 2, self.z_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1))

        # Chunking
        chunk_size = 16
        num_chunks = x_z.shape[-1] // chunk_size
        chunks = x_z.view(B, C, num_chunks, chunk_size)

        # --- Fit Cubic (with RMSE) ---
        t = torch.linspace(-1, 1, chunk_size, device=x.device)
        T_mat = torch.stack([t**3, t**2, t, torch.ones_like(t)], dim=1) # (16, 4)
        T_pinv = torch.linalg.pinv(T_mat) # (4, 16)

        y = chunks.permute(0, 2, 3, 1) # (B, Chunks, 16, C)

        # 1. Solve Coefficients
        coeffs = torch.matmul(T_pinv, y) # (B, Chunks, 4, C)

        # 2. Calculate Residuals (Sigma)
        # Reconstruct y_pred
        # T_mat (16, 4) @ coeffs (..., 4, C) -> (..., 16, C)
        # We need to broadcast T_mat
        y_pred = torch.matmul(T_mat, coeffs)

        # MSE = mean((y - y_pred)^2)
        residuals = (y - y_pred).pow(2).mean(dim=2) # Mean over chunk_len (16) -> (B, Chunks, C)
        sigma = residuals.sqrt().unsqueeze(2) # (B, Chunks, 1, C)

        # Combine [Coeffs(4), Sigma(1)] -> (B, Chunks, 5, C)
        token_data = torch.cat([coeffs, sigma], dim=2)

        # Structure Extraction (Mean over channels for PoC)
        # (B, Chunks, 5)
        structure = token_data.mean(dim=-1)

        # Output: (B, Time, 1, 5)
        return structure.unsqueeze(2)

# ============================================================
# PART 4: SYSTEM WRAPPER
# ============================================================

class VisionModule(nn.Module):
    def __init__(self, topology_dna, num_classes=10):
        super().__init__()
        self.retina = VisualRetina(window_size=32)
        self.brain = build_node(topology_dna)

        # Head input size: Segments * 5 (Coeffs+Sigma)
        # We assume the user topology ends with a specific segment count
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        stream = self.retina(x)      # (B, T, 1, 5)
        thought = self.brain(stream) # (B, T, N, 5)

        # Flatten (B, T*N*5)
        concept = thought.reshape(thought.shape[0], -1)
        return self.head(concept)

# ============================================================
# PART 5: TRAINING
# ============================================================

class TrainingPlotter:
    def __init__(self, save_path="training_metrics.png"):
        self.save_path = save_path
        self.epochs = []
        self.train_loss = []
        self.train_acc = []

    def update(self, epoch, t_loss, t_acc):
        self.epochs.append(epoch)
        self.train_loss.append(t_loss)
        self.train_acc.append(t_acc)
        self._plot()

    def _plot(self):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(self.epochs, self.train_loss, color='tab:red', linestyle='--', label='Train Loss')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy (%)', color='tab:blue')
        ax2.plot(self.epochs, self.train_acc, color='tab:blue', label='Train Acc')

        plt.title('Vision V10.9 (Trust Architecture)')
        fig.tight_layout()
        plt.savefig(self.save_path)
        plt.close()

def train_vision_model(model, epochs=5, plotter=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Device: {device} ---")
    model.to(device)

    import torchvision
    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    start_epoch = plotter.epochs[-1] + 1 if (plotter and plotter.epochs) else 1

    print(f"--- Starting Session: {epochs} Epochs ---")
    for epoch in range(epochs):
        current_epoch = start_epoch + epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Ep {current_epoch}", unit="batch")
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            # Diagnostic
            if epoch == 0 and i == 0:
                with torch.no_grad():
                    dummy_out = model.retina(inputs)
                    print(f"\n[DEBUG] Retina Out: {dummy_out.shape} (Includes Sigma)")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=loss.item(), acc=f"{100*correct/total:.2f}%")

        avg_loss = running_loss / len(trainloader)
        acc = 100 * correct / total
        if plotter: plotter.update(current_epoch, avg_loss, acc)

    return model

if __name__ == "__main__":

    # --- TOPOLOGY MACRO (5-Channel DNA) ---
    # Note: Removed "Phi Mode" from DNA since TrustGate is standard now.
    # Format: [TYPE, OP, CONTENT]

    attention_block = [
        "PARALLEL", "PRODUCT", [ # Binding
            [
                "PARALLEL", "PRODUCT", [ # Interaction
                    ["LEAF", None, 64], # Q
                    ["LEAF", None, 64]  # K
                ]
            ],
            ["LEAF", None, 64] # V
        ]
    ]

    model_topo = [
        "SERIES", None, [
            attention_block,
            attention_block,
            attention_block,
            ["LEAF", None, 64] # Final Output
        ]
    ]

    print("Initializing Vision V10.9 (Trust Architecture)...")
    model = VisionModule(model_topo)
    plotter = TrainingPlotter()

    model = train_vision_model(model, epochs=5, plotter=plotter)
