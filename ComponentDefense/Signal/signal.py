import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from tqdm import tqdm

# ============================================================
# PART 1: PHYSICS PRIMITIVES (Trust, Realization & Atoms)
# ============================================================

class TrustGate(nn.Module):
    """[The Safety Valve] Bounded Precision Normalization."""
    def __init__(self, sensitivity: float = 1.0, epsilon: float = 1e-6):
        super().__init__()
        self.sensitivity = sensitivity
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = x[..., :4]
        sigma = x[..., 4:5]
        
        amplitude = torch.norm(coeffs, p=2, dim=-1, keepdim=True)
        direction = coeffs / (amplitude + self.epsilon)
        trust_score = amplitude / (sigma + self.epsilon)
        
        trusted_gain = torch.tanh(trust_score * self.sensitivity)
        out_coeffs = direction * trusted_gain
        
        return torch.cat([out_coeffs, sigma], dim=-1)

class SplineRealizer(nn.Module):
    """
    [The Observer - V11.1 Cross-Weave]
    Converts Coeffs -> Samples.
    
    Physics:
    1. Strips Physics (Sigma/Z).
    2. Realizes: Samples the curves.
    
    Note: Does NOT pool time. It assumes the input has already been 
    summarized/warped to a fixed size by the VisionModule.
    """
    def __init__(self, resolution: int = 4):
        super().__init__()
        self.resolution = resolution
        # Matrix to sample the curve at 'resolution' points
        t = torch.linspace(-1, 1, resolution)
        self.register_buffer('T_mat', torch.stack([t**3, t**2, t, torch.ones_like(t)], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, N, Steps, 6)
        
        # 1. Strip Physics (Keep only Coeffs)
        coeffs = x[..., :4] # (B, N, Steps, 4)
        
        # 2. Realize (Sample the curves)
        # (..., 4) @ (4, R) -> (..., R)
        realized = torch.matmul(coeffs, self.T_mat.t()) 
        # Result: (B, N, Steps, Resolution)
        
        return realized

class Atom(nn.Module):
    """[The Phoneme] Resolution-Agnostic Warper."""
    def __init__(self, out_segments: int):
        super().__init__()
        self.out_segments = out_segments
        self.basis = nn.Parameter(torch.randn(out_segments, 4))
        self.gate = TrustGate(sensitivity=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input can be (B, T, M, 6) OR (B, N, T, 6) for Cross-Weave
        # We process whatever the 3rd dimension is (M) into N.
        B, T, M, C = x.shape
        N = self.out_segments
        
        # 1. Resample
        if M != N:
            flat_x = x.reshape(B * T, M, C).transpose(1, 2)
            
            # Interpolate Coeffs & Sigma
            resampled_core = F.interpolate(flat_x[:, :5, :], size=N, mode='linear', align_corners=True)
            
            # Broadcast Z-Index (or whatever is in channel 5)
            # For Cross-Weave, channel 5 might be Time or Z. We preserve it.
            resampled_z = flat_x[:, 5:, :].mean(dim=-1, keepdim=True).expand(-1, -1, N)
            
            resampled = torch.cat([resampled_core, resampled_z], dim=1)
            current_stream = resampled.transpose(1, 2).reshape(B, T, N, C)
        else:
            current_stream = x
            
        # 2. Warp
        in_coeffs = current_stream[..., :4] 
        in_sigma = current_stream[..., 4:5]
        in_z = current_stream[..., 5:6]
        
        warped_coeffs = in_coeffs * self.basis
        
        # 3. Gate
        to_gate = torch.cat([warped_coeffs, in_sigma], dim=-1)
        gated_core = self.gate(to_gate) 
        
        return torch.cat([gated_core, in_z], dim=-1)

# ============================================================
# PART 2: GRAMMAR STRUCTURES (Unchanged)
# ============================================================

class Compound(nn.Module):
    def __init__(self, dna_config):
        super().__init__()
        self.dna = dna_config
        node_type, op_mode, content = dna_config
        self.mode = node_type 
        self.op = op_mode     
        self.branches = nn.ModuleList([build_node(d) for d in content])
        self.gate = TrustGate()
        self.fog_density = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        if self.mode == "SERIES":
            results = x
            for child in self.branches:
                results = child(results)
        elif self.mode == "PARALLEL":
            branch_outputs = [child(x) for child in self.branches]
            if self.op == "SUM":
                stack = torch.stack(branch_outputs, dim=0)
                coeffs = stack[..., :4].sum(dim=0)
                sigmas = stack[..., 4]
                new_sigma = torch.sqrt((sigmas ** 2).sum(dim=0)).unsqueeze(-1)
                z_pos = stack[0, ..., 5:6]
                results = torch.cat([coeffs, new_sigma, z_pos], dim=-1)
            elif self.op == "PRODUCT":
                base = branch_outputs[0]
                c_acc = base[..., :4]
                s_acc_sq = base[..., 4] ** 2
                z_ref = base[..., 5] 
                for b in branch_outputs[1:]:
                    c_next = b[..., :4]
                    s_next = b[..., 4]
                    z_next = b[..., 5]
                    dist = torch.abs(z_ref - z_next)
                    density = F.softplus(self.fog_density) 
                    fog_variance = density * torch.log(1.0 + dist)
                    c_acc = c_acc * c_next
                    s_acc_sq = s_acc_sq + (s_next ** 2) + fog_variance
                s_final = torch.sqrt(s_acc_sq).unsqueeze(-1)
                results = torch.cat([c_acc, s_final, z_ref.unsqueeze(-1)], dim=-1)
            else:
                results = branch_outputs[0]
        core = results[..., :5]
        z_idx = results[..., 5:6]
        gated_core = self.gate(core)
        return torch.cat([gated_core, z_idx], dim=-1)

def build_node(dna):
    node_type, op, content = dna
    if node_type == "LEAF":
        return Atom(out_segments=content)
    elif node_type in ["SERIES", "PARALLEL"]:
        return Compound(dna)
    else:
        raise ValueError(f"Unknown Node Type: {node_type}")

# ============================================================
# PART 3: THE EYE (Unchanged)
# ============================================================

class VisualRetina(nn.Module):
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
        B, C, H, W = x.shape
        x_resized = F.interpolate(x, size=(self.window_size, self.window_size))
        x_flat = x_resized.view(B, C, -1)
        z_indices_flat = self.z_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        x_z = torch.gather(x_flat, 2, z_indices_flat)
        chunk_size = 16
        num_chunks = x_z.shape[-1] // chunk_size
        chunks = x_z.view(B, C, num_chunks, chunk_size)
        z_chunks = z_indices_flat.view(B, C, num_chunks, chunk_size).float()
        z_mean_chunk = z_chunks.mean(dim=-1)
        z_mean_pixel = z_mean_chunk.mean(dim=1)
        token_z_pos = z_mean_pixel.unsqueeze(-1) 
        t = torch.linspace(-1, 1, chunk_size, device=x.device)
        T_mat = torch.stack([t**3, t**2, t, torch.ones_like(t)], dim=1) 
        T_pinv = torch.linalg.pinv(T_mat) 
        y = chunks.permute(0, 2, 3, 1) 
        coeffs = torch.matmul(T_pinv, y) 
        y_pred = torch.matmul(T_mat, coeffs)
        residuals = (y - y_pred).pow(2).mean(dim=2)
        sigma = residuals.sqrt().unsqueeze(2) 
        base_data = torch.cat([coeffs, sigma], dim=2).mean(dim=-1) 
        final_stream = torch.cat([base_data, token_z_pos], dim=-1) 
        return final_stream.unsqueeze(2) 

# ============================================================
# PART 4: SYSTEM WRAPPER (Updated for Cross-Weave)
# ============================================================

class VisionModule(nn.Module):
    def __init__(self, topology_dna, num_classes=10):
        super().__init__()
        self.retina = VisualRetina(window_size=32)
        self.brain = build_node(topology_dna)
        
        # [NEW] The "Summary Atom" (Cross-Weave)
        # We warp the variable timeline into a fixed narrative of 8 steps.
        self.summarizer = Atom(out_segments=8)
        
        # The Realizer (Observer)
        # Resolution 4 means we get 4 sample points per spline segment.
        self.realizer = SplineRealizer(resolution=4)
        
        # Head Input Calculation:
        # Features = (Brain_Segments) * (Summary_Steps) * (Resolution)
        # We assume the brain topology output is 16 segments (based on DNA below).
        # 16 * 8 * 4 = 512 features.
        self.head = nn.Sequential(
            nn.Flatten(), 
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        # 1. Sensation
        stream = self.retina(x)      # (B, T, 1, 6)
        
        # 2. Perception (Brain Processing)
        thought = self.brain(stream) # (B, T, N, 6)
        
        # 3. Cross-Weave (Transpose & Summarize)
        # Swap Time (1) and Segments (2) -> (B, N, T, 6)
        transposed_thought = thought.permute(0, 2, 1, 3) 
        
        # Warp the variable T into fixed 8 steps
        summary = self.summarizer(transposed_thought) # (B, N, 8, 6)
        
        # 4. Realize (Sample the Fixed Summary)
        # Returns (B, N, 8, Resolution)
        concept = self.realizer(summary) 
        
        # 5. Classify
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
        plt.title('Vision V11.1 (Cross-Weave)')
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
    
    num_segments = 16
    attention_block = [
        "PARALLEL", "PRODUCT", [ 
            [
                "PARALLEL", "PRODUCT", [ 
                    ["LEAF", None, num_segments], 
                    ["LEAF", None, num_segments]  
                ]
            ],
            ["LEAF", None, num_segments] 
        ]
    ]

    model_topo = [
        "SERIES", None, [
            attention_block,
            attention_block,
            ["LEAF", None, num_segments]
        ]
    ]
    
    print("Initializing Vision V11.1 (Cross-Weave)...")
    model = VisionModule(model_topo)
    plotter = TrainingPlotter()
    
    model = train_vision_model(model, epochs=20, plotter=plotter)
