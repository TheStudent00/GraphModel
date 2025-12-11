import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# ============================================================
# PART 1: PHYSICS ENGINE (Nucleus & Shells)
# ============================================================
class Nucleus(nn.Module):
    """
    [V9.6 Physics] The Grayscale Shape Detector.
    Input: Coefficients (e.g. 576). Output: Features (e.g. 128).
    Logic: Spline( Permutation(x) ) + Noise
    """
    def __init__(self, num_coefficients, num_features, num_spline_bins=16, init_mode='random'):
        super().__init__()
        self.in_dim = num_coefficients
        self.out_dim = num_features 
        self.num_bins = num_spline_bins
        
        # Spectral Permutation (Frequency Analysis)
        self.perm_freqs = nn.Parameter(torch.randn(num_features, num_coefficients))
        self.perm_phase_shifts = nn.Parameter(torch.rand(num_features, 1)) 
        self.omega_scale = 1.0 
        
        # Monotonic Spline (Shape)
        self.spline_heights = nn.Parameter(torch.rand(num_features, num_spline_bins))
        self.spline_bias = nn.Parameter(torch.zeros(num_features))
        
        # Noise
        self.log_sigma = nn.Parameter(torch.ones(num_features) * -5.0)

        # Init
        with torch.no_grad():
            if init_mode == 'identity':
                self.perm_freqs.data.uniform_(-0.01, 0.01)
                self.spline_heights.data.fill_(1.0 / num_spline_bins)
            elif init_mode == 'random':
                self.perm_freqs.data.uniform_(-0.5, 0.5)
                self.spline_heights.data.uniform_(0.0, 0.1)

    def forward(self, x):
        # x: (B, S, Coeffs)
        basis_raw = F.linear(x, self.perm_freqs) + self.perm_phase_shifts.T
        basis = torch.sin(self.omega_scale * basis_raw)
        u = torch.sigmoid(basis) 
        
        w = F.softplus(self.spline_heights) 
        u_expanded = u.unsqueeze(-1)
        bin_grid = torch.linspace(0, 1, self.num_bins, device=x.device).view(1, 1, 1, -1)
        relu_basis = F.relu(u_expanded - bin_grid)
        spline_val = torch.sum(relu_basis * w.view(1, 1, self.out_dim, self.num_bins), dim=-1)
        val = spline_val + self.spline_bias.view(1, 1, self.out_dim)
        
        if self.training:
            val = val + (torch.randn_like(val) * torch.exp(self.log_sigma).view(1, 1, self.out_dim))
        return val

class OrbitalShells(nn.Module):
    """
    [V9.6 Physics] Cylindrical RoPE.
    """
    def __init__(self, dim, day_length=64): 
        super().__init__()
        self.dim = dim
        self.day_length = day_length
        self.register_buffer('inv_freq_h', 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))
        self.register_buffer('inv_freq_d', 1.0 / (100000 ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x, start_index=0):
        # x: (B, Seq, Dim)
        indices = torch.arange(start_index, start_index + x.shape[1], device=x.device).float()
        days = indices // self.day_length
        hours = indices % self.day_length
        
        angles = torch.outer(hours, self.inv_freq_h) + torch.outer(days, self.inv_freq_d)
        theta = torch.repeat_interleave(angles, 2, dim=1).unsqueeze(0)
        
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        x_rot = torch.zeros_like(x)
        # Sliced application for even/odd dimensions
        x_rot[..., 0::2] = x[..., 0::2] * cos_t[..., 0::2] - x[..., 1::2] * sin_t[..., 0::2]
        x_rot[..., 1::2] = x[..., 0::2] * sin_t[..., 1::2] + x[..., 1::2] * cos_t[..., 1::2]
        return x_rot

class Atom(nn.Module):
    def __init__(self, dim, patch_dim=None, init_mode='identity'):
        super().__init__()
        in_dim = patch_dim if patch_dim is not None else dim
        self.nucleus = Nucleus(num_coefficients=in_dim, num_features=dim, init_mode=init_mode)
        self.shells = OrbitalShells(dim)

    def forward(self, x, offset=0):
        return self.shells(self.nucleus(x), start_index=offset)

# ============================================================
# PART 2: GRAMMAR (Recursive Mixing)
# ============================================================
class MixingNode(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(1))
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, q, k, v):
        affinity = torch.matmul(q, k.transpose(-2, -1))
        weights = F.softmax(affinity * self.temperature + self.bias, dim=-1)
        out = torch.matmul(weights, v)
        return self.layer_norm(out + q)

# ============================================================
# PART 3: VISION MODULE (STL-10, Channel Stacked)
# ============================================================
class DeepCore(nn.Module):
    def __init__(self, dim, depth=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'q': Atom(dim, patch_dim=dim, init_mode='identity'),
                'k': Atom(dim, patch_dim=dim, init_mode='identity'),
                'v': Atom(dim, patch_dim=dim, init_mode='random'),
                'mixer': MixingNode(dim)
            }))
    def forward(self, x):
        curr = x
        for layer in self.layers:
            curr = layer['mixer'](
                layer['q'](curr, offset=0),
                layer['k'](curr, offset=1),
                layer['v'](curr, offset=1)
            )
        return curr

class VisionModule(nn.Module):
    def __init__(self, dim=128, patch_size=24, num_classes=10, depth=2):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        
        # Interface
        self.lens_embed = nn.Linear(4, dim)
        self.context_core = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        
        # RETINA: Input is 576 coeffs (1 channel)
        patch_dim_1ch = patch_size * patch_size * 1
        
        # Shared Atoms for R, G, B
        self.retina_k = Atom(dim, patch_dim=patch_dim_1ch, init_mode='identity')
        self.retina_v = Atom(dim, patch_dim=patch_dim_1ch, init_mode='random')
        self.lens_q = Atom(dim, patch_dim=dim, init_mode='identity')

        self.input_mixer = MixingNode(dim)
        self.deep_core = DeepCore(dim, depth=depth)
        self.head = nn.Linear(dim, num_classes)
        
        # Z-Order for 4x4 Grid
        self.register_buffer('z_indices', self._precompute_z_order(96, 96, patch_size))

    def _precompute_z_order(self, H, W, P):
        n_h, n_w = H // P, W // P
        coords = sorted([(x, y) for y in range(n_h) for x in range(n_w)], 
                        key=lambda p: self._morton_code(p[0], p[1]))
        return torch.tensor([y * n_w + x for (x, y) in coords], dtype=torch.long)

    def _morton_code(self, x, y):
        code = 0
        for i in range(16): code |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1))
        return code

    def forward(self, img):
        B, C, H, W = img.shape
        
        # 1. Interface
        scale = (H * W) / (96 * 96)
        meta = torch.cat([torch.tensor(scale).expand(B, 1).to(img.device), 
                          img.std(dim=(1,2,3), keepdim=True).squeeze().unsqueeze(1), 
                          img.mean(dim=(1,2,3), keepdim=True).squeeze().unsqueeze(1), 
                          torch.ones(B, 1).to(img.device)], dim=1)
        
        # 2. Patchify & Stack (Late Fusion)
        # Unfold -> (B, C*P*P, N)
        patches_raw = F.unfold(img, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Reshape to separate channels: (B, C, P*P, N)
        patches_reshaped = patches_raw.view(B, C, self.patch_size*self.patch_size, -1)
        
        # Transpose to (B, C, N, P*P)
        patches_transposed = patches_reshaped.permute(0, 1, 3, 2)
        
        # Apply Z-Order per channel
        patches_z = patches_transposed[:, :, self.z_indices, :]
        
        # Stack Channels: RRR...GGG...BBB...
        # Result: (B, 3*N, P*P) = (B, 48, 576)
        stream_stacked = patches_z.reshape(B, -1, self.patch_size*self.patch_size)
        
        # 3. Layer 0 (Retina)
        lens = self.context_core(self.lens_embed(meta).unsqueeze(1))
        
        q0 = self.retina_k(stream_stacked, offset=1) + lens 
        k0 = self.retina_k(stream_stacked, offset=1)
        v0 = self.retina_v(stream_stacked, offset=1)
        
        # 4. Deep Core
        state = self.input_mixer(q0, k0, v0)
        final = self.deep_core(state)
        
        return self.head(final.mean(dim=1))

# ============================================================
# PART 4: TRAINING
# ============================================================
def train():
    print("--- V9.6 Vision System (STL-10 Verified) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 0.001

    transform = transforms.Compose([
        transforms.Resize((96, 96)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    print("Downloading STL-10...")
    train_set = datasets.STL10(root='./data', split='train', download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    model = VisionModule(dim=128, patch_size=24, num_classes=10, depth=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}], Step [{i}], Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        print(f"=== Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f} ===")

if __name__ == "__main__":
    train()
