"""
FoldFlow Classifier — Image Classification via Energy Minimization.

Architectural fixes over v1:
1. Cooperativity: Lightweight 1D depthwise conv (true local interactions),
   replaces 2-layer full MHA that was harmful at small scale.
2. Chaperone: Stress-gated intervention using ||∇E|| as described in the
   paper's mathematics. Replaces the broken 0.1*softmax scaling.
3. No context skip: All discriminative info flows through particle dynamics.

Supports ablation via boolean flags for each of the 5 characteristics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dynamics.langevin import LangevinDynamics


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class WeakEncoder(nn.Module):
    """Lightweight encoder for ablation — 3 conv layers, no ResNet blocks.

    Deliberately weak so FoldFlow dynamics must carry discriminative burden.
    ~125K parameters.
    """

    def __init__(self, channels: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.fc(x.flatten(1))


class StrongEncoder(nn.Module):
    """ResNet-style encoder for best accuracy. ~2.5M parameters."""

    def __init__(self, channels: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
        )
        self.layer1 = nn.Sequential(
            _ResBlock(64, 64), _ResBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            _ResBlock(64, 128, stride=2), _ResBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            _ResBlock(128, 256, stride=2), _ResBlock(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.gelu(out)


# ---------------------------------------------------------------------------
# FoldFlow components
# ---------------------------------------------------------------------------

class EnergyNetwork(nn.Module):
    """Learnable scalar energy E(z | x).  z = particles, x = context."""

    def __init__(self, particle_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(particle_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, particles: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return total energy (scalar per sample)."""
        B, N, D = particles.shape
        ctx = context.unsqueeze(1).expand(-1, N, -1)
        inp = torch.cat([particles, ctx], dim=-1)
        per_particle = self.net(inp).squeeze(-1)  # (B, N)
        return per_particle.sum(dim=1)  # (B,)


class LocalCooperativity(nn.Module):
    """Local particle interactions via 1D depthwise-separable convolution.

    Replaces the 2-layer MHA + FFN that was harmful at small scale.
    Depthwise conv for local mixing + pointwise for cross-feature interaction.
    """

    def __init__(self, particle_dim: int, kernel_size: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(particle_dim)
        # Depthwise conv: local neighborhood mixing per feature
        self.dwconv = nn.Conv1d(
            particle_dim, particle_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=particle_dim,
            bias=False,
        )
        self.act = nn.GELU()
        # Pointwise mix: cross-feature interaction
        self.pwconv = nn.Sequential(
            nn.Linear(particle_dim, particle_dim * 2),
            nn.GELU(),
            nn.Linear(particle_dim * 2, particle_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """particles: (B, N, D) -> (B, N, D)"""
        residual = particles
        x = self.norm(particles)
        # Conv operates on (B, D, N) — local neighborhood mixing
        x = self.act(self.dwconv(x.transpose(1, 2)).transpose(1, 2))
        x = self.pwconv(x)
        return residual + self.scale * x


class StressGatedChaperone(nn.Module):
    """Chaperone that intervenes proportionally to folding stress ||∇E||.

    High stress → system is far from equilibrium → stronger correction.
    Low stress → near equilibrium → minimal intervention.
    """

    def __init__(self, particle_dim: int, hidden_dim: int):
        super().__init__()
        # Stress-conditioned intervention network
        self.net = nn.Sequential(
            nn.Linear(particle_dim + hidden_dim + 1, 128),
            nn.GELU(),
            nn.Linear(128, particle_dim),
            nn.Tanh(),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        particles: torch.Tensor,
        context: torch.Tensor,
        stress: torch.Tensor,
    ) -> torch.Tensor:
        """Apply stress-gated correction.

        Args:
            particles: (B, N, D)
            context: (B, hidden_dim)
            stress: (B,) — mean ||∇E|| from Langevin step
        """
        B, N, D = particles.shape
        # Pool particles for global state
        pooled = particles.mean(dim=1)  # (B, D)
        # Normalize stress to [0, 1] range with sigmoid
        stress_gate = torch.sigmoid(stress).unsqueeze(-1)  # (B, 1)
        inp = torch.cat([pooled, context, stress_gate], dim=-1)  # (B, D+H+1)
        correction = self.net(inp).unsqueeze(1)  # (B, 1, D)
        # Scale correction by stress: more stress → more correction
        return particles + self.scale * stress_gate.unsqueeze(-1) * correction


class DynamicTopology(nn.Module):
    """Input-conditioned connectivity via multi-head attention."""

    def __init__(self, particle_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(particle_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=particle_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        x = self.norm(particles)
        out, _ = self.attn(x, x, x)
        return particles + out


class EnvironmentSensitivity(nn.Module):
    """Context-conditioned modulation of particle states."""

    def __init__(self, hidden_dim: int, particle_dim: int):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(hidden_dim, particle_dim), nn.Softplus(),
        )
        self.shift_net = nn.Linear(hidden_dim, particle_dim)

    def forward(self, particles: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale = self.scale_net(context).unsqueeze(1) + 0.5  # (B, 1, D)
        shift = self.shift_net(context).unsqueeze(1)  # (B, 1, D)
        return particles * scale + shift


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class FoldFlowClassifier(nn.Module):
    """FoldFlow classifier with ablation flags for each characteristic.

    Args:
        num_classes: Number of output classes.
        hidden_dim: Encoder output / context dimension.
        num_particles: Number of latent particles.
        particle_dim: Dimension of each particle.
        num_steps: Number of Langevin dynamics steps.
        channels: Input image channels.
        encoder_type: 'weak' or 'strong'.
        use_energy: Enable Langevin energy minimization (C1).
        use_topology: Enable dynamic topology (C2).
        use_cooperativity: Enable cooperative local interactions (C3).
        use_chaperone: Enable stress-gated chaperone (C4).
        use_env_sensitivity: Enable environmental sensitivity (C5).
    """

    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_particles: int = 64,
        particle_dim: int = 64,
        num_steps: int = 8,
        channels: int = 3,
        dropout: float = 0.1,
        encoder_type: str = "weak",
        # Ablation flags
        use_energy: bool = True,
        use_topology: bool = True,
        use_cooperativity: bool = True,
        use_chaperone: bool = True,
        use_env_sensitivity: bool = True,
    ):
        super().__init__()
        self.num_particles = num_particles
        self.particle_dim = particle_dim
        self.num_steps = num_steps

        # Flags
        self.use_energy = use_energy
        self.use_topology = use_topology
        self.use_cooperativity = use_cooperativity
        self.use_chaperone = use_chaperone
        self.use_env_sensitivity = use_env_sensitivity

        # Encoder
        if encoder_type == "weak":
            self.encoder = WeakEncoder(channels, hidden_dim)
        else:
            self.encoder = StrongEncoder(channels, hidden_dim)

        # Particle initialiser
        self.particle_init = nn.Linear(hidden_dim, num_particles * particle_dim)

        # C1: Energy minimization
        self.energy_net = EnergyNetwork(particle_dim, hidden_dim)
        self.langevin = LangevinDynamics(hidden_dim)

        # C2: Dynamic topology
        self.topology = DynamicTopology(particle_dim, num_heads=4, dropout=dropout)

        # C3: Cooperativity (lightweight local conv)
        self.cooperativity = LocalCooperativity(particle_dim, kernel_size=5)

        # C4: Chaperone (stress-gated)
        self.chaperone = StressGatedChaperone(particle_dim, hidden_dim)

        # C5: Environmental sensitivity (applied at init AND during dynamics)
        self.env_sensitivity = EnvironmentSensitivity(hidden_dim, particle_dim)
        self.env_dynamics = EnvironmentSensitivity(hidden_dim, particle_dim)

        # Readout — NO skip from context
        self.readout = nn.Sequential(
            nn.Linear(particle_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Auxiliary classifier for intermediate supervision
        self.aux_classifier = nn.Linear(particle_dim, num_classes)

        # Temperature schedule: anneal from 1.0 → 0.1
        self.register_buffer("temp_schedule", torch.linspace(1.0, 0.1, num_steps))

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]

        # Encode
        context = self.encoder(x)

        # Initialise particles
        particles = self.particle_init(context).view(B, self.num_particles, self.particle_dim)

        # C5: Environmental sensitivity (modulate initial particles)
        if self.use_env_sensitivity:
            particles = self.env_sensitivity(particles, context)

        # C2: Dynamic topology (input-conditioned connectivity)
        if self.use_topology:
            particles = self.topology(particles)

        # Dynamics loop
        run_dynamics = self.use_energy or self.use_cooperativity or self.use_chaperone
        aux_logits = []
        last_stress = torch.zeros(B, device=x.device)

        if run_dynamics:
            for step in range(self.num_steps):
                temp = self.temp_schedule[step].item()

                # C3: Local cooperative interactions
                if self.use_cooperativity:
                    particles = self.cooperativity(particles)

                # C1: Langevin energy minimization
                if self.use_energy:
                    particles, energy, grad_norm = self.langevin.step(
                        particles,
                        self.energy_net,
                        context,
                        temp,
                    )
                    last_stress = grad_norm

                # Auxiliary prediction (only at last step to save compute)
                if self.training and step == self.num_steps - 1:
                    aux = self.aux_classifier(particles.mean(dim=1))
                    aux_logits.append(aux)

                # C4: Stress-gated chaperone (every other step)
                if self.use_chaperone and step % 2 == 1:
                    particles = self.chaperone(particles, context, last_stress)

                # C5: Environmental modulation during dynamics (every 3rd step)
                if self.use_env_sensitivity and step % 3 == 2:
                    particles = self.env_dynamics(particles, context)

        # Readout
        pooled = particles.mean(dim=1)
        features = self.readout(pooled)
        logits = self.classifier(features)

        result = {"logits": logits}
        if self.training and aux_logits:
            result["aux_logits"] = aux_logits
        return result
