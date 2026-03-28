"""
Langevin dynamics for energy minimization.

Implements:
    dz/dt = -∇_z E(z|x) + √(2T) · η(t)

with adaptive step size and temperature annealing.
"""

import math
import torch
import torch.nn as nn


class LangevinDynamics(nn.Module):
    """Langevin dynamics module with learned adaptive step size."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.step_size_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        self.base_step = 0.1

    def get_step_size(self, context: torch.Tensor) -> torch.Tensor:
        """Compute input-adaptive step size from context."""
        return self.base_step * (0.5 + self.step_size_net(context))

    def step(
        self,
        particles: torch.Tensor,
        energy_fn,
        context: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One Langevin dynamics step.

        Returns:
            (new_particles, energy, grad_norm)
            grad_norm = ||∇_z E|| averaged over batch, for stress-gated chaperone.
        """
        step_size = self.get_step_size(context)  # (B, 1)

        with torch.enable_grad():
            z = particles.detach().requires_grad_(True)
            energy = energy_fn(z, context)
            grad = torch.autograd.grad(
                energy.sum(), z, create_graph=particles.requires_grad
            )[0]

        # Gradient norm per sample for chaperone stress signal
        grad_norm = grad.detach().norm(dim=-1).mean(dim=-1)  # (B,)

        dt = step_size.unsqueeze(1) * temperature
        noise_scale = math.sqrt(2.0 * temperature) * 0.01
        noise = torch.randn_like(particles) * noise_scale
        new_particles = particles - dt * grad + noise

        return new_particles, energy.detach(), grad_norm
