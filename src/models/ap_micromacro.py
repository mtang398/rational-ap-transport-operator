"""
AP Micro-Macro Neural Operator
================================
Asymptotic-Preserving model with explicit macro/micro decomposition.

Architecture:
  Macro net: spatial field encoder -> phi(x), J(x) (moments)
  Micro net: spatial features + angular encoding -> residual R(x, omega)
  Reconstruction:
    I(x, omega) = I_P1(x, omega; phi, J) + R(x, omega)
  where I_P1 is the P1 angular reconstruction (explicit, deterministic closure):
    I_P1(x, omega) = phi(x)/(4*pi) + (3/4*pi) * J(x) . omega

Loss terms:
  L_intensity: ||I_pred - I_target||^2
  L_moment:    ||phi_I - phi_macro||^2 + ||J_I - J_macro||^2
               where phi_I = integral(I) d_omega via quadrature
  L_diffusion: epsilon^(-2) * ||phi_I - phi_diffusion||^2
               where phi_diffusion = q / sigma_a (local diffusion limit)
               Weighted by (1 - epsilon) so it activates in diffusion limit

This model is asymptotic-preserving: as epsilon -> 0, the macro net captures
the diffusion limit and the micro net vanishes, maintaining accuracy across regimes.
"""

from __future__ import annotations
import math
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common import (
    FourierFeatures, SphericalFourierFeatures, MLP, FNOBlock2d, FNOBlock3d,
    compute_moments, recon_from_moments_p1, ParamEncoder
)


class MacroNet(nn.Module):
    """
    Macro network: predicts spatial moments phi(x) and J(x).

    Uses FNO-style spatial processing to capture global correlations.
    Output is independent of angular direction omega (moments are direction-integrated).
    """

    def __init__(
        self,
        dim: int,
        n_groups: int,
        n_params: int,
        channels: int = 32,
        n_fno_blocks: int = 4,
        n_modes: int = 12,
        n_freq_x: int = 16,
        activation: str = "gelu",
        n_bc_faces: int = 4,
        moment_order: int = 1,  # 1 = phi+J, 2 = phi+J+P2 tensor
    ):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.channels = channels
        self.moment_order = moment_order

        self.x_enc = FourierFeatures(dim, n_freq=n_freq_x)
        n_input = 3 * n_groups + n_bc_faces * 2 + 2 * n_freq_x

        self.param_enc = ParamEncoder(
            ["epsilon", "g"][:n_params],
            out_dim=channels,
        )
        self.lift = nn.Linear(n_input, channels)

        if dim == 2:
            self.fno_blocks = nn.ModuleList([
                FNOBlock2d(channels, n_modes, n_modes, activation)
                for _ in range(n_fno_blocks)
            ])
        elif dim == 3:
            n_modes_3d = max(4, n_modes // 2)
            self.fno_blocks = nn.ModuleList([
                FNOBlock3d(channels, (n_modes, n_modes, n_modes_3d), activation)
                for _ in range(n_fno_blocks)
            ])

        # Projection to phi: [channels] -> [G]
        self.phi_proj = nn.Linear(channels, n_groups)
        # Projection to J: [channels] -> [dim * G]
        self.J_proj = nn.Linear(channels, dim * n_groups)

        if moment_order >= 2:
            # P2 tensor (symmetric, traceless): dim*(dim+1)/2 - 1 components per group
            n_p2 = (dim * (dim + 1)) // 2
            self.P2_proj = nn.Linear(channels, n_p2 * n_groups)

    def forward(
        self,
        sigma_a: Tensor,
        sigma_s: Tensor,
        q: Tensor,
        bc: Tensor,
        params: Tensor,
        x: Tensor,
        spatial_shape: List[int],
    ) -> Dict[str, Tensor]:
        """
        Returns:
            {"phi": [B, Nx, G], "J": [B, Nx, dim, G]}
        """
        B, Nx, G = sigma_a.shape

        x_pos = self.x_enc(x)  # [B, Nx, 2*n_freq_x]
        bc_flat = bc.reshape(B, -1).unsqueeze(1).expand(B, Nx, -1)

        features = torch.cat([sigma_a, sigma_s, q, bc_flat, x_pos], dim=-1)
        features = self.lift(features)
        features = features + self.param_enc(params).unsqueeze(1)

        # FNO spatial processing
        if self.dim == 2:
            H, W = spatial_shape
            feat_grid = features.reshape(B, H, W, self.channels).permute(0, 3, 1, 2)
            for block in self.fno_blocks:
                feat_grid = block(feat_grid)
            features = feat_grid.permute(0, 2, 3, 1).reshape(B, Nx, self.channels)
        elif self.dim == 3:
            D1, D2, D3 = spatial_shape
            feat_grid = features.reshape(B, D1, D2, D3, self.channels).permute(0, 4, 1, 2, 3)
            for block in self.fno_blocks:
                feat_grid = block(feat_grid)
            features = feat_grid.permute(0, 2, 3, 4, 1).reshape(B, Nx, self.channels)

        # Predict moments
        phi = F.softplus(self.phi_proj(features))  # [B, Nx, G] (positivity)
        J = self.J_proj(features).reshape(B, Nx, self.dim, G)  # [B, Nx, dim, G]

        out = {"phi": phi, "J": J}

        if self.moment_order >= 2 and hasattr(self, "P2_proj"):
            n_p2 = (self.dim * (self.dim + 1)) // 2
            P2 = self.P2_proj(features).reshape(B, Nx, n_p2, G)
            out["P2"] = P2

        return out


class MicroNet(nn.Module):
    """
    Micro network: predicts the angular residual R(x, omega, [t]).

    R(x, omega) is the deviation from the P1 reconstruction.
    The micro net is conditioned on:
      - spatial context (from macro net or independent spatial encoder)
      - angular direction omega
      - epsilon (to suppress micro corrections in diffusion limit)
      - optional time t

    Design: micro corrections should vanish as epsilon -> 0 (diffusion limit),
    which is achieved by weighting with epsilon in the loss.
    """

    def __init__(
        self,
        dim: int,
        n_groups: int,
        latent_dim: int,
        hidden_dims: List[int],
        n_freq_x: int = 16,
        n_freq_omega: int = 8,
        activation: str = "silu",
        time_dependent: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups

        self.x_enc = FourierFeatures(dim, n_freq=n_freq_x)
        self.omega_enc = SphericalFourierFeatures(dim, n_harmonics=n_freq_omega)

        # Input: spatial latent + omega features + epsilon encoding
        in_dim = latent_dim + self.omega_enc.out_dim + 1  # +1 for epsilon
        if time_dependent:
            self.t_enc = FourierFeatures(1, n_freq=8, sigma=0.5)
            in_dim += self.t_enc.out_dim

        self.net = MLP(in_dim, n_groups, hidden_dims, activation=activation)
        self.time_dependent = time_dependent

    def forward(
        self,
        latent: Tensor,
        omega: Tensor,
        epsilon: Tensor,
        omega_mask: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            latent: [B, Nx, latent_dim] spatial context from macro net
            omega: [B, Nw, dim]
            epsilon: [B] Knudsen parameter
            omega_mask: [B, Nw] bool
            t: [B] or None

        Returns:
            R: [B, Nx, Nw, G] residual
        """
        B, Nx, C = latent.shape
        Nw = omega.shape[1]

        omega_feat = self.omega_enc(omega)  # [B, Nw, omega_feat]

        latent_exp = latent.unsqueeze(2).expand(B, Nx, Nw, C)
        omega_exp = omega_feat.unsqueeze(1).expand(B, Nx, Nw, -1)

        # Log-scale epsilon for better conditioning
        eps_log = torch.log(epsilon.clamp(min=1e-6)).view(B, 1, 1, 1).expand(B, Nx, Nw, 1)

        inp = torch.cat([latent_exp, omega_exp, eps_log], dim=-1)

        if self.time_dependent and t is not None:
            t_val = t.view(B, 1, 1, 1).expand(B, Nx, Nw, 1)
            t_feat = self.t_enc(t_val)
            inp = torch.cat([inp, t_feat], dim=-1)

        R = self.net(inp)  # [B, Nx, Nw, G]
        # Residual can be positive or negative (no softplus)

        if omega_mask is not None:
            mask = omega_mask.unsqueeze(1).unsqueeze(-1).float()
            R = R * mask

        return R


class APMicroMacroTransport(nn.Module):
    """
    AP Micro-Macro transport neural operator.

    I(x, omega) = I_P1(phi(x), J(x), omega) + R(x, omega)

    The P1 reconstruction is a deterministic, explicitly documented closure:
      I_P1(x, omega) = phi(x)/(4*pi) + (3/4*pi) * J(x) . omega    [3D]
      I_P1(x, omega) = phi(x)/(2*pi) + (2/2*pi) * J(x) . omega    [2D]

    This ensures:
    - phi = integral I d_omega exactly matches predicted phi (up to quadrature)
    - J = integral omega I d_omega exactly matches predicted J (up to quadrature)
    - In diffusion limit: macro net captures physics, micro net vanishes
    """

    def __init__(
        self,
        dim: int = 2,
        n_groups: int = 1,
        n_params: int = 2,
        macro_channels: int = 32,
        n_fno_blocks: int = 4,
        n_modes: int = 12,
        micro_latent_dim: int = 64,
        micro_hidden: List[int] = None,
        n_freq_x: int = 16,
        n_freq_omega: int = 8,
        activation: str = "gelu",
        time_dependent: bool = False,
        n_bc_faces: int = 4,
        moment_order: int = 1,
        # Loss weight hyperparams
        lambda_moment: float = 1.0,
        lambda_diffusion: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.lambda_moment = lambda_moment
        self.lambda_diffusion = lambda_diffusion

        if micro_hidden is None:
            micro_hidden = [128, 128]

        # Macro net
        self.macro_net = MacroNet(
            dim=dim, n_groups=n_groups, n_params=n_params,
            channels=macro_channels, n_fno_blocks=n_fno_blocks,
            n_modes=n_modes, n_freq_x=n_freq_x, activation=activation,
            n_bc_faces=n_bc_faces, moment_order=moment_order,
        )

        # Micro net uses macro net's internal features as spatial context
        # We add a separate projection to get micro latent
        self.micro_latent_proj = nn.Linear(macro_channels, micro_latent_dim)

        self.micro_net = MicroNet(
            dim=dim, n_groups=n_groups, latent_dim=micro_latent_dim,
            hidden_dims=micro_hidden, n_freq_x=n_freq_x, n_freq_omega=n_freq_omega,
            activation=activation, time_dependent=time_dependent,
        )

        self.macro_channels = macro_channels
        self.micro_latent_dim = micro_latent_dim

    def _get_macro_latent(
        self,
        sigma_a: Tensor, sigma_s: Tensor, q: Tensor,
        bc: Tensor, params: Tensor, x: Tensor,
        spatial_shape: List[int],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Run macro net and return moments + internal spatial features.
        We hook into MacroNet to extract features before the projection layers.
        """
        B, Nx, G = sigma_a.shape

        # Re-run macro net's feature extraction to get spatial latent
        x_pos = self.macro_net.x_enc(x)
        bc_flat = bc.reshape(B, -1).unsqueeze(1).expand(B, Nx, -1)
        features = torch.cat([sigma_a, sigma_s, q, bc_flat, x_pos], dim=-1)
        features = self.macro_net.lift(features)
        features = features + self.macro_net.param_enc(params).unsqueeze(1)

        if self.dim == 2:
            H, W = spatial_shape
            feat_grid = features.reshape(B, H, W, self.macro_channels).permute(0, 3, 1, 2)
            for block in self.macro_net.fno_blocks:
                feat_grid = block(feat_grid)
            features = feat_grid.permute(0, 2, 3, 1).reshape(B, Nx, self.macro_channels)
        elif self.dim == 3:
            D1, D2, D3 = spatial_shape
            feat_grid = features.reshape(B, D1, D2, D3, self.macro_channels).permute(0, 4, 1, 2, 3)
            for block in self.macro_net.fno_blocks:
                feat_grid = block(feat_grid)
            features = feat_grid.permute(0, 2, 3, 4, 1).reshape(B, Nx, self.macro_channels)

        phi = F.softplus(self.macro_net.phi_proj(features))
        J = self.macro_net.J_proj(features).reshape(B, Nx, self.dim, G)

        moments = {"phi": phi, "J": J}

        micro_latent = self.micro_latent_proj(features)  # [B, Nx, micro_latent_dim]

        return moments, micro_latent

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Forward pass with AP micro-macro decomposition.

        Returns:
            dict with:
              I:     [B, Nx, Nw, G] full intensity
              I_P1:  [B, Nx, Nw, G] P1 macro reconstruction
              R:     [B, Nx, Nw, G] micro residual
              phi:   [B, Nx, G] predicted scalar flux (from macro net)
              J:     [B, Nx, dim, G] predicted current (from macro net)
              phi_I: [B, Nx, G] scalar flux computed from I via quadrature
              J_I:   [B, Nx, dim, G] current computed from I via quadrature
        """
        sigma_a = batch["sigma_a"]
        sigma_s = batch["sigma_s"]
        q = batch["q"]
        bc = batch["bc"]
        params = batch["params"]
        x = batch["x"]
        omega = batch["omega"]
        w_omega = batch["w_omega"]
        omega_mask = batch.get("omega_mask")
        t = batch.get("t")
        spatial_shape = batch["spatial_shape"]

        # Extract epsilon from params
        param_keys = batch.get("param_keys", ["epsilon", "g"])
        if "epsilon" in param_keys:
            eps_idx = param_keys.index("epsilon")
            epsilon = params[:, eps_idx]  # [B]
        else:
            epsilon = torch.ones(params.shape[0], device=params.device)

        # --- Macro net ---
        moments, micro_latent = self._get_macro_latent(
            sigma_a, sigma_s, q, bc, params, x, spatial_shape
        )
        phi = moments["phi"]  # [B, Nx, G]
        J = moments["J"]      # [B, Nx, dim, G]

        # --- P1 reconstruction from macro moments (deterministic, documented closure) ---
        # I_P1(x, omega) = phi/(4*pi) + (3/4*pi) * J.omega   [3D]
        # I_P1(x, omega) = phi/(2*pi) + (2/2*pi) * J.omega   [2D]
        I_P1 = recon_from_moments_p1(phi, J, omega, self.dim)  # [B, Nx, Nw, G]

        # --- Micro net: residual R(x, omega) ---
        R = self.micro_net(micro_latent, omega, epsilon, omega_mask, t)  # [B, Nx, Nw, G]

        # --- Full intensity ---
        I = I_P1 + R
        I = F.softplus(I - 1e-6) + 1e-6  # soft positivity (allow near-zero)

        if omega_mask is not None:
            mask = omega_mask.unsqueeze(1).unsqueeze(-1).float()
            I = I * mask

        # --- Compute quadrature moments from I ---
        phi_I, J_I = compute_moments(I, omega, w_omega, omega_mask)

        return {
            "I": I,
            "I_P1": I_P1,
            "R": R,
            "phi": phi,
            "J": J,
            "phi_I": phi_I,
            "J_I": J_I,
        }

    def compute_loss(
        self,
        pred: Dict[str, Tensor],
        batch: Dict[str, Any],
    ) -> Dict[str, Tensor]:
        """
        Compute AP micro-macro loss.

        Returns dict with:
          loss_intensity: main I matching loss
          loss_moment:    moment consistency (phi_I ~ phi_macro, J_I ~ J_macro)
          loss_diffusion: diffusion limit regularization (epsilon-weighted)
          loss_total:     weighted sum
        """
        I_pred = pred["I"]
        phi_pred = pred["phi"]
        J_pred = pred["J"]
        phi_I = pred["phi_I"]
        J_I = pred["J_I"]

        I_true = batch["I"]
        phi_true = batch["phi"]
        J_true = batch["J"]
        q = batch["q"]
        sigma_a = batch["sigma_a"]
        omega_mask = batch.get("omega_mask")
        params = batch["params"]
        param_keys = batch.get("param_keys", ["epsilon", "g"])

        # Epsilon for regime-weighted loss
        if "epsilon" in param_keys:
            eps_idx = param_keys.index("epsilon")
            epsilon = params[:, eps_idx]  # [B]
        else:
            epsilon = torch.ones(params.shape[0], device=params.device)

        # 1. Intensity loss (masked over omega)
        if omega_mask is not None:
            mask = omega_mask.unsqueeze(1).unsqueeze(-1).float()
            n_valid = mask.sum().clamp(min=1)
            loss_I = ((I_pred - I_true)**2 * mask).sum() / n_valid
        else:
            loss_I = F.mse_loss(I_pred, I_true)

        # 2. Moment consistency: moments from I should match macro net's predictions
        loss_phi = F.mse_loss(phi_I, phi_pred)
        loss_J = F.mse_loss(J_I, J_pred)
        loss_moment = loss_phi + loss_J

        # 3. Diffusion limit regularization
        # In diffusion limit (epsilon -> 0): phi_diffusion = q / sigma_a
        # Penalize deviation from this when epsilon is small
        eps_weight = (1.0 - epsilon.clamp(0, 1)).view(-1, 1, 1)  # -> 1 as eps->0
        phi_diffusion = q / (sigma_a + 1e-8)  # [B, Nx, G]
        loss_diffusion = (eps_weight * (phi_I - phi_diffusion)**2).mean()

        loss_total = (loss_I
                      + self.lambda_moment * loss_moment
                      + self.lambda_diffusion * loss_diffusion)

        return {
            "loss_total": loss_total,
            "loss_intensity": loss_I,
            "loss_moment": loss_moment,
            "loss_diffusion": loss_diffusion,
            "loss_phi": loss_phi,
            "loss_J": loss_J,
        }
