"""
FNO-based Transport Neural Operator
=====================================
FFT spectral convolution baseline with a discretization-agnostic angular query head.

Architecture:
  1. Input lifting: [sigma_a, sigma_s, q, BC, params] -> channels on spatial grid
  2. FNO blocks: spectral convolution on uniform grid (2D or 3D)
  3. Angular query head: (latent_features_at_x, omega, [t]) -> I(x, omega)
     This makes the model discretization-agnostic: omega can be any set at eval time.

The model predicts I(x, omega) for arbitrary (x, omega) combinations.
Moments phi and J are computed via quadrature of predicted I.
"""

from __future__ import annotations
import math
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common import (
    FourierFeatures, SphericalFourierFeatures, MLP, FNOBlock2d, FNOBlock3d,
    compute_moments, ParamEncoder
)


class FNOTransport(nn.Module):
    """
    FNO baseline for transport problems.

    Input fields are processed on a uniform spatial grid via FNO blocks.
    An angular query head then evaluates I at arbitrary (x, omega) pairs.
    """

    def __init__(
        self,
        dim: int = 2,
        n_groups: int = 1,
        n_params: int = 2,
        fno_channels: int = 32,
        n_fno_blocks: int = 4,
        n_modes: int = 12,
        query_head_hidden: List[int] = None,
        n_freq_x: int = 16,
        n_freq_omega: int = 8,
        activation: str = "gelu",
        time_dependent: bool = False,
        n_bc_faces: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.fno_channels = fno_channels
        self.time_dependent = time_dependent

        if query_head_hidden is None:
            query_head_hidden = [128, 128]

        # --- Input channels ---
        # Per-group: sigma_a (G), sigma_s (G), q (G)
        # BC encoding: n_bc_faces * 2
        # Positional: 2*n_freq_x (Fourier x features)
        # Params: encoded as scalar via ParamEncoder -> fno_channels
        n_field_channels = 3 * n_groups + n_bc_faces * 2 + 2 * n_freq_x
        if time_dependent:
            n_field_channels += 1  # time channel

        self.x_pos_enc = FourierFeatures(dim, n_freq=n_freq_x, sigma=1.0)
        self.param_enc = ParamEncoder(
            param_keys=["epsilon", "g"] if n_params >= 2 else ["epsilon"],
            out_dim=fno_channels,
        )

        # Lifting layer: field channels -> fno_channels
        self.lift = nn.Linear(n_field_channels, fno_channels)

        # FNO blocks
        if dim == 2:
            self.fno_blocks = nn.ModuleList([
                FNOBlock2d(fno_channels, n_modes, n_modes, activation)
                for _ in range(n_fno_blocks)
            ])
        elif dim == 3:
            n_modes_3d = max(4, n_modes // 2)  # fewer 3D modes for memory
            self.fno_blocks = nn.ModuleList([
                FNOBlock3d(fno_channels, (n_modes, n_modes, n_modes_3d), activation)
                for _ in range(n_fno_blocks)
            ])
        else:
            raise ValueError(f"FNO only supports dim=2 or dim=3, got {dim}")

        # --- Angular query head ---
        # Inputs: latent features at x (fno_channels) + omega Fourier features + [t]
        self.omega_enc = SphericalFourierFeatures(dim, n_harmonics=n_freq_omega)
        head_in = fno_channels + self.omega_enc.out_dim
        if time_dependent:
            head_in += 2 * n_freq_x  # time Fourier features

        self.angular_head = MLP(
            in_dim=head_in,
            out_dim=n_groups,
            hidden_dims=query_head_hidden,
            activation=activation,
        )

        self.n_field_channels = n_field_channels
        self.n_modes = n_modes

    def encode_inputs(
        self,
        sigma_a: Tensor,
        sigma_s: Tensor,
        q: Tensor,
        bc: Tensor,
        params: Tensor,
        x: Tensor,
        t: Optional[Tensor] = None,
        batch_param_keys: Optional[List[str]] = None,
    ) -> Tensor:
        """
        Build per-cell feature vector and lift to FNO channel space.

        Args:
            sigma_a, sigma_s, q: [B, Nx, G]
            bc: [B, n_bc_faces, 2]
            params: [B, n_params]
            x: [B, Nx, dim]
            t: [B] or [B, 1] optional time

        Returns:
            features: [B, Nx, fno_channels]
        """
        B, Nx, G = sigma_a.shape

        x_pos = self.x_pos_enc(x)  # [B, Nx, 2*n_freq_x]

        # Broadcast BC: [B, n_bc_faces*2] -> [B, Nx, n_bc_faces*2]
        bc_flat = bc.reshape(B, -1).unsqueeze(1).expand(B, Nx, -1)

        # Broadcast params encoding: [B, fno_channels] -> will be used as bias
        # (done after lift)
        # Here we just concat a summary scalar
        eps = params[:, 0:1].unsqueeze(1).expand(B, Nx, 1)  # epsilon
        g = params[:, 1:2].unsqueeze(1).expand(B, Nx, 1) if params.shape[1] > 1 else torch.zeros(B, Nx, 1, device=params.device)

        parts = [sigma_a, sigma_s, q, bc_flat, x_pos]
        if self.time_dependent and t is not None:
            t_broadcast = t.view(B, 1, 1).expand(B, Nx, 1)
            parts.append(t_broadcast)

        features = torch.cat(parts, dim=-1)  # [B, Nx, n_field_channels]
        features = self.lift(features)        # [B, Nx, fno_channels]

        # Add param encoding as spatial bias
        param_feat = self.param_enc(params, batch_param_keys).unsqueeze(1)  # [B, 1, fno_channels]
        features = features + param_feat

        return features

    def apply_fno(self, features: Tensor, spatial_shape: List[int]) -> Tensor:
        """
        Apply FNO blocks to features on the spatial grid.

        Args:
            features: [B, Nx, fno_channels] (Nx = prod(spatial_shape))
            spatial_shape: list of grid dimensions

        Returns:
            [B, Nx, fno_channels]
        """
        B, Nx, C = features.shape

        if self.dim == 2:
            H, W = spatial_shape
            x = features.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
            for block in self.fno_blocks:
                x = block(x)
            return x.permute(0, 2, 3, 1).reshape(B, Nx, C)  # [B, Nx, C]
        elif self.dim == 3:
            D1, D2, D3 = spatial_shape
            x = features.reshape(B, D1, D2, D3, C).permute(0, 4, 1, 2, 3)
            for block in self.fno_blocks:
                x = block(x)
            return x.permute(0, 2, 3, 4, 1).reshape(B, Nx, C)

    def query_intensity(
        self,
        latent: Tensor,
        omega: Tensor,
        omega_mask: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Evaluate I(x, omega) from latent features.

        Args:
            latent: [B, Nx, fno_channels]
            omega: [B, Nw, dim] (variable Nw, discretization-agnostic)
            omega_mask: [B, Nw] boolean mask
            t: [B] or None

        Returns:
            I: [B, Nx, Nw, G]
        """
        B, Nx, C = latent.shape
        Nw = omega.shape[1]

        omega_feat = self.omega_enc(omega)  # [B, Nw, omega_feat_dim]

        # Expand latent for each omega: [B, Nx, Nw, C]
        latent_exp = latent.unsqueeze(2).expand(B, Nx, Nw, C)

        # Expand omega for each x: [B, Nx, Nw, omega_feat_dim]
        omega_exp = omega_feat.unsqueeze(1).expand(B, Nx, Nw, -1)

        head_input = torch.cat([latent_exp, omega_exp], dim=-1)

        if self.time_dependent and t is not None:
            # Simple time encoding: [sin(2pi*t*k), cos(2pi*t*k)] for k=1..n_freq
            t_enc = self._encode_time(t, latent_exp.shape[-1] // 2, B, Nx, Nw, latent.device)
            head_input = torch.cat([head_input, t_enc], dim=-1)

        I = self.angular_head(head_input)  # [B, Nx, Nw, G]
        I = F.softplus(I)  # ensure positivity

        if omega_mask is not None:
            # Zero out padded directions
            mask = omega_mask.unsqueeze(1).unsqueeze(-1).expand_as(I)
            I = I * mask.float()

        return I

    def _encode_time(self, t: Tensor, n_freq: int, B: int, Nx: int, Nw: int, device) -> Tensor:
        """Simple Fourier time encoding."""
        t_val = t.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        ks = torch.arange(1, n_freq + 1, device=device, dtype=t.dtype).view(1, 1, 1, n_freq)
        t_enc = torch.cat([
            torch.sin(2 * math.pi * ks * t_val),
            torch.cos(2 * math.pi * ks * t_val),
        ], dim=-1)  # [B, 1, 1, 2*n_freq]
        return t_enc.expand(B, Nx, Nw, -1)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Forward pass.

        Args:
            batch: dict with keys sigma_a, sigma_s, q, bc, params, x, omega, w_omega,
                   omega_mask, spatial_shape, [t]

        Returns:
            dict with: I, phi, J
        """
        sigma_a = batch["sigma_a"]   # [B, Nx, G]
        sigma_s = batch["sigma_s"]
        q = batch["q"]
        bc = batch["bc"]             # [B, n_bc_faces, 2]
        params = batch["params"]     # [B, P]
        param_keys = batch.get("param_keys")
        x = batch["x"]               # [B, Nx, dim]
        omega = batch["omega"]       # [B, Nw, dim]
        w_omega = batch["w_omega"]   # [B, Nw]
        omega_mask = batch.get("omega_mask")  # [B, Nw]
        t = batch.get("t")
        spatial_shape = batch["spatial_shape"]

        # Encode inputs
        features = self.encode_inputs(sigma_a, sigma_s, q, bc, params, x, t, batch_param_keys=param_keys)

        # FNO on grid
        latent = self.apply_fno(features, spatial_shape)  # [B, Nx, C]

        # Angular query head
        I = self.query_intensity(latent, omega, omega_mask, t)  # [B, Nx, Nw, G]

        # Compute moments via quadrature
        phi, J = compute_moments(I, omega, w_omega, omega_mask)

        return {"I": I, "phi": phi, "J": J}
