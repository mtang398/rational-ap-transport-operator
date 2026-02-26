"""
DeepONet-based Transport Neural Operator
==========================================
Branch-trunk architecture for learning the transport solution operator.

Architecture:
  Branch net: encodes input fields (sigma_a, sigma_s, q, BC, params) -> basis coefficients
  Trunk net: encodes query (x, omega, [t]) -> basis functions
  Output: I(x, omega, [t]) = sum_k branch_k * trunk_k(x, omega, t) + bias

Discretization-agnostic: trunk net evaluates at any (x, omega) pair independently,
so the model naturally handles variable-size omega sets and arbitrary spatial queries.

Moment computation: quadrature over predicted I values at provided omega set.
"""

from __future__ import annotations
import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common import (
    FourierFeatures, SphericalFourierFeatures, MLP, compute_moments, ParamEncoder
)


class BranchNet(nn.Module):
    """
    Branch network: encodes input functions (fields on grid) into latent coefficients.

    For fields on a grid, uses a CNN encoder for spatial feature extraction.
    For scalar params, uses MLP encoding.
    All features are concatenated and projected to n_basis coefficients.
    """

    def __init__(
        self,
        dim: int,
        n_groups: int,
        n_params: int,
        n_basis: int,
        hidden_dims: List[int],
        activation: str = "silu",
        n_bc_faces: int = 4,
        cnn_channels: List[int] = None,
        n_freq_x: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups

        if cnn_channels is None:
            cnn_channels = [32, 64, 64]

        # CNN spatial encoder for field data
        # Input: 3*G field channels (sigma_a, sigma_s, q)
        n_field_channels = 3 * n_groups
        self.cnn = self._build_cnn(dim, n_field_channels, cnn_channels, activation)
        self.cnn_out_dim = cnn_channels[-1]

        # BC encoder
        self.bc_encoder = MLP(
            in_dim=n_bc_faces * 2,
            out_dim=32,
            hidden_dims=[64],
            activation=activation,
        )

        # Param encoder
        param_key_list = ["epsilon", "g"][:n_params]
        self.param_enc = ParamEncoder(param_key_list, out_dim=32)

        # Final projection
        total_in = self.cnn_out_dim + 32 + 32
        self.proj = MLP(total_in, n_basis, hidden_dims, activation=activation)
        self.n_basis = n_basis

    def _build_cnn(self, dim: int, in_channels: int, channels: List[int], activation: str) -> nn.Module:
        act_map = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}
        Act = act_map.get(activation, nn.SiLU)
        layers = []
        c_in = in_channels
        for c_out in channels:
            if dim == 2:
                layers += [nn.Conv2d(c_in, c_out, 3, padding=1), Act()]
            elif dim == 3:
                layers += [nn.Conv3d(c_in, c_out, 3, padding=1), Act()]
            c_in = c_out

        if dim == 2:
            layers.append(nn.AdaptiveAvgPool2d(1))
        elif dim == 3:
            layers.append(nn.AdaptiveAvgPool3d(1))

        return nn.Sequential(*layers)

    def forward(
        self,
        sigma_a: Tensor,
        sigma_s: Tensor,
        q: Tensor,
        bc: Tensor,
        params: Tensor,
        spatial_shape: List[int],
    ) -> Tensor:
        """
        Args:
            sigma_a, sigma_s, q: [B, Nx, G]
            bc: [B, n_bc_faces, 2]
            params: [B, n_params]
            spatial_shape: list of spatial dims

        Returns:
            coefficients: [B, n_basis]
        """
        B, Nx, G = sigma_a.shape

        # Reshape fields to spatial grid for CNN
        if self.dim == 2:
            H, W = spatial_shape
            fields = torch.cat([sigma_a, sigma_s, q], dim=-1)  # [B, Nx, 3G]
            fields_grid = fields.reshape(B, H, W, 3 * G).permute(0, 3, 1, 2)  # [B, 3G, H, W]
        elif self.dim == 3:
            D1, D2, D3 = spatial_shape
            fields = torch.cat([sigma_a, sigma_s, q], dim=-1)
            fields_grid = fields.reshape(B, D1, D2, D3, 3 * G).permute(0, 4, 1, 2, 3)

        cnn_out = self.cnn(fields_grid).flatten(1)  # [B, cnn_out_dim]

        bc_feat = self.bc_encoder(bc.reshape(B, -1))  # [B, 32]
        param_feat = self.param_enc(params)             # [B, 32]

        combined = torch.cat([cnn_out, bc_feat, param_feat], dim=-1)  # [B, total_in]
        return self.proj(combined)  # [B, n_basis]


class TrunkNet(nn.Module):
    """
    Trunk network: encodes query points (x, omega, [t]) into basis functions.

    Each (x, omega, [t]) triplet is encoded independently, making the model
    discretization-agnostic: any omega set can be used at evaluation time.
    """

    def __init__(
        self,
        dim: int,
        n_basis: int,
        n_groups: int,
        hidden_dims: List[int],
        n_freq_x: int = 16,
        n_freq_omega: int = 8,
        activation: str = "silu",
        time_dependent: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_basis = n_basis
        self.n_groups = n_groups
        self.time_dependent = time_dependent

        self.x_enc = FourierFeatures(dim, n_freq=n_freq_x, sigma=1.0)
        self.omega_enc = SphericalFourierFeatures(dim, n_harmonics=n_freq_omega)

        in_dim = self.x_enc.out_dim + self.omega_enc.out_dim
        if time_dependent:
            self.t_enc = FourierFeatures(1, n_freq=8, sigma=0.5)
            in_dim += self.t_enc.out_dim

        # Output: n_basis * n_groups (one trunk vector per output channel)
        self.net = MLP(in_dim, n_basis * n_groups, hidden_dims, activation=activation,
                       final_activation=True)

    def forward(
        self,
        x: Tensor,
        omega: Tensor,
        t: Optional[Tensor] = None,
        omega_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: [B, Nx, dim]
            omega: [B, Nw, dim]
            t: [B] or None
            omega_mask: [B, Nw] bool

        Returns:
            trunk: [B, Nx, Nw, n_basis * n_groups]
        """
        B, Nx, _ = x.shape
        Nw = omega.shape[1]

        x_feat = self.x_enc(x)            # [B, Nx, x_feat_dim]
        omega_feat = self.omega_enc(omega)  # [B, Nw, omega_feat_dim]

        # Cross-product: combine each (x, omega) pair
        x_exp = x_feat.unsqueeze(2).expand(B, Nx, Nw, -1)     # [B, Nx, Nw, x_feat]
        omega_exp = omega_feat.unsqueeze(1).expand(B, Nx, Nw, -1)  # [B, Nx, Nw, omega_feat]

        inp = torch.cat([x_exp, omega_exp], dim=-1)  # [B, Nx, Nw, x_feat+omega_feat]

        if self.time_dependent and t is not None:
            t_val = t.view(B, 1, 1, 1).expand(B, Nx, Nw, 1)
            t_feat = self.t_enc(t_val)  # [B, Nx, Nw, t_feat_dim]
            inp = torch.cat([inp, t_feat], dim=-1)

        trunk = self.net(inp)  # [B, Nx, Nw, n_basis * G]
        return trunk


class DeepONetTransport(nn.Module):
    """
    DeepONet baseline for transport.

    Combines branch (input encoding) and trunk (query encoding) networks.
    I(x, omega) = sum_k branch_k * trunk_k(x, omega) + bias_k
    """

    def __init__(
        self,
        dim: int = 2,
        n_groups: int = 1,
        n_params: int = 2,
        n_basis: int = 128,
        branch_hidden: List[int] = None,
        trunk_hidden: List[int] = None,
        cnn_channels: List[int] = None,
        activation: str = "silu",
        time_dependent: bool = False,
        n_bc_faces: int = 4,
        n_freq_x: int = 16,
        n_freq_omega: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.n_groups = n_groups
        self.n_basis = n_basis

        if branch_hidden is None:
            branch_hidden = [256, 256]
        if trunk_hidden is None:
            trunk_hidden = [256, 256]

        self.branch = BranchNet(
            dim=dim, n_groups=n_groups, n_params=n_params,
            n_basis=n_basis, hidden_dims=branch_hidden,
            activation=activation, n_bc_faces=n_bc_faces,
            cnn_channels=cnn_channels,
        )

        self.trunk = TrunkNet(
            dim=dim, n_basis=n_basis, n_groups=n_groups,
            hidden_dims=trunk_hidden, n_freq_x=n_freq_x,
            n_freq_omega=n_freq_omega, activation=activation,
            time_dependent=time_dependent,
        )

        # Output bias per group
        self.bias = nn.Parameter(torch.zeros(n_groups))

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        Forward pass.

        Returns:
            dict with: I [B, Nx, Nw, G], phi [B, Nx, G], J [B, Nx, dim, G]
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

        B, Nx, G = sigma_a.shape
        Nw = omega.shape[1]

        # Branch: [B, n_basis]
        branch_out = self.branch(sigma_a, sigma_s, q, bc, params, spatial_shape)

        # Trunk: [B, Nx, Nw, n_basis * G]
        trunk_out = self.trunk(x, omega, t, omega_mask)

        # Reshape trunk: [B, Nx, Nw, n_basis, G]
        trunk_out = trunk_out.reshape(B, Nx, Nw, self.n_basis, G)

        # Inner product: sum over basis
        # branch: [B, n_basis] -> [B, 1, 1, n_basis, 1]
        branch_exp = branch_out.view(B, 1, 1, self.n_basis, 1)
        I = (branch_exp * trunk_out).sum(dim=3)  # [B, Nx, Nw, G]
        I = I + self.bias.view(1, 1, 1, G)
        I = F.softplus(I)  # positivity

        if omega_mask is not None:
            mask = omega_mask.unsqueeze(1).unsqueeze(-1).float()
            I = I * mask

        # Compute moments via quadrature
        phi, J = compute_moments(I, omega, w_omega, omega_mask)

        return {"I": I, "phi": phi, "J": J}
