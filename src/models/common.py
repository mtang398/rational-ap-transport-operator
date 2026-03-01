"""
Shared modules for transport neural operators.

Includes:
- Fourier positional encodings for spatial coords x and angular directions omega
- MLP building blocks
- Spectral convolution helpers
- Quadrature moment computation utilities
"""

from __future__ import annotations
import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FourierFeatures(nn.Module):
    """
    Random Fourier Features for positional encoding of spatial coordinates x.

    Maps x in R^d -> [sin(B x), cos(B x)] in R^(2*n_freq)
    where B ~ N(0, sigma^2).

    Standard fixed or learned frequency embedding.
    """

    def __init__(self, in_dim: int, n_freq: int = 32, sigma: float = 1.0, learnable: bool = False):
        super().__init__()
        B = torch.randn(in_dim, n_freq) * sigma
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
        self.out_dim = 2 * n_freq

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., in_dim]
        Returns:
            [..., 2*n_freq]
        """
        proj = x @ self.B  # [..., n_freq]
        return torch.cat([torch.sin(2 * math.pi * proj), torch.cos(2 * math.pi * proj)], dim=-1)


class SphericalFourierFeatures(nn.Module):
    """
    Fourier features for angular directions omega on the unit sphere.

    For d=2: uses Fourier series in polar angle theta.
    For d=3: uses real spherical harmonics up to order L.

    Falls back to standard Fourier features for arbitrary d.
    """

    def __init__(self, dim: int, n_harmonics: int = 8, sigma: float = 1.0):
        super().__init__()
        self.dim = dim
        self.n_harmonics = n_harmonics

        if dim == 2:
            # Encode as Fourier series: [cos(k*theta), sin(k*theta)] for k=0..n_harmonics
            self.out_dim = 2 * n_harmonics + 1
        elif dim == 3:
            # Real spherical harmonics up to order n_harmonics
            # Total = (n_harmonics+1)^2
            self.out_dim = (n_harmonics + 1) ** 2
        else:
            # Generic Fourier features for higher dim
            B = torch.randn(dim, n_harmonics) * sigma
            self.register_buffer("B", B)
            self.out_dim = 2 * n_harmonics

    def forward(self, omega: Tensor) -> Tensor:
        """
        Args:
            omega: [..., dim] unit vectors on sphere
        Returns:
            [..., out_dim] spherical harmonic / Fourier features
        """
        if self.dim == 2:
            return self._fourier_2d(omega)
        elif self.dim == 3:
            return self._spherical_harmonics_3d(omega)
        else:
            proj = omega @ self.B
            return torch.cat([torch.sin(math.pi * proj), torch.cos(math.pi * proj)], dim=-1)

    def _fourier_2d(self, omega: Tensor) -> Tensor:
        """2D Fourier encoding via polar angle."""
        # omega: [..., 2]; compute theta = atan2(y, x)
        theta = torch.atan2(omega[..., 1:2], omega[..., 0:1])  # [..., 1]
        ks = torch.arange(1, self.n_harmonics + 1, device=omega.device, dtype=omega.dtype)
        cos_feats = torch.cos(ks * theta)  # [..., n_harmonics]
        sin_feats = torch.sin(ks * theta)  # [..., n_harmonics]
        ones = torch.ones_like(theta)      # DC component
        return torch.cat([ones, cos_feats, sin_feats], dim=-1)

    def _spherical_harmonics_3d(self, omega: Tensor) -> Tensor:
        """
        Real spherical harmonics up to degree n_harmonics.
        Uses the explicit formula for low orders (l=0,1,2) and
        falls back to Fourier features for higher orders.
        """
        x = omega[..., 0:1]
        y = omega[..., 1:2]
        z = omega[..., 2:3]
        feats = [torch.ones_like(x) * (1.0 / math.sqrt(4 * math.pi))]  # l=0, m=0

        if self.n_harmonics >= 1:
            # l=1: Y_{1,-1} = sqrt(3/4pi)*y, Y_{1,0} = sqrt(3/4pi)*z, Y_{1,1} = sqrt(3/4pi)*x
            c1 = math.sqrt(3.0 / (4 * math.pi))
            feats.extend([c1 * y, c1 * z, c1 * x])

        if self.n_harmonics >= 2:
            # l=2: 5 components
            c2a = 0.5 * math.sqrt(15.0 / math.pi)
            c2b = 0.5 * math.sqrt(5.0 / math.pi)
            c2c = 0.25 * math.sqrt(15.0 / math.pi)
            feats.extend([
                c2a * x * y,
                c2a * y * z,
                c2b * (2 * z**2 - x**2 - y**2),
                c2a * x * z,
                c2c * (x**2 - y**2),
            ])

        # For higher orders, use standard Fourier features as approximation
        if self.n_harmonics >= 3:
            phi = torch.atan2(y, x)
            costh = z
            for l in range(3, self.n_harmonics + 1):
                feats.append(torch.cos(l * phi) * costh)
                feats.append(torch.sin(l * phi) * costh)

        feats_cat = torch.cat(feats, dim=-1)
        # Trim or pad to exact out_dim
        out = feats_cat[..., :self.out_dim]
        if out.shape[-1] < self.out_dim:
            pad = torch.zeros(*out.shape[:-1], self.out_dim - out.shape[-1],
                              device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=-1)
        return out


class MLP(nn.Module):
    """
    Standard MLP with configurable hidden dims and activation.
    Supports SiLU, GELU, ReLU activations.
    """

    ACTIVATIONS = {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation: str = "silu",
        dropout: float = 0.0,
        final_activation: bool = False,
    ):
        super().__init__()
        act_cls = self.ACTIVATIONS.get(activation.lower(), nn.SiLU)
        layers: List[nn.Module] = []
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = (i == len(dims) - 2)
            if not is_last or final_activation:
                layers.append(act_cls())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SpectralConv2d(nn.Module):
    """
    2D Fourier spectral convolution (FNO-style).
    Truncates to n_modes Fourier modes in each spatial direction.
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes_1: int, n_modes_2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_1 = n_modes_1
        self.n_modes_2 = n_modes_2

        scale = 1.0 / (in_channels * out_channels)
        self.weights_re = nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes_1, n_modes_2))
        self.weights_im = nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes_1, n_modes_2))

    @property
    def weights(self) -> Tensor:
        return torch.complex(self.weights_re, self.weights_im)

    def _complex_mult2d(self, x: Tensor, w: Tensor) -> Tensor:
        """Batched complex matrix multiply: x[B,Ci,X,Y] * w[Ci,Co,X,Y] -> [B,Co,X,Y]."""
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        # Apply spectral weights on first n_modes_1 x n_modes_2 frequencies
        m1 = min(self.n_modes_1, H // 2 + 1)
        m2 = min(self.n_modes_2, W // 2 + 1)
        out_ft[:, :, :m1, :m2] = self._complex_mult2d(x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2])

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class SpectralConv3d(nn.Module):
    """3D Fourier spectral convolution."""

    def __init__(self, in_channels: int, out_channels: int,
                 n_modes_1: int, n_modes_2: int, n_modes_3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = (n_modes_1, n_modes_2, n_modes_3)

        scale = 1.0 / (in_channels * out_channels)
        self.weights_re = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes_1, n_modes_2, n_modes_3))
        self.weights_im = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes_1, n_modes_2, n_modes_3))

    @property
    def weights(self) -> Tensor:
        return torch.complex(self.weights_re, self.weights_im)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, D1, D2, D3]
        """
        B, C, D1, D2, D3 = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")
        m1, m2, m3 = [min(n, d) for n, d in zip(self.n_modes, [D1, D2, D3 // 2 + 1])]

        out_ft = torch.zeros(B, self.out_channels, D1, D2, D3 // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        w = self.weights[:, :, :m1, :m2, :m3]
        out_ft[:, :, :m1, :m2, :m3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :m1, :m2, :m3], w)

        return torch.fft.irfftn(out_ft, s=(D1, D2, D3), dim=[-3, -2, -1], norm="ortho")


class FNOBlock2d(nn.Module):
    """One FNO block: spectral conv + pointwise conv + activation."""

    def __init__(self, channels: int, n_modes_1: int, n_modes_2: int, activation: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, n_modes_1, n_modes_2)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        act_map = {"silu": nn.SiLU(), "gelu": nn.GELU(), "relu": nn.ReLU()}
        self.act = act_map.get(activation, nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.spectral(x) + self.pointwise(x))


class FNOBlock3d(nn.Module):
    """One FNO block for 3D spatial data."""

    def __init__(self, channels: int, n_modes: Tuple[int, int, int], activation: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, *n_modes)
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        act_map = {"silu": nn.SiLU(), "gelu": nn.GELU(), "relu": nn.ReLU()}
        self.act = act_map.get(activation, nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.spectral(x) + self.pointwise(x))


def compute_moments(
    I: Tensor,
    omega: Tensor,
    w_omega: Tensor,
    omega_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute scalar flux phi and current J from angular intensity I via quadrature.

    Args:
        I: [B, Nx, Nw, G] angular intensity
        omega: [B, Nw, dim] direction vectors
        w_omega: [B, Nw] quadrature weights
        omega_mask: [B, Nw] boolean mask (True = valid direction)

    Returns:
        phi: [B, Nx, G] scalar flux = integral I d_omega
        J: [B, Nx, dim, G] current = integral omega * I d_omega
    """
    B, Nx, Nw, G = I.shape

    if omega_mask is not None:
        w = w_omega * omega_mask.float()  # [B, Nw]
    else:
        w = w_omega

    # phi = sum_w w_i * I_i
    phi = torch.einsum("bw,bnwg->bng", w, I)  # [B, Nx, G]

    # J = sum_w w_i * omega_i * I_i
    J = torch.einsum("bw,bwd,bnwg->bndg", w, omega, I)  # [B, Nx, dim, G]

    return phi, J


def recon_from_moments_p1(
    phi: Tensor,
    J: Tensor,
    omega: Tensor,
    dim: int,
) -> Tensor:
    """
    P1 angular reconstruction: I(x, omega) = phi/(4*pi) + (3/4*pi) * J . omega
    This is the explicit, deterministic closure used in the AP model.

    Args:
        phi: [B, Nx, G] scalar flux
        J: [B, Nx, dim, G] current
        omega: [B, Nw, dim] direction vectors
        dim: spatial dimension (2 or 3)

    Returns:
        I_recon: [B, Nx, Nw, G]
    """
    norm = 4.0 * math.pi if dim == 3 else 2.0 * math.pi

    # Isotropic part: phi / (4*pi)
    I_iso = phi.unsqueeze(2) / norm  # [B, Nx, 1, G]

    # Streaming correction: (dim/(4*pi)) * J . omega
    # J: [B, Nx, dim, G], omega: [B, Nw, dim]
    # J_dot_omega: [B, Nx, Nw, G]
    J_dot_omega = torch.einsum("bndg,bwd->bnwg", J, omega)
    I_stream = (float(dim) / norm) * J_dot_omega

    return I_iso + I_stream  # [B, Nx, Nw, G]


class ParamEncoder(nn.Module):
    """
    Encodes scalar physics parameters (epsilon, g, etc.) as a feature vector.
    Applies log-scaling to epsilon for multi-decade range handling.
    """

    def __init__(self, param_keys: List[str], out_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.param_keys = param_keys
        n_params = len(param_keys)
        self.epsilon_idx = param_keys.index("epsilon") if "epsilon" in param_keys else -1
        self.net = MLP(n_params, out_dim, [hidden_dim], activation="silu")

    def forward(self, params: Tensor, batch_param_keys: Optional[List[str]] = None) -> Tensor:
        """
        Args:
            params: [B, n_params_in_batch]  — may have more columns than self.param_keys
            batch_param_keys: list of key names matching columns of params.
                              If provided, selects only the columns in self.param_keys.
        Returns:
            [B, out_dim]
        """
        if batch_param_keys is not None and batch_param_keys != self.param_keys:
            # Select only the columns this encoder was built for, in the right order
            indices = [batch_param_keys.index(k) for k in self.param_keys if k in batch_param_keys]
            params = params[:, indices]

        p = params.clone()
        if self.epsilon_idx >= 0:
            p[:, self.epsilon_idx] = torch.log(params[:, self.epsilon_idx].clamp(min=1e-6))
        return self.net(p)
