#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy geometry demo for Bridge-Projected Geometry-aware I2SB (BP-GI2SB).

This script is self-contained. It builds a paired low-dimensional coupling manifold
(theta -> clean x0, degraded x1), trains a non-cancelling multi-head bridge model,
and visualizes:
  1) clean/degraded/bridge manifolds;
  2) learned bridge projection m_theta;
  3) reconstructed clean endpoint x0_hat;
  4) deterministic bridge sampling trajectories with optional bridge-projection corrector.

Key non-cancelling parameterization:
    o_hat(x_t,t,x1) = (x_t - m_theta(x_t,t,x1)) / sigma_t
                      + u_theta(m_theta,t,x1)
                      + gamma(t) c_theta(x_t,m_theta,t,x1)

    x0_hat = m_theta - sigma_t * [u_theta + gamma(t)c_theta]

The bridge head m_theta therefore affects both score construction and sampler correction.
It cannot be algebraically cancelled away.

Usage:
    python toy_bp_gi2sb.py --steps 8000 --device cuda
    python toy_bp_gi2sb.py --steps 3000 --device cpu --outdir toy_outputs
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Toy paired coupling manifold
# -----------------------------

def make_pair(theta: torch.Tensor, kind: str = "swirl") -> Tuple[torch.Tensor, torch.Tensor]:
    """Return paired clean/degraded 2D samples from a shared latent theta.

    Args:
        theta: shape [B, 1], in [0, 2pi].
        kind: curve family.

    Returns:
        x0: clean endpoint, shape [B, 2].
        x1: degraded endpoint, shape [B, 2].
    """
    th = theta

    if kind == "swirl":
        # Clean: nonlinear closed curve with mild radial modulation.
        r0 = 1.0 + 0.25 * torch.sin(3.0 * th)
        x0 = torch.cat([r0 * torch.cos(th), r0 * torch.sin(th)], dim=1)

        # Degraded: compressed/rotated/smoothed counterpart sharing the same theta.
        r1 = 0.45 + 0.08 * torch.sin(2.0 * th + 0.4)
        y1 = torch.cat([r1 * torch.cos(th + 0.65), 0.55 * r1 * torch.sin(th + 0.65)], dim=1)
        x1 = y1 + torch.tensor([0.35, -0.15], device=theta.device, dtype=theta.dtype)

    elif kind == "two_moons":
        x0 = torch.cat([torch.cos(th), torch.sin(th)], dim=1)
        x0 = x0 + 0.15 * torch.cat([torch.sin(2 * th), torch.cos(3 * th)], dim=1)
        x1 = torch.cat([0.75 * torch.cos(th) + 0.25, -0.45 * torch.sin(th) - 0.25], dim=1)

    elif kind == "s_curve":
        x0 = torch.cat([torch.sin(th), torch.sign(torch.cos(th)) * (1 - torch.cos(th).abs())], dim=1)
        x1 = torch.cat([0.55 * torch.sin(th + 0.4), 0.25 * torch.cos(th) - 0.4], dim=1)

    else:
        raise ValueError(f"Unknown toy geometry kind: {kind}")

    return x0, x1


def sigma_schedule(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Endpoint-small noise schedule with peak in the middle."""
    # t in [0,1], shape [B,1]
    return sigma_min + sigma_max * torch.sqrt(torch.clamp(t * (1.0 - t), min=1e-8))


def gamma_schedule(t: torch.Tensor, gamma_max: float) -> torch.Tensor:
    return gamma_max * torch.sin(math.pi * t).pow(2)


def rho_schedule(t: torch.Tensor, rho_max: float) -> torch.Tensor:
    return rho_max * torch.sin(math.pi * t).pow(2)


@dataclass
class Batch:
    theta: torch.Tensor
    x0: torch.Tensor
    x1: torch.Tensor
    t: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    xt: torch.Tensor
    eps: torch.Tensor


def sample_batch(
    batch_size: int,
    device: torch.device,
    kind: str,
    sigma_min: float,
    sigma_max: float,
    t_eps: float,
) -> Batch:
    theta = 2.0 * math.pi * torch.rand(batch_size, 1, device=device)
    x0, x1 = make_pair(theta, kind=kind)
    t = t_eps + (1.0 - 2.0 * t_eps) * torch.rand(batch_size, 1, device=device)
    mu = (1.0 - t) * x0 + t * x1
    sigma = sigma_schedule(t, sigma_min=sigma_min, sigma_max=sigma_max)
    eps = torch.randn_like(x0)
    xt = mu + sigma * eps
    return Batch(theta=theta, x0=x0, x1=x1, t=t, mu=mu, sigma=sigma, xt=xt, eps=eps)


# -----------------------------
# Model
# -----------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("freqs", freqs)
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t * self.freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BPGI2SBToy(nn.Module):
    """Non-cancelling multi-head BP-GI2SB toy network."""

    def __init__(self, hidden: int = 128, depth: int = 3, time_dim: int = 32):
        super().__init__()
        self.temb = TimeEmbedding(time_dim)
        # Shared features consume xt, x1, t-embedding.
        self.trunk = MLP(in_dim=2 + 2 + time_dim, out_dim=hidden, hidden=hidden, depth=depth)

        # Bridge head predicts m_theta ~ mu_t.
        self.bridge_head = nn.Linear(hidden, 2)

        # Transport and correction heads are conditioned on shared h and bridge m.
        self.transport_head = MLP(in_dim=hidden + 2 + 2 + time_dim, out_dim=2, hidden=hidden, depth=2)
        self.correction_head = MLP(in_dim=hidden + 2 + 2 + 2 + time_dim, out_dim=2, hidden=hidden, depth=2)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x1: torch.Tensor,
        sigma: torch.Tensor,
        gamma_max: float = 0.1,
        stop_bridge_grad: bool = True,
    ) -> Dict[str, torch.Tensor]:
        te = self.temb(t)
        h = self.trunk(torch.cat([xt, x1, te], dim=1))
        m = self.bridge_head(h)

        m_in = m.detach() if stop_bridge_grad else m
        u_in = torch.cat([h, m_in, x1, te], dim=1)
        u = self.transport_head(u_in)

        c_in = torch.cat([h, xt, m_in, x1, te], dim=1)
        c = self.correction_head(c_in)

        gamma = gamma_schedule(t, gamma_max=gamma_max)
        score = (xt - m) / sigma + u + gamma * c
        x0_hat = m - sigma * (u + gamma * c)

        return {
            "m": m,
            "u": u,
            "c": c,
            "gamma": gamma,
            "score": score,
            "x0_hat": x0_hat,
        }


# -----------------------------
# Losses and train
# -----------------------------

def compute_losses(
    model: BPGI2SBToy,
    batch: Batch,
    gamma_max: float,
    lambda_score: float,
    lambda_bridge: float,
    lambda_transport: float,
    lambda_endpoint: float,
    lambda_corr: float,
    stop_bridge_grad: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    out = model(
        batch.xt,
        batch.t,
        batch.x1,
        batch.sigma,
        gamma_max=gamma_max,
        stop_bridge_grad=stop_bridge_grad,
    )
    target_score = (batch.xt - batch.x0) / batch.sigma
    target_u = (batch.mu - batch.x0) / batch.sigma

    loss_score = F.mse_loss(out["score"], target_score)
    loss_bridge = F.mse_loss(out["m"], batch.mu)
    loss_transport = F.mse_loss(out["u"], target_u)
    loss_endpoint = F.mse_loss(out["x0_hat"], batch.x0)
    loss_corr = out["c"].pow(2).mean()

    loss = (
        lambda_score * loss_score
        + lambda_bridge * loss_bridge
        + lambda_transport * loss_transport
        + lambda_endpoint * loss_endpoint
        + lambda_corr * loss_corr
    )

    logs = {
        "total": float(loss.detach().cpu()),
        "score": float(loss_score.detach().cpu()),
        "bridge": float(loss_bridge.detach().cpu()),
        "transport": float(loss_transport.detach().cpu()),
        "endpoint": float(loss_endpoint.detach().cpu()),
        "corr": float(loss_corr.detach().cpu()),
    }
    return loss, logs


@torch.no_grad()
def ddim_like_bridge_step(
    model: BPGI2SBToy,
    x: torch.Tensor,
    x1: torch.Tensor,
    t: float,
    t_next: float,
    sigma_min: float,
    sigma_max: float,
    gamma_max: float,
    rho_max: float,
    use_projection: bool,
) -> torch.Tensor:
    B = x.shape[0]
    tt = torch.full((B, 1), t, device=x.device, dtype=x.dtype)
    tt_next = torch.full((B, 1), t_next, device=x.device, dtype=x.dtype)
    sigma = sigma_schedule(tt, sigma_min=sigma_min, sigma_max=sigma_max)
    sigma_next = sigma_schedule(tt_next, sigma_min=sigma_min, sigma_max=sigma_max)

    out = model(x, tt, x1, sigma, gamma_max=gamma_max, stop_bridge_grad=False)
    score = out["score"]
    x0_hat = x - sigma * score

    # Predict current bridge mean and deterministic residual.
    mu_t_hat = (1.0 - tt) * x0_hat + tt * x1
    residual = x - mu_t_hat

    # DDIM-like deterministic transition to the next bridge time.
    mu_next_hat = (1.0 - tt_next) * x0_hat + tt_next * x1
    x_next = mu_next_hat + (sigma_next / sigma) * residual

    if use_projection:
        sigma_p = sigma_schedule(tt_next, sigma_min=sigma_min, sigma_max=sigma_max)
        out_p = model(x_next, tt_next, x1, sigma_p, gamma_max=gamma_max, stop_bridge_grad=False)
        rho = rho_schedule(tt_next, rho_max=rho_max)
        x_next = x_next + rho * (out_p["m"] - x_next)

    return x_next


@torch.no_grad()
def sample_trajectories(
    model: BPGI2SBToy,
    theta: torch.Tensor,
    kind: str,
    sigma_min: float,
    sigma_max: float,
    gamma_max: float,
    rho_max: float,
    n_steps: int,
    use_projection: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = theta.device
    x0, x1 = make_pair(theta, kind=kind)
    # Start from degraded endpoint with a tiny amount of noise.
    t0 = 1.0 - 1e-3
    sigma0 = sigma_schedule(torch.full((theta.shape[0], 1), t0, device=device), sigma_min, sigma_max)
    x = x1 + sigma0 * 0.25 * torch.randn_like(x1)
    traj = [x.detach().cpu()]

    times = torch.linspace(t0, 1e-3, n_steps + 1, device=device)
    for k in range(n_steps):
        x = ddim_like_bridge_step(
            model,
            x,
            x1,
            float(times[k].item()),
            float(times[k + 1].item()),
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            gamma_max=gamma_max,
            rho_max=rho_max,
            use_projection=use_projection,
        )
        traj.append(x.detach().cpu())
    return x0.detach().cpu(), x1.detach().cpu(), x.detach().cpu(), torch.stack(traj, dim=0)


# -----------------------------
# Visualization
# -----------------------------

def save_geometry_plot(outdir: str, kind: str, device: torch.device, sigma_min: float, sigma_max: float) -> None:
    theta = torch.linspace(0, 2 * math.pi, 800, device=device).view(-1, 1)
    x0, x1 = make_pair(theta, kind=kind)
    xs = []
    ts = [0.25, 0.50, 0.75]
    for tval in ts:
        t = torch.full((theta.shape[0], 1), tval, device=device)
        xs.append(((1 - t) * x0 + t * x1).detach().cpu().numpy())

    x0n = x0.detach().cpu().numpy()
    x1n = x1.detach().cpu().numpy()

    plt.figure(figsize=(7.2, 6.4), dpi=160)
    plt.plot(x0n[:, 0], x0n[:, 1], lw=2, label="clean manifold M0")
    plt.plot(x1n[:, 0], x1n[:, 1], lw=2, label="degraded manifold M1")
    for arr, tval in zip(xs, ts):
        plt.plot(arr[:, 0], arr[:, 1], lw=1.3, linestyle="--", label=f"bridge mean Mt, t={tval:.2f}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Toy paired coupling manifold and induced bridge manifolds")
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "01_toy_bridge_manifolds.png"))
    plt.close()


@torch.no_grad()
def save_prediction_plot(
    model: BPGI2SBToy,
    outdir: str,
    kind: str,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    gamma_max: float,
) -> None:
    B = 1200
    batch = sample_batch(B, device, kind, sigma_min, sigma_max, t_eps=0.02)
    out = model(batch.xt, batch.t, batch.x1, batch.sigma, gamma_max=gamma_max, stop_bridge_grad=False)

    x0 = batch.x0.detach().cpu().numpy()
    x1 = batch.x1.detach().cpu().numpy()
    xt = batch.xt.detach().cpu().numpy()
    mu = batch.mu.detach().cpu().numpy()
    m = out["m"].detach().cpu().numpy()
    x0_hat = out["x0_hat"].detach().cpu().numpy()

    plt.figure(figsize=(7.2, 6.4), dpi=160)
    plt.scatter(xt[:, 0], xt[:, 1], s=4, alpha=0.18, label="noisy x_t")
    plt.scatter(mu[:, 0], mu[:, 1], s=6, alpha=0.35, label="true bridge mean mu_t")
    plt.scatter(m[:, 0], m[:, 1], s=6, alpha=0.45, label="predicted bridge m_theta")
    plt.scatter(x0_hat[:, 0], x0_hat[:, 1], s=6, alpha=0.45, label="reconstructed x0_hat")
    plt.scatter(x0[:, 0], x0[:, 1], s=5, alpha=0.25, label="clean x0")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Learned bridge projection and endpoint reconstruction")
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02_learned_bridge_and_endpoint.png"))
    plt.close()


@torch.no_grad()
def save_trajectory_plot(
    model: BPGI2SBToy,
    outdir: str,
    kind: str,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    gamma_max: float,
    rho_max: float,
    n_steps: int,
) -> None:
    theta = torch.linspace(0, 2 * math.pi, 64, device=device).view(-1, 1)
    x0, x1, x_final_no_proj, traj_no_proj = sample_trajectories(
        model, theta, kind, sigma_min, sigma_max, gamma_max, rho_max, n_steps, use_projection=False
    )
    _, _, x_final_proj, traj_proj = sample_trajectories(
        model, theta, kind, sigma_min, sigma_max, gamma_max, rho_max, n_steps, use_projection=True
    )

    # Dense true manifolds.
    th_dense = torch.linspace(0, 2 * math.pi, 1000, device=device).view(-1, 1)
    xd0, xd1 = make_pair(th_dense, kind=kind)
    xd0 = xd0.detach().cpu().numpy()
    xd1 = xd1.detach().cpu().numpy()

    def draw(ax, traj, x_final, title):
        ax.plot(xd0[:, 0], xd0[:, 1], lw=2, label="clean M0")
        ax.plot(xd1[:, 0], xd1[:, 1], lw=2, label="degraded M1")
        tr = traj.numpy()  # [T+1, B, 2]
        for i in range(0, tr.shape[1], 4):
            ax.plot(tr[:, i, 0], tr[:, i, 1], lw=0.8, alpha=0.55)
        xf = x_final.numpy()
        ax.scatter(xf[:, 0], xf[:, 1], s=10, alpha=0.65, label="final samples")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), dpi=160)
    draw(axes[0], traj_no_proj, x_final_no_proj, "Sampler without bridge-projection corrector")
    draw(axes[1], traj_proj, x_final_proj, "Sampler with bridge-projection corrector")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "03_sampling_trajectories_projection_ablation.png"))
    plt.close()


def save_loss_plot(history: Dict[str, list], outdir: str) -> None:
    plt.figure(figsize=(7.2, 4.8), dpi=160)
    for key in ["total", "score", "bridge", "transport", "endpoint"]:
        if key in history:
            plt.plot(history["step"], history[key], label=key)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("BP-GI2SB toy training curves")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "00_training_losses.png"))
    plt.close()




# -----------------------------
# Quantitative projection metrics
# -----------------------------

@torch.no_grad()
def curve_points_at_t(
    t_value: float,
    kind: str,
    device: torch.device,
    n_dense: int = 1200,
) -> torch.Tensor:
    """Dense samples of the true bridge manifold M_t."""
    theta = torch.linspace(0, 2 * math.pi, n_dense, device=device).view(-1, 1)
    x0, x1 = make_pair(theta, kind=kind)
    t = torch.full((n_dense, 1), float(t_value), device=device)
    return (1.0 - t) * x0 + t * x1


@torch.no_grad()
def mean_nearest_distance(points: torch.Tensor, curve: torch.Tensor, chunk: int = 2048) -> float:
    """Mean Euclidean distance from points to a dense sampled curve."""
    vals = []
    for i in range(0, points.shape[0], chunk):
        d = torch.cdist(points[i:i + chunk], curve)
        vals.append(d.min(dim=1).values)
    return float(torch.cat(vals).mean().detach().cpu())


@torch.no_grad()
def mean_nearest_distance_per_point(points: torch.Tensor, curve: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Per-point nearest distance, used for mean/std reporting."""
    vals = []
    for i in range(0, points.shape[0], chunk):
        d = torch.cdist(points[i:i + chunk], curve)
        vals.append(d.min(dim=1).values)
    return torch.cat(vals)


@torch.no_grad()
def mmd_rbf(x: torch.Tensor, y: torch.Tensor, max_points: int = 512) -> float:
    """Unbiased-enough RBF MMD with median heuristic for toy evaluation."""
    if x.shape[0] > max_points:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_points]
        x = x[idx]
    if y.shape[0] > max_points:
        idx = torch.randperm(y.shape[0], device=y.device)[:max_points]
        y = y[idx]

    xy = torch.cat([x, y], dim=0)
    d2_all = torch.cdist(xy, xy).pow(2)
    # median heuristic; avoid diagonal zeros dominating too much
    tri = d2_all[d2_all > 0]
    sigma2 = torch.median(tri).clamp_min(1e-6)

    kxx = torch.exp(-torch.cdist(x, x).pow(2) / (2.0 * sigma2))
    kyy = torch.exp(-torch.cdist(y, y).pow(2) / (2.0 * sigma2))
    kxy = torch.exp(-torch.cdist(x, y).pow(2) / (2.0 * sigma2))
    return float((kxx.mean() + kyy.mean() - 2.0 * kxy.mean()).detach().cpu())


@torch.no_grad()
def trajectory_bridge_distance(
    traj: torch.Tensor,
    kind: str,
    device: torch.device,
    n_dense: int = 1200,
) -> float:
    """Average distance from trajectory states x_k to their corresponding M_{t_k}.

    traj: CPU tensor with shape [K+1, B, 2], returned by sample_trajectories().
    The time convention matches sample_trajectories(): linearly from 1-1e-3 to 1e-3.
    """
    K = traj.shape[0] - 1
    times = torch.linspace(1.0 - 1e-3, 1e-3, K + 1, device=device)
    dists = []
    for k, tv in enumerate(times):
        curve = curve_points_at_t(float(tv.item()), kind=kind, device=device, n_dense=n_dense)
        pts = traj[k].to(device)
        dists.append(mean_nearest_distance_per_point(pts, curve))
    return float(torch.cat(dists).mean().detach().cpu())




@torch.no_grad()
def trajectory_straightness_metrics(traj: torch.Tensor, x0_true: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    """Quantify whether a sampled trajectory takes unnecessary detours.

    Args:
        traj: CPU or GPU tensor with shape [K+1, B, 2]. It contains the full reverse
            trajectory from the degraded endpoint neighborhood to the clean endpoint.
        x0_true: CPU or GPU tensor with shape [B, 2]. True paired clean endpoints.
        eps: small number for numerical stability.

    Metrics:
        path_length:
            Mean accumulated Euclidean length along the sampled trajectory.
        chord_length:
            Mean direct Euclidean distance between trajectory start and final sample.
        path_tortuosity:
            path_length / chord_length. This is the standard tortuosity / detour ratio.
            It is >= 1 up to numerical error. Lower is better; 1 means perfectly straight.
        path_efficiency:
            chord_length / path_length = 1 / path_tortuosity. Higher is better.
        excess_path_length:
            path_length - chord_length. Lower is better.
        target_path_ratio:
            path_length / ||x_start - x0_true||. This compares the sampled path length
            with the ideal paired endpoint displacement. It is useful for paired toy
            tasks, but can be < 1 if the final sample under-shoots the true endpoint.
        total_turning_angle_deg:
            Sum of turning angles between consecutive segments, in degrees. Lower means
            fewer sharp turns and less zig-zag motion.
        mean_turning_angle_deg:
            Average turning angle per internal step, in degrees.
    """
    traj = traj.detach()
    x0_true = x0_true.detach().to(traj.device)

    # Segment lengths: [K, B]
    segments = traj[1:] - traj[:-1]
    seg_len = torch.linalg.norm(segments, dim=-1)
    path_len = seg_len.sum(dim=0)  # [B]

    start = traj[0]
    final = traj[-1]
    chord_len = torch.linalg.norm(final - start, dim=-1)  # [B]
    target_chord_len = torch.linalg.norm(x0_true - start, dim=-1)  # [B]

    path_tortuosity = path_len / (chord_len + eps)
    path_efficiency = chord_len / (path_len + eps)
    excess_path_length = path_len - chord_len
    target_path_ratio = path_len / (target_chord_len + eps)

    # Turning angles between consecutive trajectory segments.
    if traj.shape[0] >= 3:
        v_prev = traj[1:-1] - traj[:-2]  # [K-1, B, 2]
        v_next = traj[2:] - traj[1:-1]   # [K-1, B, 2]
        n_prev = torch.linalg.norm(v_prev, dim=-1)
        n_next = torch.linalg.norm(v_next, dim=-1)
        valid = (n_prev > eps) & (n_next > eps)
        cosang = (v_prev * v_next).sum(dim=-1) / (n_prev * n_next + eps)
        cosang = torch.clamp(cosang, -1.0, 1.0)
        angles = torch.acos(cosang)
        angles = torch.where(valid, angles, torch.zeros_like(angles))
        total_turning_angle = angles.sum(dim=0)  # [B], radians
        denom = valid.float().sum(dim=0).clamp_min(1.0)
        mean_turning_angle = angles.sum(dim=0) / denom
    else:
        total_turning_angle = torch.zeros_like(path_len)
        mean_turning_angle = torch.zeros_like(path_len)

    return {
        "path_length": float(path_len.mean().detach().cpu()),
        "chord_length": float(chord_len.mean().detach().cpu()),
        "path_tortuosity": float(path_tortuosity.mean().detach().cpu()),
        "path_efficiency": float(path_efficiency.mean().detach().cpu()),
        "excess_path_length": float(excess_path_length.mean().detach().cpu()),
        "target_path_ratio": float(target_path_ratio.mean().detach().cpu()),
        "total_turning_angle_deg": float((total_turning_angle * 180.0 / math.pi).mean().detach().cpu()),
        "mean_turning_angle_deg": float((mean_turning_angle * 180.0 / math.pi).mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate_one_sampler_setting(
    model: BPGI2SBToy,
    kind: str,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    gamma_max: float,
    rho_max: float,
    n_steps: int,
    n_eval: int,
    n_dense: int,
    use_projection: bool,
) -> Dict[str, float]:
    """Run one sampler setting and compute quantitative metrics."""
    theta = 2.0 * math.pi * torch.rand(n_eval, 1, device=device)
    x0, x1, x_final, traj = sample_trajectories(
        model=model,
        theta=theta,
        kind=kind,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        gamma_max=gamma_max,
        rho_max=rho_max,
        n_steps=n_steps,
        use_projection=use_projection,
    )
    x0 = x0.to(device)
    x1 = x1.to(device)
    x_final = x_final.to(device)

    clean_curve = curve_points_at_t(0.0, kind=kind, device=device, n_dense=n_dense)
    final_dist = mean_nearest_distance(x_final, clean_curve)
    paired_rmse = float(torch.sqrt(F.mse_loss(x_final, x0)).detach().cpu())
    paired_mae = float(torch.mean(torch.linalg.norm(x_final - x0, dim=1)).detach().cpu())
    traj_dist = trajectory_bridge_distance(traj, kind=kind, device=device, n_dense=n_dense)
    mmd = mmd_rbf(x_final, x0)
    straightness = trajectory_straightness_metrics(traj, x0.detach().cpu())

    return {
        "use_projection": float(use_projection),
        "rho_max": float(rho_max),
        "n_steps": float(n_steps),
        "final_dist_to_M0": final_dist,
        "paired_rmse": paired_rmse,
        "paired_mae": paired_mae,
        "trajectory_dist_to_Mt": traj_dist,
        "mmd_to_clean": mmd,
        **straightness,
    }


def save_metrics_csv(rows: list, path: str) -> None:
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def plot_projection_metric_bars(rows: list, outdir: str, chosen_n_steps: int, chosen_rho: float) -> None:
    """Bar plots for no-projection vs projection at one NFE/rho setting."""
    selected = [r for r in rows if int(r["n_steps"]) == int(chosen_n_steps) and abs(r["rho_max"] - chosen_rho) < 1e-12]
    if len(selected) < 2:
        return
    selected = sorted(selected, key=lambda r: r["use_projection"])
    labels = ["w/o projection", "w/ projection"]
    metrics = [
        ("final_dist_to_M0", "Final distance to clean manifold M0"),
        ("trajectory_dist_to_Mt", "Trajectory distance to bridge manifolds Mt"),
        ("paired_rmse", "Paired endpoint RMSE"),
        ("mmd_to_clean", "MMD to clean endpoint distribution"),
        ("path_tortuosity", "Trajectory tortuosity / detour ratio"),
        ("path_efficiency", "Trajectory straightness efficiency"),
        ("excess_path_length", "Excess path length"),
        ("total_turning_angle_deg", "Total turning angle"),
        ("mean_turning_angle_deg", "Mean turning angle per step"),
    ]
    for key, title in metrics:
        plt.figure(figsize=(5.2, 4.0), dpi=160)
        values = [r[key] for r in selected]
        plt.bar(labels, values)
        plt.ylabel(key)
        plt.title(f"{title}\nNFE={chosen_n_steps}, rho_max={chosen_rho}")
        for i, v in enumerate(values):
            plt.text(i, v, f"{v:.3e}", ha="center", va="bottom", fontsize=8)
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"04_bar_{key}.png"))
        plt.close()


def plot_rho_ablation(rows: list, outdir: str, chosen_n_steps: int) -> None:
    """Line plots: rho_max ablation for projection sampler."""
    selected = [r for r in rows if int(r["n_steps"]) == int(chosen_n_steps) and bool(r["use_projection"])]
    if not selected:
        return
    selected = sorted(selected, key=lambda r: r["rho_max"])
    rhos = [r["rho_max"] for r in selected]
    for key in [
        "final_dist_to_M0", "trajectory_dist_to_Mt", "paired_rmse", "mmd_to_clean",
        "path_tortuosity", "path_efficiency", "excess_path_length",
        "total_turning_angle_deg", "mean_turning_angle_deg",
    ]:
        plt.figure(figsize=(5.6, 4.0), dpi=160)
        vals = [r[key] for r in selected]
        plt.plot(rhos, vals, marker="o")
        plt.xlabel("rho_max")
        plt.ylabel(key)
        plt.title(f"Projection strength ablation, NFE={chosen_n_steps}")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"05_rho_ablation_{key}.png"))
        plt.close()


def plot_nfe_ablation(rows: list, outdir: str, chosen_rho: float) -> None:
    """Line plots: NFE ablation comparing projection vs no projection."""
    metrics = [
        "final_dist_to_M0", "trajectory_dist_to_Mt", "paired_rmse", "mmd_to_clean",
        "path_tortuosity", "path_efficiency", "excess_path_length",
        "total_turning_angle_deg", "mean_turning_angle_deg",
    ]
    for key in metrics:
        plt.figure(figsize=(5.8, 4.0), dpi=160)
        for use_proj, label in [(False, "w/o projection"), (True, "w/ projection")]:
            selected = [
                r for r in rows
                if bool(r["use_projection"]) == use_proj and abs(r["rho_max"] - chosen_rho) < 1e-12
            ]
            selected = sorted(selected, key=lambda r: r["n_steps"])
            if selected:
                xs = [r["n_steps"] for r in selected]
                ys = [r[key] for r in selected]
                plt.plot(xs, ys, marker="o", label=label)
        plt.xlabel("NFE / sampler steps")
        plt.ylabel(key)
        plt.title(f"NFE ablation, rho_max={chosen_rho}")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"06_nfe_ablation_{key}.png"))
        plt.close()


@torch.no_grad()
def save_quantitative_projection_comparison(
    model: BPGI2SBToy,
    outdir: str,
    kind: str,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    gamma_max: float,
    rho_values: list,
    nfe_values: list,
    n_eval: int,
    n_dense: int,
) -> None:
    """Evaluate projection corrector quantitatively and save csv + plots."""
    model.eval()
    rows = []
    for nfe in nfe_values:
        for rho in rho_values:
            # No projection: keep rho in the row only for matched plotting.
            rows.append(evaluate_one_sampler_setting(
                model, kind, device, sigma_min, sigma_max, gamma_max,
                rho_max=rho, n_steps=int(nfe), n_eval=n_eval, n_dense=n_dense,
                use_projection=False,
            ))
            rows.append(evaluate_one_sampler_setting(
                model, kind, device, sigma_min, sigma_max, gamma_max,
                rho_max=rho, n_steps=int(nfe), n_eval=n_eval, n_dense=n_dense,
                use_projection=True,
            ))
            print(
                f"[eval] NFE={nfe:03d}, rho={rho:.3f} done.",
                flush=True,
            )

    csv_path = os.path.join(outdir, "04_quantitative_projection_metrics.csv")
    save_metrics_csv(rows, csv_path)

    chosen_nfe = int(nfe_values[-1])
    chosen_rho = float(rho_values[-1])
    plot_projection_metric_bars(rows, outdir, chosen_nfe, chosen_rho)
    plot_rho_ablation(rows, outdir, chosen_nfe)
    plot_nfe_ablation(rows, outdir, chosen_rho)

    # Print a compact table for the default setting.
    print("\nQuantitative comparison at default plotting setting:")
    print(f"NFE={chosen_nfe}, rho_max={chosen_rho}")
    compact = [r for r in rows if int(r["n_steps"]) == chosen_nfe and abs(r["rho_max"] - chosen_rho) < 1e-12]
    compact = sorted(compact, key=lambda r: r["use_projection"])
    for r in compact:
        tag = "with projection" if bool(r["use_projection"]) else "without projection"
        print(
            f"  {tag:18s} | "
            f"final_dist={r['final_dist_to_M0']:.4e}, "
            f"traj_dist={r['trajectory_dist_to_Mt']:.4e}, "
            f"paired_rmse={r['paired_rmse']:.4e}, "
            f"mmd={r['mmd_to_clean']:.4e}, "
            f"tortuosity={r['path_tortuosity']:.4f}, "
            f"efficiency={r['path_efficiency']:.4f}, "
            f"turn_deg={r['total_turning_angle_deg']:.2f}"
        )
    print(f"Saved quantitative metrics to: {csv_path}")


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toy geometry demo for BP-GI2SB")
    p.add_argument("--outdir", type=str, default="/mnt/data/toy_bp_gi2sb_outputs")
    p.add_argument("--kind", type=str, default="swirl", choices=["swirl", "two_moons", "s_curve"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--sigma-min", type=float, default=0.02)
    p.add_argument("--sigma-max", type=float, default=0.35)
    p.add_argument("--t-eps", type=float, default=0.02)
    p.add_argument("--gamma-max", type=float, default=0.15)
    p.add_argument("--rho-max", type=float, default=0.12)
    p.add_argument("--lambda-score", type=float, default=1.0)
    p.add_argument("--lambda-bridge", type=float, default=1.0)
    p.add_argument("--lambda-transport", type=float, default=1.0)
    p.add_argument("--lambda-endpoint", type=float, default=0.1)
    p.add_argument("--lambda-corr", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--traj-steps", type=int, default=60)
    p.add_argument("--no-stop-bridge-grad", action="store_true")
    p.add_argument("--eval-samples", type=int, default=1024)
    p.add_argument("--eval-dense", type=int, default=1200)
    p.add_argument("--eval-rho-values", type=float, nargs="+", default=[0.0, 0.04, 0.08, 0.12, 0.20])
    p.add_argument("--eval-nfe-values", type=int, nargs="+", default=[5, 10, 20, 40, 60])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    model = BPGI2SBToy(hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = {"step": [], "total": [], "score": [], "bridge": [], "transport": [], "endpoint": [], "corr": []}

    save_geometry_plot(args.outdir, args.kind, device, args.sigma_min, args.sigma_max)

    model.train()
    for step in range(1, args.steps + 1):
        batch = sample_batch(
            args.batch_size,
            device,
            args.kind,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            t_eps=args.t_eps,
        )
        loss, logs = compute_losses(
            model,
            batch,
            gamma_max=args.gamma_max,
            lambda_score=args.lambda_score,
            lambda_bridge=args.lambda_bridge,
            lambda_transport=args.lambda_transport,
            lambda_endpoint=args.lambda_endpoint,
            lambda_corr=args.lambda_corr,
            stop_bridge_grad=not args.no_stop_bridge_grad,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            history["step"].append(step)
            for k in ["total", "score", "bridge", "transport", "endpoint", "corr"]:
                history[k].append(logs[k])
            print(
                f"step {step:06d} | total={logs['total']:.4e} "
                f"score={logs['score']:.4e} bridge={logs['bridge']:.4e} "
                f"trans={logs['transport']:.4e} endpoint={logs['endpoint']:.4e} corr={logs['corr']:.4e}",
                flush=True,
            )

    model.eval()
    save_loss_plot(history, args.outdir)
    save_prediction_plot(model, args.outdir, args.kind, device, args.sigma_min, args.sigma_max, args.gamma_max)
    save_trajectory_plot(
        model,
        args.outdir,
        args.kind,
        device,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        gamma_max=args.gamma_max,
        rho_max=args.rho_max,
        n_steps=args.traj_steps,
    )

    save_quantitative_projection_comparison(
        model=model,
        outdir=args.outdir,
        kind=args.kind,
        device=device,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        gamma_max=args.gamma_max,
        rho_values=args.eval_rho_values,
        nfe_values=args.eval_nfe_values,
        n_eval=args.eval_samples,
        n_dense=args.eval_dense,
    )

    ckpt_path = os.path.join(args.outdir, "bp_gi2sb_toy_model.pt")
    torch.save({"model": model.state_dict(), "args": vars(args), "history": history}, ckpt_path)
    print(f"\nSaved outputs to: {args.outdir}")
    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
