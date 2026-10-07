#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_curved_multibranch_toy_v2.py

V2 changes compared with the first curved multi-branch toy script
-----------------------------------------------------------------
1. Stronger BP-GI2SB multi-head network:
   - transport head sees backbone feature h as well as m_theta, t, x1
   - correction head sees h, x_t, m_theta, t, x1
   This keeps the non-canceling bridge-coordinate structure, but avoids making
   u_theta too weak in the curved / multi-branch setting.

2. More conservative projection defaults for downstream evaluation:
   - recommended rho_max is 0.01--0.05, not 0.2.

3. Cleaner checkpoint args and helper functions for paired-batch evaluation
   in eval_curved_multibranch_toy_v2.ipynb.

Toy definition
--------------
x0(theta,y): two clean branches.
x1(theta,y): compressed degraded branches.
M_t(theta,y;kappa) = (1-t)x0 + t x1 + kappa sin(pi t) q(theta,y).
x_t = M_t + s_t eps.

BP-GI2SB target:
o_theta = (x_t - m_theta)/s_t + u_theta(h,m_theta,t,x1)
          + gamma(t)c_theta(h,x_t,m_theta,t,x1)

x0_hat = m_theta - s_t [u_theta + gamma(t)c_theta].
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def toy_std(t: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Toy injected noise scale s_t."""
    return sigma_min + sigma_max * torch.sqrt(torch.clamp(t * (1.0 - t), min=0.0))


def gamma_schedule(t: torch.Tensor, gamma_max: float) -> torch.Tensor:
    return gamma_max * torch.sin(math.pi * t).pow(2)


def rho_schedule(t: torch.Tensor, rho_max: float, schedule: str = "sin2") -> torch.Tensor:
    if rho_max <= 0 or schedule == "none":
        return torch.zeros_like(t)
    if schedule == "constant":
        return rho_max * torch.ones_like(t)
    if schedule == "sin2":
        return rho_max * torch.sin(math.pi * t).pow(2)
    if schedule == "poly":
        return rho_max * 4.0 * t * (1.0 - t)
    raise ValueError(f"Unknown rho schedule: {schedule}")


# =============================================================================
# Curvature-controlled multi-branch manifolds
# =============================================================================

def branch_values(batch_size: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    b = torch.randint(0, 2, (batch_size,), device=device)
    y = torch.where(b == 0, -torch.ones_like(b), torch.ones_like(b)).to(dtype=dtype)
    return y


def x0_curve(theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r0 = 1.0 + 0.15 * torch.sin(3.0 * theta + 0.5 * y)
    x = r0 * torch.cos(theta)
    z = 0.75 * r0 * torch.sin(theta)
    offset = torch.stack([torch.zeros_like(y), 1.1 * y], dim=-1)
    return torch.stack([x, z], dim=-1) + offset


def x1_curve(theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r1 = 0.55 + 0.08 * torch.sin(2.0 * theta + 0.3 * y)
    x = r1 * torch.cos(theta + 0.6 * y)
    z = 0.45 * r1 * torch.sin(theta + 0.6 * y)
    offset = torch.stack([0.45 * torch.ones_like(y), 0.20 * y], dim=-1)
    return torch.stack([x, z], dim=-1) + offset


def normal_bend_vector(x0: torch.Tensor, x1: torch.Tensor, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    d = x1 - x0
    perp = torch.stack([-d[:, 1], d[:, 0]], dim=-1)
    perp = perp / (torch.linalg.norm(perp, dim=-1, keepdim=True) + 1e-8)
    amp = (0.35 + 0.15 * torch.sin(4.0 * theta + y))[:, None]
    return amp * perp


def curved_bridge_mu(
    x0: torch.Tensor,
    x1: torch.Tensor,
    theta: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    if t.dim() == 1:
        t = t[:, None]
    q = normal_bend_vector(x0, x1, theta, y)
    return (1.0 - t) * x0 + t * x1 + kappa * torch.sin(math.pi * t) * q


def linear_bridge_mu(x0_hat: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        t = t[:, None]
    return (1.0 - t) * x0_hat + t * x1


def sample_batch(batch_size: int, device: torch.device, args) -> Dict[str, torch.Tensor]:
    theta = 2.0 * math.pi * torch.rand(batch_size, device=device)
    y = branch_values(batch_size, device=device, dtype=torch.float32)

    x0 = x0_curve(theta, y)
    x1 = x1_curve(theta, y)

    t = torch.rand(batch_size, 1, device=device).clamp(args.t_eps, 1.0 - args.t_eps)
    s = toy_std(t, args.sigma_min, args.sigma_max)
    mt = curved_bridge_mu(x0, x1, theta, y, t, args.kappa)
    xt = mt + s * torch.randn_like(mt)

    return {"theta": theta, "y": y, "x0": x0, "x1": x1, "t": t, "s": s, "mt": mt, "xt": xt}


@torch.no_grad()
def make_branch_curves(n: int, device: torch.device, kappa: float, t_value: float = 0.0):
    theta = torch.linspace(0.0, 2.0 * math.pi, n, device=device)
    curves = {}
    for y_val in [-1.0, 1.0]:
        y = torch.full_like(theta, y_val)
        x0 = x0_curve(theta, y)
        x1 = x1_curve(theta, y)
        t = torch.full((n, 1), t_value, device=device)
        mt = curved_bridge_mu(x0, x1, theta, y, t, kappa)
        curves[y_val] = {"theta": theta, "y": y, "x0": x0, "x1": x1, "mt": mt}
    return curves


# =============================================================================
# Models
# =============================================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, hidden))
            layers.append(nn.SiLU())
            last = hidden
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BPModelV2(nn.Module):
    """
    Stronger non-canceling BP-GI2SB model.

    h = backbone(x_t,t,x1)
    m = bridge_head(h)

    u = u_theta(h,m,t,x1)
    c = c_theta(h,x_t,m,t,x1)

    This still forces endpoint reconstruction through m:
        x0_hat = m - s_t [u + gamma c]
    but u/c are not artificially starved of the shared feature h.
    """
    def __init__(self, hidden: int = 192, depth: int = 3, gamma_max: float = 0.1, detach_m_for_heads: bool = False):
        super().__init__()
        self.gamma_max = gamma_max
        self.detach_m_for_heads = detach_m_for_heads

        self.backbone = MLP(5, hidden, hidden, depth=depth)  # xt(2), t(1), x1(2)
        self.bridge_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

        # h(hidden) + m(2) + t(1) + x1(2)
        self.transport_head = MLP(hidden + 5, hidden, 2, depth=2)

        # h(hidden) + xt(2) + m(2) + t(1) + x1(2)
        self.corr_head = MLP(hidden + 7, hidden, 2, depth=2)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, x1: torch.Tensor, s: torch.Tensor):
        h = self.backbone(torch.cat([xt, t, x1], dim=-1))
        m = self.bridge_head(h)

        mh = m.detach() if self.detach_m_for_heads else m

        u = self.transport_head(torch.cat([h, mh, t, x1], dim=-1))
        c = self.corr_head(torch.cat([h, xt, mh, t, x1], dim=-1))

        gamma = gamma_schedule(t, self.gamma_max)
        o_hat = (xt - m) / s + u + gamma * c
        x0_hat = m - s * (u + gamma * c)

        return {"h": h, "m": m, "u": u, "c": c, "gamma": gamma, "o_hat": o_hat, "x0_hat": x0_hat}

    @torch.no_grad()
    def bridge_project(self, x: torch.Tensor, t: torch.Tensor, x1: torch.Tensor):
        h = self.backbone(torch.cat([x, t, x1], dim=-1))
        return self.bridge_head(h)


class SingleHeadModel(nn.Module):
    def __init__(self, hidden: int = 192, depth: int = 3):
        super().__init__()
        self.net = MLP(5, hidden, 2, depth=depth + 1)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, x1: torch.Tensor, s: torch.Tensor):
        o_hat = self.net(torch.cat([xt, t, x1], dim=-1))
        x0_hat = xt - s * o_hat
        return {"o_hat": o_hat, "x0_hat": x0_hat}

    @torch.no_grad()
    def bridge_project(self, x: torch.Tensor, t: torch.Tensor, x1: torch.Tensor):
        raise RuntimeError("SingleHeadModel has no bridge projection head.")


def build_model(args):
    if args.model_type == "bp":
        return BPModelV2(
            hidden=args.hidden,
            depth=args.depth,
            gamma_max=args.gamma_max,
            detach_m_for_heads=args.detach_m_for_heads,
        )
    if args.model_type == "single":
        return SingleHeadModel(hidden=args.hidden, depth=args.depth)
    raise ValueError(f"Unknown model_type: {args.model_type}")


# =============================================================================
# Losses
# =============================================================================

def compute_losses(model: nn.Module, batch: Dict[str, torch.Tensor], args):
    xt, t, x1, s = batch["xt"], batch["t"], batch["x1"], batch["s"]
    x0, mt = batch["x0"], batch["mt"]

    out = model(xt, t, x1, s)
    o_target = (xt - x0) / s

    loss_score = F.mse_loss(out["o_hat"], o_target)
    loss_endpoint = F.mse_loss(out["x0_hat"], x0)

    if args.model_type == "bp":
        u_target = (mt - x0) / s
        loss_bridge = F.mse_loss(out["m"], mt)
        loss_transport = F.mse_loss(out["u"], u_target)
        loss_corr = out["c"].pow(2).mean()

        total = (
            args.lambda_score * loss_score
            + args.lambda_bridge * loss_bridge
            + args.lambda_transport * loss_transport
            + args.lambda_endpoint * loss_endpoint
            + args.lambda_corr * loss_corr
        )
    else:
        loss_bridge = torch.zeros([], device=xt.device)
        loss_transport = torch.zeros([], device=xt.device)
        loss_corr = torch.zeros([], device=xt.device)
        total = args.lambda_score * loss_score + args.lambda_endpoint * loss_endpoint

    return total, {
        "total": float(total.detach().cpu()),
        "score": float(loss_score.detach().cpu()),
        "bridge": float(loss_bridge.detach().cpu()),
        "transport": float(loss_transport.detach().cpu()),
        "endpoint": float(loss_endpoint.detach().cpu()),
        "corr": float(loss_corr.detach().cpu()),
    }


# =============================================================================
# Sampler
# =============================================================================

@torch.no_grad()
def sample_reverse(
    model: nn.Module,
    x1: torch.Tensor,
    args,
    nfe: int,
    rho_max: float = 0.0,
    rho_sched: str = "sin2",
    projection_mode: str = "none",
    return_traj: bool = False,
):
    device = x1.device
    x = x1.clone()
    traj = [x.clone()] if return_traj else None
    times = torch.linspace(1.0 - args.t_eps, args.t_eps, nfe + 1, device=device)

    for k in range(nfe):
        tk = times[k]
        tn = times[k + 1]
        t = torch.full((x.shape[0], 1), float(tk), device=device)
        t_next = torch.full((x.shape[0], 1), float(tn), device=device)
        s = toy_std(t, args.sigma_min, args.sigma_max)
        s_next = toy_std(t_next, args.sigma_min, args.sigma_max)

        out = model(x, t, x1, s)
        x0_hat = out["x0_hat"]

        mu_t = linear_bridge_mu(x0_hat, x1, t)
        residual = x - mu_t
        mu_next = linear_bridge_mu(x0_hat, x1, t_next)
        x_pred = mu_next + (s_next / s) * residual

        if projection_mode != "none" and args.model_type == "bp" and rho_max > 0:
            m_next = model.bridge_project(x_pred, t_next, x1)
            rho = rho_schedule(t_next, rho_max, schedule=rho_sched)
            x_next = x_pred + rho * (m_next - x_pred)
        else:
            x_next = x_pred

        x = x_next
        if return_traj:
            traj.append(x.clone())

    if return_traj:
        return x, torch.stack(traj, dim=0), times
    return x, None, times


# =============================================================================
# Metrics helpers for notebooks
# =============================================================================

@torch.no_grad()
def nearest_distance_to_points(points: torch.Tensor, ref: torch.Tensor, chunk: int = 4096):
    outs = []
    for i in range(0, points.shape[0], chunk):
        outs.append(torch.cdist(points[i:i + chunk], ref).amin(dim=1))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def union_bridge_curve(n: int, device: torch.device, kappa: float, t_value: float):
    curves = make_branch_curves(n, device, kappa=kappa, t_value=t_value)
    return torch.cat([curves[-1.0]["mt"], curves[1.0]["mt"]], dim=0)


@torch.no_grad()
def branch_accuracy(final: torch.Tensor, y_true: torch.Tensor, n_curve: int, device: torch.device, kappa: float):
    curves = make_branch_curves(n_curve, device, kappa=kappa, t_value=0.0)
    d_neg = nearest_distance_to_points(final, curves[-1.0]["x0"])
    d_pos = nearest_distance_to_points(final, curves[1.0]["x0"])
    y_pred = torch.where(d_pos < d_neg, torch.ones_like(y_true), -torch.ones_like(y_true))
    return (y_pred == y_true).float().mean().item()


# =============================================================================
# Visualization
# =============================================================================

@torch.no_grad()
def plot_toy_manifolds(args, device: torch.device, outdir: Path):
    plt.figure(figsize=(6, 6), dpi=160)
    colors = {-1.0: "tab:blue", 1.0: "tab:orange"}

    for y_val in [-1.0, 1.0]:
        curves0 = make_branch_curves(args.curve_points, device, kappa=args.kappa, t_value=0.0)[y_val]
        curves1 = make_branch_curves(args.curve_points, device, kappa=args.kappa, t_value=1.0)[y_val]
        plt.plot(to_np(curves0["mt"][:, 0]), to_np(curves0["mt"][:, 1]), color=colors[y_val], lw=2, label=f"M0 y={int(y_val)}")
        plt.plot(to_np(curves1["mt"][:, 0]), to_np(curves1["mt"][:, 1]), color=colors[y_val], ls=":", lw=2, label=f"M1 y={int(y_val)}")

        for tval in [0.25, 0.50, 0.75]:
            mt = make_branch_curves(args.curve_points, device, kappa=args.kappa, t_value=tval)[y_val]["mt"]
            plt.plot(to_np(mt[:, 0]), to_np(mt[:, 1]), color=colors[y_val], ls="--", alpha=0.7, label=f"Mt y={int(y_val)}, t={tval}")

    plt.axis("equal")
    plt.grid(True, alpha=0.25)
    plt.title(f"Curved multi-branch toy, kappa={args.kappa}")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "toy_multibranch_manifolds.png", dpi=180)
    plt.close()


@torch.no_grad()
def plot_training_losses(rows: List[Dict], outdir: Path):
    if not rows:
        return
    plt.figure(figsize=(7, 4), dpi=160)
    steps = [r["step"] for r in rows]
    for key in ["total", "score", "bridge", "transport", "endpoint", "corr"]:
        vals = [r[key] for r in rows]
        plt.plot(steps, vals, label=key)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.title("Training losses")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training_losses.png", dpi=180)
    plt.close()


# =============================================================================
# Training main
# =============================================================================

def train(args):
    outdir = ensure_dir(args.outdir)
    ckpt_dir = ensure_dir(outdir / "checkpoints")
    vis_dir = ensure_dir(outdir / "visuals")

    set_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"kappa: {args.kappa}")
    print(f"hidden: {args.hidden}, depth: {args.depth}")
    if args.model_type == "bp":
        print(f"detach_m_for_heads: {args.detach_m_for_heads}")

    model = build_model(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rows = []
    for step in range(1, args.steps + 1):
        batch = sample_batch(args.batch_size, device, args)
        loss, loss_dict = compute_losses(model, batch, args)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % args.log_every == 0:
            row = {"step": step, **loss_dict}
            rows.append(row)
            print(
                f"step {step:07d} | total={loss_dict['total']:.3e} "
                f"score={loss_dict['score']:.3e} bridge={loss_dict['bridge']:.3e} "
                f"endpoint={loss_dict['endpoint']:.3e}"
            )

        if step % args.save_every == 0 or step == args.steps:
            payload = {"model": model.state_dict(), "args": vars(args)}
            torch.save(payload, ckpt_dir / f"ckpt_step_{step:07d}.pt")
            torch.save(payload, ckpt_dir / "latest.pt")

    with open(outdir / "train_losses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_training_losses(rows, vis_dir)
    plot_toy_manifolds(args, device, vis_dir)

    print(f"Done. Outputs saved to: {outdir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train curvature-controlled multi-branch toy V2.")

    p.add_argument("--outdir", type=str, default="runs_toy_curved_bp_v2")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--model_type", type=str, default="bp", choices=["bp", "single"])

    # Toy geometry.
    p.add_argument("--kappa", type=float, default=0.6)
    p.add_argument("--sigma_min", type=float, default=0.02)
    p.add_argument("--sigma_max", type=float, default=0.7)
    p.add_argument("--gamma_max", type=float, default=0.1)
    p.add_argument("--t_eps", type=float, default=1e-3)
    p.add_argument("--curve_points", type=int, default=2000)

    # Training.
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--detach_m_for_heads", action="store_true")

    # Loss weights.
    p.add_argument("--lambda_score", type=float, default=1.0)
    p.add_argument("--lambda_bridge", type=float, default=2.0)
    p.add_argument("--lambda_transport", type=float, default=1.0)
    p.add_argument("--lambda_endpoint", type=float, default=0.2)
    p.add_argument("--lambda_corr", type=float, default=1e-4)

    # Logging.
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=5000)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
