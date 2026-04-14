import torch
from torch import nn
import numpy as np
import sympy
from copy import deepcopy
from .torchmd_et import TorchMD_VQ_ET, ParamGVPFFNLayer
from .interpolant_matcher import INTERP_MATCHER
from .graph import construct_edges

import sys
sys.path.append('..')
from utils import *
from utils.constants import kB
from utils.torchmd_utils import scatter


class EnergyKernel(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, rbf_dim, expand_embed_dim, heads, layers, 
                 cutoff_lower=0.0, cutoff_upper=5.0, cutoff_H=3.5, k_neighbors=16, s_a=0.04, n_env=4):
        super(EnergyKernel, self).__init__()

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.rbf_dim = rbf_dim
        self.heads = heads
        self.layers = layers
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.cutoff_H = cutoff_H
        self.k_neighbors = k_neighbors
        self.s_a = s_a
        self.n_env = n_env

        # TorchMD-net
        self.net = TorchMD_VQ_ET(
            hidden_channels=hidden_dim, extra_channels=1, expand_embed_channels=expand_embed_dim, num_layers=layers,
            num_rbf=rbf_dim, num_heads=heads, cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, max_z=NUM_BIO_ATOM_TYPE
        )

        self.energy_ffn = nn.ModuleList(
            [ParamGVPFFNLayer(hidden_dim, ffn_dim, d_output=1) for _ in range(n_env)]
        )

        # mapping various environments to: 0 - small molecules, 1 - (AMBER) solvated peptides & proteins, 2 - denoise, 3 - (DFT) solvated proteins
        self.env_map = {0: 1, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 0, 7: 3, 8: 0}

    def _train(self, batch):
        x0, atype, abid, mask, edge_mask = batch["x0"], batch["atype"], batch["abid"], batch["mask"], batch["edge_mask"]
        N = x0.shape[0]

        t_all_one = torch.ones(N, 1, dtype=x0.dtype, device=x0.device)
        t_all_zero = torch.zeros(N, 1, dtype=x0.dtype, device=x0.device)
        # random choose t=0 or t=1
        is_t_zero = np.random.rand() < 0.5
        t_diff = t_all_zero if is_t_zero else t_all_one

        _, num_atoms_per_batch = torch.unique_consecutive(abid, return_counts=True)
        num_atoms_per_batch.unsqueeze_(1)

        env_map = self.env_map[batch["env"]]
        # denoise
        if env_map == 2:
            x, gt_energy, gt_force = self.denoise(x0, batch=abid)
        else:
            x = x0
            # formation energy per atom
            # energy unit: MJ/mole, force unit: MJ/mole/nm, distance: Angstrom
            gt_energy = batch["pot0"] / num_atoms_per_batch * 10
            gt_force = batch["force0"]

        _, pred_force = self.energy_grad(z=atype, x=x, batch=abid, t=t_diff, env=env_map, eval_=False)
        # if is_denoise:
        #     pred_force = pred_force * pred_scalar[abid]
        # loss_energy = F.mse_loss(pred_energy / num_atoms_per_batch, gt_energy, reduction='mean')
        loss_energy = 0
        loss_force = F.mse_loss(pred_force[mask], gt_force[mask], reduction='mean')
        # print(f"env: {batch['env']}, pred_force: {pred_force.std()}, gt_force: {gt_force.std()}")

        loss = 1.0 * loss_energy + 1.0 * loss_force

        return loss, (loss_energy, loss_force)

    @torch.no_grad()
    def denoise(self, x, batch):
        x -= scatter(src=x, index=batch, dim=0, reduce='mean')[batch]
        noise = torch.randn_like(x)
        x_n = x + noise * self.s_a
        x_n -= scatter(src=x_n, index=batch, dim=0, reduce='mean')[batch]
        den_energy = -1.5 * self.s_a + 0.5 * scatter(src=torch.norm(x_n - x, dim=-1, keepdim=True) ** 2,
                                                     index=batch, dim=0, reduce='mean') / self.s_a
        den_force = -(x_n - x) / self.s_a
        return x_n, den_energy, den_force

    def energy_grad(self, z, x, batch, t, env, eval_=False):
        assert (
                0 <= env < self.n_env
        ), f'environment {env} out of range'
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            # construct edges
            edge_index, edge_weight, edge_vec = construct_edges(
                Z=z, X=x_, bid=batch, mask=torch.zeros_like(z).long(),
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                cutoff_H=self.cutoff_H,
                k_neighbors=self.k_neighbors
            )
            h, vec = self(z=z, x=x_, batch=batch, t=t, edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec)
            h_out, _ = self.energy_ffn[env](h, vec)
            pred_energy = scatter(src=torch.mean(h_out, dim=-1, keepdim=True), index=batch, dim=0, reduce='sum')
            # scalar_out, _ = self.scalar_ffn(h, vec)
            # pred_scalar = scatter(src=torch.mean(scalar_out, dim=-1, keepdim=True), index=batch, dim=0, reduce='mean')
            grad_outputs = torch.ones_like(pred_energy)
            grad_x = \
                torch.autograd.grad(outputs=pred_energy, inputs=x_, grad_outputs=grad_outputs, create_graph=not eval_)[0]
        return pred_energy, -grad_x

    # NOTE: slightly modified: using self-constructed edges
    def forward(self, z, x, batch, t, edge_index, edge_weight, edge_vec):
        h, vec, _, _, _ = self.net(z=z, pos=x, batch=batch, t=t, edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec)
        return h, vec


class GeomBM(nn.Module):
    def __init__(self, encoder, ffn_dim, s_eu=0.2):
        super(GeomBM, self).__init__()

        self.encoder: EnergyKernel = encoder
        self.ffn_dim = ffn_dim
        self.hidden_dim = self.encoder.hidden_dim
        self.cutoff_lower = self.encoder.cutoff_lower
        self.cutoff_upper = self.encoder.cutoff_upper
        self.cutoff_H = self.encoder.cutoff_H
        self.k_neighbors = self.encoder.k_neighbors
        self.noise_base = s_eu

        self.vel_ffn = ParamGVPFFNLayer(self.hidden_dim, ffn_dim, vector_norm=True)
        self.den_ffn = ParamGVPFFNLayer(self.hidden_dim, ffn_dim, vector_norm=True)

        self.vel_scale = 0.5

    def adj_noise(self, temp):
        return self.noise_base * np.sqrt(kB * temp)

    def _train(self, batch, temperature=300):
        x0, x1, atype, abid, mask, edge_mask = batch["x0"], batch["x1"], batch["atype"], batch["abid"], batch["mask"], batch["edge_mask"]
        N = x0.shape[0]

        t_all_one = torch.ones(N, 1, dtype=x0.dtype, device=x0.device)
        # uniformly sample time step
        t = np.random.rand()  # use the same time step for all mini-batches
        t_diff = t_all_one * t

        # temperature steerable
        sigma = self.adj_noise(temperature)

        # sample interpolants
        xt, vel_gt, den_gt = INTERP_MATCHER.sample_conditional_flow(x0, x1, t_diff, sigma)

        # construct edges
        edge_index, edge_weight, edge_vec = construct_edges(
            Z=atype, X=xt, bid=abid, mask=edge_mask,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            cutoff_H=self.cutoff_H,
            k_neighbors=self.k_neighbors
        )

        vel_pred, den_pred, _, _ = self(z=atype, x=xt, batch=abid, t=t_diff,
                                        edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec,
                                        rescale=False)

        loss_vel = F.mse_loss(vel_pred[mask], vel_gt[mask] * self.vel_scale, reduction='mean')
        loss_den = F.mse_loss(den_pred[mask], den_gt[mask], reduction='mean')
        loss_aux = 0

        same_abid = abid.unsqueeze(-1).repeat(1, abid.shape[0])
        same_abid = same_abid == same_abid.transpose(0, 1)  # (N, N)
        same_mask = torch.outer(mask, mask)                 # (N, N)

        vel_pred /= self.vel_scale
        x0_pred = INTERP_MATCHER.expectation_on_x0(xt, vel_pred, den_pred, t_diff, sigma)
        x1_pred = INTERP_MATCHER.expectation_on_x1(xt, vel_pred, den_pred, t_diff, sigma)

        D0 = torch.cdist(x0, x0)  # (N, N)
        pred_D0 = torch.cdist(x0_pred, x0_pred)
        valid_mask0 = (D0 > 0.0) & (D0 < 6.0) & same_abid & same_mask
        loss_aux += (1 - t) * F.mse_loss(pred_D0[valid_mask0], D0[valid_mask0], reduction='mean')

        D1 = torch.cdist(x1, x1)
        pred_D1 = torch.cdist(x1_pred, x1_pred)
        valid_mask1 = (D1 > 0.0) & (D1 < 6.0) & same_abid & same_mask
        loss_aux += t * F.mse_loss(pred_D1[valid_mask1], D1[valid_mask1], reduction='mean')
        
        loss = 1.0 * loss_vel + 1.0 * loss_den + 0.25 * loss_aux

        return loss, (loss_vel, loss_den, loss_lig, loss_aux)

    def forward(self, z, x, batch, t, edge_index, edge_weight, edge_vec, rescale=False):
        h, vec = self.encoder(z=z, x=x, batch=batch, t=t, edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec)
        _, vel_out = self.vel_ffn(h, vec)
        _, den_out = self.den_ffn(h, vec)
        # SO(3)
        vel_pred = torch.sum(vel_out, dim=-1)  # (N, 3)
        den_pred = torch.sum(den_out, dim=-1)
        if rescale:
            vel_pred /= self.vel_scale
        return vel_pred, den_pred, h, vec

    def sde(self, batch, sde_step=50, temp=300, guidance=None):
        x, atype, abid, edge_mask = batch["x0"], batch["atype"], batch["abid"], batch["edge_mask"]
        N = x.shape[0]

        xt = x.clone()

        sde_step = np.linspace(0, 1. - 1. / sde_step, sde_step)
        dt = sde_step[1] - sde_step[0]

        t_all_one = torch.ones(N, 1, dtype=x.dtype, device=x.device)
        sigma = self.adj_noise(temp)

        for t in sde_step:
            t_diff = t_all_one * t
            # construct edges
            edge_index, edge_weight, edge_vec = construct_edges(
                Z=atype, X=xt, bid=abid, mask=edge_mask,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                cutoff_H=self.cutoff_H,
                k_neighbors=self.k_neighbors
            )
            vel_pred, den_pred, _, _ = self(z=atype, x=xt, batch=abid, t=t_diff,
                                            edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec,
                                            rescale=True)
            drift = INTERP_MATCHER.drift(vel_pred, den_pred, t, sigma)
            diffusion = INTERP_MATCHER.diffusion(sigma)
            xt = xt + drift * dt + diffusion * np.sqrt(dt) * torch.randn_like(xt) if t != sde_step[-1] \
                else xt + drift * dt

        return xt


class GeomFBM(nn.Module):
    def __init__(self, baseline, hidden_dim, ffn_dim, rbf_dim, expand_embed_dim, heads, layers):
        super(GeomFBM, self).__init__()

        self.base: GeomBM = baseline
        # froze parameters of baseline model
        for param in self.base.parameters():
            param.requires_grad = False

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.rbf_dim = rbf_dim
        self.heads = heads
        self.layers = layers

        self.cutoff_lower = self.base.cutoff_lower
        self.cutoff_upper = self.base.cutoff_upper
        self.cutoff_H = self.base.cutoff_H
        self.k_neighbors = self.base.k_neighbors

        self.net = TorchMD_VQ_ET(
            hidden_channels=hidden_dim, extra_channels=1, expand_embed_channels=expand_embed_dim, num_layers=layers,
            num_rbf=rbf_dim, num_heads=heads, cutoff_lower=self.cutoff_lower, cutoff_upper=self.cutoff_upper, max_z=NUM_BIO_ATOM_TYPE
        )

        self.bd0_ffn = ParamGVPFFNLayer(self.hidden_dim, ffn_dim, vector_norm=True)
        self.bd1_ffn = ParamGVPFFNLayer(self.hidden_dim, ffn_dim, vector_norm=True)
        self.iff_ffn = ParamGVPFFNLayer(self.hidden_dim, ffn_dim, vector_norm=True)

        # scaling for numerical stability
        self.iff_scale = 0.5

    def _train(self, batch, temperature):
        x0, x1, atype, abid, mask, edge_mask = batch["x0"], batch["x1"], batch["atype"], batch["abid"], batch["mask"], batch["edge_mask"]
        N = x0.shape[0]

        t_all_one = torch.ones(N, 1, dtype=x0.dtype, device=x0.device)

        # uniformly sample time step
        t = np.random.rand()  # use the same time step for all mini-batches
        t_diff = t_all_one * t

        # temperature steerable
        sigma = self.base.adj_noise(temperature)

        # sample interpolants
        xt, _, _ = INTERP_MATCHER.sample_conditional_flow(x0, x1, t_diff, sigma)

        # construct edges
        edge_index, edge_weight, edge_vec = construct_edges(
            Z=atype, X=xt, bid=abid, mask=edge_mask,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            cutoff_H=self.cutoff_H,
            k_neighbors=self.k_neighbors
        )

        # intermediate force field
        pot0, pot1 = batch["pot0"], batch["pot1"]

        # compute q_t(x_t)
        _, den_pred, h_prev, _ = self.base(z=atype, x=xt, batch=abid, t=t_diff,
                                           edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec,
                                           rescale=True)
        bd0_pred, bd1_pred, iff_pred = self(z=atype, x=xt, batch=abid, t=t_diff,
                                            edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec,
                                            h_prev=h_prev)
        iff = (1 - t) * bd0_pred.detach() + t * bd1_pred.detach() + t * (1 - t) * iff_pred
        loss_bound = F.mse_loss(bd0_pred[mask], -batch["force0"][mask], reduction='mean') + \
                     F.mse_loss(bd1_pred[mask], -batch["force1"][mask], reduction='mean')

        target = torch.exp(-(pot0 + pot1))[abid] * (
                INTERP_MATCHER.denoiser_to_score(den_pred, t, sigma) -
                INTERP_MATCHER.conditional_score(xt, x0, x1, t, sigma)
        )
        deno = 0
        for i in range(pot0.shape[0]):
            x0i, x1i, xti = x0[abid == i], x1[abid == i], xt[abid == i]
            deno += INTERP_MATCHER.likelihood(xti, x0i, x1i, t, sigma) * torch.exp(-(pot0[i] + pot1[i]))
        deno /= pot0.shape[0]
        target /= deno

        # print(f"t: {t}, force0: {batch['force0'].std()}, bd0: {bd0_pred.std()},\n"
        #       f"target: {self.iff_scale * target.std()}, iff_out: {iff_pred.std()}, iff: {iff.std()}")

        loss_iff = F.mse_loss(iff[mask], self.iff_scale * target[mask], reduction='mean')

        loss = 1.0 * loss_iff + 1.0 * loss_bound

        return loss, (loss_iff, loss_bound)

    def forward(self, z, x, batch, t, edge_index, edge_weight, edge_vec, h_prev):
        h, vec, _, _, _ = self.net(z=z, pos=x, batch=batch, t=t, edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec)
        # incorporate encoder representation
        h = h + h_prev
        _, bd0_out = self.bd0_ffn(h, vec)
        _, bd1_out = self.bd1_ffn(h, vec)
        _, iff_out = self.iff_ffn(h, vec)
        bd0_pred = torch.sum(bd0_out, dim=-1)  # (N, 3)
        bd1_pred = torch.sum(bd1_out, dim=-1)
        iff_pred = torch.sum(iff_out, dim=-1)
        return bd0_pred, bd1_pred, iff_pred

    def sde(self, batch, sde_step=50, temp=300, guidance=0.1):
        x0, atype, abid, edge_mask = batch["x0"], batch["atype"], batch["abid"], batch["edge_mask"]
        N = x0.shape[0]

        xt = x0.clone()

        sde_step = np.linspace(0, 1. - 1. / sde_step, sde_step)
        dt = sde_step[1] - sde_step[0]

        t_all_one = torch.ones(N, 1, dtype=x0.dtype, device=x0.device)
        sigma = self.base.adj_noise(temp)

        for t in sde_step:
            t_diff = t_all_one * t
            # construct edges
            edge_index, edge_weight, edge_vec = construct_edges(
                Z=atype, X=xt, bid=abid, mask=edge_mask,
                cutoff_lower=self.cutoff_lower,
                cutoff_upper=self.cutoff_upper,
                cutoff_H=self.cutoff_H,
                k_neighbors=self.k_neighbors
            )
            vel_pred, den_pred, h_prev, _ = self.base(z=atype, x=xt, batch=abid, t=t_diff,
                                                      edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec,
                                                      rescale=True)
            bd0_pred, bd1_pred, iff_pred = self(z=atype, x=xt, batch=abid, t=t_diff,
                                                edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec,
                                                h_prev=h_prev)
            iff = (1 - t) * bd0_pred + t * bd1_pred + t * (1 - t) * iff_pred
            iff /= self.iff_scale
            drift = INTERP_MATCHER.drift(vel_pred, den_pred, t, sigma) - guidance * iff
            diffusion = INTERP_MATCHER.diffusion(sigma)
            xt = xt + drift * dt + diffusion * np.sqrt(dt) * torch.randn_like(xt) if t != sde_step[-1] \
                else xt + drift * dt

        return xt
