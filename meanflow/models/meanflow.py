import torch
import torch.nn as nn
import random

from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net


class MeanFlow(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(MeanFlow, self).__init__()
        self.net = arch(**net_configs)
        self.args = args

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))
        
        self.net_ema = init_ema(self.net, arch(**net_configs), args.ema_decay)

        # maintain extra ema nets
        self.ema_decays = args.ema_decays
        for i, ema_decay in enumerate(self.ema_decays):
            self.add_module(f"net_ema{i + 1}", init_ema(self.net, arch(**net_configs), ema_decay))

    def update_ema(self):
        self.num_updates += 1
        # num_updates = self.num_updates.item()
        num_updates = self.num_updates

        update_ema_net(self.net, self.net_ema, num_updates)

        # update extra ema
        for i in range(len(self.ema_decays)):
            update_ema_net(self.net, self._modules[f"net_ema{i + 1}"], num_updates)

    def forward_with_loss(self, x, aug_cond, hybrid_ratio=0.5):

        device = x.device
        e = torch.randn_like(x).to(device)
        t, r = sample_two_timesteps(self.args, num_samples=x.shape[0], device=device)
        t, r = t.view(-1, 1, 1, 1), r.view(-1, 1, 1, 1)

        z = (1 - t) * x + t * e
        v = e - x

        # define network function
        def u_func(z, t, r):
            h = t - r
            return self.net(z, (t.view(-1), h.view(-1)), aug_cond)

        dtdt = torch.ones_like(t)
        drdt = torch.zeros_like(r)
        dtdr = torch.zeros_like(t)
        drdr = torch.ones_like(r)

        with torch.amp.autocast("cuda", enabled=False):
            u_pred, dudt = torch.func.jvp(u_func, (z, t, r), (v, dtdt, drdt))
            _, dudr = torch.func.jvp(u_func, (z, t, r), (torch.zeros_like(v), dtdr, drdr))

        
            u_tgt_t = (v - (t - r) * dudt).detach()
            u_tgt_r = (v + (t - r) * dudr).detach()


            loss_t = (u_pred - u_tgt_t)**2
            loss_r = (u_pred - u_tgt_r)**2

            mask = (torch.rand((), device=device) < hybrid_ratio).to(loss_t.dtype)
            loss = mask * loss_t + (1 - mask) * loss_r
            loss = loss.sum(dim=(1, 2, 3))  # squared l2 loss

            # adaptive weighting
            adp_wt = (loss.detach() + self.args.norm_eps) ** self.args.norm_p
            loss = loss / adp_wt

            loss = loss.mean()  # mean over batch dimension

            loss_t_val = loss_t.sum(dim=(1, 2, 3))
            adp_wt_t = (loss_t_val.detach() + self.args.norm_eps) ** self.args.norm_p
            loss_t_val = (loss_t_val / adp_wt_t).mean()

            loss_r_val = loss_r.sum(dim=(1, 2, 3))
            adp_wt_r = (loss_r_val.detach() + self.args.norm_eps) ** self.args.norm_p
            loss_r_val = (loss_r_val / adp_wt_r).mean()

        return loss, loss_t_val, loss_r_val
    
    def sample(self, samples_shape, net=None, device=None):
        net = net if net is not None else self.net_ema                

        e = torch.randn(samples_shape, dtype=torch.float32, device=device)
        z_1 = e
        t = torch.ones(samples_shape[0], device=device)
        r = torch.zeros(samples_shape[0], device=device)
        u = net(z_1, (t, t - r), aug_cond=None)
        z_0 = z_1 - u
        return z_0

    def sample_multisteps(self, samples_shape, num_steps, net=None, device=None):
        net = net if net is not None else self.net_ema

        z = torch.randn(samples_shape, dtype=torch.float32, device=device)

        dt = 1.0 / num_steps
        for i in range(num_steps, 0, -1):
            t_curr = i * dt
            t = torch.full((samples_shape[0],), t_curr, dtype=torch.float32, device=device)
            h = torch.full((samples_shape[0],), dt, dtype=torch.float32, device=device)
            u = net(z, (t, h), aug_cond=None)
            z = z - dt * u

        return z
