# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "fire>=0.5.0",
#     "matplotlib>=3.7.0",
#     "seaborn>=0.13.0",
#     "numpy>=1.24.0",
#     "lion-pytorch>=0.2.3",
#     "adam-atan2-pytorch>=0.3.6",
#     "einops>=0.8.0",
#     "torch-einops-utils>=0.1.24"
# ]
# ///
#
# Non-stationary bit-flipping benchmark (Dohare et al. 2021)
# trains a small student network against an LTU lookup-table target
# under a sequence of bit flips, comparing base optimizers, + discrete CBP, and + continuous CCBP.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from adam_atan2_pytorch import AdamAtan2, MuonAdamAtan2
from einops import reduce
from lion_pytorch import Lion
from torch import Tensor, nn
from torch.optim import SGD, Adam, Optimizer
from torch_einops_utils import pad_ndim, pad_right_ndim_to

from ccbp_pytorch import CCBP

# lion keeps ~1% of its momentum updated after a reset, any more causes instability

def lion_adjust_state(
    optimizer: Optimizer,
    param: nn.Parameter,
    alpha: Tensor,
    dim: int,
    state_name: str,
    buffer: Tensor,
    policy = 'zero',
    reset_threshold = 0.01,
    **kwargs
):
    alpha_expanded = pad_right_ndim_to(alpha, buffer.ndim) if dim == 0 else pad_ndim(alpha, (dim, buffer.ndim - dim - 1))
    buffer.masked_fill_(alpha_expanded.to(buffer.device) > reset_threshold, 0.)

CCBP.register_optimizer_handler(Lion, lion_adjust_state)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# target environment

class TargetLTUNetwork:
    def __init__(
        self,
        num_bits = 20,
        num_hidden = 100,
        beta = 0.7,
        seed = 42,
        device = 'cpu'
    ):
        self.device = device
        torch.manual_seed(seed)

        self.v = torch.randint(0, 2, (num_hidden, num_bits), device = device).float() * 2.0 - 1.0
        s = reduce((self.v == -1.0).float(), 'h b -> h', 'sum')
        self.theta = (num_bits + 1) * beta - s
        self.out_weights = torch.randn(1, num_hidden, device = device)

    def __call__(self, x: Tensor) -> Tensor:
        theta = pad_ndim(self.theta, (1, 0))
        h = (torch.einsum('b i, h i -> b h', x, self.v) > theta).float()
        return torch.einsum('b h, o h -> b o', h, self.out_weights)

class BitFlippingEnv:
    def __init__(
        self,
        num_bits = 20,
        num_flipping = 15,
        num_hidden = 100,
        beta = 0.7,
        seed = 42,
        device = 'cpu'
    ):
        self.num_bits = num_bits
        self.num_flipping = num_flipping
        self.num_iid = num_bits - num_flipping
        self.device = device

        self.target_net = TargetLTUNetwork(
            num_bits = num_bits,
            num_hidden = num_hidden,
            beta = beta,
            seed = seed,
            device = device
        )
        self.flipping_state = torch.randint(0, 2, (num_flipping,), device = device).float()

    def flip(self):
        idx = random.randint(0, self.num_flipping - 1)
        self.flipping_state[idx] = 1.0 - self.flipping_state[idx]

    def get_batch(self, batch_size = 32) -> tuple[Tensor, Tensor]:
        flips = self.flipping_state.unsqueeze(0).expand(batch_size, -1)
        iids = torch.randint(0, 2, (batch_size, self.num_iid), device = self.device).float()
        x = torch.cat([flips, iids], dim = -1)

        with torch.no_grad():
            y = self.target_net(x)

        return x, y

# student model

class StudentMLP(nn.Module):
    def __init__(self, in_dim = 20, hidden_dim = 8, out_dim = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))

def compute_effective_rank(features: Tensor, eps = 1e-6) -> float:
    if features.shape[0] < 2:
        return 1.0

    h = features - features.mean(dim = 0, keepdim = True)
    s = torch.linalg.svdvals(h)
    return (reduce(s ** 2, 'd ->', 'sum') / max(s[0] ** 2, eps)).item()

# optimizer builder

def build_optimizer(
    opt_name: str,
    model: nn.Module,
    mode: str,
    lr = None,
    replacement_rate = 0.25,
    continuous_rate = 0.25,
    steepness = 16.0,
    reset_interval = 20,
    discrete_maturity_steps = 40,
    continuous_maturity_steps = 0,
    adam_lr = 1e-2,
    lion_lr = 2e-3,
    atan2_lr = 1e-2,
    muon_lr = 1e-2,
    sgd_lr = 1e-2
) -> Optimizer:
    name = opt_name.strip()

    assert name in {'Adam', 'Lion', 'AdamAtan2', 'MuonAdamAtan2', 'SGD'}, f"opt_name '{name}' must be one of {{'Adam', 'Lion', 'AdamAtan2', 'MuonAdamAtan2', 'SGD'}}"
    assert mode in {'base', 'discrete', 'continuous'}, f"mode '{mode}' must be one of {{'base', 'discrete', 'continuous'}}"

    if name == 'Adam':
        base_opt = Adam(model.parameters(), lr = default(lr, adam_lr))
    elif name == 'Lion':
        base_opt = Lion(model.parameters(), lr = default(lr, lion_lr))
    elif name == 'AdamAtan2':
        base_opt = AdamAtan2(model.parameters(), lr = default(lr, atan2_lr))
    elif name == 'MuonAdamAtan2':
        base_lr = default(lr, muon_lr)
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        adam_params = [p for p in model.parameters() if p.ndim < 2]
        base_opt = MuonAdamAtan2(muon_params, adam_params, lr = base_lr, muon_lr = base_lr)
    elif name == 'SGD':
        base_opt = SGD(model.parameters(), lr = default(lr, sgd_lr), momentum = 0.9)

    if mode == 'base':
        return base_opt

    return CCBP(
        base_opt,
        model = model,
        continuous = mode == 'continuous',
        replacement_rate = replacement_rate,
        continuous_rate = continuous_rate,
        steepness = steepness,
        maturity_steps = continuous_maturity_steps if mode == 'continuous' else discrete_maturity_steps,
        reset_interval = reset_interval,
        utility_type = 'abs_mean',
        second_moment_policy = 'zero',
        exclude_module_names = ['fc2']
    )

# simulation

def run_single_simulation(
    opt_name: str,
    mode: str,
    tasks = 30,
    steps_per_task = 250,
    hidden_dim = 8,
    lr = None,
    batch_size = 32,
    eval_batch_size = 128,
    seeds = (42, 101, 2024),
    device = 'cpu'
) -> dict:
    all_mses, all_dorms, all_ranks = [], [], []

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)

        env = BitFlippingEnv(num_bits = 20, num_flipping = 15, seed = 42, device = device)
        model = StudentMLP(in_dim = 20, hidden_dim = hidden_dim).to(device)
        opt = build_optimizer(opt_name, model, mode, lr = lr)

        task_mses, task_dorms, task_ranks = [], [], []

        for t in range(tasks):
            if t > 0:
                env.flip()

            for _ in range(steps_per_task):
                x, y = env.get_batch(batch_size = batch_size)
                opt.zero_grad()
                loss = F.mse_loss(model(x), y)
                loss.backward()
                opt.step()

            with torch.no_grad():
                x_eval, y_eval = env.get_batch(batch_size = eval_batch_size)
                h = model.act(model.fc1(x_eval))
                task_mses.append(F.mse_loss(model.fc2(h), y_eval).item())
                task_dorms.append(reduce((h == 0).all(dim = 0).float(), 'd ->', 'mean').item() * 100.0)
                task_ranks.append(compute_effective_rank(h))

        all_mses.append(task_mses)
        all_dorms.append(task_dorms)
        all_ranks.append(task_ranks)

    half = tasks // 2

    return dict(
        mses = np.mean(all_mses, axis = 0),
        dorms = np.mean(all_dorms, axis = 0),
        ranks = np.mean(all_ranks, axis = 0),
        late_mse = float(np.mean(np.mean(all_mses, axis = 0)[half:])),
        late_dorm = float(np.mean(np.mean(all_dorms, axis = 0)[half:])),
        late_rank = float(np.mean(np.mean(all_ranks, axis = 0)[half:]))
    )

# main

def train(
    tasks = 30,
    steps_per_task = 250,
    hidden_dim = 8,
    optimizers = 'Adam,Lion,AdamAtan2,MuonAdamAtan2',
    modes = 'base,continuous',
    save_fig = 'bit_flipping_all_optimizers.png',
    device = 'cpu'
):
    if isinstance(optimizers, str):
        opt_list = [o.strip() for o in optimizers.split(',') if o.strip()]
    else:
        opt_list = [str(o).strip() for o in optimizers if str(o).strip()]

    if isinstance(modes, str):
        mode_list = [m.strip() for m in modes.split(',') if m.strip()]
    else:
        mode_list = [str(m).strip() for m in modes if str(m).strip()]
    results = dict()

    for opt_name in opt_list:
        for mode in mode_list:
            label = f"{opt_name}_{mode}"
            print(f"[{label}] running...")
            results[label] = run_single_simulation(
                opt_name, mode = mode, tasks = tasks, steps_per_task = steps_per_task,
                hidden_dim = hidden_dim, device = device
            )

    # summary table

    print('\n' + '=' * 84)
    print(f"{'Optimizer':<16} | {'Mode':<12} | {'Late MSE':<10} | {'Dormancy %':<11} | {'Rank':<8}")
    print('-' * 84)
    for label, res in results.items():
        print(f"{label.rsplit('_', 1)[0]:<16} | {label.rsplit('_', 1)[1]:<12} | {res['late_mse']:<10.4f} | {res['late_dorm']:<11.1f} | {res['late_rank']:<8.2f}")
    print('=' * 84)

    # figure

    sns.set_theme(style = 'whitegrid', font_scale = 1.05)

    palette = dict(
        Adam = '#d62728',
        Lion = '#9467bd',
        AdamAtan2 = '#ff7f0e',
        MuonAdamAtan2 = '#1f77b4',
        SGD = '#2ca02c'
    )

    comparables = [
        (o, results[f'{o}_base'], results[f'{o}_continuous'])
        for o in opt_list
        if all(f'{o}_{m}' in results for m in ('base', 'continuous'))
    ]

    fig, axes = plt.subplots(2, 2, figsize = (16, 11), dpi = 300)
    x_tasks = np.arange(1, tasks + 1)
    x_idx = np.arange(len(comparables))
    width = 0.35

    ax1 = axes[0, 0]
    for opt_name in opt_list:
        c = palette.get(opt_name, '#333333')
        for mode, style in (('base', '--'), ('continuous', '-')):
            if f"{opt_name}_{mode}" not in results:
                continue
            curves = results[f"{opt_name}_{mode}"]['mses']
            ax1.plot(x_tasks, curves, color = c, linestyle = style, alpha = 0.5 if mode == 'base' else 1.0,
                     linewidth = 1 if mode == 'base' else 2.2, label = f"{opt_name} ({'Base' if mode == 'base' else '+ CCBP'})")
    ax1.set_title('A. Tracking MSE Over Task Switches', fontsize = 13, fontweight = 'bold')
    ax1.set_xlabel('Task Switch (Sequence of Bit Flips)', fontsize = 11)
    ax1.set_ylabel('Mean Squared Error (MSE)', fontsize = 11)
    ax1.legend(fontsize = 8, ncol = 2, loc = 'upper left')

    def bar_panel(ax, title, ylabel, key, fmt, lower_better = True, annotate_both = False):
        base_vals = [b[key] for _, b, _ in comparables]
        ccbp_vals = [c[key] for _, _, c in comparables]

        ax.bar(x_idx - width / 2, base_vals, width, label = 'Base Optimizer', color = '#e74c3c', alpha = 0.85)
        ax.bar(x_idx + width / 2, ccbp_vals, width, label = '+ CCBP', color = '#2ecc71', alpha = 0.9)
        ax.set_title(title, fontsize = 13, fontweight = 'bold')
        ax.set_xticks(x_idx)
        ax.set_xticklabels([o for o, _, _ in comparables], fontsize = 10, fontweight = 'bold')
        ax.set_ylabel(ylabel, fontsize = 11)
        ax.legend(fontsize = 10)

        for i, (b, c) in enumerate(zip(base_vals, ccbp_vals)):
            if annotate_both:
                ax.annotate(fmt(b), xy = (x_idx[i] - width / 2, b), xytext = (0, 4), textcoords = 'offset points', ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold', color = '#a93226')
                ax.annotate(fmt(c), xy = (x_idx[i] + width / 2, c), xytext = (0, 4), textcoords = 'offset points', ha = 'center', va = 'bottom', fontsize = 9, fontweight = 'bold', color = '#1b7837')
            else:
                ax.annotate(fmt(b, c), xy = (x_idx[i] + width / 2, c), xytext = (0, 4), textcoords = 'offset points', ha = 'center', va = 'bottom', fontsize = 10, fontweight = 'bold', color = '#1b7837')

    bar_panel(axes[0, 1], 'B. Late-Stage Tracking MSE (Lower is Better)', 'Late MSE (Second Half of Tasks)', 'late_mse', lambda b, c: f"-{(1.0 - c / b) * 100.0:.1f}%")
    bar_panel(axes[1, 0], 'C. Proportion of Dormant Units (Lower is Better)', 'Dormant Neurons (%)', 'late_dorm', lambda v: f"{v:.1f}%", annotate_both = True)
    bar_panel(axes[1, 1], 'D. Effective Representation Rank (Higher is Better)', 'Stable Rank (srank)', 'late_rank', lambda v: f"{v:.2f}", annotate_both = True)

    sns.despine(fig = fig, top = True, right = True)
    fig.suptitle('Continual Plasticity across Optimizers (Adam, Lion, AdamAtan2, MuonAdamAtan2)', fontsize = 16, fontweight = 'bold', y = 0.99)
    plt.tight_layout()

    plt.savefig(save_fig, bbox_inches = 'tight')
    print(f"\n[saved] fig -> {save_fig}")

def main():
    fire.Fire(train)

if __name__ == '__main__':
    main()
