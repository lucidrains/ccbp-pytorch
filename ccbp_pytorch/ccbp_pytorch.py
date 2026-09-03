from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from einops import rearrange, reduce
from torch_einops_utils import (
    clamp,
    pad_ndim,
    tree_map_tensor_to_device
)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(val):
    if not exists(val):
        return ()
    return (val,) if isinstance(val, (str, int)) else tuple(val)

# tensor helpers

def isolate_dim_and_flatten(t: Tensor, dim: int) -> Tensor:
    if dim == 0:
        return rearrange(t, 'n ... -> n (...)')

    if dim == 1:
        return rearrange(t, 'd n ... -> n (d ...)')

    prefix = ' '.join(f'd{i}' for i in range(dim))
    return rearrange(t, f'{prefix} n ... -> n ({prefix} ...)')

def expand_neuron_dim(t: Tensor, dim: int, ndim: int) -> Tensor:
    return pad_ndim(t, (dim, ndim - dim - 1))

def call_init_fn(init_fn: Callable, weight: Tensor, dim: int, indices: Tensor) -> Tensor:
    shape = list(weight.shape)
    shape[dim] = len(indices)
    out = weight.new_empty(shape)
    res = init_fn(out)
    return default(res, out)

# neuron configuration

@dataclass
class NeuronConfig:
    param: Parameter
    axis: int = 0
    bias: Parameter | None = None
    outgoing_param: Parameter | None = None
    outgoing_axis: int = 1
    enabled: bool = True
    init_fn: Callable | None = None

    @property
    def is_enabled(self) -> bool:
        return self.enabled and getattr(self.param, 'cbp', True)

AdjustStateFn = Callable[..., None]

OPTIMIZER_HANDLERS: dict[type[Optimizer], AdjustStateFn] = dict()

# reset optimizer state (momentum / second moments) for the neurons being reset

def default_adjust_optimizer_state(
    optimizer: Optimizer,
    param: Parameter,
    alpha: Tensor,
    dim: int,
    state_name: str,
    buffer: Tensor,
    policy: Literal['zero', 'mean'] = 'zero',
    **kwargs
):
    device = buffer.device

    alpha_expanded = expand_neuron_dim(alpha, dim, buffer.ndim).to(device)
    retention = 1.0 - alpha_expanded

    if policy == 'mean':
        unpruned = alpha < 0.5

        if unpruned.any():
            active = unpruned.nonzero().flatten()
            active_mean = reduce(buffer.index_select(dim, active), '... ->', 'mean')
            buffer.mul_(retention).add_(active_mean * alpha_expanded)
            return

    buffer.mul_(retention)

# auto-discovery

def find_neuron_configs(
    model: Module,
    exclude_module_names = ('head', 'classifier')
) -> list[NeuronConfig]:
    exclude_module_names = cast_tuple(exclude_module_names)

    layers = [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
    ]

    configs: list[NeuronConfig] = []

    for next_i, (name, layer) in enumerate(layers, start = 1):
        if any(exc in name.lower() for exc in exclude_module_names):
            continue

        outgoing_param = None

        if next_i < len(layers):
            _, next_layer = layers[next_i]
            if hasattr(next_layer, 'weight') and next_layer.weight.ndim >= 2 and layer.weight.shape[0] == next_layer.weight.shape[1]:
                outgoing_param = next_layer.weight

        configs.append(NeuronConfig(
            param = layer.weight,
            bias = layer.bias,
            outgoing_param = outgoing_param
        ))

    return configs

# main optimizer wrapper

class CCBP(Optimizer):
    def __init__(
        self,
        base_optimizer: Optimizer,
        *,
        model: Module | None = None,
        neuron_configs: Sequence[NeuronConfig] | None = None,
        exclude_module_names = ('head', 'classifier'),
        continuous = True,
        replacement_rate = 0.02,
        continuous_rate = None,
        steepness = 16.0,
        reset_interval = 20,
        maturity_steps = 100,
        utility_type: Literal['contribution', 'abs_mean', 'norm', 'adaptable'] = 'contribution',
        utility_ema_decay = 0.99,
        eps = 1e-8,
        outgoing_eps = 1e-4,
        init_fn = nn.init.kaiming_normal_,
        second_moment_policy: Literal['zero', 'mean'] = 'mean',
        first_moment_names = ('exp_avg', 'momentum_buffer'),
        second_moment_names = ('exp_avg_sq',),
        transfer_fn: Callable[[Tensor], Tensor] | None = None,
        utility_fn: Callable[[NeuronConfig], Tensor] | None = None,
        adjust_optimizer_state_fn: AdjustStateFn | None = None
    ):
        continuous_rate = default(continuous_rate, replacement_rate)

        assert 0.0 <= replacement_rate <= 1.0
        assert maturity_steps >= 0
        assert utility_type in {'contribution', 'abs_mean', 'norm', 'adaptable'}, f"utility_type '{utility_type}' must be one of {{'contribution', 'abs_mean', 'norm', 'adaptable'}}"
        assert second_moment_policy in {'mean', 'zero'}, f"second_moment_policy '{second_moment_policy}' must be one of {{'mean', 'zero'}}"

        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.state = base_optimizer.state

        self.continuous = continuous
        self.continuous_rate = continuous_rate
        self.replacement_rate = replacement_rate
        self.steepness = steepness
        self.reset_interval = reset_interval
        self.maturity_steps = maturity_steps

        # research hooks: custom continuous transfer curve or utility metric

        self.transfer_fn = default(transfer_fn, lambda u: torch.sigmoid(-self.steepness * (u - 1.0)))
        self.utility_fn = utility_fn

        self.utility_type = utility_type
        self.utility_ema_decay = utility_ema_decay
        self.eps = eps
        self.outgoing_eps = outgoing_eps

        self.init_fn = init_fn
        self.second_moment_policy = second_moment_policy
        self.first_moment_names = tuple(first_moment_names)
        self.second_moment_names = tuple(second_moment_names)

        opt_handler = OPTIMIZER_HANDLERS.get(type(base_optimizer))
        self.adjust_optimizer_state_fn = default(adjust_optimizer_state_fn, default(opt_handler, default_adjust_optimizer_state))

        self.step_count = 0
        self.remainder = dict()

        if exists(neuron_configs):
            self.neuron_configs = list(neuron_configs)
        elif exists(model):
            self.neuron_configs = find_neuron_configs(model, exclude_module_names = exclude_module_names)
        else:
            self.neuron_configs = list()

        self.neuron_states = dict()
        self._init_neuron_states()

    @classmethod
    def register_optimizer_handler(cls, opt_cls: type[Optimizer], fn: AdjustStateFn):
        OPTIMIZER_HANDLERS[opt_cls] = fn

    def _init_neuron_states(self):
        for idx, cfg in enumerate(self.neuron_configs):
            device = cfg.param.device
            num_neurons = cfg.param.shape[cfg.axis]

            self.neuron_states[idx] = dict(
                bias_init = cfg.bias.detach().clone() if exists(cfg.bias) else None,
                age = torch.zeros(num_neurons, device = device),
                utility = torch.ones(num_neurons, device = device),
            )

    @property
    def defaults(self):
        return getattr(self.base_optimizer, 'defaults', dict())

    def zero_grad(self, set_to_none = True):
        self.base_optimizer.zero_grad(set_to_none = set_to_none)

    def reset(self, reset_base_optimizer = False):
        self.step_count = 0
        self.remainder.clear()
        self._init_neuron_states()

        if reset_base_optimizer:
            self.base_optimizer.state.clear()

    def state_dict(self):
        state = self.base_optimizer.state_dict()
        state['ccbp'] = dict(
            step_count = self.step_count,
            remainder = self.remainder,
            neuron_states = tree_map_tensor_to_device(self.neuron_states, 'cpu')
        )
        return state

    def load_state_dict(self, state_dict: dict):
        state_dict = dict(state_dict)
        ccbp_state = state_dict.pop('ccbp', None)
        self.base_optimizer.load_state_dict(state_dict)

        if not exists(ccbp_state):
            return

        self.step_count = ccbp_state.get('step_count', 0)
        self.remainder = ccbp_state.get('remainder', dict())

        for k, saved in ccbp_state.get('neuron_states', dict()).items():
            if k not in self.neuron_states:
                continue

            device = self.neuron_configs[k].param.device
            self.neuron_states[k] = tree_map_tensor_to_device(saved, device)

    def _adjust_states_for_param(self, param: Parameter, alpha: Tensor, dim: int):
        if param not in self.base_optimizer.state:
            return

        for name, buf in self.base_optimizer.state[param].items():
            if not torch.is_tensor(buf) or buf.shape != param.shape:
                continue

            is_first = any(k in name for k in self.first_moment_names)
            is_second = any(k in name for k in self.second_moment_names)
            policy = 'zero' if is_first and not is_second else self.second_moment_policy

            self.adjust_optimizer_state_fn(
                self.base_optimizer,
                param,
                alpha,
                dim,
                name,
                buf,
                policy
            )

    # core continual backprop logic: utility estimation, reset rates, and step

    def compute_neuron_utility(self, cfg: NeuronConfig) -> Tensor | None:
        if exists(self.utility_fn):
            util = self.utility_fn(cfg)
            den = clamp(reduce(util, 'n ->', 'mean'), lo = self.eps)
            return util / den

        grad = cfg.param.grad

        if not exists(grad):
            return None

        flat_grad = isolate_dim_and_flatten(grad, cfg.axis)
        util = reduce(flat_grad.abs(), 'n d -> n', 'mean') if self.utility_type in {'abs_mean', 'contribution', 'adaptable'} else flat_grad.norm(dim = 1)

        if self.utility_type in {'contribution', 'adaptable'} and exists(cfg.outgoing_param):
            flat_out = isolate_dim_and_flatten(cfg.outgoing_param, cfg.outgoing_axis)
            util = util * (reduce(flat_out.abs(), 'n d -> n', 'mean') + self.outgoing_eps)

        if self.utility_type == 'adaptable':
            flat_in = isolate_dim_and_flatten(cfg.param, cfg.axis)
            incoming_weight_mag = clamp(reduce(flat_in.abs(), 'n d -> n', 'mean'), lo = self.eps)
            util = util / incoming_weight_mag

        if exists(cfg.bias) and exists(cfg.bias.grad):
            util = util + cfg.bias.grad.abs()

        den = clamp(reduce(util, 'n ->', 'mean'), lo = self.eps)
        return util / den

    # discrete bottom-k pruning with fractional remainder accumulation,
    # or continuous partial reset rates via sigmoid transfer

    def compute_alpha(self, utility: Tensor, age: Tensor, idx: int) -> Tensor:
        device = utility.device

        if self.replacement_rate <= 0.0:
            return torch.zeros_like(utility)

        if not (divisible_by(self.step_count, self.reset_interval) and self.step_count > 0):
            return torch.zeros_like(utility)

        # continuous cbp

        if self.continuous:
            r = self.continuous_rate * self.transfer_fn(utility)

            if self.maturity_steps > 0:
                r = r * clamp(age / self.maturity_steps, hi = 1.0)

            return clamp(r, lo = 0.0, hi = 1.0)

        # discrete cbp - only mature units are eligible

        eligible = age >= self.maturity_steps
        num_eligible = eligible.sum().item()

        if num_eligible == 0:
            return torch.zeros_like(utility)

        # accumulate fractional replacements to avoid rounding bias

        ideally = (num_eligible * self.replacement_rate) + self.remainder.get(idx, 0.0)
        num_to_prune = int(ideally)
        remainder = ideally - num_to_prune

        if torch.rand((), device = device) < remainder:
            num_to_prune += 1
            remainder = 0.0

        self.remainder[idx] = remainder
        num_to_prune = clamp(num_to_prune, lo = 0, hi = num_eligible)

        if num_to_prune == 0:
            return torch.zeros_like(utility)

        # bottom-k lowest utility among eligible mature units

        masked_util = utility.masked_fill(~eligible, float('inf'))
        _, prune_indices = torch.topk(masked_util, k = num_to_prune, largest = False)

        return torch.zeros_like(utility).index_fill_(0, prune_indices, 1.0)

    @torch.no_grad()
    def step(self, closure = None):

        loss = self.base_optimizer.step(closure = closure)
        self.step_count += 1

        for idx, cfg in enumerate(self.neuron_configs):
            if not cfg.is_enabled:
                continue

            state = self.neuron_states[idx]
            weight, bias, outgoing = cfg.param, cfg.bias, cfg.outgoing_param
            utility, age = state['utility'], state['age']
            device = weight.device

            # running utility ema

            batch_util = self.compute_neuron_utility(cfg)

            if exists(batch_util):
                utility.lerp_(batch_util, 1.0 - self.utility_ema_decay)

            alpha = self.compute_alpha(utility, age, idx)

            if not alpha.any():
                age.add_(1.0)
                continue

            indices = alpha.nonzero().flatten()
            alpha_sel = alpha.index_select(0, indices)

            # lerp incoming weights toward reinitialized values

            init_fn = default(cfg.init_fn, self.init_fn)
            init_weight = call_init_fn(init_fn, weight, cfg.axis, indices)
            selected = weight.index_select(cfg.axis, indices)
            alpha_expanded = expand_neuron_dim(alpha_sel, cfg.axis, weight.ndim).to(device)

            selected.lerp_(init_weight, alpha_expanded)
            weight.index_copy_(cfg.axis, indices, selected)
            self._adjust_states_for_param(weight, alpha, cfg.axis)

            # reset bias toward init

            if exists(bias):
                init_bias = default(state['bias_init'], torch.zeros_like(bias))
                bias.lerp_(init_bias, alpha)
                self._adjust_states_for_param(bias, alpha, 0)

            # dampen outgoing weights

            if exists(outgoing):
                outgoing_device = outgoing.device
                alpha_out = expand_neuron_dim(alpha, cfg.outgoing_axis, outgoing.ndim).to(outgoing_device)
                outgoing.mul_(1.0 - alpha_out)
                self._adjust_states_for_param(outgoing, alpha, cfg.outgoing_axis)

            # age and utility bookkeeping

            age.mul_(1.0 - alpha).add_(1.0)
            utility.lerp_(torch.ones_like(utility), alpha)

        return loss

# alias

ContinualBackprop = CCBP
