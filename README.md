## CCBP (Continuous Continual Backpropagation) - Pytorch

Implementation of CCBP (Continuous Continual Backpropagation) from [Learning Continually at Peak Performance with Continuous Continual Backpropagation](https://openreview.net/forum?id=UJqXhFFzKu) (ICLR 2026) and [Continual Backpropagation](https://arxiv.org/abs/2108.06325).

A lightweight wrapper that can wrap any PyTorch optimizer to mitigate loss of plasticity in non-stationary settings

## Install

```bash
$ pip install ccbp-pytorch
```

## Usage

```python
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from ccbp_pytorch import CCBP

# define your model

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# wrap any base optimizer with CCBP

base_opt = Adam(model.parameters(), lr = 1e-3)

opt = CCBP(
    base_opt,
    model = model
)

# standard training loop

for _ in range(100):
    x = torch.randn(32, 10)
    target = torch.randn(32, 1)

    opt.zero_grad()
    F.mse_loss(model(x), target).backward()
    opt.step()
```

To run discrete Continual Backpropagation (Dohare et al.) instead:

```python
opt = CCBP(
    base_opt,
    model = model,
    continuous = False,
    replacement_rate = 0.003
)
```

## Research Hooks

Researchers can easily plug in custom continuous transfer functions or custom utility estimation functions:

```python
# custom transfer function (e.g. piecewise linear clamp)
custom_transfer = lambda u: torch.clamp(1.0 - u, min = 0.0, max = 1.0)

# custom utility metric (e.g. activation dormancy or custom gradient norm)
custom_utility = lambda cfg: cfg.param.grad.abs().sum(dim = 1)

opt = CCBP(
    base_opt,
    model = model,
    continuous = True,
    transfer_fn = custom_transfer,
    utility_fn = custom_utility
)
```

## Custom Optimizers

For non-standard optimizers (such as `Lion`, `AdamAtan2`, or `Muon`), specify their momentum / second moment buffer names:

```python
from lion_pytorch import Lion

base_opt = Lion(model.parameters(), lr = 1e-4)

opt = CCBP(
    base_opt,
    model = model,
    first_moment_names = ('exp_avg',),
    second_moment_names = ()
)
```

You can also register a custom state adjustment handler that modifies state buffers in-place:

```python
from torch_einops_utils import pad_ndim

def custom_adjust_fn(
    optimizer,    # the base optimizer instance
    param,        # the parameter associated with this state buffer
    alpha,        # 1d reset factors in [0, 1] per neuron (0 = keep, 1 = full reset)
    dim,          # neuron axis (0 for incoming, 1 for outgoing)
    state_name,   # e.g. 'exp_avg', 'exp_avg_sq'
    buffer,       # the state tensor to modify in-place
    policy,       # 'zero' or 'mean'
    **kwargs
):
    alpha_expanded = pad_ndim(alpha, (dim, buffer.ndim - dim - 1))
    buffer.mul_(1.0 - alpha_expanded)

CCBP.register_optimizer_handler(CustomOptimizer, custom_adjust_fn)
```

## Benchmarks

bit flipping

```bash
$ uv run train_bit_flipping.py
```

## Citations

```bibtex
@misc{mccutcheon2026learning,
    title   = {Learning Continually at Peak Performance with Continuous Continual Backpropagation},
    author  = {Luc McCutcheon and Evangelos Chatzaroulas and Saber Fallah},
    year    = {2026},
    url     = {https://openreview.net/forum?id=UJqXhFFzKu},
}
```

```bibtex
@article{dohare2024loss,
    title   = {Loss of plasticity in deep continual learning},
    author  = {Dohare, Shibhansh and Hernandez-Garcia, J. Fernando and Rahman, Parash and Sutton, Richard S. and Mahmood, A. Rupam},
    journal = {Nature},
    volume  = {632},
    pages   = {784--789},
    year    = {2024}
}
```

```bibtex
@misc{dohare2021continualbackprop,
    title   = {Continual Backprop: Stochastic Gradient Descent with Persistent Randomness},
    author  = {Shibhansh Dohare and Richard S. Sutton and A. Rupam Mahmood},
    year    = {2021},
    eprint  = {2108.06325},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2108.06325},
}
```

```bibtex
@article{kumar2023maintaining,
    title   = {Maintaining Plasticity in Continual Learning via Regenerative Regularization},
    author  = {Saurabh Kumar and Henrik Marklund and Benjamin Van Roy},
    journal = {arXiv preprint arXiv:2308.11958},
    year    = {2023},
    url     = {https://arxiv.org/abs/2308.11958}
}
```
