import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam

from einops import rearrange, reduce
from x_mlps_pytorch import MLP

from ccbp_pytorch import CCBP, NeuronConfig, default_adjust_optimizer_state, find_neuron_configs

param = pytest.mark.parametrize

# helpers

def train_steps(model, opt, x, target, steps):
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(model(x), target).backward()
        opt.step()

# tests

def test_find_neuron_configs():
    model = MLP(10, 20, 2)
    configs = find_neuron_configs(model, exclude_module_names = ('layers.1',))

    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.param is model.layers[0][0].weight
    assert cfg.bias is model.layers[0][0].bias
    assert cfg.outgoing_param is model.layers[1].weight
    assert cfg.axis == 0
    assert cfg.outgoing_axis == 1

def test_discrete_cbp_with_adam():
    torch.manual_seed(42)
    model = MLP(8, 16, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    cbp_opt = CCBP(
        base_opt,
        model = model,
        replacement_rate = 0.25,
        maturity_steps = 2,
        reset_interval = 5,
        continuous = False,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(4, 8)
    target = torch.randn(4, 2)
    train_steps(model, cbp_opt, x, target, 10)

    assert cbp_opt.step_count == 10
    state = cbp_opt.neuron_states[0]
    assert state['age'].shape[0] == 16
    assert (state['age'] > 0).all()

def test_replacement_rate_zero_never_prunes():
    torch.manual_seed(42)
    model = MLP(4, 8, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    cbp_opt = CCBP(
        base_opt,
        model = model,
        replacement_rate = 0.0,
        maturity_steps = 1,
        reset_interval = 2,
        continuous = False,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    train_steps(model, cbp_opt, x, target, 6)

    age = cbp_opt.neuron_states[0]['age']
    assert (age == 6.0).all()

def test_outgoing_zeroing_and_optimizer_state_reset():
    torch.manual_seed(42)
    model = MLP(4, 6, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    cbp_opt = CCBP(
        base_opt,
        model = model,
        replacement_rate = 0.34,
        maturity_steps = 1,
        reset_interval = 2,
        continuous = False,
        second_moment_policy = 'zero',
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)

    # step 1

    cbp_opt.zero_grad()
    F.mse_loss(model(x), target).backward()
    cbp_opt.step()

    fc1_weight = model.layers[0][0].weight
    fc2_weight = model.layers[1].weight
    assert (base_opt.state[fc1_weight]['exp_avg'] != 0).any()

    # step 2: triggers reset

    cbp_opt.zero_grad()
    F.mse_loss(model(x), target).backward()
    cbp_opt.step()

    age = cbp_opt.neuron_states[0]['age']
    reset_indices = rearrange((age == 1.0).nonzero(), 'n 1 -> n')
    assert len(reset_indices) > 0

    assert (fc2_weight[:, reset_indices] == 0.0).all()
    exp_avg = base_opt.state[fc1_weight]['exp_avg']
    assert (exp_avg[reset_indices] == 0.0).all()

def test_continuous_ccbp():
    torch.manual_seed(42)
    model = MLP(8, 16, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    ccbp_opt = CCBP(
        base_opt,
        model = model,
        replacement_rate = 0.01,
        maturity_steps = 2,
        continuous = True,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(4, 8)
    target = torch.randn(4, 2)
    train_steps(model, ccbp_opt, x, target, 5)

    assert ccbp_opt.step_count == 5

def test_disabling_parameter():
    model = MLP(10, 20, 2)
    model.layers[0][0].weight.cbp = False

    base_opt = SGD(model.parameters(), lr = 0.01)
    cbp_opt = CCBP(base_opt, model = model)

    assert len(cbp_opt.neuron_configs) > 0
    assert cbp_opt.neuron_configs[0].is_enabled is False

def test_custom_adjust_optimizer_state():
    model = MLP(10, 20, 2)
    called = []

    def custom_adjust(opt, param, alpha, dim, name, buf, policy, **kwargs):
        called.append((name, buf.shape))

    base_opt = Adam(model.parameters(), lr = 1e-2)
    cbp_opt = CCBP(
        base_opt,
        model = model,
        replacement_rate = 0.5,
        maturity_steps = 0,
        reset_interval = 1,
        continuous = False,
        adjust_optimizer_state_fn = custom_adjust,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(4, 10)
    reduce(model(x), '... ->', 'sum').backward()
    cbp_opt.step()

    assert len(called) > 0

def test_sgd_with_momentum():
    model = MLP(4, 8, 2)
    base_opt = SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    cbp_opt = CCBP(
        base_opt,
        model = model,
        replacement_rate = 0.25,
        maturity_steps = 1,
        reset_interval = 2,
        continuous = False,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    train_steps(model, cbp_opt, x, target, 4)

    assert cbp_opt.step_count == 4

def test_lion_optimizer():
    from lion_pytorch import Lion
    from torch_einops_utils import pad_ndim, pad_right_ndim_to

    def lion_adjust_state(optimizer, param, alpha, dim, state_name, buffer, policy = 'zero', reset_threshold = 0.01, **kwargs):
        alpha_expanded = pad_right_ndim_to(alpha, buffer.ndim) if dim == 0 else pad_ndim(alpha, (dim, buffer.ndim - dim - 1))
        buffer.masked_fill_(alpha_expanded.to(buffer.device) > reset_threshold, 0.)

    CCBP.register_optimizer_handler(Lion, lion_adjust_state)

    model = MLP(4, 8, 2)
    base_opt = Lion(model.parameters(), lr = 1e-3)
    cbp_opt = CCBP(
        base_opt,
        model = model,
        continuous = True,
        replacement_rate = 0.2,
        reset_interval = 2,
        first_moment_names = ('exp_avg',),
        second_moment_names = (),
        reset_optimizer_state = True,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    train_steps(model, cbp_opt, x, target, 4)

    assert cbp_opt.step_count == 4

def test_adam_atan2_optimizer():
    from adam_atan2_pytorch import AdamAtan2

    model = MLP(4, 8, 2)
    base_opt = AdamAtan2(model.parameters(), lr = 1e-3)
    cbp_opt = CCBP(
        base_opt,
        model = model,
        continuous = False,
        replacement_rate = 0.25,
        maturity_steps = 0,
        reset_interval = 2,
        first_moment_names = ('exp_avg',),
        second_moment_names = ('exp_avg_sq',),
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    train_steps(model, cbp_opt, x, target, 4)

    assert cbp_opt.step_count == 4

def test_muon_adam_atan2_optimizer():
    from adam_atan2_pytorch import MuonAdamAtan2

    model = MLP(4, 8, 2)
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    adam_params = [p for p in model.parameters() if p.ndim < 2]
    base_opt = MuonAdamAtan2(muon_params, adam_params, lr = 1e-3, muon_lr = 1e-3)
    cbp_opt = CCBP(
        base_opt,
        model = model,
        continuous = True,
        replacement_rate = 0.25,
        reset_interval = 2,
        first_moment_names = ('exp_avg', 'momentum'),
        second_moment_names = ('exp_avg_sq',),
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    train_steps(model, cbp_opt, x, target, 4)

    assert cbp_opt.step_count == 4

def test_custom_nn_init():
    model = MLP(4, 8, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)
    cbp_opt = CCBP(
        base_opt,
        model = model,
        continuous = False,
        replacement_rate = 0.5,
        maturity_steps = 0,
        reset_interval = 1,
        init_fn = nn.init.kaiming_uniform_,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)

    cbp_opt.zero_grad()
    F.mse_loss(model(x), target).backward()
    cbp_opt.step()

    assert cbp_opt.step_count == 1

def test_state_dict_and_load_state_dict():
    model = MLP(4, 8, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)
    cbp_opt = CCBP(
        base_opt,
        model = model,
        replacement_rate = 0.25,
        maturity_steps = 1,
        reset_interval = 2,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    train_steps(model, cbp_opt, x, target, 4)

    sd = cbp_opt.state_dict()
    assert 'ccbp' in sd
    assert sd['ccbp']['step_count'] == 4

    # create new optimizer and restore

    model2 = MLP(4, 8, 2)
    base_opt2 = Adam(model2.parameters(), lr = 1e-2)
    cbp_opt2 = CCBP(
        base_opt2,
        model = model2,
        replacement_rate = 0.25,
        maturity_steps = 1,
        reset_interval = 2,
        exclude_module_names = ('layers.1',)
    )

    cbp_opt2.load_state_dict(sd)
    assert cbp_opt2.step_count == 4
    assert torch.allclose(cbp_opt2.neuron_states[0]['age'], cbp_opt.neuron_states[0]['age'])

    train_steps(model2, cbp_opt2, x, target, 2)
    assert cbp_opt2.step_count == 6

@param('continuous', [True, False])
def test_readme_example(continuous):
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    base_opt = Adam(model.parameters(), lr = 1e-3)

    opt = CCBP(
        base_opt,
        model = model,
        continuous = continuous,
        replacement_rate = 0.25,
        maturity_steps = 10,
        reset_interval = 20
    )

    for _ in range(25):
        x = torch.randn(32, 10)
        target = torch.randn(32, 1)

        opt.zero_grad()
        F.mse_loss(model(x), target).backward()
        opt.step()

    assert opt.step_count == 25

def test_remainder_per_config():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    base_opt = Adam(model.parameters(), lr = 1e-2)
    cbp_opt = CCBP(
        base_opt,
        model = model,
        continuous = False,
        replacement_rate = 0.3,
        maturity_steps = 1,
        reset_interval = 2,
        exclude_module_names = ('2',)
    )

    x = torch.randn(4, 8)
    target = torch.randn(4, 1)
    train_steps(model, cbp_opt, x, target, 4)

    assert set(cbp_opt.remainder) <= set((0, 1))

def test_custom_neuron_config():
    model = MLP(6, 12, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    custom_config = NeuronConfig(
        param = model.layers[0][0].weight,
        bias = model.layers[0][0].bias,
        outgoing_param = model.layers[1].weight,
        axis = 0,
        outgoing_axis = 1,
        init_fn = nn.init.kaiming_normal_
    )

    opt = CCBP(
        base_opt,
        neuron_configs = [custom_config],
        continuous = True,
        replacement_rate = 0.2,
        reset_interval = 2
    )

    assert len(opt.neuron_configs) == 1
    assert opt.neuron_configs[0].param is model.layers[0][0].weight

    x = torch.randn(4, 6)
    target = torch.randn(4, 2)
    train_steps(model, opt, x, target, 4)

    assert opt.step_count == 4

def test_custom_transfer_and_utility_fn():
    model = MLP(4, 8, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    custom_transfer = lambda u: torch.clamp(1.0 - u, min = 0.0, max = 1.0)
    custom_utility = lambda cfg: cfg.param.grad.abs().sum(dim = 1)

    opt = CCBP(
        base_opt,
        model = model,
        continuous = True,
        transfer_fn = custom_transfer,
        utility_fn = custom_utility,
        reset_interval = 2,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    train_steps(model, opt, x, target, 4)

    assert opt.step_count == 4

def test_adaptable_utility():
    model = MLP(4, 8, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    opt = CCBP(
        base_opt,
        model = model,
        continuous = True,
        utility_type = 'adaptable',
        reset_interval = 2,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    train_steps(model, opt, x, target, 4)

    assert opt.step_count == 4

def test_ccbp_reset():
    model = MLP(4, 8, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    opt = CCBP(
        base_opt,
        model = model,
        continuous = True,
        reset_interval = 2,
        exclude_module_names = ('layers.1',)
    )

    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    train_steps(model, opt, x, target, 5)

    assert opt.step_count == 5
    assert (opt.neuron_states[0]['age'] > 0).any()
    assert len(base_opt.state) > 0

    # reset CCBP tracking states
    opt.reset()

    assert opt.step_count == 0
    assert (opt.neuron_states[0]['age'] == 0).all()
    assert (opt.neuron_states[0]['utility'] == 1).all()
    assert len(base_opt.state) > 0

    # reset with base optimizer cleared
    opt.reset(reset_base_optimizer = True)

    assert opt.step_count == 0
    assert len(base_opt.state) == 0

    # continue training after reset
    train_steps(model, opt, x, target, 2)
    assert opt.step_count == 2

def test_continuous_ccbp_defaults_match_paper():
    model = MLP(4, 8, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    opt = CCBP(
        base_opt,
        model = model,
        exclude_module_names = ('layers.1',)
    )

    # paper Table 3 optimal values (SlipperyAnt)

    assert opt.continuous
    assert opt.continuous_rate == 0.015
    assert opt.steepness == 16.0
    assert opt.reset_interval == 1000
    assert opt.utility_type == 'abs_mean'
    assert opt.utility_ema_decay == 0.99

    # CCBP (Algorithm 1) has no age gating and does not reset base optimizer state

    assert opt.maturity_steps == 0
    assert opt.reset_optimizer_state is False
    assert opt.adjust_optimizer_state_fn is None

def test_discrete_cbp_defaults_from_paper():
    model = MLP(4, 8, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    opt = CCBP(
        base_opt,
        model = model,
        continuous = False,
        exclude_module_names = ('layers.1',)
    )

    # paper Table 3: CBP maturity threshold of 100

    assert opt.maturity_steps == 100
    assert opt.reset_optimizer_state is True
    assert opt.adjust_optimizer_state_fn is default_adjust_optimizer_state

def test_explicit_off_overrides_mode_defaults():
    model = MLP(4, 8, 2)
    base_opt = Adam(model.parameters(), lr = 1e-2)

    opt = CCBP(
        base_opt,
        model = model,
        continuous = False,
        maturity_steps = 0,
        reset_optimizer_state = False,
        exclude_module_names = ('layers.1',)
    )

    assert opt.maturity_steps == 0
    assert opt.reset_optimizer_state is False
    assert opt.adjust_optimizer_state_fn is None

    # continuous with maturity enabled explicitly still works

    opt2 = CCBP(
        base_opt,
        model = model,
        continuous = True,
        maturity_steps = 5,
        exclude_module_names = ('layers.1',)
    )

    assert opt2.maturity_steps == 5
