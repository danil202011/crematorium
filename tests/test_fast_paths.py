"""Block-compiled scans and layer-major Sequential: same results as the reference scan."""

import pytest
import torch
import torch.nn as nn

from crematorium import CrModule, Tensor
from crematorium.module.mixins.scan import BlockScan, _block_sizes
from crematorium.nn import Sequential, is_time_distributed
from crematorium.snn import ALIF, LIF, AdEx, atan_surrogate

# Block-scan semantics (chunking, state threading, autograd) do not depend on
# the inductor backend; aot_eager keeps these tests fast and still traces
# through dynamo + AOTAutograd. One inductor test covers real codegen.
FAST = {"backend": "aot_eager"}


def _close(a, b, atol=1e-5):
    if a is None or b is None:
        assert a is None and b is None
        return
    if isinstance(a, tuple):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _close(x, y, atol)
        return
    assert a.shape == b.shape, (a.shape, b.shape)
    assert torch.allclose(a, b, atol=atol, rtol=1e-5), (a - b).abs().max()


class _TwoInput(CrModule):
    class Inputs:
        x: Tensor
        inh: Tensor

    class Specs:
        out = CrModule.OutputSpec()
        v = CrModule.StateSpec(shape="x")

    def _step(self, x, inh, v):
        v = 0.5 * v + x - inh
        return torch.tanh(v), v


class _TwoOutputs(CrModule):
    class Specs:
        a = CrModule.OutputSpec()
        b = CrModule.OutputSpec()
        s = CrModule.StateSpec()

    def _step(self, x, s):
        s = 0.9 * s + x
        return s.sin(), s.cos(), s


# _block_sizes


@pytest.mark.parametrize("time", [1, 2, 7, 63, 64, 65, 127, 128, 200, 1000])
@pytest.mark.parametrize("block", [1, 3, 16, 64, 100])
def test_block_sizes_cover_sequence(time, block):
    sizes = _block_sizes(time, block)
    assert sum(sizes) == time
    assert all(0 < s <= block for s in sizes)
    # tail sizes are distinct powers of two below the block
    tail = sizes[time // block :]
    assert len(tail) == len(set(tail))
    assert all(s & (s - 1) == 0 and s < block for s in tail)


def test_block_sizes_examples():
    assert _block_sizes(200, 64) == [64, 64, 64, 8]
    assert _block_sizes(5, 64) == [4, 1]
    assert _block_sizes(64, 64) == [64]


def test_block_sizes_bounded_set():
    seen = set()
    for time in range(1, 600):
        seen.update(_block_sizes(time, 64))
    assert seen == {64, 32, 16, 8, 4, 2, 1}


@pytest.mark.parametrize("block", [0, -1, 2.0, True])
def test_block_scan_rejects_bad_block(block):
    with pytest.raises(ValueError, match="block"):
        BlockScan(lambda i, s: (*i, *s), block, 1, False, {})


# Block-compiled scans match the reference scan


@pytest.mark.parametrize("time", [1, 5, 16, 37])
def test_block_explicit_lif_matches_reference(time):
    torch.manual_seed(0)
    x = torch.randn(time, 3, 8)
    ref_mod = LIF(learnable_beta=True)
    out_ref = ref_mod.forward_sequence(x)

    blk = LIF(learnable_beta=True).compile_sequence_scan(block=8, **FAST)
    blk.load_state_dict(ref_mod.state_dict())
    out_blk = blk.forward_sequence(x)

    _close(out_blk, out_ref)


def test_block_explicit_with_given_state_and_resume():
    torch.manual_seed(1)
    x = torch.randn(30, 2, 6)
    ref = ALIF()
    blk = ALIF().compile_sequence_scan(block=4, **FAST)
    state = ref.initial_state((2, 6))

    spk_ref, *st_ref = ref.forward_sequence(x, state)
    spk_a, *st_a = blk.forward_sequence(x[:13], state)
    spk_b, *st_b = blk.forward_sequence(x[13:], tuple(st_a))

    _close(torch.cat([spk_a, spk_b]), spk_ref)
    _close(tuple(st_b), tuple(st_ref))


def test_block_hidden_matches_reference_and_keeps_buffers():
    torch.manual_seed(2)
    x = torch.randn(21, 4, 5)
    ref = AdEx(init_hidden=True)
    blk = AdEx(init_hidden=True).compile_sequence_scan(block=8, **FAST)

    _close(blk.forward_sequence(x[:10]), ref.forward_sequence(x[:10]))
    _close(blk.forward_sequence(x[10:]), ref.forward_sequence(x[10:]))
    for name in ref._cr_state_names:
        _close(getattr(blk, name), getattr(ref, name))


def test_block_multi_input_and_multi_output():
    torch.manual_seed(3)
    x, inh = torch.randn(19, 2, 4), torch.randn(19, 2, 4)
    ref, blk = _TwoInput(), _TwoInput().compile_sequence_scan(block=4, **FAST)
    _close(blk.forward_sequence((x, inh)), ref.forward_sequence((x, inh)))
    _close(
        blk.forward_sequence({"x": x, "inh": inh}),
        ref.forward_sequence({"x": x, "inh": inh}),
    )

    ref2, blk2 = _TwoOutputs(), _TwoOutputs().compile_sequence_scan(block=4, **FAST)
    _close(blk2.forward_sequence(x), ref2.forward_sequence(x))


def test_block_gradients_match_reference():
    torch.manual_seed(4)
    x = torch.randn(23, 3, 7)
    kw = dict(
        learnable_beta=True, learnable_threshold=True, spike_grad=atan_surrogate()
    )
    ref = LIF(**kw)
    blk = LIF(**kw).compile_sequence_scan(block=8, **FAST)
    blk.load_state_dict(ref.state_dict())

    grads = []
    for m in (ref, blk):
        xi = x.clone().requires_grad_(True)
        spk, mem = m.forward_sequence(xi)
        ((spk * torch.linspace(0, 1, 7)).sum() + mem.pow(2).sum()).backward()
        grads.append((xi.grad, m.beta.grad, m.threshold.grad))

    _close(grads[1], grads[0], atol=1e-4)


def _graphs():
    from torch._dynamo.utils import counters

    return counters["stats"]["unique_graphs"]


def test_block_scanner_compiles_each_size_once():
    def scan(inputs_seq, state):
        (x,) = inputs_seq
        return (x * 2, state[0] + x.sum(0))

    scanner = BlockScan(scan, 8, 1, False, FAST)
    before = _graphs()
    for time in (3, 8, 13, 27, 27, 13):
        y, s = scanner((torch.ones(time, 2),), (torch.zeros(2),))
        assert y.shape == (time, 2)
        _close(s, torch.full((2,), float(time)))
    # sizes 1, 2, 4, 8: one graph each, later lengths reuse them
    assert sorted(scanner._compiled) == [1, 2, 4, 8]
    assert _graphs() - before == 4


def test_compiled_scan_has_no_per_step_graph_breaks():
    from torch._dynamo.utils import counters

    counters.clear()
    x = torch.randn(12, 2, 3)
    LIF().compile_sequence_scan(**FAST).forward_sequence(x)  # validation on
    assert not counters["graph_break"], dict(counters["graph_break"])


def test_many_modules_stay_compiled():
    before = _graphs()
    x = torch.randn(10, 2, 3)
    for _ in range(12):
        LIF().compile_sequence_scan(block=4, **FAST).forward_sequence(x)
    # sizes {4, 2} per module: every module got its own graphs (no fallback
    # to eager after Dynamo's per-code-object recompile limit)
    assert _graphs() - before == 12 * 2


def test_block_no_grad_uses_preallocated_output():
    torch.manual_seed(5)
    x = torch.randn(40, 2, 3)
    ref, blk = LIF(), LIF().compile_sequence_scan(block=16, **FAST)
    with torch.no_grad():
        _close(blk.forward_sequence(x), ref.forward_sequence(x))


def test_fast_sequence_passes_block():
    torch.manual_seed(6)
    x = torch.randn(33, 2, 4)
    m = LIF().fast_sequence_(block=8, **FAST)
    _close(m.forward_sequence(x), LIF().forward_sequence(x))


def test_block_inductor_matches_reference():
    torch.manual_seed(7)
    x = torch.randn(70, 2, 16)
    ref = LIF()
    blk = LIF().compile_sequence_scan(block=32)
    _close(blk.forward_sequence(x), ref.forward_sequence(x))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
def test_block_cuda_graphs_outputs_survive_next_call():
    torch.manual_seed(8)
    x = torch.randn(50, 4, 32, device="cuda")
    ref = LIF().cuda()
    blk = LIF().cuda().compile_sequence_scan(block=16, mode="reduce-overhead")
    first = blk.forward_sequence(x)
    blk.forward_sequence(torch.randn_like(x))
    _close(first, ref.forward_sequence(x))


# is_time_distributed


@pytest.mark.parametrize(
    "layer, expected",
    [
        (nn.Linear(4, 4), True),
        (nn.Conv2d(2, 3, 3), True),
        (nn.ReLU(), True),
        (nn.LayerNorm(4), True),
        (nn.Sequential(nn.Linear(4, 4), nn.GELU()), True),
        (nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)), False),
        (nn.Flatten(), True),
        (nn.Flatten(start_dim=0), False),
        (nn.Softmax(dim=-1), True),
        (nn.Softmax(dim=0), False),
        (nn.Softmax(dim=None), False),
        (nn.Dropout(0.0), True),
        (nn.Dropout(0.3), False),
        (nn.Dropout(0.3).eval(), True),
        (nn.BatchNorm1d(4), False),
        (nn.BatchNorm1d(4).eval(), True),
        (nn.BatchNorm1d(4, track_running_stats=False).eval(), False),
        (nn.LSTM(4, 4), False),
        (LIF(), False),
    ],
)
def test_is_time_distributed(layer, expected):
    assert is_time_distributed(layer) is expected


def test_time_distributed_opt_in_and_out():
    class _Scale(nn.Module):
        _cr_time_distributed = True

        def forward(self, x):
            return 2 * x

    class _NoTD(nn.Linear):
        _cr_time_distributed = False

    assert is_time_distributed(_Scale())
    assert not is_time_distributed(_NoTD(3, 3))


# Layer-major Sequential matches the per-timestep walk


def _pair(make, **kw):
    torch.manual_seed(10)
    step = make(layer_major=False, **kw)
    torch.manual_seed(10)
    layer = make(layer_major=True, **kw)
    layer.load_state_dict(step.state_dict())
    assert layer._cr_layer_major_ok() and not step._cr_layer_major_ok()
    return step, layer


def _mlp(**kw):
    return Sequential(
        nn.Linear(6, 16),
        LIF(learnable_beta=True, spike_grad=atan_surrogate()),
        nn.Linear(16, 12),
        nn.LayerNorm(12),
        ALIF(),
        nn.Linear(12, 5),
        LIF(),
        **kw,
    )


@pytest.mark.parametrize("time", [1, 9])
def test_layer_major_explicit_matches_step_major(time):
    step, layer = _pair(_mlp)
    x = torch.randn(time, 3, 6)
    _close(layer.forward_sequence(x), step.forward_sequence(x))


def test_layer_major_hidden_matches_and_resumes():
    step, layer = _pair(_mlp, init_hidden=True)
    x = torch.randn(14, 2, 6)
    _close(layer.forward_sequence(x[:6]), step.forward_sequence(x[:6]))
    _close(layer.forward_sequence(x[6:]), step.forward_sequence(x[6:]))
    for name in step._cr_state_names:
        _close(getattr(layer, name), getattr(step, name))


def test_layer_major_gradients_match():
    step, layer = _pair(_mlp)
    x = torch.randn(11, 3, 6)
    grads = []
    for net in (step, layer):
        net.zero_grad()
        xi = x.clone().requires_grad_(True)
        y, *state = net.forward_sequence(xi)
        (y.sum() + sum(s.pow(2).sum() for s in state)).backward()
        grads.append((xi.grad, *(p.grad for p in net.parameters())))
    _close(grads[1], grads[0], atol=1e-4)


def _conv(**kw):
    return Sequential(
        nn.Conv2d(2, 4, 3, padding=1),
        LIF(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 10),
        LIF(),
        **kw,
    )


def test_layer_major_conv_stack_matches():
    step, layer = _pair(_conv)
    x = torch.randn(7, 3, 2, 8, 8)
    _close(layer.forward_sequence(x), step.forward_sequence(x))


def test_layer_major_is_default_when_eligible():
    net = _mlp()
    assert net.layer_major is None and net._cr_layer_major_ok()


def test_layer_major_falls_back_for_batch_dependent_layers():
    torch.manual_seed(11)
    bn = nn.BatchNorm1d(8)
    auto = Sequential(nn.Linear(4, 8), bn, LIF())
    assert not auto._cr_layer_major_ok()
    forced = Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), LIF(), layer_major=True)
    with pytest.raises(ValueError, match="BatchNorm1d"):
        forced.forward_sequence(torch.randn(3, 2, 4))
    # the fallback is the per-timestep walk: batch statistics per step
    x = torch.randn(5, 6, 4)
    step = Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), LIF(), layer_major=False)
    step.load_state_dict(auto.state_dict())
    _close(auto.forward_sequence(x), step.forward_sequence(x))
    # eval mode uses running statistics and becomes eligible
    auto.eval()
    assert auto._cr_layer_major_ok()


def test_layer_major_dropout_train_vs_eval():
    net = Sequential(nn.Linear(4, 4), nn.Dropout(0.5), LIF())
    assert not net._cr_layer_major_ok()
    net.eval()
    assert net._cr_layer_major_ok()


def test_layer_major_disabled_for_custom_step():
    class _Custom(Sequential):
        def _step(self, x, *state):
            return super()._step(2 * x, *state)

    net = _Custom(nn.Linear(3, 3), LIF())
    assert not net._cr_layer_major_ok()
    with pytest.raises(ValueError, match="overrides _step"):
        _Custom(nn.Linear(3, 3), LIF(), layer_major=True)._cr_layer_major_ok()


def test_layer_major_rejects_bad_flag():
    with pytest.raises(TypeError, match="layer_major"):
        Sequential(LIF(), layer_major="yes")  # type: ignore[arg-type]


def test_layer_major_survives_layer_replacement():
    step, layer = _pair(_mlp)
    for net in (step, layer):
        net[3] = nn.Identity()
    x = torch.randn(6, 2, 6)
    _close(layer.forward_sequence(x), step.forward_sequence(x))
    step[3] = nn.BatchNorm1d(12)
    assert not step._cr_layer_major_ok()
    assert not Sequential(*step._layers)._cr_layer_major_ok()


# Compiled Sequential: whole-sequence and block-compiled


@pytest.mark.parametrize("block", [None, 4])
def test_compiled_sequential_matches_eager(block):
    step, layer = _pair(_mlp)
    x = torch.randn(13, 2, 6)
    ref = step.forward_sequence(x)
    layer.compile_sequence_scan(block=block, **FAST)
    _close(layer.forward_sequence(x), ref)


def test_block_compiled_sequential_hidden_matches():
    step, layer = _pair(_mlp, init_hidden=True)
    layer.compile_sequence_scan(block=4, **FAST)
    x = torch.randn(10, 2, 6)
    _close(layer.forward_sequence(x), step.forward_sequence(x))
    _close(layer.forward_sequence(x), step.forward_sequence(x))


def test_block_compiled_layer_major_compiles_each_layer_once():
    shared = LIF()
    net = Sequential(
        nn.Linear(5, 5), shared, nn.Linear(5, 5), shared, nn.Linear(5, 3), LIF()
    )
    ref = Sequential(*net._layers, layer_major=False)
    net.compile_sequence_scan(block=4, **FAST)
    x = torch.randn(10, 2, 5)
    with torch.no_grad():
        before = _graphs()
        _close(net.forward_sequence(x), ref.forward_sequence(x))
        # sizes {4, 2} x two distinct LIF instances; the repeated one reuses graphs
        assert _graphs() - before == 2 * 2
        before = _graphs()
        _close(net.forward_sequence(x[:7]), ref.forward_sequence(x[:7]))  # adds size 1
        assert _graphs() - before == 2
    # with autograd the first chunk sees a state without grad and later chunks
    # one with grad: at most one extra graph per scanner, results unchanged
    _close(net.forward_sequence(x), ref.forward_sequence(x))


def test_block_compiled_switches_order_with_train_eval():
    torch.manual_seed(12)
    net = Sequential(nn.Linear(4, 6), nn.Dropout(0.5), LIF(), nn.Linear(6, 2), LIF())
    net.compile_sequence_scan(block=4, **FAST)
    x = torch.randn(9, 3, 4)
    net.eval()
    ref = Sequential(*net._layers, layer_major=False).eval()
    _close(net.forward_sequence(x), ref.forward_sequence(x))
    net.train()  # dropout active: per-timestep walk, still runs
    y, *_ = net.forward_sequence(x)
    assert y.shape == (9, 3, 2)


def test_block_compiled_layer_major_gradients_match():
    step, layer = _pair(_mlp)
    layer.compile_sequence_scan(block=4, **FAST)
    x = torch.randn(10, 3, 6)
    grads = []
    for net in (step, layer):
        net.zero_grad()
        xi = x.clone().requires_grad_(True)
        y, *state = net.forward_sequence(xi)
        (y.sum() + sum(s.pow(2).sum() for s in state)).backward()
        grads.append((xi.grad, *(p.grad for p in net.parameters())))
    _close(grads[1], grads[0], atol=1e-4)


# Fused step math and resets


from crematorium.snn.reset import (  # noqa: E402
    _cr_reset_add,
    _cr_reset_set,
    _cr_reset_subtract,
    _cr_reset_zero,
)
from crematorium.util import mul_add  # noqa: E402

_PLAIN = {
    "subtract": lambda s, k, p: s - k * p,
    "add": lambda s, k, p: s + k * p,
    "set": lambda s, k, p: (1 - k) * s + k * p,
}
_FUSED = {"subtract": _cr_reset_subtract, "add": _cr_reset_add, "set": _cr_reset_set}


@pytest.mark.parametrize("kind", ["subtract", "add", "set"])
@pytest.mark.parametrize("graded", [False, True])
def test_fused_resets_match_plain_values_and_grads(kind, graded):
    torch.manual_seed(20)
    state = torch.randn(5, 7, requires_grad=True)
    spk = torch.rand(5, 7) if graded else (torch.rand(5, 7) > 0.5).float()
    spk.requires_grad_(True)
    p = torch.tensor(0.7, requires_grad=True)
    outs = []
    for fn in (_PLAIN[kind], _FUSED[kind]):
        y = fn(state, spk, p)
        grads = torch.autograd.grad((y * torch.arange(7.0)).sum(), (state, spk, p))
        outs.append((y, *grads))
    _close(outs[1], outs[0], atol=1e-6)


def test_fused_set_is_exact_on_binary_spikes():
    state = torch.tensor([1e8, -3.5, 0.25])
    spk = torch.tensor([1.0, 0.0, 1.0])
    p = torch.tensor(-65.0)
    assert torch.equal(_cr_reset_set(state, spk, p), torch.tensor([-65.0, -3.5, -65.0]))


def test_fused_zero_matches_plain():
    torch.manual_seed(21)
    state = torch.randn(4, 3, requires_grad=True)
    spk = torch.rand(4, 3, requires_grad=True)
    outs = []
    for fn in (lambda s, k: s * (1 - k), _cr_reset_zero):
        y = fn(state, spk)
        outs.append((y, *torch.autograd.grad(y.sum(), (state, spk))))
    _close(outs[1], outs[0], atol=1e-6)


def test_fused_helpers_fall_back_on_mixed_dtypes_and_scalars():
    s32 = torch.randn(3)
    s64 = torch.randn(3, dtype=torch.float64)
    k = torch.ones(3)
    assert _cr_reset_subtract(s32, k, 0.5).dtype == torch.float32
    assert _cr_reset_subtract(s64, k, torch.tensor(0.5)).dtype == torch.float64
    _close(_cr_reset_zero(s64, k), s64 * (1 - k))
    assert mul_add(2.0, s32, 1.0).dtype == torch.float32
    _close(mul_add(torch.tensor(0.5, dtype=torch.float64), s32, s32), 0.5 * s32 + s32)
    _close(mul_add(s32, s32, s32), s32 * s32 + s32)


def test_sequence_still_validates_input_dtype_once():
    from crematorium import CrModule as _Cr

    class _IntIn(_Cr):
        class Inputs:
            x: Tensor = _Cr.Input(dtype=float)

        class Specs:
            out = _Cr.OutputSpec()
            v = _Cr.StateSpec()

        def _step(self, x, v):
            return x, v

    m = _IntIn(validate=True)
    with pytest.raises(TypeError):
        m.forward_sequence(torch.ones(4, 2, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="state"):
        m.forward_sequence(torch.ones(4, 2, 3), (torch.zeros(2, 3), torch.zeros(2, 3)))


def test_scan_with_autograd_stacks_outputs_once():
    x = torch.randn(40, 2, 3, requires_grad=True)
    spk, mem = LIF(spike_grad=atan_surrogate()).forward_sequence(x)
    # one Stack node instead of a chain of in-place IndexCopy nodes whose
    # backward would clone the whole output gradient per chunk
    assert "Stack" in type(spk.grad_fn).__name__
    ref_spk, ref_mem = LIF(spike_grad=atan_surrogate()).forward_sequence(x.detach())
    _close(spk.detach(), ref_spk)
    _close(mem.detach(), ref_mem)
    with torch.no_grad():
        out, _ = LIF().forward_sequence(x)
    assert out.grad_fn is None


def test_auto_order_keeps_conv_stacks_per_timestep():
    assert not _conv()._cr_layer_major_ok()  # auto: conv/pool stack
    assert _conv(layer_major=True)._cr_layer_major_ok()
    assert _mlp()._cr_layer_major_ok()  # auto: matmul stack
    nested = Sequential(nn.Sequential(nn.Conv1d(2, 2, 1), nn.ReLU()), LIF())
    assert not nested._cr_layer_major_ok()


def test_fast_sequence_defaults_to_blocks():
    from crematorium.module.mixins.scan import DEFAULT_SCAN_BLOCK

    x = torch.randn(40, 2, 3)
    m = LIF().fast_sequence_(**FAST)
    before = _graphs()
    with torch.no_grad():
        _close(m.forward_sequence(x), LIF().forward_sequence(x))
    # 40 = 16 + 16 + 8: two block sizes, not one 40-step unrolled graph
    assert DEFAULT_SCAN_BLOCK == 16
    assert _graphs() - before == 2
    whole = LIF().fast_sequence_(block=None, **FAST)
    _close(whole.forward_sequence(x), LIF().forward_sequence(x))


def test_sequential_fast_sequence_default_blocks_match():
    step, layer = _pair(_mlp)
    layer.fast_sequence_(**FAST)
    x = torch.randn(37, 2, 6)
    with torch.no_grad():
        _close(layer.forward_sequence(x), step.forward_sequence(x))
