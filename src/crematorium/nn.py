from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional, cast

import torch
import torch.nn as nn

from crematorium.module import (
    CrModule,
    InputTensor,
    OutputSpec,
    StateSpec,
    StepOutput,
    Tensor,
    sequence_scan,
)
from crematorium.module.mixins.scan import (
    DEFAULT_SCAN_BLOCK,
    BlockScan,
    _store_hidden_seq_buffers,
)

__all__ = ["Sequential", "is_time_distributed"]


def _scan_layer_seq(
    layer: CrModule, inputs_seq: tuple[Tensor, ...], state: tuple[Tensor, ...]
) -> tuple[Tensor, ...]:
    # A layer with the stock forward runs its own sequence scan (inputs
    # checked once, core step per timestep). A layer that overrides forward
    # (e.g. extra device checks) is stepped through forward, exactly as the
    # per-timestep walk calls it.
    if type(layer).forward is CrModule.forward:
        return layer._cr_scan(inputs_seq, state)
    return sequence_scan(
        lambda inputs, s: cast(StepOutput, layer.forward(inputs[0], *s)),
        inputs_seq,
        state,
        1,
    )


def _scan_layer(
    layer: CrModule, x: Tensor, state: tuple[Tensor, ...]
) -> tuple[Tensor, ...]:
    return _scan_layer_seq(layer, (x,), state)


# Stateless layers that treat dim 0 as an independent batch dimension: on a
# time-major sequence flattened to (time * batch, ...) they compute exactly
# what they compute per timestep, so Sequential can apply them to the whole
# sequence at once (one large matmul instead of `time` small ones).
_TIME_DISTRIBUTED: tuple[type[nn.Module], ...] = (
    nn.Identity,
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.Upsample,
    nn.ZeroPad1d,
    nn.ZeroPad2d,
    nn.ZeroPad3d,
    nn.ConstantPad1d,
    nn.ConstantPad2d,
    nn.ConstantPad3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.Embedding,
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.PReLU,
    nn.ELU,
    nn.SELU,
    nn.CELU,
    nn.GELU,
    nn.SiLU,
    nn.Mish,
    nn.Sigmoid,
    nn.Tanh,
    nn.Hardtanh,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.Softplus,
    nn.Softsign,
    nn.Tanhshrink,
    nn.LogSigmoid,
)


# Layers whose per-step work is already large (spatial maps): batching them
# over time gains little while every intermediate sequence (time x batch x
# channels x H x W) must be materialized. Measured on a conv-LIF stack,
# layer-major was slower in training and inference and used ~10x the memory,
# so the automatic mode keeps such stacks on the per-timestep walk.
_SPATIAL: tuple[type[nn.Module], ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.Upsample,
)


def _has_spatial(layer: nn.Module) -> bool:
    return any(isinstance(m, _SPATIAL) for m in layer.modules())


def _batch_dim_ok(dim: Optional[int]) -> bool:
    # Flattening (time, batch) keeps the per-step rank, so any non-zero dim
    # (positive or negative) addresses the same axis; dim 0 / None would mix
    # timesteps.
    return dim is not None and dim != 0


def is_time_distributed(layer: nn.Module) -> bool:
    """
    True when ``layer`` can be applied to a whole time-major sequence at once.

    That holds for stateless layers that process every sample along dim 0
    independently (``Linear``, convolutions, pooling, element-wise
    activations, per-sample norms, ...). Layers whose output for one sample
    depends on the others in the batch (``BatchNorm`` using batch statistics,
    ``Dropout`` in training mode - its mask would be drawn from a different
    random stream) are not. Custom layers opt in or out with a
    ``_cr_time_distributed = True/False`` class attribute.
    """
    flag = getattr(layer, "_cr_time_distributed", None)
    if flag is not None:
        return bool(flag)
    if isinstance(layer, CrModule):
        return False
    if isinstance(layer, nn.Sequential):
        return all(is_time_distributed(child) for child in layer)
    if isinstance(
        layer, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)
    ):
        return not layer.training or layer.p == 0
    if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        # Eval mode with running stats normalizes per sample.
        return not layer.training and layer.track_running_stats
    if isinstance(layer, nn.Flatten):
        return layer.start_dim != 0
    if isinstance(layer, nn.Unflatten):
        return isinstance(layer.dim, int) and layer.dim != 0
    if isinstance(layer, (nn.Softmax, nn.LogSoftmax, nn.Softmin)):
        return _batch_dim_ok(layer.dim)
    return type(layer) in _TIME_DISTRIBUTED


# CrModule's __init_subclass__ generates a parameter-only signature that
# hides the positional *layers. The real constructor signature, restored on
# Sequential below and re-applied to subclasses without their own __init__.
_SEQUENTIAL_SIG = inspect.Signature(
    [
        inspect.Parameter("layers", inspect.Parameter.VAR_POSITIONAL),
        inspect.Parameter("init_hidden", inspect.Parameter.KEYWORD_ONLY, default=False),
        inspect.Parameter("validate", inspect.Parameter.KEYWORD_ONLY, default=None),
        inspect.Parameter("layer_major", inspect.Parameter.KEYWORD_ONLY, default=None),
    ]
)


class Sequential(CrModule):
    """
    Stack layers into a network.

    A ``CrModule`` topology manager: it moves data between layers and
    threads a flat state tuple through the time-major scan. Stateful
    ``CrModule`` layers stay pure functions (``init_hidden=False``);
    stateless ``nn.Module`` layers are applied once per timestep. The whole
    stack runs as one step function, so ``fast_sequence_()`` compiles a single
    fused scan over every layer.

    Sequence scans can run layer by layer when every stateless layer is
    time-distributed (``is_time_distributed``): a ``Linear`` is applied to
    the whole ``(time * batch, features)`` sequence in one matmul and each
    stateful layer scans its own input sequence. The stack has no feedback
    between layers, so this computes the same outputs and final state as the
    per-timestep walk with far fewer, larger kernels. By default
    (``layer_major=None``) this is used for eligible stacks without
    convolution/pooling layers, where it measured ~1.7x faster in eager
    training and inference; conv stacks stay on the per-timestep walk, which
    was faster there and needs far less memory. ``layer_major=False`` forces
    the per-timestep walk; ``layer_major=True`` forces layer-major and makes
    an ineligible stack an error instead of a silent fallback.

    ``init_hidden=True`` stateful layers are rejected: the container owns the
    state bundle, so children must run in explicit mode.

    The container reuses the ``CrModule`` machinery. Hidden state lives
    in non-persistent buffers named ``l{layer_index}_{state_name}`` (e.g.
    ``l0_mem``), and scans, compiled scans, validation, and serialization all
    come from the base mixins. State shapes are not declared: they are
    resolved from the layer stack by a fake-tensor shape walk
    (``_cr_spec_shape``).
    """

    # Container metadata is built per instance in __init__; the class-level
    # ClassVar copies from the base mixins stay empty.
    _cr_output_names: tuple[str, ...]  # pyright: ignore[reportIncompatibleVariableOverride]
    _cr_output_specs: tuple[OutputSpec, ...]  # pyright: ignore[reportIncompatibleVariableOverride]
    _cr_state_names: tuple[str, ...]  # pyright: ignore[reportIncompatibleVariableOverride]
    _cr_state_specs: tuple[StateSpec, ...]  # pyright: ignore[reportIncompatibleVariableOverride]
    _cr_spec_entries: tuple[tuple[str, Any], ...]  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Re-apply the layers signature so subclasses of Sequential keep
        # (*layers, init_hidden, validate) — unless the subclass defines
        # its own __init__, whose signature always wins.
        if "__init__" in cls.__dict__:
            cls.__signature__ = inspect.signature(cls.__init__)
        else:
            cls.__signature__ = _SEQUENTIAL_SIG

    def __init__(
        self,
        *layers: nn.Module,
        init_hidden: bool = False,
        validate: Optional[bool] = None,
        layer_major: Optional[bool] = None,
    ) -> None:
        super().__init__(init_hidden=init_hidden, validate=validate)

        if layer_major is not None and not isinstance(layer_major, bool):
            raise TypeError(f"layer_major must be a bool or None, got {layer_major!r}")
        # None = automatic (layer-major whenever every stateless layer is
        # time-distributed); False forces the per-timestep stack walk; True
        # requires layer-major and fails loudly if a layer does not allow it.
        self.layer_major: Optional[bool] = layer_major

        if not layers:
            raise ValueError("Sequential requires at least one layer")

        for layer in layers:
            if not isinstance(layer, nn.Module):
                raise TypeError(
                    f"Sequential layers must be nn.Module instances, "
                    f"got {type(layer).__name__}"
                )

            if isinstance(layer, CrModule) and layer.init_hidden:
                raise ValueError(
                    f"stateful layer {type(layer).__name__} is in "
                    f"init_hidden=True mode; Sequential owns the state, pass "
                    f"init_hidden=False"
                )

        self._layers: list[nn.Module] = list(layers)

        for i, layer in enumerate(self._layers):
            setattr(self, f"layer{i}", layer)

        # Flat state registry: one StateSpec per stateful-layer state, in
        # layer order. Defaults are resolved against the owning layer, so
        # hidden-mode allocation and the state factories fill the same values
        # the layer itself would use. Shapes stay unset ("input"): they are
        # resolved from the layer stack via _cr_spec_shape.
        state_entries: list[tuple[str, StateSpec]] = []

        for i, layer in enumerate(self._layers):
            if not isinstance(layer, CrModule):
                continue

            if len(layer._cr_input_names) != 1:
                raise ValueError(
                    f"Sequential requires single-input layers; "
                    f"{type(layer).__name__} declares "
                    f"{len(layer._cr_input_names)} inputs"
                )
            if len(layer._cr_output_specs) != 1:
                raise ValueError(
                    f"Sequential requires single-output layers; "
                    f"{type(layer).__name__} declares "
                    f"{len(layer._cr_output_specs)} outputs"
                )

            for name, spec in layer._cr_spec_entries:
                if isinstance(spec, StateSpec):
                    state_entries.append(
                        (
                            f"l{i}_{name}",
                            StateSpec(default=layer._cr_resolve_default(spec.default)),
                        )
                    )

        output_spec = OutputSpec()
        # Base mixins keep these as ClassVar per class; Sequential needs per-instance
        # values, so they are redeclared above with pyright: ignore and set here
        # via object.__setattr__ to avoid a second ignore on the assignment.
        object.__setattr__(self, "_cr_output_names", ("y",))
        object.__setattr__(self, "_cr_output_specs", (output_spec,))
        object.__setattr__(
            self, "_cr_state_names", tuple(name for name, _ in state_entries)
        )
        object.__setattr__(
            self, "_cr_state_specs", tuple(spec for _, spec in state_entries)
        )
        object.__setattr__(
            self, "_cr_spec_entries", (("y", output_spec), *state_entries)
        )

    def _cr_rebuild_state_registry(self) -> None:
        old_state_names = self.__dict__.get("_cr_state_names", ())
        state_entries: list[tuple[str, StateSpec]] = []
        for i, layer in enumerate(self._layers):
            if isinstance(layer, CrModule):
                for name, spec in layer._cr_spec_entries:
                    if isinstance(spec, StateSpec):
                        state_entries.append(
                            (
                                f"l{i}_{name}",
                                StateSpec(
                                    default=layer._cr_resolve_default(spec.default)
                                ),
                            )
                        )
        out_spec = self._cr_output_specs[0] if self._cr_output_specs else OutputSpec()
        new_state_names = tuple(n for n, _ in state_entries)
        object.__setattr__(self, "_cr_output_names", ("y",))
        object.__setattr__(self, "_cr_output_specs", (out_spec,))
        object.__setattr__(self, "_cr_state_names", new_state_names)
        object.__setattr__(self, "_cr_state_specs", tuple(s for _, s in state_entries))
        object.__setattr__(self, "_cr_spec_entries", (("y", out_spec), *state_entries))
        self.__dict__.pop("_cr_probe_cache", None)
        self._cr_compiled_sequence = None
        # When the state-name set changes (layer added/removed or state
        # arity changed), purge orphaned buffers and force re-allocation
        # so the surviving layers read correct hidden state on next forward.
        if new_state_names != old_state_names:
            valid = set(new_state_names) | set(self._cr_output_names)
            for name in list(self._buffers):
                if name not in valid:
                    self._buffers.pop(name, None)
                    self._non_persistent_buffers_set.discard(name)
            self.__dict__["_cr_allocated"] = False

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            name.startswith("layer")
            and name[5:].lstrip("-").isdigit()
            and "_layers" in self.__dict__
        ):
            if not isinstance(value, nn.Module):
                raise TypeError(
                    f"Sequential layer must be nn.Module, got {type(value).__name__}"
                )
            idx = int(name[5:])
            layers = self.__dict__["_layers"]
            if idx < 0:
                idx += len(layers)
            if not 0 <= idx < len(layers):
                raise IndexError(f"Sequential layer index out of range: {name}")
            if layers[idx] is not value:
                if isinstance(value, CrModule) and value.init_hidden:
                    raise ValueError(
                        f"stateful layer {type(value).__name__} is in init_hidden=True mode; Sequential owns state"
                    )
                layers[idx] = value
                self._cr_rebuild_state_registry()
            super().__setattr__(f"layer{idx}", value)
            return
        if name == "_layers" and "_layers" in self.__dict__:
            super().__setattr__(name, value)
            # Children live in _modules, not __dict__: iterate _modules to
            # correctly deregister stale layer{i>len-1} entries.
            for key in [
                k
                for k in list(self._modules)
                if k.startswith("layer")
                and k[5:].isdigit()
                and int(k[5:]) >= len(value)
            ]:
                try:
                    super().__delattr__(key)
                except AttributeError:
                    pass
            for i, layer in enumerate(value):
                nn.Module.__setattr__(self, f"layer{i}", layer)  # type: ignore[attr-defined]
            self._cr_rebuild_state_registry()
            return
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if (
            name.startswith("layer")
            and name[5:].isdigit()
            and "_layers" in self.__dict__
        ):
            idx = int(name[5:])
            layers = self.__dict__["_layers"]
            if 0 <= idx < len(layers):
                raise AttributeError(
                    f"Sequential layer {idx} cannot be deleted via delattr; use del net[{idx}]"
                )
        super().__delattr__(name)

    def __setitem__(self, idx: int, value: nn.Module) -> None:
        n = len(self._layers)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(idx)
        if not isinstance(value, nn.Module):
            raise TypeError(f"Sequential layer {idx} must be nn.Module")
        if isinstance(value, CrModule) and value.init_hidden:
            raise ValueError(
                f"stateful layer {type(value).__name__} is init_hidden=True"
            )
        self._layers[idx] = value
        nn.Module.__setattr__(self, f"layer{idx}", value)  # type: ignore[attr-defined]
        self._cr_rebuild_state_registry()

    def __delitem__(self, idx: int) -> None:
        n = len(self._layers)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise IndexError(idx)
        # Validate before mutating: a rejected del must leave _layers, the
        # layer{i} registrations, and the state registry untouched, not brick
        # the module half-way through the operation.
        if n == 1:
            raise ValueError("Sequential requires at least one layer")
        del self._layers[idx]
        # re-register remaining layers under correct indices (children live
        # in _modules, not __dict__)
        for key in [
            k for k in list(self._modules) if k.startswith("layer") and k[5:].isdigit()
        ]:
            try:
                super().__delattr__(key)
            except AttributeError:
                pass
        for i, layer in enumerate(self._layers):
            nn.Module.__setattr__(self, f"layer{i}", layer)  # type: ignore[attr-defined]
        self._cr_rebuild_state_registry()

    def __getitem__(self, idx: int) -> nn.Module:
        return self._layers[idx]

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):  # type: ignore[override]
        return iter(self._layers)

    # The per-step dispatcher: the compiled unit for the whole network.

    def _step(self, x: Tensor, *state: Tensor) -> StepOutput:
        offset = 0
        next_states: list[Tensor] = []

        for layer in self._layers:
            if isinstance(layer, CrModule):
                n = len(layer._cr_state_names)
                out = layer.forward(x, *state[offset : offset + n])
                assert isinstance(out, tuple)
                x = out[0]
                next_states.extend(out[1:])
                offset += n
            else:
                x = layer(x)
                # Runtime contract check: a user stateless layer may return a
                # non-tensor despite nn.Module.__call__'s annotation.
                if not isinstance(x, Tensor):
                    raise TypeError(  # pyright: ignore[reportUnreachable]
                        f"stateless layer {type(layer).__name__} in "
                        f"Sequential must return a single tensor"
                    )

        return (x, *next_states)

    # Sequence evaluation order.

    def _cr_layer_major_ok(self) -> bool:
        """
        Whether sequence scans may run layer by layer.

        Requires the stock per-step dispatcher (a subclass overriding
        ``_step`` or adding post-step hooks defines its own per-step
        semantics) and time-distributed stateless layers.
        """
        if self.layer_major is False:
            return False
        reasons: list[str] = []
        if type(self)._step is not Sequential._step:
            reasons.append(f"{type(self).__name__} overrides _step")
        if self._cr_hook_post_steps:
            reasons.append(f"{type(self).__name__} has post-step hooks")
        for i, layer in enumerate(self._layers):
            if not isinstance(layer, CrModule) and not is_time_distributed(layer):
                reasons.append(
                    f"layer {i} ({type(layer).__name__}) is not time-distributed"
                )
        if reasons and self.layer_major is True:
            raise ValueError(
                f"{type(self).__name__}(layer_major=True) cannot run layer by "
                f"layer: " + "; ".join(reasons)
            )
        if reasons:
            return False
        if self.layer_major is None:
            # Automatic: layer-major pays off for matmul stacks (Linear and
            # element-wise layers), not for convolution/pooling stacks.
            return not any(
                _has_spatial(layer)
                for layer in self._layers
                if not isinstance(layer, CrModule)
            )
        return True

    def _cr_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
        state0: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        if not self._cr_layer_major_ok():
            return super()._cr_scan(inputs_seq, state0)

        return self._cr_layer_major_run(inputs_seq, state0, _scan_layer)

    def _cr_layer_major_run(
        self,
        inputs_seq: tuple[Tensor, ...],
        state0: tuple[Tensor, ...],
        scan_layer: Callable[
            [CrModule, Tensor, tuple[Tensor, ...]], tuple[Tensor, ...]
        ],
    ) -> tuple[Tensor, ...]:
        """
        Layer-major evaluation. ``scan_layer(layer, x_seq, state)`` scans one
        stateful layer over its whole input sequence and returns
        ``(y_seq, *final_state)``: eagerly in ``_cr_scan``, through a
        per-layer compiled ``BlockScan`` in block-compiled mode.
        """
        (x,) = inputs_seq

        # Same input checks the per-step walk runs, once for the sequence.
        self._cr_check_explicit_inputs((x[0],), state0)

        offset = 0
        final_state: list[Tensor] = []

        for layer in self._layers:
            if isinstance(layer, CrModule):
                n = len(layer._cr_state_names)
                out = scan_layer(layer, x, tuple(state0[offset : offset + n]))
                x = out[0]
                final_state.extend(out[1:])
                offset += n
            else:
                time, batch = x.shape[0], x.shape[1]
                y = layer(x.reshape(time * batch, *x.shape[2:]))
                if not isinstance(y, Tensor):
                    raise TypeError(  # pyright: ignore[reportUnreachable]
                        f"stateless layer {type(layer).__name__} in "
                        f"Sequential must return a single tensor"
                    )
                x = y.reshape(time, batch, *y.shape[1:])

        return (x, *final_state)

    def _cr_block_compiled_scan(
        self,
        block: int,
        needs_clone: bool,
        kwargs: dict[str, Any],
    ) -> Callable[[tuple[Tensor, ...], tuple[Tensor, ...]], tuple[Tensor, ...]]:
        """
        Block-compiled scan for the container.

        Layer-major (decided per call, so ``train()``/``eval()`` switches of
        Dropout/BatchNorm are honored): each stateful layer gets its own
        ``BlockScan`` of just that layer, and stateless layers run eagerly on
        the whole sequence (one large matmul each). Compile cost is one small
        graph per block length per distinct stateful layer instead of one
        graph spanning every layer; a layer instance repeated in the stack
        reuses its graphs. Per-timestep (non-eligible) stacks fall back to a
        ``BlockScan`` over the whole per-step dispatcher.
        """
        n_outputs = len(self._cr_output_names)
        per_layer: dict[int, BlockScan] = {}
        whole: list[BlockScan] = []

        def scan_layer(
            layer: CrModule, x: Tensor, state: tuple[Tensor, ...]
        ) -> tuple[Tensor, ...]:
            scanner = per_layer.get(id(layer))
            if scanner is None or scanner.owner is not layer:
                scanner = BlockScan(
                    functools.partial(_scan_layer_seq, layer),
                    block,
                    1,
                    needs_clone,
                    kwargs,
                )
                scanner.owner = layer
                per_layer[id(layer)] = scanner
            return scanner((x,), state)

        def run(
            inputs_seq: tuple[Tensor, ...], state: tuple[Tensor, ...]
        ) -> tuple[Tensor, ...]:
            if self._cr_layer_major_ok():
                return self._cr_layer_major_run(inputs_seq, state, scan_layer)
            if not whole:
                whole.append(
                    BlockScan(self._cr_scan, block, n_outputs, needs_clone, kwargs)
                )
            return whole[0](inputs_seq, state)

        return run

    # Shape pass: FakeTensorMode walk.

    def _cr_probe(
        self,
        batch_shape: tuple[int, ...],
        dtype: Optional[torch.dtype],
    ) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
        # Run the stack under FakeTensorMode with a fake input: real params
        # and buffers are viewed as fake tensors for the walk, so there is
        # nothing to swap, nothing to restore, and no shared state to lock.
        # Only shapes are read from the results.
        # Imported lazily: private torch module, and the probe is cached per
        # shape so the per-call import cost is a one-time dict lookup.
        from torch._subclasses.fake_tensor import (  # pyright: ignore[reportMissingImports]
            DataDependentOutputException,
            FakeTensorMode,
        )

        with FakeTensorMode(allow_non_fake_inputs=True):
            # Walk device must match the stack's real params/buffers: fake
            # conversion keeps their device, and ops like linear reject
            # mixed-device inputs (elementwise LIF math tolerates them, so
            # bare-neuron stacks never noticed). Read-only scan, no data.
            probe_device = torch.device("cpu")
            for module in self.modules():
                for t in (*module._parameters.values(), *module._buffers.values()):
                    if isinstance(t, Tensor):
                        probe_device = t.device
                        break
                if probe_device.type != "cpu":
                    break
            x = torch.empty(batch_shape, dtype=dtype, device=probe_device)
            shapes: list[tuple[int, ...]] = []

            for layer in self._layers:
                try:
                    if isinstance(layer, CrModule):
                        state = layer.initial_state(
                            tuple(x.shape), device=probe_device, dtype=dtype
                        )
                        out = layer.forward(x, *state)
                        assert isinstance(out, tuple)
                        x = out[0]
                        shapes.extend(tuple(t.shape) for t in out[1:])
                    else:
                        x = layer(x)
                        if not isinstance(x, Tensor):
                            raise TypeError(  # pyright: ignore[reportUnreachable]
                                f"stateless layer {type(layer).__name__} in "
                                f"Sequential must return a single tensor"
                            )
                except DataDependentOutputException as exc:
                    raise RuntimeError(
                        f"{type(self).__name__} shape probe failed on layer {type(layer).__name__}: "
                        f"_step branched on tensor data or used Tensor.item() with fake tensors; "
                        f"keep _step shape-safe (no data-dependent Python control flow)"
                    ) from exc
                except RuntimeError as exc:
                    msg = str(exc)
                    if (
                        "scalar" in msg.lower()
                        or "item()" in msg
                        or "fake" in msg.lower()
                    ):
                        raise RuntimeError(
                            f"{type(self).__name__} shape probe failed on layer {type(layer).__name__}: "
                            f"_step branched on tensor data or used Tensor.item() with fake tensors; "
                            f"keep _step shape-safe (no data-dependent Python control flow)"
                        ) from exc
                    raise

        return tuple(x.shape), shapes

    def _cr_probe_shapes(
        self,
        x: Tensor,
    ) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
        """
        Shape walk with caching, so the per-spec ``_cr_spec_shape`` lookups in
        hidden allocation and validation do not re-walk the stack each time.
        """
        # Fingerprint topology + param shapes so stale cache can't survive
        # a layer replacement or weight resize without an explicit setattr.
        topo = tuple(id(layer) for layer in self._layers)
        param_shapes = tuple(
            tuple(tuple(p.shape) for p in layer.parameters()) for layer in self._layers
        )
        key = (tuple(x.shape), x.dtype, topo, param_shapes)
        cache = self.__dict__.get("_cr_probe_cache")

        if cache is None:
            cache = {}
            self.__dict__["_cr_probe_cache"] = cache

        hit = cache.get(key)

        if hit is not None:
            return hit

        result = self._cr_probe(tuple(x.shape), x.dtype)
        cache[key] = result
        return result

    # Container state shapes come from the layer stack, not from a declarative
    # spec, so resolve them with the meta walk and map back by spec identity.

    def _cr_spec_shape(self, spec: Any, inputs: tuple[Tensor, ...]) -> tuple[int, ...]:
        out_shape, state_shapes = self._cr_probe_shapes(
            inputs[self._cr_primary_input_index]
        )

        if spec is self._cr_output_specs[0]:
            return out_shape

        for i, s in enumerate(self._cr_state_specs):
            if s is spec:
                return state_shapes[i]

        raise AssertionError(
            f"{type(self).__name__}._cr_spec_shape called with a spec it does not own"
        )

    # State factories. These override StateMixin's: container state shapes are
    # resolved by the shape walk, not from ``StateSpec.shape``/``batch_shape``.

    def _cr_seq_shapes_for_batch(
        self, batch_shape: tuple[int, ...], dtype: Optional[torch.dtype]
    ) -> list[tuple[int, ...]]:
        _, shapes = self._cr_probe(tuple(batch_shape), dtype)
        return shapes

    def _cr_seq_state(
        self,
        shapes: list[tuple[int, ...]],
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        fill: str,
    ) -> tuple[Tensor, ...]:
        dtype = (
            dtype
            if dtype is None or dtype.is_floating_point
            else torch.get_default_dtype()
        )
        out: list[Tensor] = []
        for shape, spec in zip(shapes, self._cr_state_specs, strict=True):
            if fill == "full":
                out.append(
                    torch.full(
                        shape,
                        self._cr_resolve_default(spec.default),
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                out.append(torch.zeros(shape, device=device, dtype=dtype))
        return tuple(out)

    def initial_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        return self._cr_seq_state(
            self._cr_seq_shapes_for_batch(batch_shape, dtype), device, dtype, "full"
        )

    def zero_state(
        self,
        batch_shape: tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[Tensor, ...]:
        return self._cr_seq_state(
            self._cr_seq_shapes_for_batch(batch_shape, dtype), device, dtype, "zero"
        )

    def initial_state_like(
        self,
        inputs: InputTensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        inputs = self._cr_canonicalize_inputs(inputs)
        primary = inputs[self._cr_primary_input_index]
        dtype = (
            primary.dtype if primary.is_floating_point() else torch.get_default_dtype()
        )
        _, shapes = self._cr_probe_shapes(primary)
        shapes = [tuple(batch_shape) if batch_shape is not None else s for s in shapes]
        return self._cr_seq_state(shapes, primary.device, dtype, "full")

    def zero_state_like(
        self,
        inputs: InputTensor,
        batch_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[Tensor, ...]:
        inputs = self._cr_canonicalize_inputs(inputs)
        primary = inputs[self._cr_primary_input_index]
        dtype = (
            primary.dtype if primary.is_floating_point() else torch.get_default_dtype()
        )
        _, shapes = self._cr_probe_shapes(primary)
        shapes = [tuple(batch_shape) if batch_shape is not None else s for s in shapes]
        return self._cr_seq_state(shapes, primary.device, dtype, "zero")

    # Compile path. State resolution stays outside the compiled region: the
    # shape walk runs under FakeTensorMode, which would break inside a
    # ``torch.compile`` trace.

    def compile_sequence_scan(
        self, block: Optional[int] = None, **kwargs: Any
    ) -> "Sequential":
        needs_clone = kwargs.get("mode") in ("reduce-overhead", "max-autotune")

        if self.init_hidden and needs_clone:
            raise ValueError(
                f"{type(self).__name__}.compile_sequence_scan(mode="
                f"{kwargs.get('mode')!r}) is not supported with "
                f"init_hidden=True: the compiled graph owns its output "
                f"tensors, and storing them into the container's hidden "
                f"buffers would alias compiler-recycled CUDA-graph memory. "
                f"Use mode='default', or run the layers in explicit mode "
                f"(init_hidden=False) if you need CUDA graphs."
            )

        compiled: Any = (
            self._cr_block_compiled_scan(block, needs_clone, kwargs)
            if block is not None
            else torch.compile(
                lambda inputs_seq, state: self._cr_scan(inputs_seq, state),
                **kwargs,
            )
        )

        def _cr_compiled(
            x_seq: Tensor,
            state: Optional[tuple[Tensor, ...]] = None,
        ) -> Tensor | StepOutput:
            inputs_seq = self._cr_canonicalize_input_sequence(x_seq)

            if inputs_seq[0].shape[0] == 0:
                raise ValueError(
                    f"{type(self).__name__}.forward_sequence expects at least one timestep"
                )

            if self.init_hidden:
                if inputs_seq[0].shape[0] > 0:
                    self.allocate_like(self._cr_first_inputs(inputs_seq))
                state = tuple(getattr(self, name) for name in self._cr_state_names)
            elif state is None:
                state = self.initial_state_for_sequence(inputs_seq)

            out = compiled(inputs_seq, state)

            if self.init_hidden:
                y = out[0]
                _store_hidden_seq_buffers(self, (y,), out[1:])  # type: ignore[arg-type]
                return y.clone() if needs_clone else y

            if isinstance(out, Tensor):
                return out.clone() if needs_clone else out

            return tuple(t.clone() if needs_clone else t for t in out)

        self._cr_compiled_sequence = _cr_compiled
        return self

    def fast_sequence_(
        self,
        compile_scan: bool = True,
        block: Optional[int] = DEFAULT_SCAN_BLOCK,
        **compile_kwargs: Any,
    ) -> "Sequential":
        for layer in self._layers:
            if isinstance(layer, CrModule):
                layer.validate = False

        super().fast_sequence_(compile_scan, block, **compile_kwargs)
        return self

    def __repr__(self) -> str:
        inner = ",\n  ".join(repr(layer) for layer in self._layers)
        return f"{type(self).__name__}(\n  {inner}\n)"


setattr(Sequential, "__signature__", _SEQUENTIAL_SIG)  # noqa: B010
