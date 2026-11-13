from __future__ import annotations

import itertools
import types
from contextvars import ContextVar
from typing import Any, Callable, ClassVar, Optional, Self

import torch

from ..specs import (
    InputSpec,
    InputTensor,
    OutputSpec,
    Spec,
    StateSpec,
    StepOutput,
    Tensor,
)
from .forward import ForwardMixin
from .validation import (
    check_hidden_input_shape,
    is_validating,
    set_validating,
)


def _clone_last_for_buffer(t: Tensor, spec: OutputSpec) -> Tensor:
    """Clone ``t[-1]`` to a compact (B,F) buffer so ``ys`` can be freed."""
    if t.dim() > 0:
        last = t[-1].clone()
        return last.detach() if not spec.differentiable else last
    return t.detach() if not spec.differentiable else t


def _store_hidden_seq_buffers(
    host: SequenceScanMixin,
    ys: tuple[Tensor, ...],
    tail: tuple[Tensor, ...],
) -> None:
    for name, t in zip(host._cr_state_names, tail, strict=True):
        host._buffers[name] = t
    for name, spec, t in zip(
        host._cr_output_names, host._cr_output_specs, ys, strict=True
    ):
        # Output buffers are created here on first store and must be
        # non-persistent (hidden contents travel via get_extra_state).
        host._non_persistent_buffers_set.add(name)
        host._buffers[name] = _clone_last_for_buffer(t, spec)


# In eager scans, batch this many steps into a single index_copy_ scatter so
# peak memory stays at input + output (no per-step (B, F) list held for stack).
_SEQUENCE_SCAN_CHUNK_CTX: ContextVar[int] = ContextVar(
    "_SEQUENCE_SCAN_CHUNK", default=8
)
# Back-compat for ``from .scan import _SEQUENCE_SCAN_CHUNK``
_SEQUENCE_SCAN_CHUNK = 8

# Default block length for ``fast_sequence_``. Measured on LIF, MLP and
# conv stacks (T=100..10000): 16 compiles ~3x faster than 64 and ran as fast
# or faster (smaller graphs keep the fused per-step kernels cache-friendly).
DEFAULT_SCAN_BLOCK = 16


def _get_sequence_scan_chunk() -> int:
    return _SEQUENCE_SCAN_CHUNK_CTX.get()


def set_sequence_scan_chunk(chunk: int) -> None:
    """
    Set the eager-scan chunk size used by ``forward_sequence``.

    Chunking batches several steps into one ``index_copy_`` scatter. Larger
    values reduce dispatch overhead but increase peak memory; the optimal size
    depends on hardware, dtype, and batch/feature dims. Pass ``1`` to disable
    chunking.

    Context-local via ContextVar (thread/async-safe).
    """

    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError(f"scan chunk must be a positive int, got {chunk!r}")

    _SEQUENCE_SCAN_CHUNK_CTX.set(chunk)


# Time-major scan over a pure step function.


def sequence_scan(
    step: Callable[[tuple[Tensor, ...], tuple[Tensor, ...]], tuple[Tensor, ...]],
    inputs_seq: tuple[Tensor, ...],
    state0: tuple[Tensor, ...],
    n_outputs: int,
) -> tuple[Tensor, ...]:
    """
    Scan a pure step function over a time-major input.

    ``step`` is ``(inputs, state) -> (*outputs, *next_state)`` where ``inputs``
    is the canonical per-timestep input tuple (one tensor per declared input):
    the first ``n_outputs`` tensors are outputs, the rest become the state for
    the next step. Returns ``(*ys, *final_state)`` where each ``ys[k]`` is a
    preallocated ``(T, *output_shape)`` buffer.

    ``inputs_seq`` is the canonical tuple of time-major sequences (one per
    declared input, all sharing time length ``T``).

    In eager this batches ``_SEQUENCE_SCAN_CHUNK`` steps into one
    ``index_copy_`` scatter so peak memory stays at input + output. Under
    ``torch.compile`` it lowers to a flat fused loop; the whole scan becomes a
    single graph.
    """
    T = inputs_seq[0].shape[0]

    inputs0 = tuple(seq[0] for seq in inputs_seq)
    out0 = step(inputs0, state0)
    assert isinstance(out0, tuple)

    def alloc() -> tuple[Tensor, ...]:
        return tuple(
            torch.empty((T, *o.shape), dtype=o.dtype, device=o.device)
            for o in out0[:n_outputs]
        )

    if torch.compiler.is_compiling():
        ys = alloc()
        for k, y in enumerate(ys):
            y.index_copy_(0, torch.tensor([0], device=y.device), out0[k].unsqueeze(0))

        cur = out0[n_outputs:]

        for t in range(1, T):
            inputs_t = tuple(seq[t] for seq in inputs_seq)
            out = step(inputs_t, cur)

            for k, y in enumerate(ys):
                y.index_copy_(
                    0,
                    torch.tensor([t], device=y.device),
                    out[k].unsqueeze(0),
                )

            cur = out[n_outputs:]
    elif _records_grad(inputs_seq, state0, out0):
        # Autograd is recording: in-place writes into the preallocated
        # buffer would chain one IndexCopyBackward per chunk, and each of
        # those clones the gradient of the whole (T, ...) output in backward
        # (O(T^2 / chunk) traffic). Collect the steps and stack once instead:
        # one forward copy, and backward just splits the gradient.
        per_step = [seq.unbind(0) for seq in inputs_seq]
        single = per_step[0] if len(per_step) == 1 else None
        outs: list[list[Tensor]] = [[o] for o in out0[:n_outputs]]
        cur = out0[n_outputs:]

        for t in range(1, T):
            inputs_t = (
                (single[t],) if single is not None else tuple(s[t] for s in per_step)
            )
            out = step(inputs_t, cur)

            for k in range(n_outputs):
                outs[k].append(out[k])

            cur = out[n_outputs:]

        ys = tuple(torch.stack(o) for o in outs)
    else:
        ys = alloc()
        for k, y in enumerate(ys):
            y[0] = out0[k]

        cur = out0[n_outputs:]
        idx = torch.arange(T, device=inputs_seq[0].device)
        chunk = _get_sequence_scan_chunk()
        # One unbind per input yields every per-step view at once, instead of
        # a Python-level select (and an autograd SelectBackward) per step.
        per_step = [seq.unbind(0) for seq in inputs_seq]
        single = per_step[0] if len(per_step) == 1 else None

        for lo in range(1, T, chunk):
            hi = min(lo + chunk, T)

            chunks: list[list[Tensor]] = [[] for _ in range(n_outputs)]

            for t in range(lo, hi):
                inputs_t = (
                    (single[t],)
                    if single is not None
                    else tuple(steps[t] for steps in per_step)
                )
                out = step(inputs_t, cur)

                for k in range(n_outputs):
                    chunks[k].append(out[k])

                cur = out[n_outputs:]

            for k, y in enumerate(ys):
                y.index_copy_(0, idx[lo:hi], torch.stack(chunks[k]))

    return (*ys, *cur)


def _records_grad(*groups: tuple[Tensor, ...]) -> bool:
    """True when autograd records the scan (grad mode on and a tensor needs grad)."""
    if not torch.is_grad_enabled():
        return False
    return any(t.requires_grad for g in groups for t in g if isinstance(t, Tensor))


def _block_sizes(time: int, block: int) -> list[int]:
    """
    Split ``time`` steps into full ``block``-step chunks plus a tail written
    as a sum of distinct powers of two (e.g. ``block=64, time=200`` ->
    ``[64, 64, 64, 8]``). Every chunk length comes from a fixed set of at most
    ``block.bit_length() + 1`` sizes, so a compiled block scan builds that
    many graphs once and then serves any sequence length without recompiling.
    """
    sizes = [block] * (time // block)
    rest = time % block
    size = 1 << rest.bit_length()

    while rest:
        while size > rest:
            size >>= 1
        sizes.append(size)
        rest -= size

    return sizes


_BLOCK_IDS = itertools.count()


def _fresh_code(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap ``fn`` in a function with its own code object.

    Dynamo keeps its compiled-graph cache per code object. Every block length
    of every scanner gets a private entry this way, so many modules of the
    same class (or many block lengths) never pile up on one shared cache and
    hit the recompile limit, where Dynamo silently falls back to eager.
    """

    def run(inputs_seq: tuple[Tensor, ...], state: tuple[Tensor, ...]) -> Any:
        return fn(inputs_seq, state)

    code = run.__code__.replace(co_name=f"_cr_block_scan_{next(_BLOCK_IDS)}")
    return types.FunctionType(
        code, run.__globals__, code.co_name, run.__defaults__, run.__closure__
    )


class BlockScan:
    """
    Compiled time-blocked scan.

    ``scan_fn(inputs_seq, state) -> (*ys, *final_state)`` is a pure scan over
    a time-major chunk (``CrModule._cr_scan``). Instead of unrolling the
    whole sequence into one graph, the sequence is cut into ``block``-step
    chunks (``_block_sizes``) and each chunk runs through ``scan_fn``
    compiled for that chunk length. Compile time and compile memory are
    bounded by ``block`` instead of growing with the sequence length, and the
    same graphs serve every sequence length.

    State is threaded between chunks, so the result equals one scan over the
    whole sequence. With autograd on, the first chunk usually receives a state
    that does not require grad while later chunks receive one that does, so
    Dynamo builds one extra graph for that first chunk length. Under CUDA-graph modes (``needs_clone``) chunk outputs
    and the carried state are cloned before the next replay can overwrite
    them.
    """

    def __init__(
        self,
        scan_fn: Callable[[tuple[Tensor, ...], tuple[Tensor, ...]], tuple[Tensor, ...]],
        block: int,
        n_outputs: int,
        needs_clone: bool,
        compile_kwargs: dict[str, Any],
    ) -> None:
        if not isinstance(block, int) or isinstance(block, bool) or block < 1:
            raise ValueError(f"block must be a positive int, got {block!r}")

        self.block = block
        # Optional back-reference for callers that cache scanners per module.
        self.owner: Any = None
        self.n_outputs = n_outputs
        self.needs_clone = needs_clone
        self._scan_fn = scan_fn
        # Every chunk length is a separate static shape; dynamic shapes would
        # only make Dynamo retrace the unrolled loop symbolically.
        self._compile_kwargs = {"dynamic": False, **compile_kwargs}
        self._compiled: dict[int, Callable[..., tuple[Tensor, ...]]] = {}
        # Wrap the main block length now (torch.compile itself is lazy), so
        # bad compile kwargs fail at configuration time, not on first use.
        self._for_size(block)

    def _for_size(self, size: int) -> Callable[..., tuple[Tensor, ...]]:
        fn = self._compiled.get(size)
        if fn is None:
            fn = torch.compile(_fresh_code(self._scan_fn), **self._compile_kwargs)
            self._compiled[size] = fn
        return fn

    def __call__(
        self,
        inputs_seq: tuple[Tensor, ...],
        state0: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        time = inputs_seq[0].shape[0]
        n_out = self.n_outputs
        # With autograd on, chunk outputs are concatenated (cat is
        # differentiable); without it they are copied into a preallocated
        # buffer so peak memory stays at one output sequence plus one chunk.
        collect = torch.is_grad_enabled()
        parts: list[list[Tensor]] = [[] for _ in range(n_out)]
        ys: list[Tensor] = []
        state = tuple(state0)
        lo = 0

        for size in _block_sizes(time, self.block):
            chunk = tuple(seq[lo : lo + size] for seq in inputs_seq)
            out = self._for_size(size)(chunk, state)

            for k in range(n_out):
                if collect:
                    y = out[k]
                    parts[k].append(y.clone() if self.needs_clone else y)
                else:
                    if not ys or len(ys) <= k:
                        ys.append(
                            torch.empty(
                                (time, *out[k].shape[1:]),
                                dtype=out[k].dtype,
                                device=out[k].device,
                            )
                        )
                    ys[k][lo : lo + size].copy_(out[k])

            carried = out[n_out:]
            state = (
                tuple(s.clone() for s in carried)
                if self.needs_clone
                else tuple(carried)
            )
            lo += size

        if collect:
            ys = [p[0] if len(p) == 1 else torch.cat(p, 0) for p in parts]

        return (*ys, *state)


class SequenceScanMixin(ForwardMixin):
    """Time-major sequence scans over a CrModule.

    Depends on ForwardMixin (explicit step, alloc, canonicalization, state
    factories); the annotations below are typing-only and set no runtime
    attribute. The stub surface also structurally satisfies
    ``_ValidationHost`` so the free validation helpers accept ``self``.
    """

    # Host attributes referenced by the scan methods (typing-only).

    _validate_override: Optional[bool]
    init_hidden: bool
    _cr_allocated: bool
    _cr_compiled_sequence: Optional[Callable[..., Any]]
    _cr_output_names: ClassVar[tuple[str, ...]]
    _cr_state_names: ClassVar[tuple[str, ...]]
    _cr_state_specs: ClassVar[tuple[StateSpec, ...]]
    _cr_input_names: ClassVar[tuple[str, ...]]
    _cr_input_specs: ClassVar[tuple[InputSpec, ...]]
    _cr_spec_entries: ClassVar[tuple[tuple[str, Spec], ...]]
    _cr_output_specs: ClassVar[tuple[OutputSpec, ...]]
    _buffers: dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: set[str]

    # Sequence scan implementation.

    def _cr_canonicalize_input_sequence(
        self,
        x_seq: InputTensor,
    ) -> tuple[Tensor, ...]:
        """
        Canonicalize a time-major sequence input (or a tuple/dict of
        sequences) into an ordered tuple of sequences. Every sequence must be
        at least ``(time, ...)`` and all sequences must share the time length.
        """
        inputs_seq = self._cr_canonicalize_inputs(x_seq)

        for seq in inputs_seq:
            if seq.dim() < 3:
                raise ValueError(
                    f"{type(self).__name__} expects (time, batch, features) "
                    f"sequence inputs, got {seq.dim()} dims"
                )

        time = inputs_seq[0].shape[0]

        if any(seq.shape[0] != time for seq in inputs_seq):
            raise ValueError(
                f"{type(self).__name__} requires all input sequences to share "
                f"the same time length"
            )

        return inputs_seq

    @staticmethod
    def _cr_first_inputs(inputs_seq: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Per-timestep inputs for the first timestep of a sequence scan.
        """
        return tuple(seq[0] for seq in inputs_seq)

    def forward_sequence(
        self,
        x_seq: InputTensor,
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        """
        Time-major sequence scan.

        Input shape:
            (time, batch, features)

        Multi-input modules pass a tuple/dict of sequences, one per declared
        input (all sharing the time length):

            (x_seq, inh_seq)

        Hidden mode:
            single output -> output sequence:
                (time, batch, features)
            multiple outputs -> tuple of output sequences:
                ((time, batch, features), ...)

        Explicit mode:
            returns:
                (output_sequence(s), *final_state)

        After ``fast_sequence_()`` this routes through the compiled scan and
        clones returned tensors (safe across ``torch.compile`` / CUDA graphs).

        Note: the eager path uses ``torch.compiler.is_compiling()`` to branch
        between the chunked and the flat scan. Under ``torch.compile`` each
        wrapper sees its own trace; recompiling or nesting multiple compiled
        wrappers is fine, but a manually written wrapper that calls
        ``forward_sequence`` outside a proper compile context may take the
        eager branch even while surrounding code is compiled. Prefer
        ``fast_sequence_()`` over wrapping this method yourself.
        """
        compiled = self._cr_compiled_sequence

        if compiled is not None:
            return compiled(x_seq, state)

        return self._cr_reference_sequence_scan(
            self._cr_canonicalize_input_sequence(x_seq),
            state,
        )

    def _cr_reference_sequence_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
        state: Optional[tuple[Tensor, ...]] = None,
    ) -> Tensor | StepOutput:
        """
        The reference per-step scan. This is the compile unit for
        ``compile_sequence_scan``; keep it free of state-allocation side
        effects that break tracing. ``inputs_seq`` must already be canonical.
        """
        if inputs_seq[0].shape[0] == 0:
            raise ValueError(
                f"{type(self).__name__}.forward_sequence expects at least one timestep"
            )

        if self.init_hidden:
            if state is not None:
                raise ValueError(
                    f"{type(self).__name__} is in init_hidden=True mode; "
                    f"forward_sequence does not accept explicit state"
                )

            return self._cr_hidden_sequence_scan(inputs_seq)

        return self._cr_explicit_sequence_scan(inputs_seq, state)

    def _cr_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
        state0: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """
        The pure scan unit: ``(inputs_seq, state0) -> (*ys, *final_state)``.

        Every sequence path (eager hidden/explicit, compiled, block-compiled)
        runs through this method, so a container can change how a sequence
        is evaluated (e.g. ``Sequential`` evaluating layer by layer) without
        touching the state and buffer bookkeeping around it. It must stay
        free of side effects so it can be compiled.
        """
        # Input arity and dtypes are fixed for the whole sequence: check them
        # once on the first timestep, then run the core step (``_step``, the
        # output contract check and post-step hooks) per timestep.
        self._cr_check_explicit_inputs(tuple(seq[0] for seq in inputs_seq), state0)
        return sequence_scan(
            self._cr_forward_core,
            inputs_seq,
            state0,
            len(self._cr_output_names),
        )

    def _cr_hidden_sequence_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
    ) -> Tensor | StepOutput:
        first_inputs = self._cr_first_inputs(inputs_seq)

        if not self._cr_allocated:
            self._cr_alloc_hidden(first_inputs)
        elif is_validating(self):
            check_hidden_input_shape(self, first_inputs)

        n_outputs = len(self._cr_output_names)
        state0 = tuple(getattr(self, name) for name in self._cr_state_names)

        # Hidden mode is an explicit scan plus buffer bookkeeping at the
        # edges: the scan itself is pure, so it can share _cr_scan with
        # explicit mode (and with multi-module containers).
        result = self._cr_scan(inputs_seq, state0)

        ys = result[:n_outputs]
        _store_hidden_seq_buffers(self, ys, result[n_outputs:])

        if n_outputs == 1:
            return ys[0]

        return ys

    def _cr_explicit_sequence_scan(
        self,
        inputs_seq: tuple[Tensor, ...],
        state: Optional[tuple[Tensor, ...]],
    ) -> StepOutput:
        if state is None:
            state = self.initial_state_for_sequence(inputs_seq)

        n_outputs = len(self._cr_output_names)

        result = self._cr_scan(inputs_seq, state)

        if n_outputs == 1:
            return (result[0], *result[1:])

        return result

    def compile_sequence_scan(
        self,
        block: Optional[int] = None,
        **kwargs: Any,
    ) -> Self:
        """
        Compile the reference sequence scan and route forward_sequence through it.

        Output tensors are cloned before returning only when CUDA graphs are in
        play (``mode="reduce-overhead"`` or ``mode="max-autotune"``): a subsequent
        graph run would otherwise overwrite the previously returned tensor. With
        any other mode the outputs are returned as-is - cloning would cost a full
        extra copy of the spike tensor on every call.

        Explicit mode and ``state=None``: the initial state is allocated inside
        the compiled call on every invocation. That is functionally correct in
        all modes (the returned state is freshly allocated, and graph modes
        clone it), but it is a per-call allocation the compiler must plan for.
        If you call the compiled scan repeatedly on chunks of a long sequence,
        allocate the state once with ``initial_state`` and pass it explicitly to
        skip that allocation and keep the output state un-cloned in default mode.

        Note on sequence length: the compiled unit is the fully unrolled T-step
        scan, so compilation cost and peak memory grow with T. It is fast up to
        roughly T~1000 and impractical (``RecursionError`` in inductor, or
        minutes-long compiles) beyond T~3000. This is not specific to crematorium:
        norse's compiled sequence and any fully unrolled scan hit the same wall.
        For very long sequences pass ``block``: the scan is then compiled in
        ``block``-step chunks (see ``BlockScan``) with the state threaded
        between them, so compile cost is bounded by ``block`` and any
        sequence length runs on the same few graphs. Results match the
        unblocked scan.
        """
        needs_clone = kwargs.get("mode") in ("reduce-overhead", "max-autotune")

        # Hidden mode: compile a pure scan (no buffer side-effects) so the
        # inductor can reuse memory like the explicit path. Buffer writes
        # happen outside the graph; otherwise the ``self._buffers`` stores
        # force an extra live ``(T,B,F)`` output buffer (380 MiB vs 252 MiB)
        # and split the graph.
        if self.init_hidden:
            if needs_clone:
                raise ValueError(
                    f"{type(self).__name__}.compile_sequence_scan(mode="
                    f"{kwargs.get('mode')!r}) is not supported with "
                    f"init_hidden=True: the compiled graph owns its output "
                    f"tensors, and storing them into the module's hidden "
                    f"buffers would alias compiler-recycled CUDA-graph "
                    f"memory. Use mode='default' (and call allocate_like(x) "
                    f"before the first call), or switch to explicit mode "
                    f"(init_hidden=False) if you need CUDA graphs."
                )

            n_outputs = len(self._cr_output_names)

            def _pure_hidden(
                inputs_seq: tuple[Tensor, ...],
                state0: tuple[Tensor, ...],
            ) -> tuple[Tensor, ...]:
                return self._cr_scan(inputs_seq, state0)

            compiled_pure: Callable[..., tuple[Tensor, ...]] = (
                BlockScan(self._cr_scan, block, n_outputs, False, kwargs)
                if block is not None
                else torch.compile(_pure_hidden, **kwargs)
            )

            def _cr_compiled_hidden(
                x_seq: InputTensor,
                state: Optional[tuple[Tensor, ...]] = None,
            ) -> Tensor | StepOutput:
                if state is not None:
                    raise ValueError(
                        f"{type(self).__name__} is in init_hidden=True mode; "
                        f"forward_sequence does not accept explicit state"
                    )
                inputs_seq = self._cr_canonicalize_input_sequence(x_seq)
                if inputs_seq[0].shape[0] == 0:
                    raise ValueError(
                        f"{type(self).__name__}.forward_sequence expects at least one timestep"
                    )
                if not self._cr_allocated:
                    self._cr_alloc_hidden(self._cr_first_inputs(inputs_seq))
                elif is_validating(self):
                    check_hidden_input_shape(self, self._cr_first_inputs(inputs_seq))
                state0 = tuple(getattr(self, name) for name in self._cr_state_names)
                result = compiled_pure(inputs_seq, state0)
                ys = result[:n_outputs]
                _store_hidden_seq_buffers(self, ys, result[n_outputs:])
                if n_outputs == 1:
                    out = ys[0]
                    return out.clone() if needs_clone else out
                if needs_clone:
                    return tuple(t.clone() for t in ys)
                return ys

            self._cr_compiled_sequence = _cr_compiled_hidden
            return self

        if block is not None:
            n_outputs = len(self._cr_output_names)
            scanner = BlockScan(self._cr_scan, block, n_outputs, needs_clone, kwargs)

            def _cr_compiled_blocks(
                x_seq: InputTensor,
                state: Optional[tuple[Tensor, ...]] = None,
            ) -> Tensor | StepOutput:
                inputs_seq = self._cr_canonicalize_input_sequence(x_seq)
                if inputs_seq[0].shape[0] == 0:
                    raise ValueError(
                        f"{type(self).__name__}.forward_sequence expects at least one timestep"
                    )
                if state is None:
                    state = self.initial_state_for_sequence(inputs_seq)
                result = scanner(inputs_seq, state)
                if n_outputs == 1:
                    return (result[0], *result[1:])
                return result

            self._cr_compiled_sequence = _cr_compiled_blocks
            return self

        compiled = torch.compile(self._cr_reference_sequence_scan, **kwargs)

        def _cr_compiled(
            x_seq: InputTensor,
            state: Optional[tuple[Tensor, ...]] = None,
        ) -> Tensor | StepOutput:
            inputs_seq = self._cr_canonicalize_input_sequence(x_seq)

            out = compiled(inputs_seq, state)

            if isinstance(out, Tensor):
                return out.clone() if needs_clone else out

            return tuple(t.clone() for t in out) if needs_clone else out

        self._cr_compiled_sequence = _cr_compiled

        return self

    def fast_sequence_(
        self,
        compile_scan: bool = True,
        block: Optional[int] = DEFAULT_SCAN_BLOCK,
        **compile_kwargs: Any,
    ) -> Self:
        """
        Enable a fast research path: validation off + optional compiled scan.

        The scan is compiled in ``block``-step chunks (``DEFAULT_SCAN_BLOCK``
        = 16 by default; see ``compile_sequence_scan``): compile time stays a
        few graphs regardless of sequence length, and small blocks measured
        as fast as or faster than longer ones at run time. ``block=None``
        compiles the fully unrolled sequence instead.

        ``mode`` defaults to ``"default"``. CUDA-graph modes
        (``"reduce-overhead"``, ``"max-autotune"``) are refused in hidden
        mode (``init_hidden=True``): graph-owned outputs would alias the
        compiler's recycled memory when stored into the module's hidden
        buffers. Pass them only on explicit-mode (``init_hidden=False``)
        modules, where outputs are cloned under graph modes.
        """
        if self.init_hidden and compile_kwargs.get("mode") in (
            "reduce-overhead",
            "max-autotune",
        ):
            raise ValueError(
                f"{type(self).__name__}.fast_sequence_(mode="
                f"{compile_kwargs.get('mode')!r}) is not supported with "
                f"init_hidden=True: CUDA-graph outputs would alias the "
                f"module's hidden buffers. Use mode='default' (the default), "
                f"or switch to explicit mode (init_hidden=False)."
            )

        set_validating(self, False)

        if compile_scan:
            compile_kwargs.setdefault("mode", "default")
            self.compile_sequence_scan(block=block, **compile_kwargs)

        return self

    def initial_state_for_sequence(
        self,
        x_seq: InputTensor,
    ) -> tuple[Tensor, ...]:
        inputs_seq = self._cr_canonicalize_input_sequence(x_seq)

        return self.initial_state_like(self._cr_first_inputs(inputs_seq))
