"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.float.3_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 1, 12), (12, 12, 1))`` — logical input,
  dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 1, 12), (12, 12, 1), torch.float32,
  SpyreTensorLayout(device_size=[1, 2, 1, 32], ...))`` — logical output,
  same logical shape as the input but dtype ``torch.float32``.
- ``triton_unk_fused__to_copy_0.run(arg0_1, buf0, 12, stream=stream0)`` —
  the kernel is launched with ``xnumel=12``.
- ``triton_meta={..., 'spyre_grid': (1,)}`` — a single-program grid (grid
  (1) != xnumel (12)).

Unlike the other examples in this batch, the two tensor descriptors here
have *different dtypes and physical shapes*: ``in_ptr0`` is fp16 with an
innermost 64-element stick (``[1, 1, 1, 64]``), while ``out_ptr0`` is fp32
with an innermost 32-element stick (``[1, 2, 1, 32]``) — casting to a wider
dtype halves the stick width and doubles the stick count, so the physical
buffer is reshaped even though nothing about the *logical* `[1, 1, 12]`
shape changes.

Both descriptors' `block_shape` equal their full `shape` (single tile, no
per-program slicing), and working out each descriptor's address formula
from its `strides` (`in`: `dim0*64+dim1*64+dim2*64+dim3` with only `dim3`
non-constant, i.e. a plain flat 64-index; `out`:
`dim0*64+dim1*32+dim2*32+dim3` with only `dim1`/`dim3` non-constant, i.e. a
flat 64-index unrolled as `(2, 32)` row-major) shows both reduce to the
*same* flat 0..63 index space in the *same* order — so the physical
`[1, 2, 1, 32]` output buffer is exactly `reshape(64 -> (1, 2, 1, 32))` of
the physical `[1, 1, 1, 64]` input buffer, element-for-element, after the
fp16->fp32 cast. That identity is what the oracle below relies on (a plain
`.astype(np.float32).reshape(...)`, not a round-trip back to fp16) --
however, see the `disabled` note in `VARIANTS["default"]`: the kernel
itself never performs this reshape before its store, so it cannot actually
compile.

BUG (present verbatim in the traced source, see kernel.py): `desc_0.load`
returns a value of shape `[1, 1, 1, 64]` (desc_0's own block_shape); this
is cast in place (`.to(tl.float32)`, shape-preserving) and then passed
directly to `desc_1.store`, whose block_shape is `[1, 2, 1, 32]` -- same
element count (64), different shape, with no intervening `tl.reshape`.
This is the same category of bug seen across the `torch.cat.*_spyre`
examples (a store whose value shape doesn't match its descriptor's
block_shape), rejected by Triton's frontend `validate_store_like` check
before any TTIR is built. See `VARIANTS["default"]["disabled"]`.

NOTE on ``XBLOCK``: the source's own ``XBLOCK`` is ``12`` (not a power of
2). ``XBLOCK`` only feeds the dead ``xoffset``/``xindex``/``tl.arange(0,
XBLOCK)`` boilerplate — ``xmask = xindex < xnumel`` here is a real bounds
check (unlike the trivial all-``True`` mask in the `add.*`/`neg.*`/`pow.*`
examples in this batch), but it is still never read by the descriptor
load/store (``dim3 = c0 = 0``, a literal — see ``kernel.py``). Triton's
frontend nonetheless rejects ``tl.arange(0, 12)`` at compile time with
"arange's range must be a power of 2" regardless of whether the result is
used. Since this constexpr has no effect on the computed result, ``params``
below substitutes the next power of 2 (``16``) purely to satisfy that
frontend check — a reviewer should double-check this reasoning (that
``XBLOCK`` is truly dead here) rather than take it on faith.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(xnumel: int, XBLOCK: int, **_unused) -> dict:
    """Build pointer-tensor inputs for the kernel.

    ``xnumel``/``XBLOCK`` are accepted so the signature matches the full
    param set, but neither shapes the data: the kernel body reassigns
    ``xnumel = 12`` and never reads ``xindex`` for indexing (only for the
    otherwise-unused ``xmask`` bounds check), so the buffers are built
    directly at each descriptor's hardcoded shape — ``in_ptr0`` at
    ``[1, 1, 1, 64]`` (fp16) and ``out_ptr0`` at ``[1, 2, 1, 32]`` (fp32,
    zero-initialized) — per ``kernel.py``.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((1, 1, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((1, 2, 1, 32), dtype=np.float32)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: elementwise fp16->fp32 cast (`aten._to_copy`/
    `prims.convert_element_type.default`), followed by the same
    flat-index-preserving relayout from a `[1, 1, 1, 64]` (64-wide fp16
    stick) buffer to a `[1, 2, 1, 32]` (two 32-wide fp32 sticks) buffer that
    the kernel's own descriptor address arithmetic performs (see
    ``meta.py`` module docstring for the derivation) — a plain `.astype` +
    `.reshape`, with no round-trip back to fp16."""
    x = inputs["in_ptr0"].astype(np.float32)
    return x.reshape(1, 2, 1, 32)


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*fp16",
    "out_ptr0": "*fp32",
    "xnumel":   "i32",
    "XBLOCK":   "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "dtype-cast", "program-id-1d"],
        "summary": (
            "fp16 -> fp32 dtype cast (`.float()`) on Meta-Llama-3.1-8B-"
            "Instruct's traced `torch.float.3` op, single program, "
            "relayouts a 64-wide fp16 stick into two 32-wide fp32 sticks."
        ),
        "doc": (
            "Casts every element of a logical `[1, 1, 12]` f16 tensor to "
            "f32 (`prims.convert_element_type.default`). On the Spyre "
            "device layout the fp16 input's innermost dim pads to a "
            "64-element stick (`[1, 1, 1, 64]`), while the fp32 output's "
            "innermost dim uses a narrower 32-element stick, giving "
            "physical shape `[1, 2, 1, 32]` — a dtype cast that widens the "
            "element size also reshapes the physical buffer even though "
            "the logical shape is untouched. The grid is a single "
            "program (`spyre_grid=(1,)`, `xnumel=12` — grid != xnumel, so "
            "no `distribution_loop: False`); both descriptors' "
            "`block_shape` equal their full `shape` (one tile covers the "
            "whole buffer, no per-program slicing via `tl.program_id`). "
            "`xindex`/`xmask` follow Inductor's default pointwise codegen "
            "(`xmask = xindex < xnumel` is a real bounds check here, "
            "unlike the trivial all-`True` mask in the `pow.*` examples), "
            "but both are still dead for the descriptor load/store."
        ),
        "kernel_fn":  kernel.triton_unk_fused__to_copy_0,
        "constexpr":  ["XBLOCK"],
        # XBLOCK substituted from the source's 12 to the next power of 2
        # (16) — see the "NOTE on XBLOCK" in the module docstring above:
        # XBLOCK only feeds dead xindex/xmask boilerplate, but Triton's
        # `tl.arange` requires a power-of-2 range at compile time.
        "params":     {"xnumel": [12], "XBLOCK": [16]},
        "grid":       [1],
        # desc_1.store() passes desc_0.load()'s value (shape [1,1,1,64],
        # unchanged by the shape-preserving fp32 cast) directly into a
        # descriptor whose block_shape is [1,2,1,32] -- same element count,
        # different shape, with no tl.reshape. Present verbatim in the
        # traced output_code.py (see kernel.py/module docstring), rejected
        # by Triton's frontend validate_store_like check before any TTIR
        # is built. Same bug category as the torch.cat.*_spyre examples.
        "disabled": {
            "reason": (
                "desc_1.store() passes a [1,1,1,64] value (from "
                "desc_0.load(), shape-preserved through the fp32 cast) "
                "into a descriptor with block_shape [1,2,1,32] with no "
                "reshape -- a shape mismatch present verbatim in the "
                "traced source, rejected by Triton's frontend "
                "validate_store_like check before TTIR construction."
            ),
        },
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
