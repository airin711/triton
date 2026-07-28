"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.pow.1_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 12, 4096), (49152, 4096, 1))`` — logical
  input, dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 12, 4096), (49152, 4096, 1),
  torch.float16, SpyreTensorLayout(device_size=[12, 64, 1, 64], ...))`` —
  logical output, same shape/dtype as the input.
- ``triton_unk_fused_pow_0.run(arg0_1, buf0, 49152, stream=stream0)`` — the
  kernel is launched with ``xnumel=49152``.
- ``triton_meta={..., 'spyre_grid': (32,)}`` — 32-program grid (grid !=
  xnumel): each program covers a `[12, 2, 1, 64]` block (1536 elements,
  matching `XBLOCK=1536`) sliced out of the flattened `4096`-wide row via
  `tl.program_id(0) * 128`.

The traced graph fragment records `aten.pow.Tensor_Scalar(arg0_1, 2)`, but
Inductor's own codegen lowers the integer exponent 2 to a repeated
multiplication rather than a literal `**`/`libdevice.pow` call — the kernel
body computes `tmp1 = tmp0 * tmp0` (see ``kernel.py``). The oracle below
mirrors that lowering exactly rather than using `np.power`, so it matches
what the kernel actually computes bit-for-bit (aside from the f32
accumulation reproduced by both).

The kernel itself hardcodes shape ``[12, 64, 1, 64]`` into both tensor
descriptors, so the pointer args here are built directly at that shape.

NOTE on ``XBLOCK``: the source's own ``XBLOCK`` is ``1536`` (not a power of
2). ``XBLOCK`` only feeds the dead ``xoffset``/``xindex``/``tl.arange(0,
XBLOCK)``/``xmask`` boilerplate (never read by the descriptor load/store —
see ``kernel.py``), but Triton's frontend still rejects
``tl.arange(0, 1536)`` at compile time with "arange's range must be a power
of 2" regardless of whether the result is used. Since this constexpr has no
effect on the computed result, ``params`` below substitutes the next
power of 2 (``2048``) purely to satisfy that frontend check — a reviewer
should double-check this reasoning (that ``XBLOCK`` is truly dead here)
rather than take it on faith.
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
    ``xnumel = 49152`` and never reads ``xindex``/``xmask`` (``xmask`` is a
    trivial all-``True`` mask), so the buffers are built directly at the
    descriptors' hardcoded shape ``[12, 64, 1, 64]`` (see ``kernel.py``).
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((12, 64, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((12, 64, 1, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.pow.Tensor_Scalar(x, 2)``, computed the same way
    the kernel computes it — as `x * x` in the kernel's own compute
    precision (extend to f32, square, truncate back to f16) — rather than
    via `np.power`/`x ** 2`."""
    x = inputs["in_ptr0"].astype(np.float32)
    return (x * x).astype(np.float16)


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*fp16",
    "out_ptr0": "*fp16",
    "xnumel":   "i32",
    "XBLOCK":   "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d"],
        "summary": (
            "Elementwise `out = in * in` (lowering of `pow(in, 2)`) on "
            "Meta-Llama-3.1-8B-Instruct's traced `torch.pow.1` op, one "
            "program per `[12, 2, 1, 64]` block."
        ),
        "doc": (
            "Squares every element of a logical `[1, 12, 4096]` f16 "
            "tensor, `aten.pow.Tensor_Scalar(x, 2)` lowered by Inductor to "
            "`tmp0 * tmp0` rather than a literal `**`. On the Spyre device "
            "layout the flattened `4096`-wide row splits into a "
            "`[64, 1, 64]` stick shape, giving a physical shape "
            "`[12, 64, 1, 64]`; the grid is sized to 32 programs, each "
            "covering a `[12, 2, 1, 64]` block (1536 elements, matching "
            "`XBLOCK=1536`) via `tl.program_id(0) * 128` — no "
            "`num_programs`/`cdiv` distribution loop in the source. "
            "`xindex`/`xmask` are boilerplate inherited from Inductor's "
            "pointwise codegen: `xnumel` is immediately overwritten with "
            "the literal `49152` and `xmask` is a trivial all-`True` mask "
            "(`tl.full([XBLOCK], True, tl.int1)`), never read for masking."
        ),
        "kernel_fn":  kernel.triton_unk_fused_pow_0,
        "constexpr":  ["XBLOCK"],
        # XBLOCK substituted from the source's 1536 to the next power of 2
        # (2048) — see the "NOTE on XBLOCK" in the module docstring above:
        # XBLOCK only feeds dead xindex/xmask boilerplate, but Triton's
        # `tl.arange` requires a power-of-2 range at compile time.
        "params":     {"xnumel": [49152], "XBLOCK": [2048]},
        # grid (32) exactly tiles desc_0's shape[1] (64) via block_shape[1]
        # == 2: 64/2 == 32, nothing left over, so DistributeWork emits no
        # residual scf.for -- same generalized rule as add.1/matmul.1.
        "distribution_loop": False,
        "grid":       [32],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
