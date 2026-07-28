"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.mul.10_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 1, 4096), (4096, 4096, 1))`` — logical
  ``x``, dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (1, 1, 1), (1, 1, 1))`` — logical ``y``, a
  true fully-scalar tensor broadcast over every element of ``x``.
- ``buf0 = spyre_empty_with_layout((1, 1, 4096), (4096, 4096, 1),
  torch.float16, SpyreTensorLayout(device_size=[1, 64, 1, 64], ...))`` —
  logical output, SAME device layout as ``in_ptr0`` (no permute/axis-move,
  unlike ``torch.mul.2/4/6_spyre``): ``desc_0`` and ``desc_2`` both hardcode
  shape ``[1, 64, 1, 64]`` and the store even reuses ``desc_0``'s own
  ``dim0..dim3`` offsets.
- ``triton_unk_fused_mul_0.run(arg0_1, arg1_1, buf0, 4096, stream=stream0)``
  — the kernel is launched with ``xnumel=4096``.
- ``triton_meta={..., 'spyre_grid': (32,)}`` — 32-program grid, each program
  covering a 128-wide (2-stick) slab of the 64-stick-wide dim.

Because ``in_ptr0``/``out_ptr0`` share the identical device shape, no
reshape/permute of ``x`` is needed at all — only ``in_ptr1`` (the true
scalar) is reshaped and broadcast to match the tile shape before
multiplying. The oracle below relies on plain NumPy broadcasting rather
than an explicit ``broadcast_to`` call, since ``in_ptr1`` is built with all
64 lanes identical (a true scalar), matching the kernel's own semantics.
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
    ``xnumel = 4096`` and never reads ``xindex``/``xmask`` to gate the
    descriptor load/store, so the buffers are built directly at each
    descriptor's hardcoded shape (see ``kernel.py``).

    ``in_ptr1``'s logical shape is ``[1, 1, 1]`` — a true fully-scalar
    tensor — so every one of its 64 physical lane values must be
    identical, built via ``np.full`` rather than independent random draws.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((1, 64, 1, 64)).astype(np.float16)
    scalar_val = float(rng.standard_normal())
    in_ptr1 = np.full((1, 1, 1, 64), scalar_val, dtype=np.float16)
    out_ptr0 = np.zeros((1, 64, 1, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.mul.Tensor(x, y)`` with ``y`` a true scalar
    broadcast over every element of ``x``. ``in_ptr0`` and ``out_ptr0``
    share the same device-layout shape (no reshape/permute in the kernel
    body), so this is a plain elementwise multiply — NumPy's own
    broadcasting rules handle ``in_ptr1``'s reshape/broadcast_to
    automatically since every one of its 64 lanes already holds the same
    value.
    """
    x = inputs["in_ptr0"].astype(np.float32)
    y = inputs["in_ptr1"].astype(np.float32)
    return (x * y).astype(np.float16)


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*fp16",
    "in_ptr1":  "*fp16",
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
            "Broadcasting `out = x * y` on Meta-Llama-3.1-8B-Instruct's "
            "traced `torch.mul.10` op: a true fully-scalar `y` broadcast "
            "over every element of `x`, with in_ptr0/out_ptr0 sharing the "
            "same device layout."
        ),
        "doc": (
            "Multiplies every element of a logical `[1, 1, 4096]` f16 "
            "tensor `x` by a logical `[1, 1, 1]` f16 scalar `y`. On the "
            "Spyre device layout `x`'s 4096-wide innermost dim splits "
            "into 64 64-element f16 sticks, giving physical shape `[1, "
            "64, 1, 64]`; `out_ptr0` uses the identical device layout "
            "(no axis reordering, unlike `torch.mul.2/4/6_spyre`), and the "
            "kernel even reuses `in_ptr0`'s own tile offsets "
            "(`dim0..dim3`) for the store. `y`'s device layout is `[1, 1, "
            "1, 64]`, replicated across all 64 lanes of its single stick "
            "(a true scalar). The grid is sized to exactly 32 programs, "
            "each covering a 128-wide (2-stick) slab. `xnumel`/`XBLOCK`/"
            "`xindex` are boilerplate inherited from Inductor's pointwise "
            "codegen: `xnumel` is immediately overwritten with the "
            "literal `4096`, and `xmask` is hardcoded to all-`True` "
            "(`tl.full([XBLOCK], True, tl.int1)`) rather than derived "
            "from `xindex`, confirmed dead by the traced `.ttir` (no "
            "`tt.load`/`tt.store` depends on either)."
        ),
        "kernel_fn":  kernel.triton_unk_fused_mul_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [4096], "XBLOCK": [128]},
        "grid":       [32],
        # grid (32) exactly partitions the 64-wide stick axis into 32
        # 2-stick tiles — no residual scf.for (see
        # torch.mul.1_spyre/meta.py for the full rationale; empirically
        # verified via test_work_distribution).
        "distribution_loop": False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
