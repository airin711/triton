"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.mul.12_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 32, 1, 128), (4096, 128, 128, 1))`` —
  logical ``x``, dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (1, 1, 1, 128), (128, 128, 128, 1))`` —
  logical ``y``, broadcast over ``x``'s head dim (dim 1, size 32). ``y``'s
  last logical dim is a real 128-wide vector, not a true scalar.
- ``buf0 = spyre_empty_with_layout((1, 32, 1, 128), (4096, 128, 128, 1),
  torch.float16, SpyreTensorLayout(device_size=[1, 1, 2, 32, 64], ...))``
  — logical output, same logical shape as ``x`` but a *different* physical
  device layout (head axis moved from front to back).
- ``triton_unk_fused_mul_0.run(arg0_1, arg1_1, buf0, 4096, stream=stream0)``
  — the kernel is launched with ``xnumel=4096``.
- ``triton_meta={..., 'spyre_grid': (32,)}`` — 32-program grid, one program
  per head (``tl.program_id(0)`` directly indexes the 32-wide head axis).

Same broadcast pattern as ``torch.mul.4_spyre`` (head-tile size 1, so no
in-kernel ``tl.permute`` needed, just reshapes), but with the "row" axis
collapsed to 1 (this op has no sequence-length dim, only heads), so the
descriptors here are 5D with a size-1 axis where ``torch.mul.4_spyre`` had
its 12-row axis. See that example's docstring for the general reasoning.
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

    ``in_ptr1``'s logical last dim is a real 128-wide vector (not a true
    scalar), so its 64-lane sticks hold genuinely distinct random values —
    no lane replication.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((32, 1, 2, 1, 64)).astype(np.float16)
    in_ptr1 = rng.standard_normal((1, 1, 2, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((1, 1, 2, 32, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.mul.Tensor(x, y)`` (``y`` broadcast over ``x``'s
    head axis), replicated at full scale in ``out_ptr0``'s own
    device-layout order:

    - ``in_ptr0``: ``[32, 1, 2, 1, 64]`` -> squeeze dims 1, 3 -> ``[32, 2,
      64]`` -> transpose(1, 0, 2) -> ``[2, 32, 64]`` (head axis moved from
      front to middle, matching the per-head-tile offset mapping into
      ``desc_2``'s dim3).
    - ``in_ptr1``: ``[1, 1, 2, 1, 64]`` -> reshape (order-preserving) ->
      ``[2, 64]`` -> broadcast over the new head axis.
    - multiply, then reshape ``[2, 32, 64]`` -> ``[1, 1, 2, 32, 64]``
      (order-preserving insert of two size-1 axes) to match ``desc_2``'s
      shape.
    """
    x = inputs["in_ptr0"].astype(np.float32)
    y = inputs["in_ptr1"].astype(np.float32)

    x = x.reshape(32, 2, 64)
    x = np.transpose(x, (1, 0, 2))  # -> (2, 32, 64)

    y = y.reshape(2, 64)

    result = x * y[:, None, :]
    result = result.reshape(1, 1, 2, 32, 64)
    return result.astype(np.float16)


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
            "traced `torch.mul.12` op: `y` broadcast over `x`'s 32-wide "
            "head axis, one program per head, with a head-axis "
            "front-to-back move between `x`'s and `out`'s device layouts."
        ),
        "doc": (
            "Multiplies a logical `[1, 32, 1, 128]` f16 tensor `x` by a "
            "logical `[1, 1, 1, 128]` f16 tensor `y` broadcast over `x`'s "
            "head dim. On the Spyre device layout `x`'s 128-wide innermost "
            "dim splits into 2 64-element f16 sticks, giving physical "
            "shape `[32, 1, 2, 1, 64]`; the output stores the same data "
            "with the head axis moved to the back, `[1, 1, 2, 32, 64]`. "
            "The grid is sized to exactly 32 programs, one per head "
            "(`tl.program_id(0)` directly indexes the head axis); each "
            "program loads its single head slice of `x` plus the entire "
            "(un-tiled) `y`, multiplies, and stores at the head's new "
            "offset — no `tl.permute` needed since the head-tile size is "
            "1 (same pattern as `torch.mul.4_spyre`, just without a "
            "sequence-length/row axis). `xnumel`/`XBLOCK`/`xindex` are "
            "boilerplate inherited from Inductor's pointwise codegen: "
            "`xnumel` is immediately overwritten with the literal `4096`, "
            "and `xmask` is hardcoded to all-`True` (`tl.full([XBLOCK], "
            "True, tl.int1)`) rather than derived from `xindex`, "
            "confirmed dead by the traced `.ttir` (no `tt.load`/"
            "`tt.store` depends on either)."
        ),
        "kernel_fn":  kernel.triton_unk_fused_mul_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [4096], "XBLOCK": [128]},
        "grid":       [32],
        # grid (32) exactly partitions the 32-wide head axis into one
        # head per program — no residual scf.for (see
        # torch.mul.1_spyre/meta.py for the full rationale; empirically
        # verified via test_work_distribution).
        "distribution_loop": False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
