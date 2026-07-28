"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.mul.14_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 8, 1, 128), (1024, 128, 128, 1))`` —
  logical ``x``, dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (1, 1, 1, 128), (128, 128, 128, 1))`` —
  logical ``y``, broadcast over ``x``'s head dim (dim 1, size 8). ``y``'s
  last logical dim is a real 128-wide vector, not a true scalar.
- ``buf0 = spyre_empty_with_layout((1, 8, 1, 128), (1024, 128, 128, 1),
  torch.float16, SpyreTensorLayout(device_size=[1, 1, 2, 8, 64], ...))``
  — logical output, same logical shape as ``x`` but a *different* physical
  device layout (head axis moved from front to back).
- ``triton_unk_fused_mul_0.run(arg0_1, arg1_1, buf0, 1024, stream=stream0)``
  — the kernel is launched with ``xnumel=1024``.
- ``triton_meta={..., 'spyre_grid': (16,)}`` — 16-program grid: `c0 =
  program_id // 2` (8 heads) times `c1 = (program_id % 2) * 64` (2 sticks
  of the 128-wide vector), i.e. 8 * 2 = 16 programs, each covering one
  head and one stick.

Same broadcast pattern as ``torch.mul.12_spyre`` (head-tile size 1, no
in-kernel ``tl.permute``), just with a head count of 8 instead of 32 and
the stick axis also tiled 1-at-a-time (2 tiles instead of one 2-wide tile),
which does not change the full-scale oracle formula. See
``torch.mul.12_spyre/meta.py`` for the general reasoning.
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
    ``xnumel = 1024`` and the resulting ``xmask`` (``xindex < xnumel``,
    here always true for the tiny ``XBLOCK=64``) is never read to gate the
    descriptor load/store, so the buffers are built directly at each
    descriptor's hardcoded shape (see ``kernel.py``).

    ``in_ptr1``'s logical last dim is a real 128-wide vector (not a true
    scalar), so its 64-lane sticks hold genuinely distinct random values —
    no lane replication.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((8, 1, 2, 1, 64)).astype(np.float16)
    in_ptr1 = rng.standard_normal((1, 1, 2, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((1, 1, 2, 8, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.mul.Tensor(x, y)`` (``y`` broadcast over ``x``'s
    head axis), replicated at full scale in ``out_ptr0``'s own
    device-layout order. Same derivation as ``torch.mul.12_spyre``, just
    with a head count of 8 instead of 32:

    - ``in_ptr0``: ``[8, 1, 2, 1, 64]`` -> squeeze dims 1, 3 -> ``[8, 2,
      64]`` -> transpose(1, 0, 2) -> ``[2, 8, 64]`` (head axis moved from
      front to middle, matching the per-head-tile offset mapping into
      ``desc_2``'s dim3).
    - ``in_ptr1``: ``[1, 1, 2, 1, 64]`` -> reshape (order-preserving) ->
      ``[2, 64]`` -> broadcast over the new head axis.
    - multiply, then reshape ``[2, 8, 64]`` -> ``[1, 1, 2, 8, 64]``
      (order-preserving insert of two size-1 axes) to match ``desc_2``'s
      shape.
    """
    x = inputs["in_ptr0"].astype(np.float32)
    y = inputs["in_ptr1"].astype(np.float32)

    x = x.reshape(8, 2, 64)
    x = np.transpose(x, (1, 0, 2))  # -> (2, 8, 64)

    y = y.reshape(2, 64)

    result = x * y[:, None, :]
    result = result.reshape(1, 1, 2, 8, 64)
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
            "traced `torch.mul.14` op: `y` broadcast over `x`'s 8-wide "
            "head axis, one program per (head, stick) pair, with a "
            "head-axis front-to-back move between `x`'s and `out`'s "
            "device layouts."
        ),
        "doc": (
            "Multiplies a logical `[1, 8, 1, 128]` f16 tensor `x` by a "
            "logical `[1, 1, 1, 128]` f16 tensor `y` broadcast over `x`'s "
            "head dim. On the Spyre device layout `x`'s 128-wide innermost "
            "dim splits into 2 64-element f16 sticks, giving physical "
            "shape `[8, 1, 2, 1, 64]`; the output stores the same data "
            "with the head axis moved to the back, `[1, 1, 2, 8, 64]`. "
            "The grid is sized to exactly 16 programs (8 heads times 2 "
            "sticks); no `tl.permute` is needed since both the head-tile "
            "and stick-tile sizes are 1 (same pattern as "
            "`torch.mul.12_spyre`, just with head count 8 and the stick "
            "axis tiled 1-at-a-time instead of loaded whole). "
            "`xnumel`/`XBLOCK`/`xindex`/`xmask` are boilerplate inherited "
            "from Inductor's pointwise codegen: `xnumel` is immediately "
            "overwritten with the literal `1024`, and this kernel is the "
            "one member of the mul.N family where `xmask` (`xindex < "
            "xnumel`) is a live expression rather than hardcoded "
            "all-`True` — still confirmed dead by the traced `.ttir`, "
            "since no `tt.load`/`tt.store` depends on it (only `dim0..4`, "
            "derived from `tl.program_id(0)`, gate the descriptor "
            "load/store)."
        ),
        "kernel_fn":  kernel.triton_unk_fused_mul_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [1024], "XBLOCK": [64]},
        "grid":       [16],
        # grid (16) exactly partitions (8 heads x 2 sticks) into one
        # program per (head, stick) pair — no residual scf.for (see
        # torch.mul.1_spyre/meta.py for the full rationale; empirically
        # verified via test_work_distribution).
        "distribution_loop": False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
