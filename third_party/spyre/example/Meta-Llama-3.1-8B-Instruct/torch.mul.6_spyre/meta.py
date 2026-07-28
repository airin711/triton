"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.mul.6_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 8, 12, 128), (12288, 1536, 128, 1))`` —
  logical ``x``, dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (1, 1, 12, 128), (1536, 1536, 128, 1))`` —
  logical ``y``, broadcast over ``x``'s head dim (dim 1, size 8). ``y``'s
  last logical dim is a real 128-wide vector, not a true scalar.
- ``buf0 = spyre_empty_with_layout((1, 8, 12, 128), (12288, 1536, 128, 1),
  torch.float16, SpyreTensorLayout(device_size=[12, 1, 2, 8, 64], ...))``
  — logical output, same logical shape as ``x`` but a *different* physical
  device layout (head axis moved from front to back).
- ``triton_unk_fused_mul_0.run(arg0_1, arg1_1, buf0, 12288, stream=stream0)``
  — the kernel is launched with ``xnumel=12288``.
- ``triton_meta={..., 'spyre_grid': (24,)}`` — 24-program grid: `c0 =
  (program_id // 12) * 4` (2 head-tiles of width 4) and `c1 = program_id %
  12` (12 rows), i.e. 2 * 12 = 24 programs, each covering one row and a
  4-wide head-tile.

Structurally the same broadcast pattern as ``torch.mul.4_spyre`` (head axis
of ``x`` broadcast against by ``y``, then moved front-to-back in the output
device layout), just with a smaller head count (8 instead of 32) and a
head-tile size >1 (4 instead of 1) — which is why this kernel body *does*
contain an explicit ``tl.permute`` (``torch.mul.4``'s tile-size-1 head axis
needed none). The full-scale NumPy oracle below is nonetheless identical in
structure to ``torch.mul.4_spyre``'s, since aggregating over all tiles of
either tile size reproduces the same global head-axis transpose.
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
    ``xnumel = 12288`` and never reads ``xindex``/``xmask`` to gate the
    descriptor load/store, so the buffers are built directly at each
    descriptor's hardcoded shape (see ``kernel.py``).

    ``in_ptr1``'s logical last dim is a real 128-wide vector (not a true
    scalar), so its 64-lane sticks hold genuinely distinct random values —
    no lane replication.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((8, 12, 2, 1, 64)).astype(np.float16)
    in_ptr1 = rng.standard_normal((1, 12, 2, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((12, 1, 2, 8, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.mul.Tensor(x, y)`` (``y`` broadcast over ``x``'s
    head axis), replicated at full scale in ``out_ptr0``'s own
    device-layout order. Same derivation as ``torch.mul.4_spyre``, just
    with a head count of 8 instead of 32:

    - ``in_ptr0``: ``[8, 12, 2, 1, 64]`` -> squeeze dim3 -> ``[8, 12, 2,
      64]`` -> transpose(1, 2, 0, 3) -> ``[12, 2, 8, 64]`` (head axis moved
      from front to back, matching the per-head-tile offset mapping into
      ``desc_2``'s dim3 — the kernel's explicit ``tl.permute`` operates on
      each 4-wide head-tile individually, but aggregated over both tiles
      this is the same global transpose).
    - ``in_ptr1``: ``[1, 12, 2, 1, 64]`` -> reshape (order-preserving) ->
      ``[12, 2, 64]`` -> broadcast over the new head axis.
    - multiply, then reshape ``[12, 2, 8, 64]`` -> ``[12, 1, 2, 8, 64]``
      (order-preserving insert of a size-1 axis) to match ``desc_2``'s
      shape.
    """
    x = inputs["in_ptr0"].astype(np.float32)
    y = inputs["in_ptr1"].astype(np.float32)

    x = x.reshape(8, 12, 2, 64)
    x = np.transpose(x, (1, 2, 0, 3))  # -> (12, 2, 8, 64)

    y = y.reshape(12, 2, 64)

    result = x * y[:, :, None, :]
    result = result.reshape(12, 1, 2, 8, 64)
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
            "traced `torch.mul.6` op: `y` broadcast over `x`'s 8-wide "
            "head axis, one program per (row, 4-wide head-tile) pair, "
            "with an explicit `tl.permute` and a head-axis front-to-back "
            "move between `x`'s and `out`'s device layouts."
        ),
        "doc": (
            "Multiplies a logical `[1, 8, 12, 128]` f16 tensor `x` by a "
            "logical `[1, 1, 12, 128]` f16 tensor `y` broadcast over `x`'s "
            "head dim. On the Spyre device layout `x`'s 128-wide innermost "
            "dim splits into 2 64-element f16 sticks, giving physical "
            "shape `[8, 12, 2, 1, 64]`; the output stores the same data "
            "with the head axis moved to the back, `[12, 1, 2, 8, 64]`. "
            "The grid is sized to exactly 24 programs (2 head-tiles of "
            "width 4, times 12 rows); since the head-tile size (4) is "
            ">1, the kernel needs an explicit `tl.permute([1, 2, 0, 3])` "
            "to reorder each tile's (head, row, stick) axes before "
            "storing (compare `torch.mul.4_spyre`'s tile-size-1 head "
            "axis, which needs no permute). `xnumel`/`XBLOCK`/`xindex` "
            "are boilerplate inherited from Inductor's pointwise codegen: "
            "`xnumel` is immediately overwritten with the literal "
            "`12288`, and `xmask` is hardcoded to all-`True` "
            "(`tl.full([XBLOCK], True, tl.int1)`) rather than derived "
            "from `xindex`, confirmed dead by the traced `.ttir` (no "
            "`tt.load`/`tt.store` depends on either)."
        ),
        "kernel_fn":  kernel.triton_unk_fused_mul_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [12288], "XBLOCK": [512]},
        "grid":       [24],
        # grid (24) exactly partitions (2 head-tiles x 12 rows) into one
        # program per (head-tile, row) pair — no residual scf.for (see
        # torch.mul.1_spyre/meta.py for the full rationale; empirically
        # verified via test_work_distribution).
        "distribution_loop": False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
