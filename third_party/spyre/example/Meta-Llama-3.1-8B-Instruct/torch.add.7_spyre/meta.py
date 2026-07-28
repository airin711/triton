"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.add.7_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 8, 1, 128), (1024, 128, 128, 1))`` / same
  for ``arg1_1`` — two logical inputs, both dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 8, 1, 128), (1024, 128, 128, 1),
  torch.float16, SpyreTensorLayout(device_size=[8, 1, 2, 1, 64], ...))`` —
  logical output, same shape/dtype as the inputs.
- ``triton_unk_fused_add_0.run(arg0_1, arg1_1, buf0, 1024, stream=stream0)``
  — the kernel is launched with ``xnumel=1024``.
- ``triton_meta={..., 'spyre_grid': (16,)}`` — 16-program grid (grid !=
  xnumel): `dim0`/`dim2` come from `tl.program_id(0) // 2` and `% 2`
  respectively, each program covering a `[1, 1, 1, 1, 64]` block (64
  elements). Note `XBLOCK=64` here does NOT equal the per-program element
  count implied by the block shape's `xmask = xindex < xnumel` guard — this
  kernel (unlike the other `add` variants above) keeps a real bounds check,
  even though `xindex` is still never read by the descriptor indexing.

The kernel itself hardcodes shape ``[8, 1, 2, 1, 64]`` into all three tensor
descriptors, so the pointer args here are built directly at that shape.
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
    ``xnumel = 1024`` and never reads ``xindex``/``xmask`` for the actual
    descriptor indexing (`dim0`..`dim4` are derived straight from
    `tl.program_id(0)`), so the buffers are built directly at the
    descriptors' hardcoded shape ``[8, 1, 2, 1, 64]`` (see ``kernel.py``).
    """
    del xnumel, XBLOCK
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)
    in_ptr0 = rng0.standard_normal((8, 1, 2, 1, 64)).astype(np.float16)
    in_ptr1 = rng1.standard_normal((8, 1, 2, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((8, 1, 2, 1, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.add.Tensor(x, y)`` in the kernel's own compute
    precision (extend both operands to f32, add, truncate back to f16)."""
    x = inputs["in_ptr0"].astype(np.float32)
    y = inputs["in_ptr1"].astype(np.float32)
    return (x + y).astype(np.float16)


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
            "Elementwise `out = in0 + in1` on Meta-Llama-3.1-8B-Instruct's "
            "traced `torch.add.7` op, one program per `[1, 1, 1, 1, 64]` "
            "block."
        ),
        "doc": (
            "Adds two logical `[1, 8, 1, 128]` f16 tensors elementwise. On "
            "the Spyre device layout the innermost dim splits into a "
            "`[2, 1, 64]` stick shape, giving a physical shape "
            "`[8, 1, 2, 1, 64]`; the grid is sized to 16 programs, with "
            "`dim0`/`dim2` derived from `tl.program_id(0) // 2` and `% 2` — "
            "no `num_programs`/`cdiv` distribution loop in the source. "
            "`xindex`/`xmask` are boilerplate inherited from Inductor's "
            "pointwise codegen: `xnumel` is immediately overwritten with the "
            "literal `1024`, and while this variant keeps a real "
            "`xmask = xindex < xnumel` (unlike its `tl.full(..., True)` "
            "siblings), the descriptor indexing never reads `xindex`/`xmask` "
            "either way — confirmed dead by the traced `.ttir`."
        ),
        "kernel_fn":  kernel.triton_unk_fused_add_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [1024], "XBLOCK": [64]},
        "grid":       [16],
        # grid (16) exactly tiles desc_0's shape[0] (8) and shape[2] (2) via
        # block_shape[0]==1, block_shape[2]==1: (8/1) * (2/1) == 16, nothing
        # left over, so DistributeWork emits no residual scf.for -- same
        # generalized rule as add.1/matmul.1.
        "distribution_loop": False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
