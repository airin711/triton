"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.rsqrt.1_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 12, 1), (12, 1, 1))`` — logical input,
  dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 12, 1), (12, 1, 1), torch.float16,
  SpyreTensorLayout(device_size=[12, 1, 1, 64], ...))`` — logical output,
  same shape/dtype as the input.
- ``triton_unk_fused_rsqrt_0.run(arg0_1, buf0, 12, stream=stream0)`` — the
  kernel is launched with ``xnumel=12``.
- ``triton_meta={..., 'spyre_grid': (12,)}`` — 12-program grid, one core
  per row of the device-layout shape ``[12, 1, 1, 64]``, exactly like
  ``torch.add.1_spyre``.

The kernel itself hardcodes shape ``[12, 1, 1, 64]`` into both tensor
descriptors, so the pointer args here are built directly at that shape.
Since ``rsqrt`` is undefined for non-positive inputs, the input generator
draws strictly positive values (unlike the signed-normal draws used by the
elementwise-add examples).
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
    ``xnumel = 12`` and never reads ``xindex``/``xmask``, so the buffers are
    built directly at the descriptor's hardcoded shape ``[12, 1, 1, 64]``
    (see ``kernel.py``). Values are drawn strictly positive (uniform in
    ``[0.1, 10.1)``) since ``rsqrt`` of a non-positive input is undefined.
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = (rng.random((12, 1, 1, 64)) * 10.0 + 0.1).astype(np.float16)
    out_ptr0 = np.zeros((12, 1, 1, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.rsqrt.default(x)`` in the kernel's own compute
    precision (the kernel itself casts to f32 for `tl.rsqrt`, then back to
    f16: ``tl.rsqrt(tmp0.to(tl.float32)).to(tl.float16)``)."""
    x = inputs["in_ptr0"].astype(np.float32)
    return (1.0 / np.sqrt(x)).astype(np.float16)


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
            "Elementwise `out = 1/sqrt(in)` on Meta-Llama-3.1-8B-Instruct's "
            "traced `torch.rsqrt.1` op, one program per row."
        ),
        "doc": (
            "Computes `rsqrt` of every element of a logical `[1, 12, 1]` "
            "f16 tensor. On the Spyre device layout the innermost dim pads "
            "to a 64-element f16 stick, giving a physical shape "
            "`[12, 1, 1, 64]`; the grid is sized to exactly 12 programs "
            "(one per row), and each program reads `tl.program_id(0)` "
            "directly as its row index — no `num_programs`/`cdiv` "
            "distribution loop in the source. `xnumel`/`XBLOCK`/`xindex`/"
            "`xmask` are boilerplate inherited from Inductor's pointwise "
            "codegen: `xnumel` is immediately overwritten with the literal "
            "`12` and `xindex`/`xmask` are never read, confirmed dead by "
            "the traced `.ttir` (no `tt.load`/`tt.store` depends on them)."
        ),
        "kernel_fn":  kernel.triton_unk_fused_rsqrt_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [12], "XBLOCK": [1]},
        "grid":       [12],
        # grid (12) == xnumel (12): one program per row, so DistributeWork
        # emits ktdp.get_compute_tile_id but no residual scf.for — there is
        # no remaining work per program to loop over. Same shape as
        # torch.add.1_spyre; verified empirically via
        # test_work_distribution rather than assumed.
        "distribution_loop": False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
