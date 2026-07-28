"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.mul.7_spyre``.

Structurally identical to ``torch.mul.6_spyre`` — same shapes, strides, and
kernel body, just a distinct traced call site (see ``kernel.py`` for the
exact source path). See ``torch.mul.6_spyre/meta.py`` for the full
derivation of the reference oracle and device-layout reasoning; this file
duplicates it verbatim.

Caller-side settings recap:

- logical ``x``: ``[1, 8, 12, 128]`` f16; logical ``y``: ``[1, 1, 12,
  128]`` f16, broadcast over ``x``'s head dim; logical output: same shape
  as ``x``.
- device shapes: ``in_ptr0`` ``[8, 12, 2, 1, 64]``, ``in_ptr1`` ``[1, 12,
  2, 1, 64]``, ``out_ptr0`` ``[12, 1, 2, 8, 64]``.
- ``xnumel = 12288``, ``spyre_grid = (24,)``, one program per (row,
  4-wide head-tile) pair.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(xnumel: int, XBLOCK: int, **_unused) -> dict:
    """Build pointer-tensor inputs for the kernel (see
    ``torch.mul.6_spyre/meta.py`` for the full rationale).
    """
    del xnumel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((8, 12, 2, 1, 64)).astype(np.float16)
    in_ptr1 = rng.standard_normal((1, 12, 2, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((12, 1, 2, 8, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: identical derivation to ``torch.mul.6_spyre/meta.py``.
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
            "traced `torch.mul.7` op (structurally identical to "
            "`torch.mul.6`): `y` broadcast over `x`'s 8-wide head axis, "
            "one program per (row, 4-wide head-tile) pair."
        ),
        "doc": (
            "Multiplies a logical `[1, 8, 12, 128]` f16 tensor `x` by a "
            "logical `[1, 1, 12, 128]` f16 tensor `y` broadcast over `x`'s "
            "head dim; same kernel body, shapes, and strides as "
            "`torch.mul.6_spyre` (see that example's `doc` for the full "
            "device-layout explanation), just a distinct traced call site."
        ),
        "kernel_fn":  kernel.triton_unk_fused_mul_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [12288], "XBLOCK": [512]},
        "grid":       [24],
        "distribution_loop": False,  # see torch.mul.6_spyre/meta.py
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
