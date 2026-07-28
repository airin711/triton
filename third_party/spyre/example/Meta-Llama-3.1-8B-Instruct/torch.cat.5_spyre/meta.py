"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.cat.5_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 32, 1, 64), (2048, 64, 64, 1))`` / same
  for ``arg1_1`` — two logical inputs, both dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 32, 1, 128), (4096, 128, 128, 1),
  torch.float16, SpyreTensorLayout(device_size=[32, 1, 2, 1, 64], ...))``
  — logical output ``f16[1, 32, 1, 128]``.
- ``triton_bundle_0.run(arg0_1, arg1_1, buf0, stream=stream0)`` — the entry
  takes only the three pointer args; the ``xnumel``/``XBLOCK`` values used
  by the two helpers (``2048``/``64``) are baked in as literals at the
  call sites inside ``triton_bundle_0`` itself.
- ``triton_meta={..., 'spyre_grid': (32,)}`` — 32-program grid, one core
  per row of the device-layout shape ``[32, 1, 1, 1, 64]`` /
  ``[32, 1, 2, 1, 64]``.

Both helpers hardcode the device shapes ``[32, 1, 1, 1, 64]`` (inputs) and
``[32, 1, 2, 1, 64]`` (output) into their tensor descriptors, so the
pointer args here are built directly at those shapes. ``kernel_0`` writes
``in_ptr0`` into index 0 of the doubled axis (axis 2); ``kernel_1`` writes
``in_ptr1`` into index 1 of that same axis — together they implement
``aten.cat.default([in_ptr0, in_ptr1], dim=-1)`` on the physical buffers.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

# Device-layout axis along which the two operands are concatenated (the
# axis whose size goes from 1 in each input descriptor to 2 in the output
# descriptor -- see kernel.py: kernel_0 stores at index 0, kernel_1 at
# index 1 of this axis).
CONCAT_AXIS = 2

_IN_SHAPE = (32, 1, 1, 1, 64)
_OUT_SHAPE = (32, 1, 2, 1, 64)


def make_inputs(**_unused) -> dict:
    """Build pointer-tensor inputs for the kernel.

    No runtime scalar params exist at the entry function's own signature
    (``xnumel``/``XBLOCK`` are internal to the two ``noinline`` helpers,
    baked in as call-site literals -- see ``kernel.py``), so the buffers
    are simply built at the descriptors' hardcoded device shapes.
    """
    del _unused
    rng0 = np.random.default_rng(0)
    rng1 = np.random.default_rng(1)
    in_ptr0 = rng0.standard_normal(_IN_SHAPE).astype(np.float16)
    in_ptr1 = rng1.standard_normal(_IN_SHAPE).astype(np.float16)
    out_ptr0 = np.zeros(_OUT_SHAPE, dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: ``aten.cat.default([in_ptr0, in_ptr1], dim=-1)`` on the
    physical (device-layout) buffers. Both helpers are pure copies (no
    arithmetic), so no precision extension is needed -- the output is
    exactly the two f16 inputs concatenated along ``CONCAT_AXIS``."""
    return np.concatenate(
        [inputs["in_ptr0"], inputs["in_ptr1"]], axis=CONCAT_AXIS,
    )


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py. Note this is the ENTRY
# function's own signature (``triton_bundle_0``), not either helper's.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*fp16",
    "in_ptr1":  "*fp16",
    "out_ptr0": "*fp16",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d", "bundled-multi-kernel"],
        "summary": (
            "Bundled two-kernel `aten.cat` implementing Meta-Llama-3.1-8B-"
            "Instruct's traced `torch.cat.5` op: concatenates two logical "
            "`[1, 32, 1, 64]` f16 tensors along the last dim, one program "
            "per row."
        ),
        "doc": (
            "`aten.cat.default([x, y], dim=-1)` on two logical "
            "`[1, 32, 1, 64]` f16 tensors, producing `[1, 32, 1, 128]`. On "
            "the Spyre device layout each input's innermost 64-wide dim "
            "stays a single stick (`[32, 1, 1, 1, 64]`); the output pads "
            "the 128-wide concatenated dim out to two 64-wide sticks "
            "(`[32, 1, 2, 1, 64]`). The traced entry `triton_bundle_0` is "
            "a thin dispatcher that calls two `noinline=True` helpers in "
            "sequence: `triton_bundle_0_kernel_0` loads `in_ptr0` (shape "
            "`[1,1,1,1,64]`) and stores it directly into stick 0 of the "
            "output's doubled axis (axis 2) with no reshape/broadcast, "
            "while `triton_bundle_0_kernel_1` reshapes+broadcasts "
            "`in_ptr1` to `[1,1,2,1,64]` before storing into stick 1 — "
            "see the `disabled` note below, this asymmetry is a genuine "
            "bug in the traced source, not a mistake in this extraction. "
            "The grid is sized to exactly 32 programs (one per `dim0` row), "
            "and each program reads `tl.program_id(0)` directly as its row "
            "index — no `num_programs`/`cdiv` distribution loop in the "
            "source. `xnumel`/`XBLOCK`/`xindex`/`xmask` inside the helpers "
            "are boilerplate inherited from Inductor's pointwise codegen: "
            "`xnumel` is immediately overwritten with the literal `2048` "
            "and `xindex`/`xmask` are never read."
        ),
        "kernel_fn":  kernel.triton_bundle_0,
        "constexpr":  [],
        "params":     {},
        "grid":       [32],
        # kernel_0 stores its loaded value (shape [1,1,1,1,64], from
        # desc_0's block_shape) directly into desc_1, whose block_shape is
        # [1,1,2,1,64] (the doubled concat axis) -- with no reshape/
        # broadcast to match, unlike kernel_1's otherwise-identical store.
        # This is present verbatim in the traced output_code.py (see
        # kernel.py's docstring) and reproduces identically across every
        # torch.cat.*_spyre example in this batch, so it looks like a
        # genuine bug in Inductor/torch-spyre's bundled-cat codegen for
        # this op, not an artifact of extraction. Triton's ASTSource
        # frontend rejects the store at compile time (validate_store_like
        # shape-equality assertion), before any TTIR is produced.
        "disabled": {
            "reason": (
                "kernel_0's desc_1.store() passes a [1,1,1,1,64] value "
                "into a descriptor with block_shape [1,1,2,1,64] with no "
                "reshape/broadcast (kernel_1's store does reshape+"
                "broadcast) -- a shape mismatch present verbatim in the "
                "traced source, rejected by Triton's frontend "
                "validate_store_like check before TTIR construction. "
                "Reproduces identically across all torch.cat.*_spyre "
                "examples."
            ),
        },
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
