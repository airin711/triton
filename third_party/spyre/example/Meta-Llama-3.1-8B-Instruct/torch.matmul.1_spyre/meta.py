"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.matmul.1_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py``'s docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 64, 1), (64, 1, 1))`` and
  ``assert_size_stride(arg1_1, (1, 1, 12), (12, 12, 1))`` — logical inputs,
  dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 64, 12), (768, 12, 1), torch.float16,
  SpyreTensorLayout(device_size=[1, 1, 64, 64], ...))`` — logical output
  ``f16[1, 64, 12]``.
- ``triton_unk_fused_bmm_0.run(arg0_1, arg1_1, buf0, 64, 12, stream=stream0)``
  — launched with ``ynumel=64``, ``xnumel=12``.
- ``triton_meta={..., 'spyre_grid': (32,)}`` — 32-program 1D grid.

``aten.bmm.default`` here contracts over a size-1 dimension, so it
degenerates into a broadcast elementwise multiply: ``tl.program_id(0)``
selects a 2-row tile (``YBLOCK=2``) of the 64-row first operand; the second
operand's single 64-wide stick (holding all 12 real output columns, padded
to 64) is broadcast across those 2 rows and multiplied in. ``XBLOCK`` (=12,
covering the full ``xnumel``) and every ``x*``/``y*`` boilerplate variable
(``xoffset``/``xindex``/``xmask``/``yoffset``/``yindex``/``ymask``) are dead
— none of them feed the descriptor indices (``c0``/``c1``), which are
derived directly from ``tl.program_id(0)`` (``c1`` is always the literal
``0``: there is only one tile along that axis since ``XBLOCK == xnumel``).
The kernel body works on the full padded 64-wide sticks in both operands
uniformly, so the NumPy oracle below just replicates that same stick-wise
multiply on the physical (padded) buffers — it does not need to reproduce
the original logical ``bmm`` shapes.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(xnumel: int, ynumel: int, XBLOCK: int, YBLOCK: int, **_unused) -> dict:
    """Build pointer-tensor inputs for the kernel.

    ``xnumel``/``ynumel``/``XBLOCK``/``YBLOCK`` are accepted so the
    signature matches the full param set, but none of them shape the data:
    the kernel body reassigns ``xnumel = 12``/``ynumel = 64`` and never
    reads ``xindex``/``xmask``/``yindex``/``ymask``, so the buffers are
    built directly at the descriptors' hardcoded physical shapes (see
    ``kernel.py``): ``in_ptr0`` is ``[64, 1, 1, 64]``, ``in_ptr1`` is
    ``[1, 1, 1, 64]``, ``out_ptr0`` is ``[1, 1, 64, 64]``.
    """
    del xnumel, ynumel, XBLOCK, YBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((64, 1, 1, 64)).astype(np.float16)
    in_ptr1 = rng.standard_normal((1, 1, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((1, 1, 64, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "in_ptr1": in_ptr1, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: broadcast elementwise multiply of the two padded
    sticks, in the kernel's own compute precision (extend to f32,
    multiply, truncate back to f16)."""
    x = inputs["in_ptr0"].astype(np.float32).reshape(64, 64)   # [row, stick]
    w = inputs["in_ptr1"].astype(np.float32).reshape(64)        # [stick]
    out = x * w[None, :]
    return out.reshape(1, 1, 64, 64).astype(np.float16)


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*fp16",
    "in_ptr1":  "*fp16",
    "out_ptr0": "*fp16",
    "ynumel":   "i32",
    "xnumel":   "i32",
    "YBLOCK":   "i32",
    "XBLOCK":   "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d"],
        "summary": (
            "Broadcast elementwise multiply implementing a degenerate "
            "`torch.matmul.1` (`aten.bmm`) with contracted dim size 1, on "
            "Meta-Llama-3.1-8B-Instruct's traced op."
        ),
        "doc": (
            "`aten.bmm.default` of logical `f16[1, 64, 1]` and `f16[1, 1, 12]` "
            "tensors contracts over a size-1 dimension, so it degenerates "
            "into a broadcast multiply rather than a real matmul. The first "
            "operand's device layout pads its size-1 feature dim out to a "
            "full 64-wide f16 stick (`[64, 1, 1, 64]`, only element 0 of "
            "each row logically real); the second operand's 12 real "
            "columns live in a single 64-wide stick (`[1, 1, 1, 64]`). "
            "`tl.program_id(0)` selects a 2-row tile (`YBLOCK=2`) of the "
            "64-row first operand; the second operand's stick is loaded "
            "once per program and broadcast across those 2 rows. "
            "`XBLOCK` (=12, covering the whole `xnumel`) and every "
            "`x*`/`y*` boilerplate variable are dead — confirmed by the "
            "traced source: `c0`/`c1` (the only values that actually feed "
            "the descriptor indices) are derived directly from "
            "`tl.program_id(0)` and the literal `0`, never from "
            "`xindex`/`xmask`/`yindex`/`ymask`."
        ),
        "kernel_fn":  kernel.triton_unk_fused_bmm_0,
        "constexpr":  ["YBLOCK", "XBLOCK"],
        # XBLOCK deviates from the traced decorator's literal config value
        # (12): XBLOCK only feeds the dead xoffset/xindex/xmask chain (never
        # read by the real descriptor-driven computation, see kernel.py's
        # docstring), but Triton's frontend still eagerly compiles
        # `tl.arange(0, XBLOCK)` and requires a power-of-2 range. 12 isn't
        # one, so it's rounded up to 16 here — this has zero effect on the
        # kernel's actual output since xindex/xmask are never used.
        "params":     {"ynumel": [64], "xnumel": [12], "YBLOCK": [2], "XBLOCK": [16]},
        "grid":       [32],
        # grid (32) * YBLOCK (2) == ynumel (64), and XBLOCK covers the whole
        # xnumel in one tile: every program gets a fixed 2-row chunk with
        # nothing left over, so DistributeWork emits ktdp.get_compute_tile_id
        # but no residual scf.for — the same underlying reason as add.1's
        # grid == xnumel case, just with a tile size (YBLOCK=2) > 1.
        "distribution_loop": False,
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
    },
}
