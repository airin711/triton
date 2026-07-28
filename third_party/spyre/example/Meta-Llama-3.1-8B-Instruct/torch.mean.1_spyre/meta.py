"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.mean.1_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 12, 4096), (49152, 4096, 1))`` — logical
  input, dtype ``torch.float16``.
- ``buf0 = spyre_empty_with_layout((1, 12, 1), (12, 1, 1), torch.float16,
  SpyreTensorLayout(device_size=[1, 1, 1, 12, 64], ...))`` — logical output
  ``f16[1, 12, 1]`` (the true `aten.mean.dim` result), physically padded to
  device size ``[1, 1, 1, 12, 64]``.
- ``triton_unk_fused_mean_0.run(arg0_1, buf0, 12, 4096, stream=stream0)`` —
  the kernel is launched with ``xnumel=12``, ``r0_numel=4096``.
- ``triton_meta={..., 'spyre_grid': (12,)}`` — 12-program grid, one core per
  row of the logical ``[1, 12, 4096]`` input.

The kernel hardcodes both tensor-descriptor shapes into its body, so the
pointer args here are built directly at those physical shapes:
``in_ptr0`` at ``desc_0``'s ``shape=[12, 64, 1, 64]`` and ``out_ptr0`` at
``desc_1``'s ``shape=[1, 1, 1, 12, 64]``.

Reduction-shape analysis (this is the load-bearing part of this example)
--------------------------------------------------------------------------
The logical op is ``aten.mean.dim(x, [-1], True)`` on ``f16[1, 12, 4096]``,
i.e. a full reduction of the last (4096-element) dimension down to 1,
per row. Inductor's own reduction bookkeeping agrees: ``r0_numel = 4096``
and ``R0_BLOCK: tl.constexpr = 4096`` both say "one tile covers the whole
row, no residual loop".

But the Spyre device layout splits that logical 4096-element row into two
physical axes — a 64-wide "stick" axis (``dim1``, stride 64) and a 64-wide
"lane" axis (``dim3``, stride 1), since ``64 * 64 == 4096`` and the strides
(``[4096, 64, 64, 1]`` for ``shape=[12, 64, 1, 64]``) confirm this is just
the contiguous 4096-element row reinterpreted as ``[stick=64, 1, lane=64]``.

Critically, ``desc_0``'s ``block_shape=[1, 32, 1, 64]`` only spans **32**
of the 64 stick positions (``dim1 in [0, 32)``, since ``r0_offset`` is
always 0 — there is no loop over ``r0_offset`` in this traced kernel body).
So only 2048 of the 4096 logical elements (the first half of the
contiguous row) are ever loaded by ``desc_0.load(...)``; the other 2048
(stick positions 32-63) are never read.

``tmp2 = tl.mean(tmp1, 1)`` then reduces axis 1 of the loaded 4D tile
(the 32-element stick axis) — *not* the full logical reduction — and the
64-element lane axis (axis 3, unchanged through the reshape) is left
**entirely unreduced**: ``tmp4`` has shape ``[1, 1, 1, 1, 64]`` and
``desc_1.store(...)`` writes all 64 of those (distinct, un-averaged-together)
lane values into the output, per row. The logical output is declared 1
element per row (``f16[1, 12, 1]``), but this kernel's own store places
64 values per row into the padded device buffer.

Conclusion: this is a **partial physical-axis reduction** — this traced
kernel body only ever computes a mean over 32 of the 64 stick positions
(half the logical row) and does not touch the lane axis at all. Per the
task's guidance, the NumPy oracle below reproduces exactly this kernel's
own arithmetic (partial stick-axis mean over the loaded slice, lane axis
untouched) rather than the full logical `torch.mean` semantics — matching
what *this* kernel computes, not what `aten.mean.dim` means at the
graph level. Whether a downstream pipeline stage is expected to finish
the reduction (over the remaining 32 stick positions and over the 64 lane
positions) is outside the scope of what this single kernel does.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(xnumel: int, r0_numel: int, XBLOCK: int, **_unused) -> dict:
    """Build pointer-tensor inputs for the kernel.

    ``xnumel``/``r0_numel``/``XBLOCK`` are accepted so the signature
    matches the full param set, but none of them shape the data: the
    kernel body reassigns ``xnumel = 12``/``r0_numel = 4096`` and hardcodes
    both tensor-descriptor shapes, so the buffers are built directly at
    those shapes (see ``kernel.py``): ``in_ptr0`` at ``[12, 64, 1, 64]``
    (``desc_0``) and ``out_ptr0`` at ``[1, 1, 1, 12, 64]`` (``desc_1``).
    """
    del xnumel, r0_numel, XBLOCK
    rng = np.random.default_rng(0)
    in_ptr0 = rng.standard_normal((12, 64, 1, 64)).astype(np.float16)
    out_ptr0 = np.zeros((1, 1, 1, 12, 64), dtype=np.float16)
    return {"in_ptr0": in_ptr0, "out_ptr0": out_ptr0}


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle matching this kernel's own (partial) arithmetic.

    Loads only the ``dim1 in [0, 32)`` slice of the ``[12, 64, 1, 64]``
    input (matching ``desc_0``'s ``block_shape=[1, 32, 1, 64]`` at
    offset 0), computes the mean over that 32-element stick axis in f32
    (matching ``tmp1.to(tl.float32)`` -> ``tl.mean(tmp1, 1)``), truncates
    back to f16 once (matching ``tmp2.to(tl.float16)``), and reshapes to
    the ``[1, 1, 1, 12, 64]`` store shape. The 64-element lane axis is
    left unreduced throughout, exactly as the kernel leaves it.
    """
    x = inputs["in_ptr0"].astype(np.float32)  # [12, 64, 1, 64]
    x_block = x[:, 0:32, :, :]                 # [12, 32, 1, 64] -- desc_0's loaded slice
    mean = x_block.mean(axis=1)                 # [12, 1, 64], f32
    return mean.astype(np.float16).reshape(1, 1, 1, 12, 64)


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":   "*fp16",
    "out_ptr0":  "*fp16",
    "xnumel":    "i32",
    "r0_numel":  "i32",
    "XBLOCK":    "i32",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["descriptor-load-static", "descriptor-store-static", "program-id-1d", "reduction"],
        "summary": (
            "Partial stick-axis `tl.mean` reduction on Meta-Llama-3.1-8B-"
            "Instruct's traced `torch.mean.1` op, one program per row."
        ),
        "doc": (
            "Traced from `aten.mean.dim(x, [-1], True)` on a logical "
            "`f16[1, 12, 4096]` tensor. The Spyre device layout splits the "
            "4096-element reduction row into a 64-wide stick axis and a "
            "64-wide lane axis (`64 * 64 == 4096`), giving `in_ptr0` a "
            "physical shape `[12, 64, 1, 64]`. This kernel's tensor "
            "descriptor only loads `dim1 in [0, 32)` (block_shape "
            "`[1, 32, 1, 64]`) -- half the stick axis, i.e. 2048 of the "
            "4096 logical elements -- and `tl.mean(tmp1, 1)` reduces only "
            "that loaded 32-element stick slice, leaving the 64-element "
            "lane axis completely unreduced: the store writes 64 values "
            "per row (`out_ptr0` physical shape `[1, 1, 1, 12, 64]`), not "
            "the single true mean per row that the logical `f16[1, 12, 1]` "
            "output shape implies. Grid is exactly 12 programs (one per "
            "row), matching `xnumel=12`; each program reads "
            "`tl.program_id(0)` directly as its row index -- no "
            "`num_programs`/`cdiv` distribution loop in the source. This "
            "is treated here as a genuine partial physical-axis reduction "
            "(see meta.py module docstring for the full trace-through): "
            "the oracle reproduces exactly this kernel's own arithmetic, "
            "not the full logical `torch.mean`."
        ),
        "kernel_fn":  kernel.triton_unk_fused_mean_0,
        "constexpr":  ["XBLOCK"],
        "params":     {"xnumel": [12], "r0_numel": [4096], "XBLOCK": [1]},
        "grid":       [12],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr0",
        # See the module docstring above for the reduction-shape analysis
        # (a genuine partial physical-axis reduction) -- that analysis is
        # moot for testing purposes because this kernel cannot be compiled
        # at all: `tl.mean` is not a real Triton language primitive on any
        # backend (see triton/python/triton/language/standard.py -- the
        # reduction ops are sum/max/min/xor_sum, no mean). Compiling this
        # kernel_fn via ASTSource raises
        # `AttributeError: module 'triton.language' has no attribute
        # 'mean'` at the Python AST-to-TTIR frontend stage, before any
        # Spyre-specific TTIR->KTIR lowering runs -- so every structural
        # test (which compiles in setup_method) would ERROR, not just the
        # numerical test, ruling out `xfail_numerical` (which only guards
        # test_numerical). Confirmed independently by torch-spyre's own
        # torch_spyre/_inductor_triton/spyre_triton_patches.py, whose
        # `_spyre_triton_decomp_layer_norm`/`_spyre_triton_decomp_rms_norm`
        # docstrings state plainly: "aten.mean.dim reaches codegen as an
        # ops.reduction(..., 'mean') that get_triton_reduction_function
        # maps to tl.mean -- an op the Spyre Triton backend does not
        # provide, so the bundle fails TTIR generation" -- and work around
        # it by rewriting mean as sum(...)/N for those two decompositions.
        # This traced kernel is a direct (non-decomposed) `torch.mean` call
        # that hits the exact same unsupported-op path; no such workaround
        # exists for a bare `torch.mean`. Kept verbatim (kernel.py is
        # untouched) per this example library's convention of preserving
        # traced kernel bodies as-is rather than "fixing" them.
        "disabled": {
            "reason": (
                "Traced kernel body calls tl.mean(tmp1, 1), but tl.mean is "
                "not a real Triton language primitive (any backend) -- "
                "compiling via ASTSource raises AttributeError: module "
                "'triton.language' has no attribute 'mean' at the Python "
                "AST-to-TTIR frontend stage, before Spyre's own TTIR->KTIR "
                "pipeline runs. See torch_spyre/_inductor_triton/"
                "spyre_triton_patches.py, which documents this exact gap "
                "for layer_norm/rms_norm's internal mean usage (worked "
                "around there via sum(...)/N) -- this traced op is a "
                "direct, non-decomposed torch.mean call with no such "
                "workaround applied."
            ),
        },
    },
}
