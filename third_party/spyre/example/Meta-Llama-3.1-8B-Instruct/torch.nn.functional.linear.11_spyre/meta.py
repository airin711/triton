"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.nn.functional.linear.11_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 1, 4096), (4096, 4096, 1))`` — logical
  activation ``x``, dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (128256, 4096), (4096, 1))`` — logical
  weight, dtype ``torch.float16`` (``128256`` = the tokenizer vocab
  size, i.e. this is the traced LM-head projection).
  ``fx_graph_readable.py`` confirms the traced op is
  ``nn.functional.linear(x, weight)`` with **no bias**:
  ``permute(weight,[1,0])`` -> unsqueeze/expand -> ``aten.bmm`` against
  the (unsqueezed/expanded) activation.
- ``triton_bundle_0.run(arg1_1, arg0_1, buf1, buf0, stream=stream0)`` —
  the entry function's own pointer args are, in order,
  ``(weight, x, buf1_scratch, out)``; ``buf1`` is an intermediate
  scratch buffer (weight repacked into a device-tiled layout by
  ``kernel_0``) that ``kernel_1`` consumes as its second input.
- ``triton_meta={..., 'spyre_grids': {'triton_bundle_0_kernel_0': (16,),
  'triton_bundle_0_kernel_1': (4,)}, 'spyre_grid': (16,)}`` — note the
  grid here is ``16``, not the ``32`` used by ``linear.7``-``.10``.

The two helper kernels hardcode their own tensor-descriptor shapes, so
the pointer args here are built directly at those hardcoded physical
shapes (mirroring ``torch.add.1_spyre``'s convention), not at the
logical torch shapes:

- ``in_ptr0`` (weight, consumed by ``kernel_0``'s ``desc_0``): physical
  shape ``[64, 128256, 64]`` (``64*128256*64 == 128256*4096``), a plain
  contiguous reshape of the logical ``(128256, 4096)`` weight.
- ``in_ptr1`` (x, consumed by ``kernel_1``'s ``desc_0``): physical shape
  ``[1, 1, 64, 64]`` (``64*64 == 4096``), a plain contiguous reshape of
  the logical ``(1, 1, 4096)`` activation.
- ``out_ptr0`` (scratch, written by ``kernel_0``'s ``desc_1``): physical
  shape ``[2004, 4096, 64]`` (``2004*4096*64 == 525336576``; note
  ``2004*64 == 128256``). Only needs to be large enough; initial content
  is irrelevant since ``kernel_0`` fully overwrites it before
  ``kernel_1`` reads it.
- ``out_ptr1`` (final output, ``kernel_1``'s ``desc_2``): physical shape
  ``[1, 1, 2004, 64]`` (``2004*64 == 128256``), a plain contiguous
  reshape of the logical ``(1, 1, 128256)`` output.

Numerical caveat (see ``xfail_numerical`` below): as documented in
``kernel.py``, ``kernel_1``'s activation descriptor (``desc_0``,
``block_shape=[1, 1, 8, 64]`` of a ``shape=[1, 1, 64, 64]`` descriptor)
loads only ``8*64 == 512`` of the true ``K=4096`` reduction elements,
and the matching weight descriptor truncates identically -- this is a
*static* reading of the literal source (Triton's inliner/canonicalizer
do not alter hardcoded descriptor shapes; not independently re-verified
via ``ktir_cpu`` for this kernel specifically, since compilation fails
before numerics ever run -- see the primary blocker below). Taken
literally, this kernel's single ``tl.dot`` only ever contracts over the
*first 512* of the 4096 true reduction elements, for every output tile,
on every program -- more severe than ``linear.7``'s half-K truncation.
The oracle below intentionally reproduces this literal (truncated)
math, not the true full-``K=4096`` `nn.functional.linear` result, for
the same reasons documented at length in ``linear.7_spyre/meta.py``.

**Primary blocker (see ``disabled`` below), CONFIRMED BY RUNNING PYTEST
-- and it is a *different* failure than ``linear.7``/``.8_spyre``:** this
kernel fails even earlier than the KTIR pipeline, at plain TTIR
construction (``compile_to_ttir``, before any Spyre-specific lowering
runs at all). ``kernel_0`` (the weight-repack helper) is invoked by the
entry with ``XBLOCK=32833536``, and its body does
``xindex = xoffset + tl.arange(0, XBLOCK)[:]`` -- but ``32833536`` is
**not a power of 2** (``525336576 == 2**20 * 501``, and the odd factor
survives the ``/16`` split into ``32833536 == 2**16 * 501``, ``501 ==
3*167``; this traces back to this op's ``N=128256 == 2004*64`` with
``2004 == 4*501``). Triton's own ``arange()`` unconditionally rejects
any non-power-of-2 range (``python/triton/language/semantic.py``:
``"arange's range must be a power of 2"``), raising
``CompilationError`` -- confirmed by actually running pytest against
this variant (``test_no_tt_ops`` et al. all fail with this exact error,
not a KTIR/``PassManager`` error). Even if this were somehow worked
around, the entry's ``tt.call`` into its ``noinline=True`` helpers would
still fail to compile through the KTIR pipeline for the same
``ConvertFunctions``/``tt.call`` reason documented in
``linear.7_spyre/meta.py`` (confirmed there for ``linear.7``/
``.8_spyre``, whose ``kernel_0`` ``XBLOCK`` values happen to be powers of
2 and so get past this earlier check only to hit that one instead).
Either way, ``setup_method()`` never succeeds, so every structural check
and this numerical comparison fail identically. Not an extraction or
oracle problem.
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(**_unused) -> dict:
    """Build pointer-tensor inputs for the kernel, at each tensor
    descriptor's hardcoded physical shape (see module docstring)."""
    del _unused
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((128256, 4096)).astype(np.float16)
    x = rng.standard_normal((1, 1, 4096)).astype(np.float16)
    in_ptr0 = weight.reshape(64, 128256, 64)
    in_ptr1 = x.reshape(1, 1, 64, 64)
    out_ptr0 = np.zeros((2004, 4096, 64), dtype=np.float16)
    out_ptr1 = np.zeros((1, 1, 2004, 64), dtype=np.float16)
    return {
        "in_ptr0": in_ptr0,
        "in_ptr1": in_ptr1,
        "out_ptr0": out_ptr0,
        "out_ptr1": out_ptr1,
    }


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle matching the kernel's *own* (truncated) contraction.

    Reproduces ``out = x[..., :512] @ weight[:, :512].T`` -- i.e. only
    the first ``512`` of the true ``K=4096`` reduction -- because that
    is what ``kernel_1``'s literal, compile-time-constant descriptor
    offset actually contracts over (see module docstring). This
    intentionally does **not** match true
    ``torch.nn.functional.linear(x, weight)`` semantics over the full
    ``K=4096``.
    """
    x = inputs["in_ptr1"].reshape(1, 1, 4096).astype(np.float32)
    weight = inputs["in_ptr0"].reshape(128256, 4096).astype(np.float32)
    x_trunc = x[..., :512]
    w_trunc = weight[:, :512]
    out = x_trunc @ w_trunc.T  # (1, 1, 128256)
    return out.reshape(1, 1, 2004, 64).astype(np.float16)


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg of the top-level entry function,
# taken from the decorator's ``triton_meta['signature']`` in output_code.py.
# ---------------------------------------------------------------------------

SIGNATURE = {
    "in_ptr0":  "*fp16",
    "in_ptr1":  "*fp16",
    "out_ptr0": "*fp16",
    "out_ptr1": "*fp16",
}


# ---------------------------------------------------------------------------
# VARIANTS
# ---------------------------------------------------------------------------

VARIANTS = {
    "default": {
        "tags": ["bundled-multi-kernel", "tl-dot", "weight-repack"],
        "summary": (
            "`nn.functional.linear` (no bias) on Meta-Llama-3.1-8B-Instruct's "
            "traced `torch.nn.functional.linear.11` op (the LM-head / "
            "vocab-logits projection): x[1,1,4096] @ weight[128256,4096].T "
            "-> out[1,1,128256], as a bundle of a weight-repack helper "
            "feeding a `tl.dot`-based matmul helper."
        ),
        "doc": (
            "`triton_bundle_0` calls `triton_bundle_0_kernel_0` (16 "
            "programs, 2D-flattened program-id indexing) to repack the "
            "raw weight into a device-tiled scratch buffer, then "
            "`triton_bundle_0_kernel_1` (4 programs, one per "
            "32064-wide output-feature tile) to load an activation tile "
            "and a repacked-weight tile, contract them with a single "
            "`tl.dot(..., input_precision=\"ieee\")`, and store the f16 "
            "result. See `kernel.py`/module docstrings for the K-reduction "
            "caveat that this variant's oracle intentionally reproduces "
            "(kernel's own 512-of-4096 truncated contraction, not true "
            "full-K `nn.functional.linear`). This description is of the "
            "traced source's intended computation only -- see `disabled` "
            "below: the kernel does not actually compile, failing at "
            "plain TTIR construction on kernel_0's literal XBLOCK, so "
            "none of this has been verified to execute end to end."
        ),
        "kernel_fn":  kernel.triton_bundle_0,
        "constexpr":  [],
        "params":     {},
        "grid":       [16],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr1",
        # triton_bundle_0's own body calls kernel_0 with a literal
        # XBLOCK=32833536 (non-power-of-2) baked into the call site --
        # same category of bug and same value as linear.5_spyre (both
        # share N=128256). Not fixable via meta.py: the compiled entry
        # (triton_bundle_0) exposes no constexpr.
        "disabled": {
            "reason": (
                "Fails at plain TTIR construction (compile_to_ttir), before "
                "any Spyre-specific lowering: kernel_0 is invoked with a "
                "literal XBLOCK=32833536, a non-power-of-2 value (525336576 == "
                "2**20*501, so 32833536 == 2**16*501, 501 == 3*167; this "
                "factor comes from this op's N=128256 == 2004*64 == "
                "4*501*64), and Triton's own tl.arange(0, XBLOCK) "
                "unconditionally rejects non-power-of-2 ranges ('arange's "
                "range must be a power of 2', "
                "python/triton/language/semantic.py), raising "
                "CompilationError -- confirmed by running pytest against "
                "this variant. Even absent that, the entry's tt.call into "
                "its noinline=True helpers would still hit the KTIR-level "
                "ConvertFunctions/tt.call gap described in "
                "linear.7_spyre/meta.py. Either way setup_method() never "
                "succeeds, so every structural check and the numerical "
                "comparison fail identically. Not an extraction or oracle "
                "problem."
            ),
        },
    },
}
