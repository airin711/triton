"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.nn.functional.linear.10_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 1, 14336), (14336, 14336, 1))`` —
  logical activation ``x``, dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (4096, 14336), (14336, 1))`` — logical
  weight, dtype ``torch.float16``. ``fx_graph_readable.py`` confirms the
  traced op is ``nn.functional.linear(x, weight)`` with **no bias**:
  ``permute(weight,[1,0])`` -> unsqueeze/expand -> ``aten.bmm`` against
  the (unsqueezed/expanded) activation. Note ``in_features=14336``,
  ``out_features=4096`` here — the transpose of ``linear.9``'s shape.
- ``triton_bundle_0.run(arg1_1, arg0_1, buf1, buf0, stream=stream0)`` —
  the entry function's own pointer args are, in order,
  ``(weight, x, buf1_scratch, out)``; ``buf1`` is an intermediate
  scratch buffer (weight repacked into a device-tiled layout by
  ``kernel_0``) that ``kernel_1`` consumes as its second input.
- ``triton_meta={..., 'spyre_grids': {'triton_bundle_0_kernel_0': (32,),
  'triton_bundle_0_kernel_1': (32,)}, 'spyre_grid': (32,)}``.

The two helper kernels hardcode their own tensor-descriptor shapes, so
the pointer args here are built directly at those hardcoded physical
shapes (mirroring ``torch.add.1_spyre``'s convention), not at the
logical torch shapes:

- ``in_ptr0`` (weight, consumed by ``kernel_0``'s ``desc_0``): physical
  shape ``[224, 4096, 64]`` (``224*4096*64 == 4096*14336``), a plain
  contiguous reshape of the logical ``(4096, 14336)`` weight.
- ``in_ptr1`` (x, consumed by ``kernel_1``'s ``desc_0``): physical shape
  ``[1, 1, 224, 64]`` (``224*64 == 14336``), a plain contiguous reshape
  of the logical ``(1, 1, 14336)`` activation.
- ``out_ptr0`` (scratch, written by ``kernel_0``'s ``desc_1``): physical
  shape ``[64, 14336, 64]`` (``64*14336*64 == 58720256``). Only needs
  to be large enough; initial content is irrelevant since ``kernel_0``
  fully overwrites it before ``kernel_1`` reads it.
- ``out_ptr1`` (final output, ``kernel_1``'s ``desc_2``): physical shape
  ``[1, 1, 64, 64]`` (``64*64 == 4096``), a plain contiguous reshape of
  the logical ``(1, 1, 4096)`` output.

``kernel_1``'s activation descriptor here (``desc_0``,
``block_shape=[1, 1, 224, 64]``) is fully equal to its own ``shape`` --
no K-axis truncation (``K=14336`` divides evenly into ``224*64``). The
single ``tl.dot`` genuinely contracts over the entire ``K=14336``
(``tmp6`` reshaped to ``[1, 14336]`` against ``tmp5`` reshaped to
``[14336, 128]``), so the oracle below (full-``K`` ``nn.functional.linear``,
no bias) matches the kernel's own literal math with no caveat needed on
that front. (The unusual ``R0_BLOCK=16384``/``r0_mask`` in ``kernel_1``
is Inductor persistent-reduction register-tile padding and does not
change what the descriptors themselves load — see ``kernel.py``.)

**Primary blocker (see ``disabled`` below), CONFIRMED BY RUNNING PYTEST
-- and it is a *different* failure than ``linear.7``/``.8_spyre``:** this
kernel fails even earlier than the KTIR pipeline, at plain TTIR
construction (``compile_to_ttir``, before any Spyre-specific lowering
runs at all). ``kernel_0`` (the weight-repack helper) is invoked by the
entry with ``XBLOCK=1835008``, and its body does
``xindex = xoffset + tl.arange(0, XBLOCK)[:]`` -- but ``1835008`` is
**not a power of 2** (``58720256 == 7 * 2**23``, and the odd factor of
``7`` survives the ``/32`` split into ``1835008 == 7 * 2**18``; the
``7`` traces back to this op's ``K=14336 == 224*64`` with
``224 == 32*7``). Triton's own ``arange()`` unconditionally rejects any
non-power-of-2 range (``python/triton/language/semantic.py``: ``"arange's
range must be a power of 2"``), raising ``CompilationError`` -- confirmed
by actually running pytest against this variant (``test_no_tt_ops`` et
al. all fail with this exact error, not a KTIR/``PassManager`` error).
Even if this were somehow worked around, the entry's ``tt.call`` into
its ``noinline=True`` helpers would still fail to compile through the
KTIR pipeline for the same ``ConvertFunctions``/``tt.call`` reason
documented in ``linear.7_spyre/meta.py`` (confirmed there for
``linear.7``/``.8_spyre``, whose ``kernel_0`` ``XBLOCK`` values happen to
be powers of 2 and so get past this earlier check only to hit that one
instead). Either way, ``setup_method()`` never succeeds, so every
structural check and this numerical comparison fail identically. Not an
extraction or oracle problem.
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
    weight = rng.standard_normal((4096, 14336)).astype(np.float16)
    x = rng.standard_normal((1, 1, 14336)).astype(np.float16)
    in_ptr0 = weight.reshape(224, 4096, 64)
    in_ptr1 = x.reshape(1, 1, 224, 64)
    out_ptr0 = np.zeros((64, 14336, 64), dtype=np.float16)
    out_ptr1 = np.zeros((1, 1, 64, 64), dtype=np.float16)
    return {
        "in_ptr0": in_ptr0,
        "in_ptr1": in_ptr1,
        "out_ptr0": out_ptr0,
        "out_ptr1": out_ptr1,
    }


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: full-K ``nn.functional.linear(x, weight)``, no bias.

    ``out = x @ weight.T`` over the entire ``K=14336`` -- this matches
    both true `torch.nn.functional.linear` semantics *and* what
    ``kernel_1``'s own (untruncated) descriptor/`tl.dot` literally
    computes (see module docstring).
    """
    x = inputs["in_ptr1"].reshape(1, 1, 14336).astype(np.float32)
    weight = inputs["in_ptr0"].reshape(4096, 14336).astype(np.float32)
    out = x @ weight.T  # (1, 1, 4096)
    return out.reshape(1, 1, 64, 64).astype(np.float16)


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
            "traced `torch.nn.functional.linear.10` op: x[1,1,14336] @ "
            "weight[4096,14336].T -> out[1,1,4096], as a bundle of a "
            "weight-repack helper feeding a `tl.dot`-based matmul helper."
        ),
        "doc": (
            "`triton_bundle_0` calls `triton_bundle_0_kernel_0` (32 "
            "programs) to repack the raw weight into a device-tiled "
            "scratch buffer, then `triton_bundle_0_kernel_1` (32 "
            "programs, one per 128-wide output-feature tile) to load an "
            "activation tile and a repacked-weight tile, contract them "
            "with a single `tl.dot(..., input_precision=\"ieee\")` over "
            "the full K=14336, and store the f16 result. This description "
            "is of the traced source's intended computation only -- see "
            "`disabled` below: the kernel does not actually compile, "
            "failing at plain TTIR construction on kernel_0's literal "
            "XBLOCK, so none of this has been verified to execute end "
            "to end."
        ),
        "kernel_fn":  kernel.triton_bundle_0,
        "constexpr":  [],
        "params":     {},
        "grid":       [32],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr1",
        # triton_bundle_0's own body calls kernel_0 with a literal
        # XBLOCK=1835008 (non-power-of-2) baked into the call site --
        # same category of bug as linear.1-.5_spyre/.9_spyre, discovered
        # here via the transposed K=14336 shape. Not fixable via
        # meta.py: the compiled entry (triton_bundle_0) exposes no
        # constexpr.
        "disabled": {
            "reason": (
                "Fails at plain TTIR construction (compile_to_ttir), before "
                "any Spyre-specific lowering: kernel_0 is invoked with a "
                "literal XBLOCK=1835008, a non-power-of-2 value (58720256 == "
                "7*2**23, so 1835008 == 7*2**18; the factor of 7 comes from "
                "this op's K=14336 == 224*64 == 32*7*64), and Triton's own "
                "tl.arange(0, XBLOCK) unconditionally rejects non-power-of-2 "
                "ranges ('arange's range must be a power of 2', "
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
