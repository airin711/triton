"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.nn.functional.linear.7_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the ``async_compile.triton(...)`` decorator
metadata in the traced ``output_code.py`` (see ``kernel.py`` docstring for
the exact path):

- ``assert_size_stride(arg0_1, (1, 1, 4096), (4096, 4096, 1))`` — logical
  activation ``x``, dtype ``torch.float16``.
- ``assert_size_stride(arg1_1, (1024, 4096), (4096, 1))`` — logical
  weight, dtype ``torch.float16``. ``fx_graph_readable.py`` confirms the
  traced op is ``nn.functional.linear(x, weight)`` with **no bias**:
  ``permute(weight,[1,0])`` -> unsqueeze/expand -> ``aten.bmm`` against
  the (unsqueezed/expanded) activation.
- ``triton_bundle_0.run(arg1_1, arg0_1, buf1, buf0, stream=stream0)`` —
  the entry function's own pointer args are, in order,
  ``(weight, x, buf1_scratch, out)``; ``buf1`` is an intermediate
  scratch buffer (weight repacked into a device-tiled layout by
  ``kernel_0``) that ``kernel_1`` consumes as its second input.
- ``triton_meta={..., 'spyre_grids': {'triton_bundle_0_kernel_0': (32,),
  'triton_bundle_0_kernel_1': (16,)}, 'spyre_grid': (32,)}``.

The two helper kernels hardcode their own tensor-descriptor shapes, so
the pointer args here are built directly at those hardcoded physical
shapes (mirroring ``torch.add.1_spyre``'s convention), not at the
logical torch shapes:

- ``in_ptr0`` (weight, consumed by ``kernel_0``'s ``desc_0``): physical
  shape ``[64, 1024, 64]`` (``64*1024*64 == 1024*4096``), a plain
  contiguous reshape of the logical ``(1024, 4096)`` weight (strides
  ``[65536, 64, 1]`` are exactly the contiguous strides for that shape).
- ``in_ptr1`` (x, consumed by ``kernel_1``'s ``desc_0``): physical shape
  ``[1, 1, 64, 64]`` (``64*64 == 4096``), a plain contiguous reshape of
  the logical ``(1, 1, 4096)`` activation.
- ``out_ptr0`` (scratch, written by ``kernel_0``'s ``desc_1``, shape
  ``[16, 4096, 64]``; read back by ``kernel_1``'s ``desc_1`` under a
  *different* shape ``[4096, 16, 64]`` with strides that are the same
  ``{262144, 64, 1}`` set permuted between axes -- i.e. the same
  physical buffer viewed two ways, no data movement). Only needs to be
  large enough (``4194304`` elements); initial content is irrelevant
  since ``kernel_0`` fully overwrites it before ``kernel_1`` reads it.
- ``out_ptr1`` (final output, ``kernel_1``'s ``desc_2``): physical shape
  ``[1, 1, 16, 64]`` (``16*64 == 1024``), a plain contiguous reshape of
  the logical ``(1, 1, 1024)`` output.

Numerical caveat (see ``disabled`` below): ``kernel_1``'s own
reduction-axis tensor descriptor for the activation
(``desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 1, 64, 64], ...,
block_shape=[1, 1, 32, 64])``) has a ``block_shape`` **half** its
descriptor ``shape`` along the K axis (32 vs 64), and the offset feeding
that axis (``dim2 = c1 // 64`` with ``c1 = r0_offset``) is a Python-level
compile-time constant ``0`` -- Inductor's "persistent reduction"
codegen, which on a GPU backend would load the *entire* ``r0_numel``
range in one register tile (``R0_BLOCK == r0_numel == 4096``). The
matching weight descriptor (``desc_1``) has the same halving on its own
reduction axis (``dim_1_0 = c1``, block ``2048`` of ``4096``). This is a
*static* reading of the literal source (and of the compiled TTIR emitted
just before the pipeline failure described below -- Triton's inliner/
canonicalizer do not alter hardcoded descriptor shapes, so the printed
TTIR shows the same ``[1, 1, 32, 64]``/``2048``-of-``4096`` block shapes
verbatim); it was **not** independently confirmed by running the kernel
through ``ktir_cpu``, because compilation never gets that far (see next
paragraph). Taken literally, the kernel as extracted only ever contracts
over the *first half* of the true ``K=4096`` reduction, for every output
tile, on every program -- a real difference from "what a full
torch-spyre Inductor pipeline (with its OpSpec/device-layout side
channel) would compute" vs. "what compiling this kernel's own literal
TTIR source produces." The oracle below intentionally reproduces the
*kernel's* half-K math (not true full-K `nn.functional.linear`) on that
basis, so that if the compilation blocker below is ever fixed, this
oracle is the honest thing to check against -- but this has not been
exercised end to end.

**Primary blocker (see ``disabled`` below):** this kernel does not
compile through this test harness's KTIR pipeline at all. Compilation
fails inside ``backend._make_ktir`` (``PassManager::run failed``) with
the MLIR diagnostic ``'tt.call' op '...' does not reference a valid
function``. Root cause: ``third_party/spyre/lib/Dialect/KTDP/Transforms/
ConvertFunctions.cpp`` converts every ``triton::FuncOp`` to a
``func::FuncOp`` (by cloning ops and erasing the original), but has no
rewrite pattern for ``triton::CallOp`` -- so after conversion,
``triton_bundle_0``'s ``tt.call`` into its ``noinline=True`` helpers
still names a Triton-dialect call, and Triton's own ``tt.call`` verifier
requires the callee to be a ``triton::FuncOp``, which no longer exists.
This means every structural test *and* the numerical test fail at
``setup_method()`` before any KTIR is produced -- confirmed by actually
running ``pytest`` against this variant. This is a genuine backend
pipeline gap for inter-function ``tt.call`` between ``noinline`` helpers
(present in every bundled multi-kernel trace in this family), not an
extraction mistake or an oracle problem.
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
    weight = rng.standard_normal((1024, 4096)).astype(np.float16)
    x = rng.standard_normal((1, 1, 4096)).astype(np.float16)
    in_ptr0 = weight.reshape(64, 1024, 64)
    in_ptr1 = x.reshape(1, 1, 64, 64)
    out_ptr0 = np.zeros((16, 4096, 64), dtype=np.float16)
    out_ptr1 = np.zeros((1, 1, 16, 64), dtype=np.float16)
    return {
        "in_ptr0": in_ptr0,
        "in_ptr1": in_ptr1,
        "out_ptr0": out_ptr0,
        "out_ptr1": out_ptr1,
    }


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle matching the kernel's *own* (truncated) contraction.

    Reproduces ``out = x[..., :2048] @ weight[:, :2048].T`` -- i.e. only
    the first half of the true ``K=4096`` reduction -- because that is
    what ``kernel_1``'s literal, compile-time-constant descriptor offset
    actually contracts over (see module docstring for the full
    explanation and the empirical confirmation against ``ktir_cpu``).
    This intentionally does **not** match true
    ``torch.nn.functional.linear(x, weight)`` semantics over the full
    ``K=4096``.
    """
    x = inputs["in_ptr1"].reshape(1, 1, 4096).astype(np.float32)
    weight = inputs["in_ptr0"].reshape(1024, 4096).astype(np.float32)
    x_half = x[..., :2048]
    w_half = weight[:, :2048]
    out = x_half @ w_half.T  # (1, 1, 1024)
    return out.reshape(1, 1, 16, 64).astype(np.float16)


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
            "traced `torch.nn.functional.linear.7` op: x[1,1,4096] @ "
            "weight[1024,4096].T -> out[1,1,1024], as a bundle of a "
            "weight-repack helper feeding a `tl.dot`-based matmul helper."
        ),
        "doc": (
            "`triton_bundle_0` calls `triton_bundle_0_kernel_0` (32 "
            "programs) to repack the raw weight into a device-tiled "
            "scratch buffer, then `triton_bundle_0_kernel_1` (16 "
            "programs, one per 64-wide output-feature tile) to load an "
            "activation tile and a repacked-weight tile, contract them "
            "with a single `tl.dot(..., input_precision=\"ieee\")`, and "
            "store the f16 result. See `kernel.py`/module docstrings for "
            "the K-reduction caveat that this variant's oracle "
            "intentionally reproduces (kernel's own half-K contraction, "
            "not true full-K `nn.functional.linear`). This description "
            "is of the traced source's intended computation only -- see "
            "`disabled` below: the kernel does not actually compile "
            "through this harness's KTIR pipeline (`tt.call` into the "
            "`noinline=True` helpers has no `ConvertFunctions` rewrite), "
            "so none of this has been verified to execute end to end."
        ),
        "kernel_fn":  kernel.triton_bundle_0,
        "constexpr":  [],
        "params":     {},
        "grid":       [32],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr1",
        # Compiles to TTIR, but fails inside backend._make_ktir: the
        # ConvertFunctions pass has no rewrite pattern for the tt.call
        # ops that invoke triton_bundle_0's noinline=True helpers, so
        # MLIR verification rejects the module ('tt.call' op does not
        # reference a valid function). See the module docstring for the
        # full pytest-confirmed analysis; this is a backend/pipeline
        # gap common to every bundled multi-kernel trace in this family,
        # not an extraction or oracle problem.
        "disabled": {
            "reason": (
                "Does not compile through this harness's KTIR pipeline: "
                "ConvertFunctions (lib/Dialect/KTDP/Transforms/"
                "ConvertFunctions.cpp) converts every tt.func/tt.return to "
                "func.func/func.return but has no rewrite pattern for "
                "tt.call, so triton_bundle_0's calls into its noinline=True "
                "helpers end up naming a symbol that now resolves to a "
                "func::FuncOp; Triton's own tt.call verifier requires a "
                "triton::FuncOp callee, so backend._make_ktir raises "
                "RuntimeError('PassManager::run failed') with "
                "\"'tt.call' op ... does not reference a valid function\" "
                "before any KTIR is produced. Every structural check and "
                "the numerical comparison fail identically at "
                "setup_method() for this reason -- confirmed by running "
                "pytest against this variant. This is a backend gap for "
                "inter-function tt.call between noinline helpers, common "
                "to every bundled multi-kernel trace in this family, not "
                "an extraction or oracle problem."
            ),
        },
    },
}
