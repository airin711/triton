"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.nn.functional.linear.6_spyre``.

Same bundled-kernel structure as ``.1_spyre`` (see that folder's
``meta.py`` docstring for the full field-by-field rationale). M=1 (a
single decode-step token), K=4096, N=4096 — the M=1 analogue of
``.1_spyre``'s square weight. ``spyre_grid: (32,)`` for both bundled
kernels.

Per-pointer device layout (plain C-contiguous reshapes, no permutation
at this level — see ``kernel.py``):

- ``in_ptr0``  (raw weight,     logical ``[4096, 4096]`` = N x K) -> device ``[64, 4096, 64]``
- ``in_ptr1``  (raw activation, logical ``[1, 4096]``     = M x K) -> device ``[1, 1, 64, 64]``
- ``out_ptr0`` (scratch)                                          -> device ``[64, 4096, 64]``
- ``out_ptr1`` (raw output,     logical ``[1, 4096]``     = M x N) -> device ``[1, 1, 64, 64]``

Unlike every other ``.N_spyre`` variant in this bundle (``.1``-``.5``,
all of which have at least one non-power-of-2 tile constant driving a
dead ``tl.arange`` that Triton's frontend unconditionally rejects), every
constexpr tile size in *both* bundled sub-kernels here is a power of 2
(kernel_0: XBLOCK=524288=2**19; kernel_1: XBLOCK=128=2**7,
R0_BLOCK=4096=2**12; kernel_1 also has no y-tree at all, same as
``.5_spyre``, since M=1). This is the only one of the six expected to
compile cleanly all the way through TTIR (and, pending the KTIR lowering
pass actually supporting `tl.dot`/tensor-descriptor bundled kernels, on
to KTIR).
"""

import numpy as np

from . import kernel


M, K, N = 1, 4096, 4096
WEIGHT_DEV_SHAPE = (64, 4096, 64)   # N*K = 16777216
ACT_DEV_SHAPE = (M, 1, 64, 64)      # M*K = 4096
SCRATCH_DEV_SHAPE = (64, 4096, 64)  # N*K = 16777216
OUT_DEV_SHAPE = (M, 1, 64, 64)      # M*N = 4096


def make_inputs(**_unused) -> dict:
    del _unused
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((N, K)).astype(np.float16)
    x = rng.standard_normal((M, K)).astype(np.float16)
    return {
        "in_ptr0": weight.reshape(WEIGHT_DEV_SHAPE),
        "in_ptr1": x.reshape(ACT_DEV_SHAPE),
        "out_ptr0": np.zeros(SCRATCH_DEV_SHAPE, dtype=np.float16),
        "out_ptr1": np.zeros(OUT_DEV_SHAPE, dtype=np.float16),
    }


def run(inputs: dict) -> np.ndarray:
    """NumPy oracle: `out = x @ weight.T` (no bias), f32 contraction
    truncated back to f16, matching `tl.dot(..., input_precision="ieee")`
    + immediate `.to(tl.float16)`."""
    weight = inputs["in_ptr0"].reshape(N, K).astype(np.float32)
    x = inputs["in_ptr1"].reshape(M, K).astype(np.float32)
    out = x @ weight.T
    return out.astype(np.float16).reshape(OUT_DEV_SHAPE)


SIGNATURE = {
    "in_ptr0":  "*fp16",
    "in_ptr1":  "*fp16",
    "out_ptr0": "*fp16",
    "out_ptr1": "*fp16",
}


VARIANTS = {
    "default": {
        "tags": ["bundled-kernel", "tl-dot", "descriptor-load-static", "descriptor-store-static"],
        "summary": (
            "`nn.functional.linear` (no bias) on Meta-Llama-3.1-8B-Instruct's "
            "traced `torch.nn.functional.linear.6` op: M=1 (decode step), "
            "K=4096, N=4096."
        ),
        "doc": (
            "Same structure as `.1_spyre`, with M=1 collapsing kernel_1's "
            "y-tree entirely (no `ynumel`/`YBLOCK` in its signature, same "
            "as `.5_spyre`). All constexpr tile sizes in both bundled "
            "sub-kernels are powers of 2, so — unlike `.1_spyre`-`.5_spyre` "
            "— this kernel compiles cleanly to TTIR. It still fails to "
            "lower to KTIR: the `ConvertFunctions` pass (`third_party/"
            "spyre/lib/Dialect/KTDP/Transforms/ConvertFunctions.cpp`) "
            "converts every `tt.func`/`tt.return` to `func.func`/"
            "`func.return` but never rewrites the `tt.call` ops that "
            "invoke the two `noinline=True` helpers, leaving them "
            "pointing at symbols that are no longer `triton::FuncOp`s; "
            "MLIR verification then rejects the module with \"'tt.call' "
            "op ... does not reference a valid function\". The upstream "
            "TTIR inliner (`_make_ttir`'s `add_inliner`) does not help "
            "either, since it correctly honors `noinline=True` and leaves "
            "these calls un-inlined by design — this is a real KTIR-"
            "lowering gap for bundled/multi-function kernels, not an "
            "artifact of this test harness."
        ),
        "kernel_fn":  kernel.triton_bundle_0,
        "constexpr":  [],
        "params":     {},
        "grid":       [32],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr1",
        # Compiles cleanly through Triton's frontend to TTIR (every
        # constexpr tile size in both bundled sub-kernels happens to be
        # a power of 2 here), but the ConvertFunctions pass
        # (third_party/spyre/lib/Dialect/KTDP/Transforms/
        # ConvertFunctions.cpp) has no rewrite pattern for the tt.call
        # ops that invoke triton_bundle_0's noinline=True helpers, so
        # MLIR verification rejects the module once ConvertFunctions has
        # converted the callees to func::FuncOp. This is a genuine
        # backend/pipeline gap (confirmed against the production
        # inliner too, which correctly leaves noinline calls un-inlined),
        # not an extraction artifact -- see the module docstring.
        "disabled": {
            "reason": (
                "Compiles cleanly to TTIR (all constexpr tile sizes are "
                "powers of 2), but `make_ktir_mod` fails while lowering "
                "to KTIR: the `ConvertFunctions` pass converts every "
                "`triton::FuncOp` to `func::FuncOp` but has no rewrite "
                "pattern for the `tt.call` ops that invoke "
                "`triton_bundle_0`'s `noinline=True` helpers, so MLIR "
                "verification rejects the resulting module ('tt.call' "
                "op does not reference a valid function). Confirmed via "
                "direct `compile_to_ttir` + `make_ktir_mod` invocation "
                "-- a genuine KTIR-lowering-pipeline gap for bundled "
                "multi-function kernels, not a numerical-oracle "
                "mismatch or extraction artifact."
            ),
        },
    },
}
