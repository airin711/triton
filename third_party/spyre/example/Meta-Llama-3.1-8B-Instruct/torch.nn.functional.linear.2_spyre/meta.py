"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.nn.functional.linear.2_spyre``.

Same bundled-kernel structure as ``.1_spyre`` (see that folder's
``meta.py`` docstring for the full field-by-field rationale); only the
weight's `out_features` (N) dimension differs: N=1024 here vs. N=4096
there. M=12, K=4096, N=1024 (``assert_size_stride(arg0_1 [weight], (1024,
4096), ...)``, ``assert_size_stride(arg1_1 [activation], (1, 12, 4096),
...)``). ``spyre_grid: (32,)``.

Per-pointer device layout (plain C-contiguous reshapes, no permutation
at this level — see ``kernel.py``):

- ``in_ptr0``  (raw weight,     logical ``[1024, 4096]`` = N x K) -> device ``[64, 1024, 64]``
- ``in_ptr1``  (raw activation, logical ``[12, 4096]``   = M x K) -> device ``[12, 1, 64, 64]``
- ``out_ptr0`` (scratch)                                          -> device ``[16, 4096, 64]``
- ``out_ptr1`` (raw output,     logical ``[12, 1024]``   = M x N) -> device ``[12, 1, 16, 64]``
"""

import numpy as np

from . import kernel


M, K, N = 12, 4096, 1024
WEIGHT_DEV_SHAPE = (64, 1024, 64)   # N*K = 4194304
ACT_DEV_SHAPE = (M, 1, 64, 64)      # M*K = 49152
SCRATCH_DEV_SHAPE = (16, 4096, 64)  # N*K = 4194304
OUT_DEV_SHAPE = (M, 1, 16, 64)      # M*N = 12288


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
            "traced `torch.nn.functional.linear.2` op: M=12, K=4096, N=1024."
        ),
        "doc": (
            "Same structure as `.1_spyre`. kernel_0's XBLOCK=131072 is a "
            "power of 2, but kernel_1's YBLOCK=6 is not: the dead "
            "`yindex = yoffset + tl.arange(0, YBLOCK)[...]` line (never "
            "consumed by any load/store) still executes at Triton frontend "
            "compile time and trips the unconditional power-of-2 gate in "
            "`arange()`, so this kernel fails to compile even to TTIR."
        ),
        "kernel_fn":  kernel.triton_bundle_0,
        "constexpr":  [],
        "params":     {},
        "grid":       [32],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr1",
        # triton_bundle_0's own body calls kernel_1 with a literal
        # YBLOCK=6 (non-power-of-2) baked into the call site (see
        # kernel.py: `triton_bundle_0_kernel_1(in_ptr1, out_ptr0,
        # out_ptr1, 12, 1024, 4096, 6, 64)`) -- the same category of bug
        # as `.1_spyre`, just a different YBLOCK value. Not fixable via
        # meta.py: the compiled entry (`triton_bundle_0`) exposes no
        # constexpr, only the 4 pointer args.
        "disabled": {
            "reason": (
                "triton_bundle_0's call site passes kernel_1 a literal "
                "YBLOCK=6 (non-power-of-2, baked into kernel.py's own "
                "source text, not an overridable meta.py constexpr); "
                "that value feeds a dead `yindex = yoffset + "
                "tl.arange(0, YBLOCK)[...]` line (never consumed by any "
                "load/store), but Triton's frontend `arange()` "
                "unconditionally rejects non-power-of-2 ranges "
                "regardless of whether the result is used, so the "
                "kernel fails to compile even to TTIR -- confirmed by "
                "direct `compile_to_ttir` invocation."
            ),
        },
    },
}
