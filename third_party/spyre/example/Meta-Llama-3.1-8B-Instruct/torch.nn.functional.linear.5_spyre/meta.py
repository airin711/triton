"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.nn.functional.linear.5_spyre``.

Same bundled-kernel structure as ``.1_spyre`` (see that folder's
``meta.py`` docstring for the full field-by-field rationale). M=1 (a
single decode-step token), K=4096, N=128256 (the LM-head projection to
the full vocabulary). ``spyre_grids: {kernel_0: (16,), kernel_1: (4,)}``,
overall ``spyre_grid: (16,)`` (the entry's actual launch grid is sized to
the larger of the two — kernel_1's own extra idle programs, id 4..15,
are masked out by its `xmask = xindex < xnumel`).

Per-pointer device layout (plain C-contiguous reshapes, no permutation
at this level — see ``kernel.py``):

- ``in_ptr0``  (raw weight,     logical ``[128256, 4096]`` = N x K) -> device ``[64, 128256, 64]``
- ``in_ptr1``  (raw activation, logical ``[1, 4096]``       = M x K) -> device ``[1, 1, 64, 64]``
- ``out_ptr0`` (scratch)                                            -> device ``[2004, 4096, 64]``
- ``out_ptr1`` (raw output,     logical ``[1, 128256]``     = M x N) -> device ``[1, 1, 2004, 64]``
"""

import numpy as np

from . import kernel


M, K, N = 1, 4096, 128256
WEIGHT_DEV_SHAPE = (64, 128256, 64)   # N*K = 525336576
ACT_DEV_SHAPE = (M, 1, 64, 64)        # M*K = 4096
SCRATCH_DEV_SHAPE = (2004, 4096, 64)  # N*K = 525336576
OUT_DEV_SHAPE = (M, 1, 2004, 64)      # M*N = 128256


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
            "traced `torch.nn.functional.linear.5` op: M=1 (decode step), "
            "K=4096, N=128256 (LM head)."
        ),
        "doc": (
            "Same structure as `.1_spyre`, but M=1 collapses kernel_1's "
            "y-tree entirely (no `ynumel`/`YBLOCK` in its signature — only "
            "an x (N) tile and an r0 (K) reduction). kernel_0's own XBLOCK "
            "is non-power-of-2 (32833536 = 525336576 / 16), so the (dead) "
            "`xindex = xoffset + tl.arange(0, XBLOCK)[:]` line in "
            "kernel_0 — which runs *first* inside the entry — trips the "
            "power-of-2 gate before kernel_1 is even reached (kernel_1's "
            "own XBLOCK=32064 is also non-power-of-2, so it would fail "
            "the same way even if kernel_0 were fixed)."
        ),
        "kernel_fn":  kernel.triton_bundle_0,
        "constexpr":  [],
        "params":     {},
        "grid":       [16],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr1",
        # triton_bundle_0's own body calls kernel_0 with a literal
        # XBLOCK=32833536 (non-power-of-2) baked into the call site (see
        # kernel.py: `triton_bundle_0_kernel_0(in_ptr0, out_ptr0,
        # 525336576, 32833536)`) -- kernel_0 runs first, so this trips
        # the frontend before kernel_1 is reached. kernel_1's own call
        # site (`..., 128256, 4096, 32064`) also passes a
        # non-power-of-2 XBLOCK=32064, so this would fail the same way
        # even if kernel_0's literal were somehow worked around. Neither
        # is fixable via meta.py: the compiled entry (`triton_bundle_0`)
        # exposes no constexpr, only the 4 pointer args.
        "disabled": {
            "reason": (
                "triton_bundle_0's call site passes kernel_0 a literal "
                "XBLOCK=32833536 (non-power-of-2, baked into kernel.py's "
                "own source text, not an overridable meta.py "
                "constexpr); that value feeds a dead "
                "`xindex = xoffset + tl.arange(0, XBLOCK)[:]` line "
                "(never consumed by any load/store), but Triton's "
                "frontend `arange()` unconditionally rejects "
                "non-power-of-2 ranges regardless of whether the result "
                "is used, so the kernel fails to compile even to TTIR "
                "-- confirmed by direct `compile_to_ttir` invocation. "
                "kernel_1's own call-site XBLOCK=32064 is independently "
                "non-power-of-2 too, so this kernel would fail the same "
                "way even if kernel_0's literal were fixed."
            ),
        },
    },
}
