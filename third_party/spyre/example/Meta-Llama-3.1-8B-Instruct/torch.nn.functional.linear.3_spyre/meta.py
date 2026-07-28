"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.nn.functional.linear.3_spyre``.

Same bundled-kernel structure as ``.1_spyre`` (see that folder's
``meta.py`` docstring for the full field-by-field rationale). M=12,
K=4096, N=14336 (MLP up-projection width). ``spyre_grid: (32,)``.

Per-pointer device layout (plain C-contiguous reshapes, no permutation
at this level — see ``kernel.py``):

- ``in_ptr0``  (raw weight,     logical ``[14336, 4096]`` = N x K) -> device ``[64, 14336, 64]``
- ``in_ptr1``  (raw activation, logical ``[12, 4096]``    = M x K) -> device ``[12, 1, 64, 64]``
- ``out_ptr0`` (scratch)                                           -> device ``[224, 4096, 64]``
- ``out_ptr1`` (raw output,     logical ``[12, 14336]``   = M x N) -> device ``[12, 1, 224, 64]``
"""

import numpy as np

from . import kernel


M, K, N = 12, 4096, 14336
WEIGHT_DEV_SHAPE = (64, 14336, 64)   # N*K = 58720256
ACT_DEV_SHAPE = (M, 1, 64, 64)       # M*K = 49152
SCRATCH_DEV_SHAPE = (224, 4096, 64)  # N*K = 58720256
OUT_DEV_SHAPE = (M, 1, 224, 64)      # M*N = 172032


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
            "traced `torch.nn.functional.linear.3` op: M=12, K=4096, "
            "N=14336 (MLP up-projection)."
        ),
        "doc": (
            "Same structure as `.1_spyre`. Here kernel_0's own XBLOCK is "
            "non-power-of-2 (1835008 = 58720256 / 32), so the (dead) "
            "`xindex = xoffset + tl.arange(0, XBLOCK)[:]` line in "
            "kernel_0 — which runs *first* inside the entry — trips the "
            "power-of-2 gate before kernel_1 is even reached. Fails to "
            "compile to TTIR at the very first bundled sub-kernel."
        ),
        "kernel_fn":  kernel.triton_bundle_0,
        "constexpr":  [],
        "params":     {},
        "grid":       [32],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr1",
        # triton_bundle_0's own body calls kernel_0 with a literal
        # XBLOCK=1835008 (non-power-of-2) baked into the call site (see
        # kernel.py: `triton_bundle_0_kernel_0(in_ptr0, out_ptr0,
        # 58720256, 1835008)`) -- kernel_0 runs first inside the entry,
        # so this trips the frontend before kernel_1 is even reached.
        # Not fixable via meta.py: the compiled entry (`triton_bundle_0`)
        # exposes no constexpr, only the 4 pointer args.
        "disabled": {
            "reason": (
                "triton_bundle_0's call site passes kernel_0 a literal "
                "XBLOCK=1835008 (non-power-of-2, baked into kernel.py's "
                "own source text, not an overridable meta.py "
                "constexpr); that value feeds a dead "
                "`xindex = xoffset + tl.arange(0, XBLOCK)[:]` line "
                "(never consumed by any load/store), but Triton's "
                "frontend `arange()` unconditionally rejects "
                "non-power-of-2 ranges regardless of whether the result "
                "is used, so the kernel fails to compile even to TTIR "
                "-- confirmed by direct `compile_to_ttir` invocation."
            ),
        },
    },
}
