"""SIGNATURE + VARIANTS + reference oracle + input generator for
``torch.nn.functional.linear.1_spyre``.

Caller-side settings (pointer shapes/dtypes/grid) are taken from the
``Runner.call`` method and the bundled-kernel metadata in the traced
``output_code.py`` (see ``kernel.py`` docstring for the exact path):

- ``assert_size_stride(arg0_1 [weight], (4096, 4096), (4096, 1))``,
  ``assert_size_stride(arg1_1 [activation], (1, 12, 4096), (49152, 4096, 1))``
  — logical shapes, dtype ``torch.float16``. M=12, K=4096, N=4096
  (square weight; out_features == in_features here).
- ``triton_bundle_0.run(arg0_1, arg1_1, buf0, buf1, stream=stream0)`` —
  the entry takes exactly 4 raw ``*fp16`` pointers, no scalar/constexpr
  args (the tile constants are baked as literals into the call sites
  inside the entry body — see ``kernel.py``).
- ``triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16',
  'out_ptr0': '*fp16', 'out_ptr1': '*fp16'}}``.
- ``spyre_grid: (32,)`` — both bundled kernels are sized for a 32-program
  grid (``triton_bundle_0_kernel_0``: 16777216 / 524288 = 32;
  ``triton_bundle_0_kernel_1``: 4096 / 128 = 32, with its y-axis
  (``ynumel=12``, ``YBLOCK=12``) needing only 1 program on axis 1).

Per-pointer device layout (all four are *plain C-contiguous reshapes* of
the logical row-major tensors — confirmed by matching each tensor
descriptor's ``shape=``/``strides=`` against the logical tensor's own
row-major strides; no permutation happens at this level, only inside
``kernel_0``'s body):

- ``in_ptr0``  (raw weight,     logical ``[4096, 4096]``   = N x K) -> device ``[64, 4096, 64]``
- ``in_ptr1``  (raw activation, logical ``[12, 4096]``      = M x K) -> device ``[12, 1, 64, 64]``
- ``out_ptr0`` (scratch: repacked weight, written by kernel_0, read by
  kernel_1 through a *different* descriptor shape over the same flat
  buffer) -> device ``[64, 4096, 64]`` (kernel_0's own write view; sized
  N*K total, reinterpreted by kernel_1 as ``[4096, 64, 64]``, same total)
- ``out_ptr1`` (raw output,     logical ``[12, 4096]``      = M x N) -> device ``[12, 1, 64, 64]``
"""

import numpy as np

from . import kernel


# ---------------------------------------------------------------------------
# Shape constants
# ---------------------------------------------------------------------------

M, K, N = 12, 4096, 4096
WEIGHT_DEV_SHAPE = (64, 4096, 64)   # N*K = 16777216
ACT_DEV_SHAPE = (M, 1, 64, 64)      # M*K = 49152
SCRATCH_DEV_SHAPE = (64, 4096, 64)  # N*K = 16777216
OUT_DEV_SHAPE = (M, 1, 64, 64)      # M*N = 49152


# ---------------------------------------------------------------------------
# Reference (NumPy oracle) + input maker
# ---------------------------------------------------------------------------

def make_inputs(**_unused) -> dict:
    """Build the four raw ``*fp16`` pointer buffers for the entry function.

    ``in_ptr0``/``in_ptr1`` hold random logical weight/activation data
    reshaped (no permutation — see module docstring) into the device
    shape each tensor descriptor declares. ``out_ptr0`` (scratch) and
    ``out_ptr1`` (final output) are zero-initialized.
    """
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
    """NumPy oracle: ``nn.functional.linear(x, weight)`` (no bias), i.e.
    ``out = x @ weight.T``, computed in the kernel's own compute precision
    (extend both operands to f32 for the contraction, truncate the result
    back to f16 — matching the traced kernel's ``tl.dot(...,
    input_precision="ieee")`` + immediate ``.to(tl.float16)``)."""
    weight = inputs["in_ptr0"].reshape(N, K).astype(np.float32)
    x = inputs["in_ptr1"].reshape(M, K).astype(np.float32)
    out = x @ weight.T
    return out.astype(np.float16).reshape(OUT_DEV_SHAPE)


# ---------------------------------------------------------------------------
# SIGNATURE — dtype per @triton.jit arg, taken from the decorator's
# ``triton_meta['signature']`` in output_code.py. The entry (``triton_
# bundle_0``) takes only the four raw pointers — no constexpr/scalar args.
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
        "tags": ["bundled-kernel", "tl-dot", "descriptor-load-static", "descriptor-store-static"],
        "summary": (
            "`nn.functional.linear` (no bias) on Meta-Llama-3.1-8B-Instruct's "
            "traced `torch.nn.functional.linear.1` op: M=12, K=4096, N=4096."
        ),
        "doc": (
            "Bundled trace: `triton_bundle_0_kernel_0` repacks the raw "
            "weight into a device-tiled layout via reshape/permute/reshape, "
            "then `triton_bundle_0_kernel_1` loads a tile of the repacked "
            "weight and a tile of the activation and contracts them with a "
            "single `tl.dot(..., input_precision=\"ieee\")`. All constexpr "
            "tile sizes here (XBLOCK=524288 in kernel_0; YBLOCK=12, "
            "XBLOCK=128, R0_BLOCK=4096 in kernel_1) are powers of 2 "
            "*except* YBLOCK=12 — a dead `yindex = yoffset + tl.arange(0, "
            "YBLOCK)[...]` line (never consumed by any load/store; real "
            "row-indexing runs through `c0`/`yoffset` directly into the "
            "tensor descriptors) that nonetheless executes at Triton "
            "frontend compile time and trips the unconditional power-of-2 "
            "gate in `arange()`. Production Spyre Inductor codegen "
            "(`spyre_triton_kernel.py`'s `codegen_range_tree`) strips this "
            "exact class of dead boilerplate before compilation; the "
            "verbatim `output_code.py` debug dump extracted here still "
            "contains it, so this kernel fails to compile even to TTIR via "
            "the standard `ASTSource` frontend used by this test suite."
        ),
        "kernel_fn":  kernel.triton_bundle_0,
        "constexpr":  [],
        "params":     {},
        "grid":       [32],
        "reference":  run,
        "inputs":     make_inputs,
        "output_key": "out_ptr1",
        # triton_bundle_0's own body calls kernel_1 with a literal
        # YBLOCK=12 (non-power-of-2) baked into the call site (see
        # kernel.py: `triton_bundle_0_kernel_1(in_ptr1, out_ptr0,
        # out_ptr1, 12, 4096, 4096, 12, 128)`), unlike the non-bundled
        # examples in this suite where XBLOCK is a real constexpr on the
        # compiled entry that meta.py's `params` can substitute -- the
        # compiled entry here (`triton_bundle_0`) takes only 4 pointers,
        # no constexpr, so this literal cannot be overridden without
        # editing kernel.py's verbatim traced body.
        "disabled": {
            "reason": (
                "triton_bundle_0's call site passes kernel_1 a literal "
                "YBLOCK=12 (non-power-of-2, baked into kernel.py's own "
                "source text, not an overridable meta.py constexpr); "
                "that value feeds a dead `yindex = yoffset + "
                "tl.arange(0, YBLOCK)[...]` line (never consumed by any "
                "load/store -- real row-indexing runs through "
                "c0/yoffset directly into the tensor descriptors), but "
                "Triton's frontend `arange()` unconditionally rejects "
                "non-power-of-2 ranges regardless of whether the result "
                "is used, so the kernel fails to compile even to TTIR "
                "-- confirmed by direct `compile_to_ttir` invocation."
            ),
        },
    },
}
