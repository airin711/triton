"""``torch.mean.1_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body (lines 75-110 of the ``async_compile.triton(...)``
source string) verbatim, from torch-spyre's Inductor output for
Meta-Llama-3.1-8B-Instruct's traced ``torch.mean.1`` op
(``aten.mean.dim(x, [-1], True)`` on a logical ``f16[1, 12, 4096]`` tensor,
reducing to ``f16[1, 12, 1]``); see
``torch-spyre/test_results_triton_20260714_120532/torch.mean.1_spyre/
torch_compile_debug/run_2026_07_14_12_19_32_086019-pid_257232/
torchinductor/model__0_inference_0.0/output_code.py``.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'XBLOCK': 1}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.

IMPORTANT (see ``meta.py`` docstring for the full analysis): despite
``r0_numel = 4096`` and ``R0_BLOCK: tl.constexpr = 4096`` suggesting a
full-row reduction, ``desc_0``'s ``block_shape=[1, 32, 1, 64]`` only spans
32 of the 64 "stick" positions in ``shape=[12, 64, 1, 64]`` (the physical
split of the logical 4096-element row into a 64-stick x 64-lane device
layout) — i.e. only 2048 of the 4096 logical elements are ever loaded by
this kernel. ``tl.mean(tmp1, 1)`` then reduces only that 32-element stick
axis, leaving the 64-element lane axis entirely unreduced in both the
compute and the store (``desc_1`` stores 64 values per row, not 1). This is
copied verbatim; the indexing/reduction math is untouched.
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused_mean_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 12
    r0_numel = 4096
    R0_BLOCK: tl.constexpr = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    # Triton -> Logical layouts
    c0 = tl.program_id(0)
    c1 = r0_offset
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = c1 // 64
    dim2 = 0
    dim3 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[12, 64, 1, 64], strides=[4096, 64, 64, 1], block_shape=[1, 32, 1, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = 0
    dim_1_1 = 0
    dim_1_2 = 0
    dim_1_3 = c0
    dim_1_4 = 0
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[1, 1, 1, 12, 64], strides=[768, 768, 768, 64, 1], block_shape=[1, 1, 1, 1, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.mean(tmp1, 1)
    tmp3 = tmp2.to(tl.float16)
    tmp4 = tl.reshape(tmp3, [1, 1, 1, 1, 64])
    desc_1.store([dim_1_0, dim_1_1, dim_1_2, dim_1_3, dim_1_4], tmp4)
