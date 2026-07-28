"""``torch.nn.functional.linear.6_spyre`` kernel extracted from a
torch-spyre Inductor trace.

The three ``@triton.jit`` functions verbatim (lines 61-142 of the
``async_compile.triton(...)`` source string) from torch-spyre's Inductor
output for Meta-Llama-3.1-8B-Instruct's traced ``torch.nn.functional.linear.6``
op (``aten.mm`` between a logical ``f16[1, 1, 4096]`` activation — a
single decode-step token, M=1 — and a logical ``f16[4096, 4096]`` weight,
i.e. `nn.functional.linear` without bias, the same square weight shape as
``.1_spyre`` but for a single-token decode step instead of a 12-token
prefill); see ``torch-spyre/test_results_triton_20260714_120532/
torch.nn.functional.linear.6_spyre/torch_compile_debug/
run_2026_07_14_13_06_54_770622-pid_270398/torchinductor/
model__0_inference_0.0/output_code.py``.

Bundled multi-kernel trace, same structure as ``torch.nn.functional.linear.1
_spyre`` (see that folder's ``kernel.py`` docstring for the full rationale):
``triton_bundle_0_kernel_0`` repacks the raw weight into a device-tiled
layout, ``triton_bundle_0_kernel_1`` is the ``tl.dot``-based matmul, and
``triton_bundle_0`` is the entry that calls both in sequence. Like
``.5_spyre`` (also M=1), ``triton_bundle_0_kernel_1`` has no y-tree
(``ynumel``/``YBLOCK`` absent from its signature) — only x (N) and r0 (K).

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)``
decorator that wraps ``triton_bundle_0`` in the Inductor output (autotuning/
Inductor-side metadata only; the compiled ``@triton.jit`` function is
unchanged). This test suite compiles ``kernel_fn`` (the top-level entry)
directly via ``ASTSource`` (see ``meta.py``) — Triton transparently pulls in
the two ``noinline`` helpers it calls.
"""

import triton
import triton.language as tl


@triton.jit(noinline=True)
def triton_bundle_0_kernel_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    # Triton -> Logical layouts
    c0 = (tl.program_id(0)) * 128
    c1 = 0
    # Logical layouts -> Device layouts
    dim0 = c1 // 64
    dim1 = c0
    dim2 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[64, 4096, 64], strides=[262144, 64, 1], block_shape=[64, 128, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c0 // 64
    dim_1_1 = c1
    dim_1_2 = c0 % 64
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[64, 4096, 64], strides=[262144, 64, 1], block_shape=[2, 4096, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp1 = tl.reshape(tmp0, [64, 2, 64, 64])
    tmp2 = tl.permute(tmp1, [1, 0, 3, 2])
    tmp3 = tl.reshape(tmp2, [2, 4096, 64])
    desc_1.store([dim_1_0, dim_1_1, dim_1_2], tmp3)

@triton.jit(noinline=True)
def triton_bundle_0_kernel_1(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 4096
    R0_BLOCK: tl.constexpr = 4096
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    # Triton -> Logical layouts
    c0 = xoffset
    c1 = r0_offset
    # Logical layouts -> Device layouts
    dim0 = 0
    dim1 = 0
    dim2 = c1 // 64
    dim3 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 1, 64, 64], strides=[4096, 64, 64, 1], block_shape=[1, 1, 64, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c1
    dim_1_1 = c0 // 64
    dim_1_2 = c0 % 64
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[4096, 64, 64], strides=[64, 262144, 1], block_shape=[4096, 2, 64])
    # Logical layouts -> Device layouts
    dim_2_0 = 0
    dim_2_1 = 0
    dim_2_2 = c0 // 64
    dim_2_3 = c0 % 64
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[1, 1, 64, 64], strides=[4096, 64, 64, 1], block_shape=[1, 1, 2, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp2 = desc_1.load([dim_1_0, dim_1_1, dim_1_2])
    tmp1 = tmp0.to(tl.float16)
    tmp3 = tmp2.to(tl.float16)
    tmp4 = tl.reshape(tmp1, [1, 1, 4096])
    tmp5 = tl.reshape(tmp3, [4096, 128])
    tmp6 = tl.reshape(tmp4, [1, 4096])
    tmp7 = tl.dot(tmp6, tmp5, input_precision="ieee")
    tmp8 = tmp7.to(tl.float16)
    tmp9 = tl.reshape(tmp8, [1, 1, 2, 64])
    desc_2.store([dim_2_0, dim_2_1, dim_2_2, dim_2_3], tmp9)

@triton.jit
def triton_bundle_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1):
    triton_bundle_0_kernel_0(in_ptr0, out_ptr0, 16777216, 524288)
    triton_bundle_0_kernel_1(in_ptr1, out_ptr0, out_ptr1, 4096, 4096, 128)
