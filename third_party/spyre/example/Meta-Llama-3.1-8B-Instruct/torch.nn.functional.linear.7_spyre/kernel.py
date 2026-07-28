"""``torch.nn.functional.linear.7_spyre`` kernel extracted from a
torch-spyre Inductor trace.

The three ``@triton.jit`` functions verbatim (lines 61-142 of the
``async_compile.triton(...)`` source string) from torch-spyre's Inductor
output for Meta-Llama-3.1-8B-Instruct's traced ``torch.nn.functional.linear.7``
op — ``aten.bmm`` between a logical ``f16[1, 1, 4096]`` activation
(unsqueezed/expanded) and a logical ``f16[1, 4096, 1024]`` weight
(``arg1_1`` permuted ``[1, 0]``, unsqueezed, expanded), i.e.
``nn.functional.linear`` without bias, ``out = x @ weight.T``; see
``torch-spyre/test_results_triton_20260714_120532/
torch.nn.functional.linear.7_spyre/torch_compile_debug/
run_2026_07_14_13_09_19_055718-pid_270907/torchinductor/
model__0_inference_0.0/output_code.py``.

This is a *bundled* multi-kernel trace: the top-level entry
``triton_bundle_0`` sequentially calls two ``noinline=True`` helpers; it
never appears alone in isolation in the traced output:

- ``triton_bundle_0_kernel_0`` — repacks the raw weight tensor
  (``arg1_1``, logical shape ``[1024, 4096]``) into a device-friendly
  tiled layout (via ``tl.reshape``/``tl.permute``/``tl.reshape``) — a
  pure device-layout re-tiling of the same logical values (confirmed by
  ``ir_post_fusion.txt``'s ``op1_loop_body``: a straight elementwise copy
  ``buf1[4096*p0+p1] = arg1_1[4096*p0+p1]``, no transpose at the
  scheduler-IR level — the logical transpose implied by
  ``aten.permute.default(arg1_1, [1, 0])`` in the traced fx graph is
  folded into how the matmul helper indexes the repacked buffer, not
  into this copy).
- ``triton_bundle_0_kernel_1`` — the actual matmul: loads a tile of the
  activation and a tile of the repacked weight, both cast to
  ``float16``, contracts them with a single ``tl.dot(..., input_precision
  ="ieee")``, casts the result back to ``float16``, and stores it.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)``
decorator that wraps ``triton_bundle_0`` in the Inductor output. It only
carries autotuning/Inductor-side metadata (``config={}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a
dependency; the ``@triton.jit`` function it wraps is unchanged either way.
This test suite compiles ``kernel_fn`` (the top-level entry) directly via
``ASTSource`` (see ``meta.py``) — Triton transparently pulls in the two
``noinline`` helpers it calls, never going through that decorator.
"""

import triton
import triton.language as tl


@triton.jit(noinline=True)
def triton_bundle_0_kernel_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    # Triton -> Logical layouts
    c0 = 0
    c1 = (tl.program_id(0)) * 128
    # Logical layouts -> Device layouts
    dim0 = c1 // 64
    dim1 = c0
    dim2 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[64, 1024, 64], strides=[65536, 64, 1], block_shape=[2, 1024, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c0 // 64
    dim_1_1 = c1
    dim_1_2 = c0 % 64
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[16, 4096, 64], strides=[262144, 64, 1], block_shape=[16, 128, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp1 = tl.reshape(tmp0, [2, 16, 64, 64])
    tmp2 = tl.permute(tmp1, [1, 0, 3, 2])
    tmp3 = tl.reshape(tmp2, [16, 128, 64])
    desc_1.store([dim_1_0, dim_1_1, dim_1_2], tmp3)

@triton.jit(noinline=True)
def triton_bundle_0_kernel_1(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
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
    c0 = xoffset
    c1 = r0_offset
    # Logical layouts -> Device layouts
    dim0 = 0
    dim1 = 0
    dim2 = c1 // 64
    dim3 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 1, 64, 64], strides=[4096, 64, 64, 1], block_shape=[1, 1, 32, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c1
    dim_1_1 = c0 // 64
    dim_1_2 = c0 % 64
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[4096, 16, 64], strides=[64, 262144, 1], block_shape=[2048, 1, 64])
    # Logical layouts -> Device layouts
    dim_2_0 = 0
    dim_2_1 = 0
    dim_2_2 = c0 // 64
    dim_2_3 = c0 % 64
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[1, 1, 16, 64], strides=[1024, 64, 64, 1], block_shape=[1, 1, 1, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp2 = desc_1.load([dim_1_0, dim_1_1, dim_1_2])
    tmp1 = tmp0.to(tl.float16)
    tmp3 = tmp2.to(tl.float16)
    tmp4 = tl.reshape(tmp1, [1, 1, 2048])
    tmp5 = tl.reshape(tmp3, [2048, 64])
    tmp6 = tl.reshape(tmp4, [1, 2048])
    tmp7 = tl.dot(tmp6, tmp5, input_precision="ieee")
    tmp8 = tmp7.to(tl.float16)
    tmp9 = tl.reshape(tmp8, [1, 1, 1, 64])
    desc_2.store([dim_2_0, dim_2_1, dim_2_2, dim_2_3], tmp9)

@triton.jit
def triton_bundle_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1):
    triton_bundle_0_kernel_0(in_ptr0, out_ptr0, 4194304, 131072)
    triton_bundle_0_kernel_1(in_ptr1, out_ptr0, out_ptr1, 1024, 4096, 64)
