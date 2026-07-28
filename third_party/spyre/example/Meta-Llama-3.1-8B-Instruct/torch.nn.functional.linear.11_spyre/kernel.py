"""``torch.nn.functional.linear.11_spyre`` kernel extracted from a
torch-spyre Inductor trace.

The three ``@triton.jit`` functions verbatim from torch-spyre's Inductor
output for Meta-Llama-3.1-8B-Instruct's traced ``torch.nn.functional.linear.11``
op — ``aten.bmm`` between a logical ``f16[1, 1, 4096]`` activation
(unsqueezed/expanded) and a logical ``f16[1, 4096, 128256]`` weight
(``arg1_1`` permuted ``[1, 0]``, unsqueezed, expanded), i.e.
``nn.functional.linear`` without bias, ``out = x @ weight.T``; the
vocab-sized (``128256``) output feature dimension makes this the traced
LM-head / final-logits projection; see ``torch-spyre/
test_results_triton_20260714_120532/torch.nn.functional.linear.11_spyre/
torch_compile_debug/run_2026_07_14_13_35_53_762855-pid_276646/
torchinductor/model__0_inference_0.0/output_code.py``.

This is a *bundled* multi-kernel trace: the top-level entry
``triton_bundle_0`` sequentially calls two ``noinline=True`` helpers; it
never appears alone in isolation in the traced output:

- ``triton_bundle_0_kernel_0`` — repacks the raw weight tensor
  (``arg1_1``, logical shape ``[128256, 4096]``) into a device-friendly
  tiled layout (via ``tl.reshape``/``tl.permute``/``tl.reshape``). Its
  program-id indexing is 2D-flattened (``c0 = (pid // 4) * 32064``,
  ``c1 = (pid % 4) * 1024``), unlike the 1D indexing in ``linear.7``-
  ``.10``'s ``kernel_0`` — a consequence of ``spyre_grids['kernel_0'] ==
  (16,)`` needing to cover a larger logical tile than a single
  ``program_id(0)*step`` can express in one dimension here.
- ``triton_bundle_0_kernel_1`` — the actual matmul: loads a tile of the
  activation and a tile of the repacked weight, both cast to
  ``float16``, contracts them with a single ``tl.dot(..., input_precision
  ="ieee")``, casts the result back to ``float16``, and stores it. This
  kernel's activation descriptor ``block_shape`` (``[1, 1, 8, 64]``) is
  only **1/8** of its own ``shape`` (``[1, 1, 64, 64]``) along the K
  axis (``8*64 == 512`` of the true ``K=4096``) — the most severe
  K-axis truncation in this batch (``linear.7``'s is a half, ``.8``/
  ``.9``/``.10`` have none). The matching weight descriptor (``desc_1``)
  has the same ``512``-of-``4096`` truncation on its own reduction axis.
  As with ``linear.7``, the offset feeding that axis
  (``dim2/dim3 = c1 // 64, c1 % 64`` with ``c1 = r0_offset = 0``) is a
  Python-level compile-time constant, so — taken literally — every
  program on every K-tile only ever contracts over the *same* first 512
  of 4096 reduction elements (``tmp6`` reshaped to ``[1, 512]`` against
  ``tmp5`` reshaped to ``[512, 32064]``).
- ``spyre_grids: {'triton_bundle_0_kernel_0': (16,), 'triton_bundle_0_
  kernel_1': (4,)}``, top-level ``spyre_grid=(16,)`` — differs from the
  ``(32,)`` used by ``linear.7``-``.10`` in this batch.

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
    xnumel = 525336576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    # Triton -> Logical layouts
    c0 = ((tl.program_id(0) // 4)) * 32064
    c1 = ((tl.program_id(0) % 4)) * 1024
    # Logical layouts -> Device layouts
    dim0 = c1 // 64
    dim1 = c0
    dim2 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[64, 128256, 64], strides=[8208384, 64, 1], block_shape=[16, 32064, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c0 // 64
    dim_1_1 = c1
    dim_1_2 = c0 % 64
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[2004, 4096, 64], strides=[262144, 64, 1], block_shape=[501, 1024, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp1 = tl.reshape(tmp0, [16, 501, 64, 64])
    tmp2 = tl.permute(tmp1, [1, 0, 3, 2])
    tmp3 = tl.reshape(tmp2, [501, 1024, 64])
    desc_1.store([dim_1_0, dim_1_1, dim_1_2], tmp3)

@triton.jit(noinline=True)
def triton_bundle_0_kernel_1(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128256
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
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 1, 64, 64], strides=[4096, 64, 64, 1], block_shape=[1, 1, 8, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c1
    dim_1_1 = c0 // 64
    dim_1_2 = c0 % 64
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[4096, 2004, 64], strides=[64, 262144, 1], block_shape=[512, 501, 64])
    # Logical layouts -> Device layouts
    dim_2_0 = 0
    dim_2_1 = 0
    dim_2_2 = c0 // 64
    dim_2_3 = c0 % 64
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[1, 1, 2004, 64], strides=[128256, 64, 64, 1], block_shape=[1, 1, 501, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp2 = desc_1.load([dim_1_0, dim_1_1, dim_1_2])
    tmp1 = tmp0.to(tl.float16)
    tmp3 = tmp2.to(tl.float16)
    tmp4 = tl.reshape(tmp1, [1, 1, 512])
    tmp5 = tl.reshape(tmp3, [512, 32064])
    tmp6 = tl.reshape(tmp4, [1, 512])
    tmp7 = tl.dot(tmp6, tmp5, input_precision="ieee")
    tmp8 = tmp7.to(tl.float16)
    tmp9 = tl.reshape(tmp8, [1, 1, 501, 64])
    desc_2.store([dim_2_0, dim_2_1, dim_2_2, dim_2_3], tmp9)

@triton.jit
def triton_bundle_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1):
    triton_bundle_0_kernel_0(in_ptr0, out_ptr0, 525336576, 32833536)
    triton_bundle_0_kernel_1(in_ptr1, out_ptr0, out_ptr1, 128256, 4096, 32064)
