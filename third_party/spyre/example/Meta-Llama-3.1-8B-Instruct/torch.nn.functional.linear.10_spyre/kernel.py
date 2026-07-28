"""``torch.nn.functional.linear.10_spyre`` kernel extracted from a
torch-spyre Inductor trace.

The three ``@triton.jit`` functions verbatim from torch-spyre's Inductor
output for Meta-Llama-3.1-8B-Instruct's traced ``torch.nn.functional.linear.10``
op — ``aten.bmm`` between a logical ``f16[1, 1, 14336]`` activation
(unsqueezed/expanded) and a logical ``f16[1, 14336, 4096]`` weight
(``arg1_1`` permuted ``[1, 0]``, unsqueezed, expanded), i.e.
``nn.functional.linear`` without bias, ``out = x @ weight.T``; see
``torch-spyre/test_results_triton_20260714_120532/
torch.nn.functional.linear.10_spyre/torch_compile_debug/
run_2026_07_14_13_34_12_548817-pid_276275/torchinductor/
model__0_inference_0.0/output_code.py``.

This is a *bundled* multi-kernel trace: the top-level entry
``triton_bundle_0`` sequentially calls two ``noinline=True`` helpers; it
never appears alone in isolation in the traced output:

- ``triton_bundle_0_kernel_0`` — repacks the raw weight tensor
  (``arg1_1``, logical shape ``[4096, 14336]``) into a device-friendly
  tiled layout (via ``tl.reshape``/``tl.permute``/``tl.reshape``). Note
  the ``c0``/``c1`` roles are swapped relative to ``linear.7``/``.8``/
  ``.9`` (``c0 = 0``, ``c1 = program_id(0) * 448``, rather than
  ``c0 = program_id(0) * ..., c1 = 0``) — a consequence of this weight's
  transposed logical shape (``[4096, 14336]`` vs. the other kernels'
  ``[N, 4096]``), not a bug.
- ``triton_bundle_0_kernel_1`` — the actual matmul: loads a tile of the
  activation and a tile of the repacked weight, both cast to
  ``float16``, contracts them with a single ``tl.dot(..., input_precision
  ="ieee")``, casts the result back to ``float16``, and stores it. This
  kernel's reduction axis is ``K=14336`` (not 4096, since this op's
  in_features is 14336): its activation descriptor ``block_shape``
  (``[1, 1, 224, 64]``) is fully equal to its own ``shape``
  (``224*64 == 14336``) — no truncation; note also the unusual
  ``R0_BLOCK: tl.constexpr = 16384`` (padded up from ``r0_numel=14336``)
  with a genuine ``r0_mask = r0_index < r0_numel`` (unlike the other
  kernels in this batch, which all have ``R0_BLOCK == r0_numel`` and a
  trivial all-``True`` mask) — this is Inductor persistent-reduction
  register-tile padding to a rounder block size; it does not affect the
  tensor-descriptor loads, which are still sized exactly to ``K=14336``.

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
    xnumel = 58720256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    # Triton -> Logical layouts
    c0 = 0
    c1 = (tl.program_id(0)) * 448
    # Logical layouts -> Device layouts
    dim0 = c1 // 64
    dim1 = c0
    dim2 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[224, 4096, 64], strides=[262144, 64, 1], block_shape=[7, 4096, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c0 // 64
    dim_1_1 = c1
    dim_1_2 = c0 % 64
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[64, 14336, 64], strides=[917504, 64, 1], block_shape=[64, 448, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2])
    tmp1 = tl.reshape(tmp0, [7, 64, 64, 64])
    tmp2 = tl.permute(tmp1, [1, 0, 3, 2])
    tmp3 = tl.reshape(tmp2, [64, 448, 64])
    desc_1.store([dim_1_0, dim_1_1, dim_1_2], tmp3)

@triton.jit(noinline=True)
def triton_bundle_0_kernel_1(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 14336
    R0_BLOCK: tl.constexpr = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
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
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 1, 224, 64], strides=[14336, 64, 64, 1], block_shape=[1, 1, 224, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c1
    dim_1_1 = c0 // 64
    dim_1_2 = c0 % 64
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[14336, 64, 64], strides=[64, 917504, 1], block_shape=[14336, 2, 64])
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
    tmp4 = tl.reshape(tmp1, [1, 1, 14336])
    tmp5 = tl.reshape(tmp3, [14336, 128])
    tmp6 = tl.reshape(tmp4, [1, 14336])
    tmp7 = tl.dot(tmp6, tmp5, input_precision="ieee")
    tmp8 = tmp7.to(tl.float16)
    tmp9 = tl.reshape(tmp8, [1, 1, 2, 64])
    desc_2.store([dim_2_0, dim_2_1, dim_2_2, dim_2_3], tmp9)

@triton.jit
def triton_bundle_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1):
    triton_bundle_0_kernel_0(in_ptr0, out_ptr0, 58720256, 1835008)
    triton_bundle_0_kernel_1(in_ptr1, out_ptr0, out_ptr1, 4096, 14336, 128)
