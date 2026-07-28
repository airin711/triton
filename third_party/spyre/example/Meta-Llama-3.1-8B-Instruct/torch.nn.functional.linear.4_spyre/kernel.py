"""``torch.nn.functional.linear.4_spyre`` kernel extracted from a
torch-spyre Inductor trace.

The three ``@triton.jit`` functions verbatim (lines 61-147 of the
``async_compile.triton(...)`` source string) from torch-spyre's Inductor
output for Meta-Llama-3.1-8B-Instruct's traced ``torch.nn.functional.linear.4``
op (``aten.mm`` between a logical ``f16[1, 12, 14336]`` activation and a
logical ``f16[4096, 14336]`` weight, i.e. `nn.functional.linear` without
bias — the MLP down-projection, K=14336, N=4096, the mirror image of
``.3_spyre``'s up-projection); see ``torch-spyre/test_results_triton_20260714
_120532/torch.nn.functional.linear.4_spyre/torch_compile_debug/
run_2026_07_14_12_48_49_226897-pid_266646/torchinductor/
model__0_inference_0.0/output_code.py``.

Bundled multi-kernel trace, same structure as ``torch.nn.functional.linear.1
_spyre`` (see that folder's ``kernel.py`` docstring for the full rationale):
``triton_bundle_0_kernel_0`` repacks the raw weight into a device-tiled
layout, ``triton_bundle_0_kernel_1`` is the ``tl.dot``-based matmul, and
``triton_bundle_0`` is the entry that calls both in sequence. Note
``r0_numel``/``R0_BLOCK`` differ here (14336 / 16384, i.e. the reduction
(K) dimension is *not* a multiple of the 4096-wide R0_BLOCK padding used
by ``.1_spyre``-``.3_spyre``): K=14336 rounds up to R0_BLOCK=16384, but the
tensor-descriptor block shapes used for the actual loads/dot are sized to
the exact K=14336 (``desc_0``/``desc_1`` block shapes both total 14336),
so this does not introduce a K-reduction loop — it is still a single
``tl.dot`` call.

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
def triton_bundle_0_kernel_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4096
    r0_numel = 14336
    R0_BLOCK: tl.constexpr = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[None, :, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    # Triton -> Logical layouts
    c0 = yoffset
    c1 = xoffset
    c2 = r0_offset
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = 0
    dim2 = c2 // 64
    dim3 = c2 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[12, 1, 224, 64], strides=[14336, 64, 64, 1], block_shape=[12, 1, 224, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c2
    dim_1_1 = c1 // 64
    dim_1_2 = c1 % 64
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[14336, 64, 64], strides=[64, 917504, 1], block_shape=[14336, 2, 64])
    # Logical layouts -> Device layouts
    dim_2_0 = c0
    dim_2_1 = 0
    dim_2_2 = c1 // 64
    dim_2_3 = c1 % 64
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[12, 1, 64, 64], strides=[4096, 64, 64, 1], block_shape=[12, 1, 2, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp2 = desc_1.load([dim_1_0, dim_1_1, dim_1_2])
    tmp1 = tmp0.to(tl.float16)
    tmp3 = tmp2.to(tl.float16)
    tmp4 = tl.reshape(tmp1, [12, 1, 14336])
    tmp5 = tl.reshape(tmp3, [14336, 128])
    tmp6 = tl.reshape(tmp4, [12, 14336])
    tmp7 = tl.dot(tmp6, tmp5, input_precision="ieee")
    tmp8 = tmp7.to(tl.float16)
    tmp9 = tl.reshape(tmp8, [12, 1, 2, 64])
    desc_2.store([dim_2_0, dim_2_1, dim_2_2, dim_2_3], tmp9)

@triton.jit
def triton_bundle_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1):
    triton_bundle_0_kernel_0(in_ptr0, out_ptr0, 58720256, 1835008)
    triton_bundle_0_kernel_1(in_ptr1, out_ptr0, out_ptr1, 12, 4096, 14336, 12, 128)
