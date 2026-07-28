"""``torch.nn.functional.linear.1_spyre`` kernel extracted from a
torch-spyre Inductor trace.

The three ``@triton.jit`` functions verbatim (lines 61-147 of the
``async_compile.triton(...)`` source string) from torch-spyre's Inductor
output for Meta-Llama-3.1-8B-Instruct's traced ``torch.nn.functional.linear.1``
op (``aten.mm`` between a logical ``f16[1, 12, 4096]`` activation and a
logical ``f16[4096, 4096]`` weight, i.e. `nn.functional.linear` without
bias); see ``torch-spyre/test_results_triton_20260714_120532/
torch.nn.functional.linear.1_spyre/torch_compile_debug/
run_2026_07_14_12_23_57_368015-pid_258872/torchinductor/
model__0_inference_0.0/output_code.py``.

This is a *bundled* multi-kernel trace: the top-level entry
``triton_bundle_0`` sequentially calls two ``noinline=True`` helpers it
never appears alone in isolation in the traced output:

- ``triton_bundle_0_kernel_0`` — repacks the raw weight tensor (loaded
  through one tensor-descriptor view) into a device-friendly tiled
  layout (via ``tl.reshape``/``tl.permute``/``tl.reshape``), written out
  through a second tensor-descriptor view into a scratch buffer.
- ``triton_bundle_0_kernel_1`` — the actual matmul: loads a tile of the
  activation and a tile of the repacked weight, both cast to
  ``float16``, contracts them with a single ``tl.dot(..., input_precision
  ="ieee")`` (accumulates in the dot's default higher-precision
  accumulator, cast back to ``float16`` immediately after), and stores
  the result tile.

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
def triton_bundle_0_kernel_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4096
    r0_numel = 4096
    R0_BLOCK: tl.constexpr = 4096
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
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, None, :]
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
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[12, 1, 64, 64], strides=[4096, 64, 64, 1], block_shape=[12, 1, 64, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c2
    dim_1_1 = c1 // 64
    dim_1_2 = c1 % 64
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[4096, 64, 64], strides=[64, 262144, 1], block_shape=[4096, 2, 64])
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
    tmp4 = tl.reshape(tmp1, [12, 1, 4096])
    tmp5 = tl.reshape(tmp3, [4096, 128])
    tmp6 = tl.reshape(tmp4, [12, 4096])
    tmp7 = tl.dot(tmp6, tmp5, input_precision="ieee")
    tmp8 = tmp7.to(tl.float16)
    tmp9 = tl.reshape(tmp8, [12, 1, 2, 64])
    desc_2.store([dim_2_0, dim_2_1, dim_2_2, dim_2_3], tmp9)

@triton.jit
def triton_bundle_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1):
    triton_bundle_0_kernel_0(in_ptr0, out_ptr0, 16777216, 524288)
    triton_bundle_0_kernel_1(in_ptr1, out_ptr0, out_ptr1, 12, 4096, 4096, 12, 128)
