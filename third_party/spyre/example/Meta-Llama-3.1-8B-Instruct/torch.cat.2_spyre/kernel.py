"""``torch.cat.2_spyre`` kernel extracted from a torch-spyre Inductor trace.

The three ``@triton.jit`` functions verbatim (lines 61-120 of the
``async_compile.triton(...)`` source string), from torch-spyre's Inductor
output for Meta-Llama-3.1-8B-Instruct's traced ``torch.cat.2`` op
(``aten.cat.default([x, y], dim=-1)`` on two logical ``f16[1, 32, 12, 64]``
tensors, producing a logical ``f16[1, 32, 12, 128]`` tensor); see
``torch-spyre/test_results_triton_20260714_120532/torch.cat.2_spyre/
torch_compile_debug/run_2026_07_14_12_31_00_891526-pid_260991/
torchinductor/model__0_inference_0.0/output_code.py``.

This is a *bundled* multi-kernel trace: the top-level entry
``triton_bundle_0`` sequentially calls two ``noinline=True`` helpers; it
never appears alone in isolation in the traced output.

- ``triton_bundle_0_kernel_0`` — loads a tile of ``in_ptr0`` (the first cat
  operand) through one tensor-descriptor view and stores it through a
  second descriptor view into the front slot (index 0 of the doubled
  axis 2) of the output's device layout.
- ``triton_bundle_0_kernel_1`` — loads a tile of ``in_ptr1`` (the second cat
  operand), reshapes/broadcasts it to match the doubled-axis output block
  shape, and stores it into the back slot (index 1 of the doubled axis 2)
  of the output's device layout.
- ``triton_bundle_0`` (the entry) calls ``kernel_0`` then ``kernel_1`` in
  sequence, so the output buffer ends up holding ``in_ptr0`` followed by
  ``in_ptr1`` along the concatenated axis — i.e. ``aten.cat``.

Unlike ``torch.cat.1_spyre`` (one program per row, 12-program grid), here
each program's descriptor block spans the *entire* middle dim (12) in one
shot (``block_shape=[1, 12, 1, 1, 64]``); the grid is sized to 32 programs,
one per the outermost (``dim0``) axis.

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
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    # Triton -> Logical layouts
    c0 = tl.program_id(0)
    c1 = 0
    c2 = 0
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = c1
    dim2 = 0
    dim3 = 0
    dim4 = c2
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[32, 12, 1, 1, 64], strides=[768, 64, 64, 64, 1], block_shape=[1, 12, 1, 1, 64])
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[32, 12, 2, 1, 64], strides=[1536, 128, 64, 64, 1], block_shape=[1, 12, 2, 1, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3, dim4])
    desc_1.store([dim0, dim1, dim2, dim3, dim4], tmp0)

@triton.jit(noinline=True)
def triton_bundle_0_kernel_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    # Triton -> Logical layouts
    c0 = tl.program_id(0)
    c1 = 0
    c2 = 0
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = c1
    dim2 = 0
    dim3 = 0
    dim4 = c2
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[32, 12, 1, 1, 64], strides=[768, 64, 64, 64, 1], block_shape=[1, 12, 1, 1, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c0
    dim_1_1 = c1
    dim_1_2 = 1
    dim_1_3 = 0
    dim_1_4 = c2
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[32, 12, 2, 1, 64], strides=[1536, 128, 64, 64, 1], block_shape=[1, 12, 2, 1, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3, dim4])
    tmp1 = tl.reshape(tmp0, [1, 12, 1, 1, 64])
    tmp2 = tl.broadcast_to(tmp1, [1, 12, 2, 1, 64])
    desc_1.store([dim_1_0, dim_1_1, dim_1_2, dim_1_3, dim_1_4], tmp2)

@triton.jit
def triton_bundle_0(in_ptr0, in_ptr1, out_ptr0):
    triton_bundle_0_kernel_0(in_ptr0, out_ptr0, 24576, 768)
    triton_bundle_0_kernel_1(in_ptr1, out_ptr0, 24576, 768)
