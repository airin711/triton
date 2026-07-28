"""``torch.cat.1_spyre`` kernel extracted from a torch-spyre Inductor trace.

The three ``@triton.jit`` functions verbatim (lines 61-116 of the
``async_compile.triton(...)`` source string), from torch-spyre's Inductor
output for Meta-Llama-3.1-8B-Instruct's traced ``torch.cat.1`` op
(``aten.cat.default([x, y], dim=-1)`` on two logical ``f16[1, 12, 64]``
tensors, producing a logical ``f16[1, 12, 128]`` tensor); see
``torch-spyre/test_results_triton_20260714_120532/torch.cat.1_spyre/
torch_compile_debug/run_2026_07_14_12_13_21_715447-pid_255771/
torchinductor/model__0_inference_0.0/output_code.py``.

This is a *bundled* multi-kernel trace: the top-level entry
``triton_bundle_0`` sequentially calls two ``noinline=True`` helpers; it
never appears alone in isolation in the traced output.

- ``triton_bundle_0_kernel_0`` — loads a tile of ``in_ptr0`` (the first cat
  operand) through one tensor-descriptor view and stores it unchanged
  through a second descriptor view into the front half (index 0 of the
  doubled axis) of the output's device layout.
- ``triton_bundle_0_kernel_1`` — loads a tile of ``in_ptr1`` (the second cat
  operand), reshapes/broadcasts it to match the doubled-axis output block
  shape, and stores it into the back half (index 1 of the doubled axis) of
  the output's device layout.
- ``triton_bundle_0`` (the entry) calls ``kernel_0`` then ``kernel_1`` in
  sequence, so the output buffer ends up holding ``in_ptr0`` followed by
  ``in_ptr1`` along the concatenated axis — i.e. ``aten.cat``.

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
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    # Triton -> Logical layouts
    c0 = tl.program_id(0)
    c1 = 0
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = 0
    dim2 = 0
    dim3 = c1
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[12, 1, 1, 64], strides=[64, 64, 64, 1], block_shape=[1, 1, 1, 64])
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[12, 2, 1, 64], strides=[128, 64, 64, 1], block_shape=[1, 2, 1, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    desc_1.store([dim0, dim1, dim2, dim3], tmp0)

@triton.jit(noinline=True)
def triton_bundle_0_kernel_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    # Triton -> Logical layouts
    c0 = tl.program_id(0)
    c1 = 0
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = 0
    dim2 = 0
    dim3 = c1
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[12, 1, 1, 64], strides=[64, 64, 64, 1], block_shape=[1, 1, 1, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c0
    dim_1_1 = 1
    dim_1_2 = 0
    dim_1_3 = c1
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[12, 2, 1, 64], strides=[128, 64, 64, 1], block_shape=[1, 2, 1, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp1 = tl.reshape(tmp0, [1, 1, 1, 64])
    tmp2 = tl.broadcast_to(tmp1, [1, 2, 1, 64])
    desc_1.store([dim_1_0, dim_1_1, dim_1_2, dim_1_3], tmp2)

@triton.jit
def triton_bundle_0(in_ptr0, in_ptr1, out_ptr0):
    triton_bundle_0_kernel_0(in_ptr0, out_ptr0, 768, 64)
    triton_bundle_0_kernel_1(in_ptr1, out_ptr0, 768, 64)
