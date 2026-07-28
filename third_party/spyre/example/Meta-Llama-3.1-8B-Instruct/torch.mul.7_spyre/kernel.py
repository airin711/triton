"""``torch.mul.7_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body (from the ``async_compile.triton(...)``
source string) verbatim, from torch-spyre's Inductor output for
Meta-Llama-3.1-8B-Instruct's traced ``torch.mul.7`` op
(``aten.mul.Tensor(x, y)`` broadcasting a logical ``f16[1, 1, 12, 128]``
tensor ``y`` over the head dim of a logical ``f16[1, 8, 12, 128]`` tensor
``x``; structurally identical to ``torch.mul.6_spyre`` — same shapes,
strides and kernel body, just a distinct traced call site); see
``torch-spyre/test_results_triton_20260714_120532/torch.mul.7_spyre/
torch_compile_debug/run_2026_07_14_12_36_11_883431-pid_262675/
torchinductor/model__0_inference_0.0/output_code.py``.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'XBLOCK': 512}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    # Triton -> Logical layouts
    c0 = ((tl.program_id(0) // 12)) * 4
    c1 = (tl.program_id(0) % 12)
    c2 = 0
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = c1
    dim2 = c2 // 64
    dim3 = 0
    dim4 = c2 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[8, 12, 2, 1, 64], strides=[1536, 128, 64, 64, 1], block_shape=[4, 1, 2, 1, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = 0
    dim_1_1 = c1
    dim_1_2 = c2 // 64
    dim_1_3 = 0
    dim_1_4 = c2 % 64
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[1, 12, 2, 1, 64], strides=[1536, 128, 64, 64, 1], block_shape=[1, 1, 2, 1, 64])
    # Logical layouts -> Device layouts
    dim_2_0 = c1
    dim_2_1 = 0
    dim_2_2 = c2 // 64
    dim_2_3 = c0
    dim_2_4 = c2 % 64
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[12, 1, 2, 8, 64], strides=[1024, 1024, 512, 64, 1], block_shape=[1, 1, 2, 4, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3, dim4])
    tmp4 = desc_1.load([dim_1_0, dim_1_1, dim_1_2, dim_1_3, dim_1_4])
    tmp1 = tl.reshape(tmp0, [4, 1, 2, 64])
    tmp2 = tl.permute(tmp1, [1, 2, 0, 3])
    tmp3 = tl.reshape(tmp2, [1, 1, 2, 4, 64])
    tmp5 = tl.reshape(tmp4, [1, 1, 2, 1, 64])
    tmp6 = tl.broadcast_to(tmp5, [1, 1, 2, 4, 64])
    tmp7 = tmp3 * tmp6
    desc_2.store([dim_2_0, dim_2_1, dim_2_2, dim_2_3, dim_2_4], tmp7)
