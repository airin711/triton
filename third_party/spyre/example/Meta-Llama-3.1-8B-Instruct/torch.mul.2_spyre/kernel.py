"""``torch.mul.2_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body (from the ``async_compile.triton(...)``
source string) verbatim, from torch-spyre's Inductor output for
Meta-Llama-3.1-8B-Instruct's traced ``torch.mul.2`` op
(``aten.mul.Tensor(x, y)`` broadcasting a logical ``f16[1, 12, 1]`` tensor
``y`` over the last dim of a logical ``f16[1, 12, 4096]`` tensor ``x``); see
``torch-spyre/test_results_triton_20260714_120532/torch.mul.2_spyre/
torch_compile_debug/run_2026_07_14_12_21_52_076513-pid_258137/
torchinductor/model__0_inference_0.0/output_code.py``.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'XBLOCK': 1536}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    # Triton -> Logical layouts
    c0 = 0
    c1 = (tl.program_id(0)) * 128
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = c1 // 64
    dim2 = 0
    dim3 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[12, 64, 1, 64], strides=[4096, 64, 64, 1], block_shape=[12, 2, 1, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = c0
    dim_1_1 = 0
    dim_1_2 = 0
    dim_1_3 = 0
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[12, 1, 1, 64], strides=[64, 64, 64, 1], block_shape=[12, 1, 1, 64])
    # Logical layouts -> Device layouts
    dim_2_0 = 0
    dim_2_1 = c1 // 64
    dim_2_2 = c0
    dim_2_3 = c1 % 64
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[1, 64, 12, 64], strides=[49152, 768, 64, 1], block_shape=[1, 2, 12, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp4 = desc_1.load([dim_1_0, dim_1_1, dim_1_2, dim_1_3])
    tmp1 = tl.reshape(tmp0, [12, 2, 64])
    tmp2 = tl.permute(tmp1, [1, 0, 2])
    tmp3 = tl.reshape(tmp2, [1, 2, 12, 64])
    tmp5 = tl.reshape(tmp4, [1, 1, 12, 64])
    tmp6 = tl.broadcast_to(tmp5, [1, 2, 12, 64])
    tmp7 = tmp3 * tmp6
    desc_2.store([dim_2_0, dim_2_1, dim_2_2, dim_2_3], tmp7)
