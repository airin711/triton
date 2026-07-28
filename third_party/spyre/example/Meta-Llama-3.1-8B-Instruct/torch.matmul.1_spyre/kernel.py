"""``torch.matmul.1_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body (lines 75-112 of the ``async_compile.triton(...)``
source string) verbatim, from torch-spyre's Inductor output for
Meta-Llama-3.1-8B-Instruct's traced ``torch.matmul.1`` op
(``aten.bmm.default(expand[1,64,1], expand_1[1,1,12])`` -> ``f16[1,64,12]``);
see ``torch-spyre/test_results_triton_20260714_120532/torch.matmul.1_spyre/
torch_compile_debug/run_2026_07_14_12_11_50_661280-pid_255363/
torchinductor/model__0_inference_0.0/output_code.py``.

The batched matmul's contracted dimension has size 1, so ``aten.bmm``
degenerates into a broadcast elementwise multiply: each of the 64 rows of
the first operand (itself padded to a full 64-wide f16 "stick", though only
its first element is logically real) is multiplied by the single 64-wide
stick of the second operand (which holds the 12 real columns of the output,
padded up to 64). The kernel operates on the full padded sticks in both
operands uniformly, with no masking.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'YBLOCK': 2, 'XBLOCK': 12}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused_bmm_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 12
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    # Triton -> Logical layouts
    c0 = (tl.program_id(0)) * 2
    c1 = 0
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = 0
    dim2 = 0
    dim3 = 0
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[64, 1, 1, 64], strides=[64, 64, 64, 1], block_shape=[2, 1, 1, 64])
    # Logical layouts -> Device layouts
    dim_1_0 = 0
    dim_1_1 = 0
    dim_1_2 = 0
    dim_1_3 = c1
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[1, 1, 1, 64], strides=[64, 64, 64, 1], block_shape=[1, 1, 1, 64])
    # Logical layouts -> Device layouts
    dim_2_0 = 0
    dim_2_1 = 0
    dim_2_2 = c0
    dim_2_3 = c1
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[1, 1, 64, 64], strides=[4096, 4096, 64, 1], block_shape=[1, 1, 2, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp2 = desc_1.load([dim_1_0, dim_1_1, dim_1_2, dim_1_3])
    tmp1 = tl.reshape(tmp0, [1, 1, 2, 64])
    tmp3 = tl.reshape(tmp2, [1, 1, 1, 64])
    tmp4 = tl.broadcast_to(tmp3, [1, 1, 2, 64])
    tmp5 = tmp1 * tmp4
    desc_2.store([dim_2_0, dim_2_1, dim_2_2, dim_2_3], tmp5)
