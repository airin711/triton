"""``torch.pow.1_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body (lines 75-92 of the ``async_compile.triton(...)``
source string) verbatim, from torch-spyre's Inductor output for
Meta-Llama-3.1-8B-Instruct's traced ``torch.pow.1`` op
(``aten.pow.Tensor_Scalar(arg0_1, 2)`` — unary, integer exponent 2 — on a
logical ``f16[1, 12, 4096]`` tensor); see
``torch-spyre/test_results_triton_20260714_120532/torch.pow.1_spyre/
torch_compile_debug/run_2026_07_14_12_18_39_322315-pid_256867/
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
def triton_unk_fused_pow_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[12, 64, 1, 64], strides=[4096, 64, 64, 1], block_shape=[12, 2, 1, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp1 = tmp0 * tmp0
    desc_1.store([dim0, dim1, dim2, dim3], tmp1)
