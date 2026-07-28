"""``torch.add.6_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body (lines 76-96 of the ``async_compile.triton(...)``
source string) verbatim, from torch-spyre's Inductor output for
Meta-Llama-3.1-8B-Instruct's traced ``torch.add.6`` op
(``aten.add.Tensor(arg0_1, arg1_1)`` — tensor+tensor, both f16 — on a logical
``f16[1, 32, 1, 128]`` tensor); see
``torch-spyre/test_results_triton_20260714_120532/torch.add.6_spyre/
torch_compile_debug/run_2026_07_14_13_17_39_081995-pid_272578/
torchinductor/model__0_inference_0.0/output_code.py``.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'XBLOCK': 128}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    # Triton -> Logical layouts
    c0 = tl.program_id(0)
    c1 = 0
    # Logical layouts -> Device layouts
    dim0 = c0
    dim1 = 0
    dim2 = c1 // 64
    dim3 = 0
    dim4 = c1 % 64
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[32, 1, 2, 1, 64], strides=[128, 128, 64, 64, 1], block_shape=[1, 1, 2, 1, 64])
    desc_1 = tl.make_tensor_descriptor(in_ptr1, shape=[32, 1, 2, 1, 64], strides=[128, 128, 64, 64, 1], block_shape=[1, 1, 2, 1, 64])
    desc_2 = tl.make_tensor_descriptor(out_ptr0, shape=[32, 1, 2, 1, 64], strides=[128, 128, 64, 64, 1], block_shape=[1, 1, 2, 1, 64])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3, dim4])
    tmp1 = desc_1.load([dim0, dim1, dim2, dim3, dim4])
    tmp2 = tmp0 + tmp1
    desc_2.store([dim0, dim1, dim2, dim3, dim4], tmp2)
