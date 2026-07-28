"""``torch.float.3_spyre`` kernel extracted from a torch-spyre Inductor trace.

The ``@triton.jit`` function body (lines 75-91 of the ``async_compile.triton(...)``
source string) verbatim, from torch-spyre's Inductor output for
Meta-Llama-3.1-8B-Instruct's traced ``torch.float.3`` op
(``prims.convert_element_type.default(arg0_1, torch.float32)`` — an
``fp16 -> fp32`` dtype cast, i.e. the ``.float()`` call — on a logical
``f16[1, 1, 12]`` tensor); see
``torch-spyre/test_results_triton_20260714_120532/torch.float.3_spyre/
torch_compile_debug/run_2026_07_14_12_10_58_739028-pid_255093/
torchinductor/model__0_inference_0.0/output_code.py``.

Dropped on extraction: the ``@triton_heuristics.fixed_config(...)`` decorator
that wraps the kernel in the Inductor output. It only carries autotuning/
Inductor-side metadata (``config={'XBLOCK': 12}``, ``triton_meta``,
``inductor_meta``) and pulls in ``torch._inductor.runtime`` as a dependency;
the ``@triton.jit`` function it wraps is unchanged either way, and this test
suite compiles ``kernel_fn`` directly via ``ASTSource`` (see ``meta.py``),
never through that decorator.

Note the two tensor descriptors have different *dtypes and physical shapes*
(``in_ptr0``: fp16, ``[1, 1, 1, 64]``; ``out_ptr0``: fp32, ``[1, 2, 1, 32]``)
— casting to a wider dtype changes the device-layout "stick" width (64
fp16 elements vs. 32 fp32 elements per stick), so the physical shape of the
output differs from the input even though the logical shape is unchanged.
"""

import triton
import triton.language as tl


@triton.jit
def triton_unk_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    # Triton -> Logical layouts
    c0 = 0
    # Logical layouts -> Device layouts
    dim0 = 0
    dim1 = 0
    dim2 = 0
    dim3 = c0
    desc_0 = tl.make_tensor_descriptor(in_ptr0, shape=[1, 1, 1, 64], strides=[64, 64, 64, 1], block_shape=[1, 1, 1, 64])
    desc_1 = tl.make_tensor_descriptor(out_ptr0, shape=[1, 2, 1, 32], strides=[64, 32, 32, 1], block_shape=[1, 2, 1, 32])
    tmp0 = desc_0.load([dim0, dim1, dim2, dim3])
    tmp1 = tmp0.to(tl.float32)
    desc_1.store([dim0, dim1, dim2, dim3], tmp1)
