#!/usr/bin/env python3
"""
Structural and numerical KTIR tests for every kernel variant discovered
from ``example/**/meta.py``.

Mirrors ``test/test_ktir_examples.py``'s :class:`TestExample` structure,
but runs against this directory's own :data:`EXAMPLES` registry (see
``conftest.py`` in this folder). Kept self-contained and independent of
``test/`` — see this folder's ``conftest.py`` docstring for why.

One class :class:`TestExample` drives everything via pytest parametrize.
Each test method falls into one of three categories (see the class
docstring for detail):

  a. **Pipeline invariants** — kernel-agnostic properties of the final
     KTIR (no ``tt.*`` ops, ``ktdp.*`` ops present, DistributeWork ran,
     ...). Parametrized over every variant uniformly.
  b. **Per-variant structural hook** — ``test_extra_checks`` runs the
     variant's own ``extra_checks`` callable from ``meta.py`` for
     claims that depend on variant shape.
  c. **Numerical** — ``test_numerical`` runs the kernel through
     ``ktir_cpu`` and compares to the variant's NumPy oracle, with
     per-variant ``xfail_numerical`` marks.
"""

import pytest
from conftest import EXAMPLES, KTIRCpuTester, KTIRStructuralTester

# ---------------------------------------------------------------------------
# Discovered variants — subset of EXAMPLES that came in via meta.py discovery
# (anything with ``kernel_fn`` is compiled from Triton source; path-based
# legacy entries are excluded).
# ---------------------------------------------------------------------------

DISCOVERED = sorted(k for k, v in EXAMPLES.items() if "kernel_fn" in v)


def _keys():
    """Build pytest params for structural tests.

    Every discovered variant becomes a ``pytest.param``. Variants that
    declare ``"disabled"`` in their ``meta.py`` entry get a
    ``pytest.mark.skip`` with the declared reason, so the structural
    tests surface a visible ``SKIPPED`` line explaining why the variant
    is off (e.g. a known compile-time gap).
    """
    params = []
    for key in DISCOVERED:
        disabled = EXAMPLES[key].get("disabled")
        if disabled is None:
            params.append(pytest.param(key, id=key))
        else:
            mark = pytest.mark.skip(reason=disabled["reason"])
            params.append(pytest.param(key, marks=[mark], id=key))
    return params


def _keys_with_numerical_xfail():
    """Param list for the numerical test — attaches each variant's mark.

    ``disabled`` variants skip (they can't compile, so there is nothing to
    run numerically). For the rest, ``xfail_numerical`` in meta.py is
    either a short reason string or a dict forwarded to
    ``pytest.mark.xfail(**d)`` (so ``raises=ValueError`` etc. work); it
    is built at collection time so failures are reported as proper
    XFAIL, not SKIP.
    """
    params = []
    for k in DISCOVERED:
        entry = EXAMPLES[k]
        marks = []
        disabled = entry.get("disabled")
        if disabled is not None:
            marks.append(pytest.mark.skip(reason=disabled["reason"]))
        else:
            xfm = entry.get("xfail_numerical")
            if xfm is not None:
                kw = xfm if isinstance(xfm, dict) else {"reason": xfm, "strict": True}
                marks.append(pytest.mark.xfail(**kw))
        params.append(pytest.param(k, marks=marks, id=k))
    return params


class TestExample(KTIRCpuTester, KTIRStructuralTester):
    """Parametrized suite for every kernel variant under ``example/``.

    The test methods split into three groups:

    1. **Pipeline invariants** — properties of the final KTIR that hold for
       every descriptor-based Triton kernel regardless of what the kernel
       computes. If one of these fails, a pipeline pass has regressed, not
       the kernel itself.
    2. **Per-variant structural hook** (``test_extra_checks``) — runs the
       variant's own ``extra_checks`` callable from ``meta.py``.
    3. **Numerical** (``test_numerical``) — runs the kernel on ``ktir_cpu``
       and compares against the NumPy oracle declared in ``meta.py``.

    New example folders (grouped by traced model, e.g.
    ``Meta-Llama-3.1-8B-Instruct``) automatically pick up group (1) from
    discovery; they add group (2) / (3) content in their own ``meta.py``.
    """

    # ------------------------------------------------------------------
    # Group 1: pipeline invariants (kernel-agnostic)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("key", _keys())
    def test_no_tt_ops(self, key):
        """No Triton dialect ops should survive the TTIR→KTIR lowering."""
        self.EXAMPLE = key
        self.setup_method()
        self.assert_absent(
            "tt.descriptor_load", "tt.descriptor_store",
            "tt.make_tensor_descriptor", "tt.get_program_id",
            "tt.func", "tt.return",
        )

    @pytest.mark.parametrize("key", _keys())
    def test_no_raw_ptr_ops(self, key):
        """Raw-pointer Triton ops (pre-descriptor idiom) should be absent."""
        self.EXAMPLE = key
        self.setup_method()
        self.assert_absent("tt.splat", "tt.addptr", "tt.load", "tt.store")

    @pytest.mark.parametrize("key", _keys())
    def test_no_tt_ptr_type(self, key):
        """No SSA value in the final KTIR should have a ``!tt.ptr`` type."""
        self.EXAMPLE = key
        self.setup_method()
        for op in self.ops:
            for t in op.result_types:
                assert "!tt.ptr" not in t, (
                    f"Found !tt.ptr in result type of '{op.name}': {t}"
                )

    @pytest.mark.parametrize("key", _keys())
    def test_ktdp_ops_present(self, key):
        """The lowering should emit the expected KTDP memory-access ops."""
        entry = EXAMPLES[key]
        self.EXAMPLE = key
        self.setup_method()
        expected = ["ktdp.construct_memory_view", "ktdp.load", "ktdp.store"]
        if entry.get("direct_access_tile", True):
            expected.append("ktdp.construct_access_tile")
        self.assert_present(*expected)

    @pytest.mark.parametrize("key", _keys())
    def test_work_distribution(self, key):
        """DistributeWork must lower ``tl.program_id`` and wrap the body.

        Applies only to variants that use ``tl.program_id`` — such
        variants opt out by setting ``"parallel": False`` in ``meta.py``
        (default is ``True``). ``scf.for`` is always required: Spyre-
        compliant kernels must handle any input size for a fixed grid
        size and tile size, so every parallel kernel distributes work via
        a loop even when the grid maps one axis per output dimension
        (e.g. grid == xnumel).
        """
        entry = EXAMPLES[key]
        if not entry.get("parallel", True):
            pytest.skip(f"{key}: not a parallel kernel (no tl.program_id)")
        self.EXAMPLE = key
        self.setup_method()
        self.assert_present("ktdp.get_compute_tile_id")
        if entry.get("distribution_loop", True):
            self.assert_present("scf.for")

    @pytest.mark.parametrize("key", _keys())
    def test_access_tile_type(self, key):
        """``ktdp.construct_access_tile`` result must be index-typed."""
        entry = EXAMPLES[key]
        if not entry.get("direct_access_tile", True):
            pytest.skip(f"{key}: no direct construct_access_tile (all-indirect kernel)")
        self.EXAMPLE = key
        self.setup_method()
        self.assert_result_type("ktdp.construct_access_tile", "xindex")

    @pytest.mark.parametrize("key", _keys())
    def test_memref_type(self, key):
        """``ktdp.construct_memory_view`` must produce a ``memref<...>`` type."""
        self.EXAMPLE = key
        self.setup_method()
        self.assert_result_type("ktdp.construct_memory_view", "memref<")

    # ------------------------------------------------------------------
    # Group 2: per-variant structural hook
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("key", _keys())
    def test_extra_checks(self, key):
        """Run the variant's ``extra_checks`` callable from ``meta.py``."""
        entry = EXAMPLES[key]
        if entry.get("extra_checks") is None:
            pytest.skip(f"{key}: no extra_checks")
        self.EXAMPLE = key
        self.setup_method()
        entry["extra_checks"](self)

    # ------------------------------------------------------------------
    # Group 3: numerical (ktir_cpu execution + NumPy oracle)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("key", _keys_with_numerical_xfail())
    def test_numerical(self, key):
        """Execute the kernel on ``ktir_cpu`` and compare to the NumPy oracle."""
        import numpy as np

        entry = EXAMPLES[key]
        if entry.get("reference") is None:
            pytest.skip(f"{key}: no numerical oracle")
        self.EXAMPLE = key
        self.setup_method()

        param_values = entry["param_values"]
        inputs = entry["inputs"](**param_values)
        runtime_scalars = {
            k: v for k, v in param_values.items()
            if k not in entry["constexprs"] and k not in inputs
        }
        func_name = entry.get("func_name") or entry["kernel_fn"].__name__
        outputs = self.run_cpu(
            func_name, kernel_fn=entry["kernel_fn"],
            **inputs, **runtime_scalars,
        )

        ref = entry["reference"](inputs)
        output_key = entry["output_key"]
        np.testing.assert_allclose(outputs[output_key], ref,
                                   rtol=entry.get("rtol", 1e-6),
                                   atol=entry.get("atol", 0))
