#!/usr/bin/env python3
"""
Unit tests for individual LowerSpyreOps conversion patterns.

Organization: one test class per math op (TestSqrt, …). Each class
contains positive tests for the supported scalar types plus negative
tests for expected failure modes.

Test modules must *use* the op result (e.g. ``tt.return %0``) so
cleanupDeadOps doesn't erase the newly created ops.

Negative tests use ``pytest.raises`` + ``assert_stderr(capfd, ...)``
to verify both the RuntimeError and the MLIR diagnostic on stderr.
"""

import tempfile

import pytest
from conftest import SinglePassTester, StructuralAssertions, make_ktir_mod, walk_module
from utils_pattern import pattern


class LowerSpyreOpsTester(SinglePassTester):
    """Shared base for all LowerSpyreOps pattern tests."""
    PASS = "add_lower_spyre_ops"


# =========================================================================
# math.sqrt -> spyreop.sqrt
# =========================================================================

class TestSqrt(LowerSpyreOpsTester):
    # math.sqrt on a scalar f16/f32 -> spyreop.sqrt. spyreop's intrinsics
    # are scalar-only, so a math.sqrt still on a tensor is left untouched
    # (see test_tensor_operand_untouched) until ConvertElementwiseToLinalg
    # has scalarized it.
    #
    # test_f32                    — scalar f32 operand
    # test_f16                    — scalar f16 operand
    # test_tensor_operand_untouched — tensor operand is not matched
    # test_f64_fails               — unsupported scalar type is illegal

    @pattern("math-sqrt", category="compute", example=[
        "y = tl.sqrt(x)  # math.sqrt on a scalar",
    ])
    def test_f32(self):
        self.run("""
        module {
          tt.func @k(%s: f32) -> f32 {
            %0 = math.sqrt %s : f32
            tt.return %0 : f32
          }
        }
        """)
        self.assert_present("spyreop.sqrt")
        self.assert_absent("math.sqrt")
        self.assert_result_type("spyreop.sqrt", "f32")

    def test_f16(self):
        self.run("""
        module {
          tt.func @k(%s: f16) -> f16 {
            %0 = math.sqrt %s : f16
            tt.return %0 : f16
          }
        }
        """)
        self.assert_present("spyreop.sqrt")
        self.assert_absent("math.sqrt")
        self.assert_result_type("spyreop.sqrt", "f16")

    def test_tensor_operand_untouched(self):
        # spyreop.sqrt is scalar-only, so a tensor-typed math.sqrt (not yet
        # scalarized by ConvertElementwiseToLinalg) is left legal as-is.
        self.run("""
        module {
          tt.func @k(%t: tensor<4xf32>) -> tensor<4xf32> {
            %0 = math.sqrt %t : tensor<4xf32>
            tt.return %0 : tensor<4xf32>
          }
        }
        """)
        self.assert_present("math.sqrt")
        self.assert_absent("spyreop.sqrt")

    @pattern("math-sqrt-unsupported-type", category="compute", negative=True,
             example=[
                 "# Not yet supported: math.sqrt on f64",
                 "y = tl.sqrt(x)  # x: f64",
             ])
    def test_f64_fails(self, capfd):
        """math.sqrt on f64 has no spyreop intrinsic -> stays illegal.

        spyreop.sqrt only accepts f16/df16/f32, so an f64 operand is never
        matched by ConvertMathSqrt and dialect conversion reports it as an
        unlegalized op.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%s: f64) -> f64 {
                %0 = math.sqrt %s : f64
                tt.return %0 : f64
              }
            }
            """)
        self.assert_stderr(capfd,
            "failed to legalize operation 'math.sqrt'",
            "LowerSpyreOps: failed to convert math ops",
        )


# =========================================================================
# lower_spyre_ops as a default required_fix (SpyreBackend.parse_options)
#
# TestSqrt above runs LowerSpyreOps in isolation. These drive the real
# default _make_ktir pipeline instead -- ConvertElementwiseToLinalg
# scalarizes the tensor math.sqrt into a linalg.generic body, and
# LowerSpyreOps (anchored on rewrite_descriptor_layout, after it) now
# lowers that scalar op with no caller having to name required_fixes.
# =========================================================================

class DefaultPipelineTester(StructuralAssertions):
    """Compiles TTIR text through the real default pipeline via ``make_ktir_mod``,
    with no explicit ``required_fixes``, then exposes StructuralAssertions
    over the result.
    """

    def compile(self, mlir_text: str):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".mlir", delete_on_close=False) as f:
            f.write(mlir_text)
            f.flush()
            mod = make_ktir_mod(f.name)
        self.ops = walk_module(mod)
        self._def_map = None
        return mod


class TestSqrtDefaultPipeline(DefaultPipelineTester):
    # test_tensor_sqrt_f32_default_pipeline      — reaches spyreop.sqrt with
    #                                               no options at all
    # test_tensor_sqrt_f64_default_pipeline_fails — accepted tradeoff: an
    #                                               unsupported scalar type
    #                                               now fails a compile that
    #                                               used to pass through

    def test_tensor_sqrt_f32_default_pipeline(self):
        self.compile("""
        module {
          tt.func public @k(%t: tensor<8xf32>) -> tensor<8xf32> {
            %0 = math.sqrt %t : tensor<8xf32>
            tt.return %0 : tensor<8xf32>
          }
        }
        """)
        self.assert_present("spyreop.sqrt", parent="linalg.generic")
        self.assert_absent("math.sqrt")

    def test_tensor_sqrt_f64_default_pipeline_fails(self, capfd):
        """f64 has no spyreop intrinsic. Before lower_spyre_ops became a
        default fix, this compiled and left math.sqrt untouched; now the
        default pipeline reports it and the compile fails instead.
        """
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.compile("""
            module {
              tt.func public @k(%t: tensor<8xf64>) -> tensor<8xf64> {
                %0 = math.sqrt %t : tensor<8xf64>
                tt.return %0 : tensor<8xf64>
              }
            }
            """)
        self.assert_stderr(capfd,
            "failed to legalize operation 'math.sqrt'",
            "LowerSpyreOps: failed to convert math ops",
        )
