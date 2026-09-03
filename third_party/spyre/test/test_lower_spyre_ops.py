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

import pytest
from conftest import SinglePassTester
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
