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
# math.exp -> spyreop.exp
# =========================================================================

class TestExp(LowerSpyreOpsTester):
    @pattern("math-exp", category="compute", example=[
        "y = tl.exp(x)  # math.exp on a scalar",
    ])
    def test_f32(self):
        self.run("""
        module {
          tt.func @k(%s: f32) -> f32 {
            %0 = math.exp %s : f32
            tt.return %0 : f32
          }
        }
        """)
        self.assert_present("spyreop.exp")
        self.assert_absent("math.exp")
        self.assert_result_type("spyreop.exp", "f32")

    def test_tensor_operand_untouched(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<4xf32>) -> tensor<4xf32> {
            %0 = math.exp %t : tensor<4xf32>
            tt.return %0 : tensor<4xf32>
          }
        }
        """)
        self.assert_present("math.exp")
        self.assert_absent("spyreop.exp")

    @pattern("math-exp-unsupported-type", category="compute", negative=True,
             example=[
                 "# Not yet supported: math.exp on f64",
                 "y = tl.exp(x)  # x: f64",
             ])
    def test_f64_fails(self, capfd):
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%s: f64) -> f64 {
                %0 = math.exp %s : f64
                tt.return %0 : f64
              }
            }
            """)
        self.assert_stderr(capfd,
            "failed to legalize operation 'math.exp'",
            "LowerSpyreOps: failed to convert math ops",
        )


# =========================================================================
# math.rsqrt -> spyreop.rsqrt
# =========================================================================

class TestRsqrt(LowerSpyreOpsTester):
    @pattern("math-rsqrt", category="compute", example=[
        "y = tl.rsqrt(x)  # math.rsqrt on a scalar",
    ])
    def test_f32(self):
        self.run("""
        module {
          tt.func @k(%s: f32) -> f32 {
            %0 = math.rsqrt %s : f32
            tt.return %0 : f32
          }
        }
        """)
        self.assert_present("spyreop.rsqrt")
        self.assert_absent("math.rsqrt")
        self.assert_result_type("spyreop.rsqrt", "f32")

    def test_tensor_operand_untouched(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<4xf32>) -> tensor<4xf32> {
            %0 = math.rsqrt %t : tensor<4xf32>
            tt.return %0 : tensor<4xf32>
          }
        }
        """)
        self.assert_present("math.rsqrt")
        self.assert_absent("spyreop.rsqrt")

    @pattern("math-rsqrt-unsupported-type", category="compute", negative=True,
             example=[
                 "# Not yet supported: math.rsqrt on f64",
                 "y = tl.rsqrt(x)  # x: f64",
             ])
    def test_f64_fails(self, capfd):
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%s: f64) -> f64 {
                %0 = math.rsqrt %s : f64
                tt.return %0 : f64
              }
            }
            """)
        self.assert_stderr(capfd,
            "failed to legalize operation 'math.rsqrt'",
            "LowerSpyreOps: failed to convert math ops",
        )


# =========================================================================
# arith.divf -> spyreop.realdiv
# =========================================================================

class TestRealDiv(LowerSpyreOpsTester):
    @pattern("arith-divf", category="compute", example=[
        "y = x / z  # arith.divf on scalars",
    ])
    def test_f32(self):
        self.run("""
        module {
          tt.func @k(%a: f32, %b: f32) -> f32 {
            %0 = arith.divf %a, %b : f32
            tt.return %0 : f32
          }
        }
        """)
        self.assert_present("spyreop.realdiv")
        self.assert_absent("arith.divf")
        self.assert_result_type("spyreop.realdiv", "f32")

    def test_tensor_operand_untouched(self):
        self.run("""
        module {
          tt.func @k(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
            %0 = arith.divf %a, %b : tensor<4xf32>
            tt.return %0 : tensor<4xf32>
          }
        }
        """)
        self.assert_present("arith.divf")
        self.assert_absent("spyreop.realdiv")

    @pattern("arith-divf-unsupported-type", category="compute", negative=True,
             example=[
                 "# Not yet supported: arith.divf on f64",
                 "y = x / z  # x, z: f64",
             ])
    def test_f64_fails(self, capfd):
        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            self.run("""
            module {
              tt.func @k(%a: f64, %b: f64) -> f64 {
                %0 = arith.divf %a, %b : f64
                tt.return %0 : f64
              }
            }
            """)
        self.assert_stderr(capfd,
            "failed to legalize operation 'arith.divf'",
            "LowerSpyreOps: failed to convert math ops",
        )


# =========================================================================
# arith.addi / arith.muli -> spyreop.{addi32toi32,addi64toi64,muli32toi32}
#
# Unlike the math ops above, plain scalar integer add/mul is also used for
# loop indices, offsets, and tile addressing -- not just scalarized tensor
# compute. So these patterns only match inside a linalg.generic body (the
# structural signal ConvertElementwiseToLinalg leaves behind), and only at
# the bit-widths spyreop has an intrinsic for. Anything else (a different
# width, or arith.addi/muli outside a linalg.generic entirely) is left
# legal rather than reported -- there is no accepted-regression tradeoff
# here the way there is for an unsupported float type above.
# =========================================================================

def _generic_i32_body(op: str) -> str:
    return f"""
    module {{
      tt.func @k(%t: tensor<4xi32>) -> tensor<4xi32> {{
        %init = tensor.empty() : tensor<4xi32>
        %0 = linalg.generic {{
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}}
            ins(%t : tensor<4xi32>) outs(%init : tensor<4xi32>) {{
        ^bb0(%in: i32, %out: i32):
          %1 = {op} %in, %in : i32
          linalg.yield %1 : i32
        }} -> tensor<4xi32>
        tt.return %0 : tensor<4xi32>
      }}
    }}
    """


class TestAddIToSpyreInt(LowerSpyreOpsTester):
    @pattern("arith-addi-i32", category="compute", example=[
        "y = x + x  # arith.addi i32, scalarized inside a linalg.generic",
    ])
    def test_i32_inside_generic(self):
        self.run(_generic_i32_body("arith.addi"))
        self.assert_present("spyreop.addi32toi32", parent="linalg.generic")
        self.assert_absent("arith.addi")

    def test_i64_inside_generic(self):
        self.run("""
        module {
          tt.func @k(%t: tensor<4xi64>) -> tensor<4xi64> {
            %init = tensor.empty() : tensor<4xi64>
            %0 = linalg.generic {
                indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                iterator_types = ["parallel"]}
                ins(%t : tensor<4xi64>) outs(%init : tensor<4xi64>) {
            ^bb0(%in: i64, %out: i64):
              %1 = arith.addi %in, %in : i64
              linalg.yield %1 : i64
            } -> tensor<4xi64>
            tt.return %0 : tensor<4xi64>
          }
        }
        """)
        self.assert_present("spyreop.addi64toi64", parent="linalg.generic")
        self.assert_absent("arith.addi")

    def test_i16_inside_generic_untouched(self):
        """i16 has no spyreop add intrinsic -- left legal, not reported."""
        self.run("""
        module {
          tt.func @k(%t: tensor<4xi16>) -> tensor<4xi16> {
            %init = tensor.empty() : tensor<4xi16>
            %0 = linalg.generic {
                indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                iterator_types = ["parallel"]}
                ins(%t : tensor<4xi16>) outs(%init : tensor<4xi16>) {
            ^bb0(%in: i16, %out: i16):
              %1 = arith.addi %in, %in : i16
              linalg.yield %1 : i16
            } -> tensor<4xi16>
            tt.return %0 : tensor<4xi16>
          }
        }
        """)
        self.assert_present("arith.addi")
        self.assert_absent("spyreop.addi32toi32", "spyreop.addi64toi64")

    def test_i32_outside_generic_untouched(self):
        """Plain scalar arith.addi outside any linalg.generic is index/address
        arithmetic, not scalarized compute -- must be left alone.
        """
        self.run("""
        module {
          tt.func @k(%a: i32, %b: i32) -> i32 {
            %0 = arith.addi %a, %b : i32
            tt.return %0 : i32
          }
        }
        """)
        self.assert_present("arith.addi")
        self.assert_absent("spyreop.addi32toi32")


class TestMulIToSpyreInt(LowerSpyreOpsTester):
    @pattern("arith-muli-i32", category="compute", example=[
        "y = x * x  # arith.muli i32, scalarized inside a linalg.generic",
    ])
    def test_i32_inside_generic(self):
        self.run(_generic_i32_body("arith.muli"))
        self.assert_present("spyreop.muli32toi32", parent="linalg.generic")
        self.assert_absent("arith.muli")

    def test_i64_inside_generic_untouched(self):
        """No spyreop.muli64toi64 intrinsic exists -- i64 is left legal."""
        self.run("""
        module {
          tt.func @k(%t: tensor<4xi64>) -> tensor<4xi64> {
            %init = tensor.empty() : tensor<4xi64>
            %0 = linalg.generic {
                indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                iterator_types = ["parallel"]}
                ins(%t : tensor<4xi64>) outs(%init : tensor<4xi64>) {
            ^bb0(%in: i64, %out: i64):
              %1 = arith.muli %in, %in : i64
              linalg.yield %1 : i64
            } -> tensor<4xi64>
            tt.return %0 : tensor<4xi64>
          }
        }
        """)
        self.assert_present("arith.muli")
        self.assert_absent("spyreop.muli32toi32")

    def test_i32_outside_generic_untouched(self):
        self.run("""
        module {
          tt.func @k(%a: i32, %b: i32) -> i32 {
            %0 = arith.muli %a, %b : i32
            tt.return %0 : i32
          }
        }
        """)
        self.assert_present("arith.muli")
        self.assert_absent("spyreop.muli32toi32")


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
