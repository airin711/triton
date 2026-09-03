//===- LowerSpyreOps.cpp - Lower scalar math ops to spyreop intrinsics ---===//
//
// Lowers scalar math dialect ops to spyreop dialect intrinsics. spyreop's
// intrinsics are scalar-only (f16/df16/f32), so this pass only matches a
// math op that is already scalar -- typically the body of a linalg.generic
// after ConvertElementwiseToLinalg has scalarized a tensor-level math op.
//
//===----------------------------------------------------------------------===//

#include "Dialect/KTDP/Transforms/Passes.h"
#include "ktir/Dialect/SpyreOp/SpyreOp.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir::triton::ktdp {
#define GEN_PASS_DEF_LOWERSPYREOPS
#include "Dialect/KTDP/Transforms/Passes.h.inc"
} // namespace mlir::triton::ktdp

namespace {

/// Whether spyreop's scalar intrinsics accept this operand type.
static bool isSpyreOpScalarType(Type type) {
  return isa<Float16Type, Float32Type, spyreop::DF16Type>(type);
}

//===----------------------------------------------------------------------===//
// math.sqrt -> spyreop.sqrt
//===----------------------------------------------------------------------===//

struct ConvertMathSqrt : public OpConversionPattern<math::SqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(math::SqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isSpyreOpScalarType(op.getType()))
      return failure();
    rewriter.replaceOpWithNewOp<spyreop::Sqrt>(op, op.getType(),
                                               adaptor.getOperand());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LowerSpyreOpsPass
    : public mlir::triton::ktdp::impl::LowerSpyreOpsBase<LowerSpyreOpsPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    // A math.sqrt still on a tensor/vector hasn't been scalarized yet (that's
    // ConvertElementwiseToLinalg's job) -- leave it legal, quietly, rather
    // than reporting it. Any scalar type is illegal here: the pattern
    // converts the ones spyreop supports and leaves the rest illegal so
    // conversion reports them instead of silently dropping them.
    target.addDynamicallyLegalOp<math::SqrtOp>([](math::SqrtOp op) {
      return isa<ShapedType>(op.getType());
    });
    target.addLegalDialect<spyreop::SpyreOpDialect>();
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertMathSqrt>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError("LowerSpyreOps: failed to convert math ops");
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton::ktdp {
std::unique_ptr<OperationPass<ModuleOp>> createLowerSpyreOpsPass() {
  return std::make_unique<LowerSpyreOpsPass>();
}
} // namespace mlir::triton::ktdp
