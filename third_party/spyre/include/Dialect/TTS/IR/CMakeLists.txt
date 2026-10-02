# The tts dialect's declarations.  Two tablegen inputs: the dialect itself, and
# the authoring ops.  No -gen-typedef-decls / -gen-attrdef-decls -- the dialect
# defines no type and no attribute type, and must not, since the point of a
# discardable `tts.` name is that a consumer which does not load THIS dialect can
# still parse it.  Every `tts.` attribute's value is builtin for that reason -- see
# the note on `kPinAttrName` in TTSDialect.td.
#
# third_party/spyre/CMakeLists.txt already puts the source and binary include/
# trees on the path, which is what every `#include "Dialect/TTS/..."` resolves
# against.  The Triton include tree is on the path from the top-level
# CMakeLists.txt, which is what lets TTSOps.td pull in TritonTypes.td for
# TT_TensorDescType.
#
# Nothing of the ktir frontend's tree is on the path here, and nothing needs to be:
# no .td under this directory includes one of theirs.  `tts.pin` holds its memory
# space as a builtin string rather than as `#ktdp.memory_space`, for the reason
# TTSOps.td gives -- the op is built during tracing, where loading ktdp would load
# `func` and break the `ttir` stage's Inliner.
set(LLVM_TARGET_DEFINITIONS TTSDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=tts)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=tts)

set(LLVM_TARGET_DEFINITIONS TTSOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
# No add_mlir_doc: it resolves its output against MLIR_BINARY_DIR, which no
# CMakeLists under third_party/spyre sets, so a doc target here writes to /docs.
# Nothing in this tree generates dialect docs, so there is nothing to set it for.

# One target for both, so every dependent gets all four generated files from a
# single DEPENDS entry.
add_public_tablegen_target(TTSIncGen)
