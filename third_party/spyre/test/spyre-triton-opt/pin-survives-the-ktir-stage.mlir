// RUN: spyre-triton-opt %s --spyre-ttir-to-ktir | FileCheck %s

// A pinned value with NO OTHER USE must survive the `ktir` stage. This is a
// whole-stage test because no single pass can show it: the hazard is an interaction
// between LowerTTSMarkers and the DCE that the stage's closing canonicalize performs.
//
// The hazard. A marker op is not memory-effect-free, so DCE leaves it -- and while it
// stands it USES the value it marks, which is what keeps that value alive. The
// `tts.pin` ATTRIBUTE does not: it rides on the producer, and a producer whose results
// are otherwise unused is trivially dead. So if LowerTTSMarkers ran before this
// stage's closing canonicalize, that canonicalize would delete the very value the pin
// asked to place, and the attribute would go with it. The pass therefore runs LAST,
// after the canonicalize: the marker holds the value across the DCE, and the attribute
// is written when nothing left in the stage deletes anything.
//
// `math.exp` here has exactly one use -- the marker -- which is what makes this the
// case that FAILS under the wrong order rather than merely passing under the right
// one. It is not a contrived shape: a relayout's share looks like this by the time a
// compose has consumed it, because the compose consumes it by erasing the marker.
//
// What the stage is asked for is an ATTRIBUTE and nothing more. Whether anything
// honours it is the next stage's question and is not asked here -- see
// pin-survives-the-stage.mlir, which drives both stages.

// CHECK-LABEL: func.func @pinned_value_with_no_other_use
// The producer survived, carrying the request.
// CHECK: math.exp
// CHECK-SAME: tts.pin = {memory_space = "ct_local", offset = 4096 : i32}
// And the marker op did not survive: an op from this dialect would fail to parse in a
// consumer that does not load it, which is why the annotation is an attribute.
// CHECK-NOT: tts.pin %

tt.func public @pinned_value_with_no_other_use(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}
