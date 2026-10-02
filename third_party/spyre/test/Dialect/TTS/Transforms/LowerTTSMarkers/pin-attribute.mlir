// RUN: spyre-triton-opt %s -split-input-file --lower-tts-markers | FileCheck %s

// The `tts.pin` marker becoming the `tts.pin` attribute.
//
// Two things happen, and the first is the one with a choice in it:
//
//   1. the marker's memory_space and offset land on the op DEFINING the pinned
//      value, as a `tts.pin` dictionary attribute;
//   2. the marker op is erased.
//
// (1) is where this marker differs from `tts.tensor_layout`. A layout names a
// descriptor, which has resolved to one particular op -- a memory view -- and the
// pass checks that it did. A pin names a VALUE, and a value's only op is the one
// defining it, whatever that op happens to be. So there is no admissibility test
// on the carrier: the cases below pin a `math` result and a named `linalg` one --
// the two forms a pinned value actually takes at this point in the pipeline -- and
// the pass treats them the same way, because a consumer reads the attribute and
// never the op's identity.
//
// The dead-cast cleanup is a case here rather than an absence, which it used to be.
// A pin's operand is a `tensor` throughout -- no pass retypes it -- so no cast ever
// stands BETWEEN a pin and its value, the way one does for a layout. What can happen
// is the other thing: the pin's producer IS a cast, and then the cleanup would erase
// the very op the attribute was just written on. See @cast_producer_is_the_carrier.
//
// Both fields are carried THROUGH rather than interpreted. Whatever number the
// offset holds and whatever name the space holds, the attribute holds what the op
// held: capacity and disjointness are arithmetic for the consumer, and
// turning the space's name into `#ktdp.memory_space` is the consumer's too -- this
// pass constructs no dialect attribute, which is what lets it declare no dependent
// dialects. MaterializePinnedBuffers does both.

// An elementwise producer, the common case. `math.exp` survives this stage as
// itself -- ConvertElementwiseToLinalg runs in `spyrecode` -- so the attribute
// lands on the math op.
// CHECK-LABEL: tt.func @elementwise_producer
// CHECK: math.exp {{.*}} {tts.pin = {memory_space = "ct_local", offset = 4096 : i32}}
// CHECK-NOT: tts.pin %
tt.func @elementwise_producer(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// A named linalg producer. This is what a pinned `tl.sum` is by the time the pass
// runs, and the reason the pass sits after LowerComputeOps: before it the value
// is produced by a tt.reduce, which that pass REPLACES, and an attribute written
// on the tt.reduce would be dropped with it.
// CHECK-LABEL: tt.func @linalg_producer
// CHECK: linalg.reduce
// CHECK-SAME: {tts.pin = {memory_space = "ct_local", offset = 8192 : i32}}
// CHECK-NOT: tts.pin %
tt.func @linalg_producer(%x: tensor<4x64xf32>) -> tensor<4xf32> {
  %init = tensor.empty() : tensor<4xf32>
  %r = linalg.reduce { arith.addf } ins(%x : tensor<4x64xf32>) outs(%init : tensor<4xf32>) dimensions = [1]
  tts.pin %r {memory_space = "ct_local", offset = 8192 : i32} : tensor<4xf32>
  tt.return %r : tensor<4xf32>
}

// -----
// The offset is reused as the attribute's `offset` entry unchanged: the same i32,
// not restated and not widened. This pass moves the fields the marker carried and
// decides nothing about them.
// CHECK-LABEL: tt.func @offset_moves_unchanged
// CHECK: math.exp {{.*}} {tts.pin = {memory_space = "ct_local", offset = 0 : i32}}
// CHECK-NOT: tts.pin %
tt.func @offset_moves_unchanged(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 0 : i32} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----
// Two pins on two values produced by the same KIND of op, to show the attribute
// is per-op and not per-function: each lands on its own producer with its own
// offset.
// CHECK-LABEL: tt.func @two_pins
// CHECK: math.exp {{.*}} {tts.pin = {memory_space = "ct_local", offset = 0 : i32}}
// CHECK: math.sqrt {{.*}} {tts.pin = {memory_space = "ct_local", offset = 512 : i32}}
// CHECK-NOT: tts.pin %
tt.func @two_pins(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %s = math.sqrt %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 0 : i32} : tensor<4x64xf16>
  tts.pin %s {memory_space = "ct_local", offset = 512 : i32} : tensor<4x64xf16>
  %y = arith.addf %e, %s : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// A pin whose position in the block is BELOW a use of the value. The marker's
// location does not matter, because the annotation is about the value and the
// carrier is its producer -- so this lowers exactly like the same pin written
// above the use. That is the difference between an annotation on a value and a
// marker with a program point.
// CHECK-LABEL: tt.func @pin_below_a_use
// CHECK: math.exp {{.*}} {tts.pin = {memory_space = "ct_local", offset = 4096 : i32}}
// CHECK-NOT: tts.pin %
tt.func @pin_below_a_use(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  %e = math.exp %x : tensor<4x64xf16>
  %y = math.sqrt %e : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return %y : tensor<4x64xf16>
}

// -----
// The pin's producer IS a `builtin.unrealized_conversion_cast`, which is the one
// shape that made the driver's dead-cast cleanup destructive. That cleanup exists for
// a marker whose operand is reached THROUGH a cast -- `tts.tensor_layout`, whose
// subject is the memory view behind it -- and a pin resolves to its operand's
// producer, so here the cast IS the subject. Without the `def != subject` guard step 3
// makes it use-empty and step 4 erases it, taking the attribute with it and leaving a
// function body with nothing in it. Silent: no diagnostic, and the pin simply stops
// existing.
//
// Hand-written, because no pass in this pipeline produces a cast a pin would name.
// That is the point of driving it anyway -- the guard is about the driver's contract
// with a future marker, not about IR this tree emits today.
// CHECK-LABEL: tt.func @cast_producer_is_the_carrier
// CHECK: builtin.unrealized_conversion_cast
// CHECK-SAME: {tts.pin = {memory_space = "ct_local", offset = 4096 : i32}}
// CHECK-NOT: tts.pin %
tt.func @cast_producer_is_the_carrier(%m: memref<4x64xf16>) {
  %c = builtin.unrealized_conversion_cast %m : memref<4x64xf16> to tensor<4x64xf16>
  tts.pin %c {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}
