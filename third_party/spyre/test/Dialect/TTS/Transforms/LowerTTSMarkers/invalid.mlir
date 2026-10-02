// RUN: spyre-triton-opt %s -split-input-file --lower-tts-markers -verify-diagnostics

// A marker whose operand does not resolve to anything that can carry its
// attribute.
//
// This is the per-marker half of the pass stated as a refusal: `tts.tensor_layout`
// admits exactly one resolution -- the lowered-descriptor bridge
// LowerDescriptorMemory leaves, and the ktdp.construct_memory_view behind it --
// and anything else is an error rather than a skip. Silently skipping would take
// the kernel's layout with the marker and leave a logical artifact that looks
// correct; asserting would be wrong for a pass that is invocable on hand-written
// IR, which is what these two cases are.

// A descriptor that was never lowered: the operand is a function argument of
// !tt.tensordesc type, so there is no bridge cast at all. This is also what
// reaching the pass out of order looks like -- before LowerDescriptorMemory, or
// on a descriptor that pass declined because it could not recover the shape.
tt.func @not_lowered(%desc: !tt.tensordesc<64x64xf32>) {
  // expected-error @+1 {{tts.tensor_layout does not annotate a lowered tensor descriptor: expected its operand to be the builtin.unrealized_conversion_cast that lower-descriptor-memory leaves bridging a memref back to !tt.tensordesc}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// A bridge cast of the right shape over the wrong memref: the operand really is
// an unrealized_conversion_cast from a memref, so the first check passes, but the
// memref is a bare allocation and no memory view defines it. The layout has
// nothing to land on, and the two checks are separate for exactly this reason --
// the first says "is this a bridge", the second says "does it bridge a view".
tt.func @no_memory_view() {
  %alloc = memref.alloc() : memref<64x64xf32>
  %desc = builtin.unrealized_conversion_cast %alloc
      : memref<64x64xf32> to !tt.tensordesc<64x64xf32>
  // expected-error @+1 {{tts.tensor_layout does not annotate a lowered tensor descriptor: its operand bridges a memref that no ktdp.construct_memory_view defines, so there is no memory view to carry the layout}}
  tts.tensor_layout %desc
    {phys_src = array<i64: 1, 0, 1>,
     phys_op = array<i64: 1, 0, 2>,
     phys_arg = array<i64: 64, 0, 64>} : !tt.tensordesc<64x64xf32>
  tt.return
}

// -----

// Two markers reaching one subject and DISAGREEING. `setAttr` would resolve this by
// overwriting, so the later marker would win and the earlier would be gone with
// nothing said -- and which one survived would be a fact about the driver's loop
// order rather than about anything the author wrote.
//
// Disagreeing is the whole rule: two markers stating the same thing write the value
// that is already there and are accepted, which
// tensor-layout-attribute.mlir's @two_identical_markers drives. So the message names
// both statements, since what the author has to fix is the difference between them.
//
// Two pins on one VALUE is the reachable shape of it: they share a producer, and
// the producer is the carrier. The generic driver holds the check rather than the
// pin, because the hazard is one subject reachable from two markers, which
// `tts.tensor_layout` has its own spelling of -- two layouts on one descriptor
// share a memory view.
tt.func @two_pins_on_one_value(%x: tensor<4x64xf16>) -> tensor<4x64xf16> {
  // expected-note @+1 {{the op both resolve to is here}}
  %e = math.exp %x : tensor<4x64xf16>
  tts.pin %e {memory_space = "ct_local", offset = 0 : i32} : tensor<4x64xf16>
  // expected-error @+1 {{second tts.pin resolving to the same op with a different statement, which can carry only one; this one states {memory_space = "ct_local", offset = 512 : i32}, the first states {memory_space = "ct_local", offset = 0 : i32}}}
  tts.pin %e {memory_space = "ct_local", offset = 512 : i32} : tensor<4x64xf16>
  tt.return %e : tensor<4x64xf16>
}

// -----

// A producer with several results. UNSUPPORTED rather than ill formed: the value
// is a good thing to pin and what is missing is a spelling, since one attribute on
// the producer can carry one pin and has no way to say which result it is for.
//
// ARGMAX is what reaches it, now that this pass runs at the end of the stage:
// `tl.max(..., return_indices=True)` is one `linalg.reduce` with two results by here,
// a value and an index. Written as that linalg.reduce rather than as a stand-in,
// because the shape is the point -- there is no kernel-level rewrite to suggest, the
// way there would be for a loop carrying two values, since one reduction producing
// two results cannot be written as two reductions.
//
// A loop carrying two values is no longer this pass's case at all: its iter_args are
// block arguments, and a pin on one is refused at trace time.
//
// The pinned VALUE is well formed, which is why this is the lowering's rule and
// not the op's: the op holds its value as an operand and can check that, while
// only resolution knows which op is about to carry the annotation.
tt.func @multi_result_producer(%x: tensor<4x64xf32>, %idx: tensor<4x64xi32>)
    -> tensor<4xf32> {
  %vinit = tensor.empty() : tensor<4xf32>
  %iinit = tensor.empty() : tensor<4xi32>
  // expected-note @+1 {{the producer is here}}
  %v, %i = linalg.reduce
      ins(%x, %idx : tensor<4x64xf32>, tensor<4x64xi32>)
      outs(%vinit, %iinit : tensor<4xf32>, tensor<4xi32>)
      dimensions = [1]
      (%in: f32, %inidx: i32, %accv: f32, %acci: i32) {
        %gt = arith.cmpf ogt, %in, %accv : f32
        %nv = arith.select %gt, %in, %accv : f32
        %ni = arith.select %gt, %inidx, %acci : i32
        linalg.yield %nv, %ni : f32, i32
      }
  // expected-error @+1 {{pinning one result of a 2-result op is not supported: the annotation is one attribute on the producer, so it cannot say which result it is for. An argmax is the shape that reaches this}}
  tts.pin %v {memory_space = "ct_local", offset = 4096 : i32} : tensor<4xf32>
  tt.return %v : tensor<4xf32>
}

// -----

// A pin naming a BLOCK ARGUMENT, which is this pass's refusal now that the op's
// verifier has stopped making it. Two causes remain by the time a module is here and
// the message names both, because the second is the one that will surprise someone:
// the kernel pinned a value an op produced, and a fold replaced that op with its own
// input. Tracing has already excluded the author writing one.
//
// Hand-written IR is the other cause, and is what this case is.
tt.func @pin_names_a_block_argument(%x: tensor<4x64xf16>) {
  // expected-error @+1 {{tts.pin names a block argument, which no op produces, so there is nothing to carry the annotation}}
  tts.pin %x {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}
