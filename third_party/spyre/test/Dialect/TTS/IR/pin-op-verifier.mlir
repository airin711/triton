// RUN: spyre-triton-opt %s -split-input-file -verify-diagnostics

// The tts.pin OP's verifier. No pass runs here.
//
// The offset is an ATTRIBUTE, and that removes most of what this file used to
// test rather than relocating it. A shape rule needs an expression to have a
// shape: when the offset was an SSA operand, a run-time value, a non-constant
// coefficient, a `program_id` on the wrong axis and a sum of two `program_id`
// terms were all well-formed arith that had to be refused one at a time. None of
// them is spellable in an attribute, so none of them needs a rule.
//
// What is left is what a type constraint cannot say. ODS narrows the offset to a
// single i32 and the memory space to a string; the verifier adds the rules that are
// about MEANING rather than shape: whether the space is a name ktdp defines, which
// of those kinds a pin may name, and that it states an offset.
//
// Note what this verifier does NOT ask, which is anything about the pinned value's
// PROVENANCE -- see @a_block_argument_verifies below.
//
// Note what is NOT here, and why it is absent rather than relocated: a ct_id. The
// memory space is a NAME and not `#ktdp.memory_space`, so `ct_local, ct_id = 7` --
// core 7's scratchpad, which is a different request from a pin -- cannot be written
// at all. See TTSOps.td for why the field is a name: the op is built during
// tracing, and constructing ktdp's attribute there would load `func` and break the
// `ttir` stage's Inliner.
//
// The numeric rules -- a range that fits the scratchpad, ranges that do not
// overlap -- are arithmetic on those numbers and belong to the consumer, which is
// MaterializePinnedBuffers. Alignment is not among them anywhere: an offset counts
// from a base the scratchpad allocator assigns, and aligning that base is the
// allocator's rule rather than the author's.

// A name ktdp does not define. Refused here and not by a type constraint, which is
// the cost of the string: a `#ktdp.memory_space` operand would have made this a
// parse error, and nothing earlier can reject a name the enum never heard of.
tt.func @not_a_memory_space_kind(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{'lx' is not a ktdp memory space kind}}
  tts.pin %e {memory_space = "lx", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// Case matters, so a plausible-looking near-miss is refused with the same message
// rather than silently matching.
tt.func @wrong_case_memory_space(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{'CT_LOCAL' is not a ktdp memory space kind}}
  tts.pin %e {memory_space = "CT_LOCAL", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// `global` is a KNOWN kind and still not pinnable, so it gets its own message
// rather than being reported as a misspelling. lx-placement.md's first
// assumption puts HBM intermediates outside a pin: one is written as a
// tl.make_tensor_descriptor with an explicit store and load, and nothing here
// allocates an anonymous device buffer.
tt.func @global_memory_space(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{memory space 'global' cannot be pinned: only 'ct_local' is}}
  tts.pin %e {memory_space = "global"} : tensor<4x64xf16>
  tt.return
}

// -----
// And with an offset too, so the message is the space's either way rather than
// changing depending on what else the pin carries.
tt.func @global_with_offset(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{memory space 'global' cannot be pinned}}
  tts.pin %e {memory_space = "global", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// An offset that is not a single i32. There is one spelling, so anything else is
// refused by the type constraint rather than by a hand-written rule -- including
// the per-core array, which is the form a reader is most likely to expect here.
tt.func @array_offset(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{failed to satisfy constraint: 32-bit signless integer attribute}}
  tts.pin %e {memory_space = "ct_local", offset = array<i32: 4096, 6144>} : tensor<4x64xf16>
  tt.return
}

// -----
// A dynamic extent cannot be placed: there is no buffer size, so no capacity
// answer and nothing to build a view over. Refused by the operand's type
// constraint rather than by a hand-written rule.
tt.func @dynamic_shape(%x: tensor<?x64xf16>) {
  %e = math.exp %x : tensor<?x64xf16>
  // expected-error @+1 {{operand #0 must be statically shaped tensor of any type values}}
  tts.pin %e {memory_space = "ct_local", offset = 4096 : i32} : tensor<?x64xf16>
  tt.return
}

// The forms that DO verify, so the rejections above are read as rules and not as
// the op being hard to satisfy.
//
// Note there are no rules of dashes anywhere in this file, and no comment quotes
// the split marker either: -split-input-file matches the marker as a substring,
// so both would start a new chunk and the text after them would be parsed as IR.

// -----
tt.func @accepted_forms(%x: tensor<4x64xf16>) {
  %e0 = math.exp %x : tensor<4x64xf16>
  %e1 = math.exp %x : tensor<4x64xf16>
  %r = tensor.empty() : tensor<f16>

  // One i32, which is the only offset spelling: the same offset on every core.
  tts.pin %e0 {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>

  // Nothing here constrains the NUMBER -- 0 is an offset like any other, and
  // whether the range it starts fits the scratchpad is answerable only against a
  // device description, which is MaterializePinnedBuffers' rule.
  tts.pin %e1 {memory_space = "ct_local", offset = 0 : i32} : tensor<4x64xf16>

  // Rank 0. A scalar is a value with a buffer like any other.
  tts.pin %r {memory_space = "ct_local", offset = 8192 : i32} : tensor<f16>

  tt.return
}

// -----
// A BLOCK ARGUMENT, which VERIFIES -- and this case exists to say that the silence
// is chosen rather than overlooked.
//
// A pin does need a defining op, since that op is what carries the annotation. But
// whether the operand has one is a fact about the operand's PROVENANCE, which is IR
// outside this op, and a verifier runs after every pass. The canonicalizer that runs
// before LowerTTSMarkers folds `x * 1`, `x + 0` and an identity
// reshape/broadcast/transpose by RAUWing this op's operand to the fold's input --
// without touching this op. So a provenance rule here would let a legal rewrite of a
// NEIGHBOURING op invalidate the module and report it against a `tts.pin` the author
// wrote correctly. Contrast the field rules above, which read attributes stored on
// this op that no pass can change without rewriting it.
//
// The rule did not disappear; it moved to the two places that can hold it. Tracing
// refuses a block-argument handle at the kernel line, where an entry input and a
// loop-carried value are still distinguishable and the author is looking at which of
// them they wrote (test_frontend_guards.py). `resolvePin` refuses one it is handed,
// naming both of the causes that remain by then (Transforms/invalid.mlir).
tt.func @a_block_argument_verifies(%x: tensor<4x64xf16>) {
  tts.pin %x {memory_space = "ct_local", offset = 4096 : i32} : tensor<4x64xf16>
  tt.return
}

// -----
// No offset. The field is OPTIONAL in ODS and required here, which is the honest
// statement of what exists: the offsetless form is the design's baseline -- the
// compiler places every intermediate and a pin only overrides where -- but nothing
// in this tree can act on one, so accepting it would put an annotation in the
// artifact that no consumer could honour.
//
// Keeping the field optional is what lets this refusal be lifted without the
// surface changing shape.
tt.func @no_offset(%x: tensor<4x64xf16>) {
  %e = math.exp %x : tensor<4x64xf16>
  // expected-error @+1 {{no offset, and nothing here can choose one}}
  tts.pin %e {memory_space = "ct_local"} : tensor<4x64xf16>
  tt.return
}
