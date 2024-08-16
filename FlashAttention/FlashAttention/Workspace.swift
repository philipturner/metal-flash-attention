//
//  Workspace.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import Metal
import QuartzCore

/// The repo author's own workspace for running tests and developing kernels.
/// The contents of this function have no meaning, and ideally will be blank
/// when the 'main' branch is in a stable state. Clients can utilize this
/// function to script tests in their fork.

// Refactoring:
// - Verify that the new GEMM shader cache works correctly. [DONE]
// - Run the adversarial shape tests with the new caching mechanism. [DONE]
// - Incorporate the new design philosophy into the attention test. [DONE]
//
// Mixed precision:
// - Allow input, L/D, and output memory types to be FP16/BF16. [DONE]
//   - Debug the numerical correctness of frequently truncating O to FP16
//     during forward. Do larger traversal block sizes help? [DONE]
// - Allow input, L/D, and output register types to be FP16/BF16. [DONE]
// - Allow attention matrix register types to be FP16/BF16. [DONE]
//   - Try a situation where the memory precision is FP32, and the register
//     precision is forced to be FP16. This should be flagged as a misuse and
//     precondition failure IRL. But during development, I need to have
//     feedback that the correct register precision is being employed. There
//     should be a reduction in numerical accuracy. [DONE]
//   - Examine generated code when memory precision is BF16. "load_bfloat"
//     and "store_bfloat" should appear on M1, and not appear on M3. [DONE]
//   - Benchmark the numerical accuracy of accumulating the attention matrix
//     in FP16, provided both inputs are FP16. [DONE]
// - Check for improved occupancy on M1. [DONE]
//   - Does BF16 compression in memory make the occupancy worse? [DONE]
//   - Reduce the threadgroup memory allocation with mixed precision. [DONE]
//   - Change the GEMM and Attention kernels, so the threadgroup memory
//     allocation is computed beforehand. [DONE]
