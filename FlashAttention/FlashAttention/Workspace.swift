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
// - Allow input, L/D, and output memory types to be FP16/BF16.
//   - Debug the numerical correctness of frequently truncating O to FP16
//     during forward. Do larger traversal block sizes help? Yes [DONE]
// - Allow input, L/D, and output register types to be FP16/BF16.
// - Allow attention matrix register types to be FP16/BF16.
//   - Benchmark the numerical accuracy of accumulating the attention matrix
//     or O in FP16, provided both inputs are FP16. Try adding a test case
//     that covers larger sequence lengths, where the accuracy of FP16 O would
//     suffer.
// - Check for improved occupancy on M1.
