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
// - Verify that the new GEMM shader cache works correctly.
// - Run the adversarial shape tests with the new caching mechanism.
// - Incorporate the new design philosophy into the current attention test.
//
// Mixed precision:
// - Debug the numerical correctness of frequently truncating O to FP16
//   during forward. Do larger traversal block sizes help?
