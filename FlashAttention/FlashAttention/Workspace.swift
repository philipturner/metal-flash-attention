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

// Tasks:
// - change the attention kernel to rely on a two-level descriptor API [DONE]
// - add two optional pathways to elide async copies
//   - reads that would be shared among threads (like GEMM load)
//     - L and D terms during backward key-value
//   - reads that have one-to-one mapping to threads (like GEMM store)
//     - applies to anything that is cacheable
// - Find a way to benchmark the 90% iterations direct, 10% iterations async
//   case on M4, when the problem size is divisible by the block size.
//   - Most severe impact is when the D block is very small
//
// To implement the pathways:
// - start by modifying the L_term/D_term code to rely on 'preferAsyncLoad'
// - benchmark the impact on occupancy
