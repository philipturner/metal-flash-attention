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

// Tasks for M3 optimization:
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
// - start by modifying the L/D code to rely on 'preferAsyncLoad'
// - benchmark the impact on occupancy
// - expand 'preferAsyncLoad' to the following specific areas:
//   - dO * O (RHS) [NOT NEEDED]
//   - outer product (RHS) [DONE]
//   - accumulate (RHS) [DONE]
// - compress the threads along the parallelization dimension
//   - when writing at a matrix edge, we now need SIMD barrier(device) instead
//     of threadgroup barrier(threadgroup)
// - where might preferAsyncCache apply?
//   - when loading/storing any cached variables
//     - O(n), so we can just always use device memory for these
//     - NO - there was a performance regression with GEMM bias, when switching
//       from threadgroup load to device load on M3. Allow the option to use
//       threadgroup memory here.
//     - To save time, we will not correct dO * O to match the data about
//       the optimal access pattern.
//   - when loading the LHS in outer product
//   - when paging the accumulator during accumulate
