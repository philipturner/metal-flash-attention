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
// - add two optional pathways to elide async copies [DONE]
//   - reads that would be shared among threads (like GEMM load) [DONE]
//     - L and D terms during backward key-value
//   - reads that have one-to-one mapping to threads (like GEMM store) [DONE]
//     - applies to anything that is cacheable
// - Find a way to benchmark the 90% iterations direct, 10% iterations async
//   case on M4, when the problem size is divisible by the block size.
//   - Most severe impact is when the D block is very small
//   - Figure out which option is fastest on M1, as well.
// - Specify the usage of async load/store on a per-operand level,
//   in attention kernel.
// - Add a randomized correctness test for combinations of AttentionKernel
//   config options, problem sizes, and eventually precisions. [DONE]
//
// To implement the pathways:
// - start by modifying the L/D code to rely on 'preferAsyncLoad'
// - benchmark the impact on occupancy
// - expand 'preferAsyncLoad' to the following specific areas:
//   - dO * O (RHS) [NOT NEEDED]
//   - outer product (RHS) [DONE]
//   - accumulate (RHS) [DONE]
// - compress the threads along the parallelization dimension
// - where might preferAsyncCache apply?
//   - when loading/storing any cached variables [DONE]
//   - when loading the LHS in outer product [DONE]
//   - when paging the accumulator during accumulate [DONE]
//
// To benchmark the pathways:
// - Check for correctness regressions on M3. [DONE]
// - Confirm the effectiveness of async copy elisions.
//   - M1 performance should not regress (addressSpace = threadgroup).
//   - M3 performance should improve (addressSpace = device).
//   - Find optimal address spaces on each architecture.
// - Check for performance regressions with indivisible problem sizes.
//   - Compare side by side:
//   - One less than the power of 2 (both sequence and head decrease)
//   - The power of 2
// - Test for coupling betwen optimal address space and problem divisibility.
//   - Is this affected by whether the problem size is divisible? If so,
//     something is going wrong.
