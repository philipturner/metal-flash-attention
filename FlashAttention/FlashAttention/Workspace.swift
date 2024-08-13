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
// - Confirm the effectiveness of async copy elisions. [DONE]
//   - M1 performance should not regress (addressSpace = threadgroup). [DONE]
//   - M3 performance should improve (addressSpace = device). [DONE]
//   - Find optimal address spaces on each architecture. [DONE]
// - Find the reason for oscillatory performance that suspiciously looks like
//   the compute work along the head edge, is not being elided.
//   - Can we find a patch that doesn't harm occupancy?
// - Find additional regressions due to indivisible problem sizes. [DONE]
//   - Compare side by side: [DONE]
//   - One less than the power of 2 (both sequence and head decrease) [DONE]
//   - The power of 2 [DONE]
// - Test for coupling betwen optimal address space and problem divisibility.
//   - Is this affected by whether the problem size is divisible? If so,
//     something is going wrong.

// TODO: To fix the issue of performance depending on head size divisibility.
// - Analyze the generated source code.
// - Find how it might be refactored, to bring the extra branch outside of
//   the inner loop.
// - Test how the fix changes occupancy.
// - Test how the fix changes performance.
// - Try optimizing for when the head size is divisible by 8, but just not
//   divisible by the block size.
// - As one final optimization, try duplicating the GEMM work at the edge of
//   the matrix (when possible) to fuse the final iteration with the rest. Or
//   make the async copy pad the entire edge with zeroes.
