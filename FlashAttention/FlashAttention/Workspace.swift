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

// Zeroinit branch in GEMM
// - Profile the kernel with "load previous C" done through a function
//   constant, on both M1 and M4. Are there any regressions?
//
// Noncontiguous strides
// - Include leading dimension A, B, and C in the function constants, just
//   like BLAS.
// - Change "paddedBlockDimensions" to "leadingBlockDimensions".
//
// Attention
// - Backward Key-Value gains the ability to work on a subregion of the
//   attention matrix at a time.
