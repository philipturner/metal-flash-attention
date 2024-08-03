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
// - runtime boolean shader argument
// - negligible performance delta, although omitted from reference impl.
// - easiest way to support in-place accumulation
// - include "previous C matrix" in shape tests
// - check for a performance regression in the Laplacian test, both when C
//   is and isn't loaded from a previous value. Is it correct and performant
//   on both M1 Max and M4?
//
// if (accumulate) {
//   load
// } else {
//   zeroinit
// }
//
// Noncontiguous strides
// - include leading dimension A, B, and C in the function constants, just
//   like BLAS
