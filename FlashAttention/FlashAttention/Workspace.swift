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
func executeScript() {
  // Next, implement "naive attention" with the unified GEMM kernel. Measure
  // performance of the forward and backward pass with various problem configs.
  //
  // This will require a few custom GPU kernels:
  // - softmax
  //   - load all of the matrix elements into registers
  //   - requires knowledge of problem size up-front
  // - D[i] term
  // - dS elementwise
  //   - find a workable way to copy the entire unified GEMM kernel source,
  //     without adding technical debt to the minimal implementation in-tree
  //   - initialize accumulator for dP as D[i]
  //   - "use async store" becomes "load previous C value with async copy"
  //   - C will always be written to threadgroup memory
  //
  // Task 1:
  // - Modify the finite differencing test and "network" to correctly scale
  //   the GEMM output by sqrt(D).
  //
  // Task 2:
  // - Design a theoretically optimal softmax kernel.
  //
  // Task 3:
  // - Make a copy of the in-tree GEMM kernel, which fuses some operations
  //   during computation of dS.
}
