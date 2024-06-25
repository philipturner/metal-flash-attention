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
  // TODO: Next, implement "naive attention" with the unified GEMM kernel.
  // Measure performance of the forward and backward pass with various problem
  // configurations.
  //
  // This will require a few custom GPU kernels:
  // - softmax
  // - D[i] term
  // - dS elementwise
}
