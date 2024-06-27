//
//  Workspace.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/20/24.
//

import Metal
import QuartzCore

#if true
/// The repo author's own workspace for running tests and developing kernels.
/// The contents of this function have no meaning, and ideally will be blank
/// when the 'main' branch is in a stable state. Clients can utilize this
/// function to script tests in their fork.
func executeScript() {
  print("Hello, console.")
  
  // Implement "naive attention" with the unified GEMM kernel. Measure
  // performance of the forward and backward pass with various problem configs.
  //
  // Forward:
  // S = Q K^T
  // P = softmax(S / sqrt(D))
  // O = P V
  //
  // Backward:
  // D[i] = generateDTerms(dO, O)
  // dV = P^T dO
  // dP = dO V^T
  // dS = derivativeSoftmax(dP, P, D[i]) / sqrt(D)
  // dK = dS^T Q
  // dQ = dS K
  //
  // To simplify the code, only two precision configurations will be supported.
  // Full 32-bit and full 16-bit.
  //
  // Maybe the code complexity is an inherent issue with naive attention.
  // FlashAttention would decrease the number of memory allocations and the
  // number of kernels dispatched.
  
}
#endif
