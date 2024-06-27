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
  
  // Design a set of simple kernels for forward and backward FlashAttention:
  // - FP32 (hardcoded data type keyword)
  // - 32x32 block, 4 splits (hardcoded block size)
  // - all GEMM operands accessed like with standard GEMM + M1
  //   - use async copies
  //   - transposes are supported
  // - no masking, dropout, etc.
  //
  // Kernel 1:
  // - in: Q, K, V
  // - out: O
  // - out: logsumexp
  //
  // Kernel 2:
  // - in: Q, K, V
  // - in: O
  // - in: logsumexp
  // - in: dO
  // - out: dQ
  // - out: D[i]
  //
  // Kernel 3:
  // - in: Q, K, V
  // - in: D[i]
  // - in: logsumexp
  // - in: dO
  // - out: dK, dV
  
}
#endif
