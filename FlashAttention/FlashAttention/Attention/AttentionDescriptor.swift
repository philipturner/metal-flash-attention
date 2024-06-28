//
//  AttentionDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/28/24.
//

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
//
// ## Reference Pseudocode for the Attention Operation
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

enum AttentionKernelType {
  /// Forward attention, computing O and softmax\_logsumexp.
  ///
  /// Variants:
  /// - `false`: compute O
  /// - `true`: compute O and softmax\_logsumexp
  case forward(Bool)
  
  /// Backward attention, computing D[i] and dQ.
  ///
  /// Variants:
  /// - `false`: compute D[i]
  /// - `true`: compute D[i] and dQ
  ///
  /// Depends on: softmax\_logsumexp
  case backwardQuery(Bool)
  
  /// Backward attention, computing dK and dV.
  ///
  /// Variants:
  /// - `false`: compute dV, store the intermediate dS
  /// - `true`: compute dV and dK
  ///
  /// Depends on: softmax\_logsumexp, D[i]
  case backwardKeyValue(Bool)
}

struct AttentionDescriptor {
  /// The dimensions of the input and output matrices.
  /// - Parameters R: Number of rows in the attention matrix.
  /// - Parameters C: Number of columns in the attention matrix.
  /// - Parameters D: Size of the head.
  ///
  /// If you wish to support multiple heads (e.g. an 'H' dimension), you will
  /// have to modify the kernel to compute multiple attention matrices
  /// simultaneously. Multi-head attention typically interleaves the data
  /// along the 'D' dimension in memory. If so, the stride between rows of the
  /// Q/K/V operands will change from 'D' to 'H * D'. The code must be modified
  /// accordingly.
  ///
  /// Recommendation: bake the 'H' dimension into the GPU assembly code, either
  /// when generating the shader source (JIT) or through function constants.
  /// Entering 'H' (or any other info about the data's layout in memory) through
  /// a buffer table argument at runtime will seriously harm performance.
  var matrixDimensions: (R: UInt32, C: UInt32, D: UInt16)?
  
  /// Whether each operand is transposed in RAM.
  ///
  /// If the layout is row-major, where a row spans D contiguous elements in
  /// memory, enter `false`. If the layout is column-major, where a row spans
  /// D widely separated elements in memory, enter `true`.
  ///
  /// The transpose state of the corresponding derivative (e.g. dQ for Q) must
  /// be the same as the input from the forward pass.
  var transposeState: (Q: Bool, K: Bool, V: Bool, O: Bool)?
}
