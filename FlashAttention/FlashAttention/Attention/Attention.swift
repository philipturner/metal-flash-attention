//
//  Attention.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/27/24.
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
}

func createForwardAttention(descriptor: AttentionDescriptor) -> String {
  guard let (R, C, D) = descriptor.matrixDimensions else {
    fatalError("Descriptor was incomplete.")
  }
  
  let paddedD = (D + 8 - 1) / 8 * 8
  
  return """
// What are the Metal function arguments?
// - Q [FP32, FP16]
// - K [FP32, FP16]
// - V [FP32, FP16]
// - O [FP32, FP16]
// - softmax_logsumexp [FP32, FP16]
// - threadgroup_block [uchar]

kernel void forward(
  device float *Q [[buffer(0)]],
  device float *K [[buffer(1)]],
  device float *V [[buffer(2)]],
  device float *O [[buffer(3)]],
  device float *softmax_logsumexp [[buffer(4)]],
  
  threadgroup uchar [[threadgroup(0)]],
  
  uint gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]]
) {
  // What registers are needed before the first matmul can be done?
  // - Q (cached)
  // - O accumulator
  // - m accumulator
  // - l accumulator
  // - location where SIMD 0 will async copy from

  simdgroup_matrix_storage<float> Q_sram[\(paddedD / 8)];
  simdgroup_matrix_storage<float> O_sram[\(paddedD / 8)];
  float m = -numeric_limits<float>::max();
  float l = numeric_limits<float>::denorm_min();
  
}

"""
}

func createBackwardQueryAttention() -> String {
  fatalError("Not implemented.")
}

func createBackwardKeyValueAttention() -> String {
  fatalError("Not implemented.")
}
