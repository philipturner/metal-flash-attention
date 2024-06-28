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

func createForwardAttention(descriptor: AttentionDescriptor) -> String {
  guard let matrixDimensions = descriptor.matrixDimensions,
        let transposeState = descriptor.transposeState else {
    fatalError("Descriptor was incomplete.")
  }
  var source: String = ""
  
  // Inject the contents of the headers.
  source += """
\(createMetalSimdgroupEvent())
\(createMetalSimdgroupMatrixStorage())
using namespace metal;

"""
  
  // Declare the size of the register allocation.
  let paddedD = (matrixDimensions.D + 8 - 1) / 8 * 8
  
  // Determine the block dimensions from the transpose state.
  var leadingDimensionQ: String = "D"
  var leadingDimensionK: String = "D"
  var leadingDimensionV: String = "D"
  var leadingDimensionO: String = "D"
  var leadingBlockDimensionQ: UInt16 = paddedD
  var leadingBlockDimensionK: UInt16 = paddedD
  var leadingBlockDimensionV: UInt16 = paddedD
  var leadingBlockDimensionO: UInt16 = paddedD
  if transposeState.Q {
    leadingDimensionQ = "R"
    leadingBlockDimensionQ = 32
  }
  if transposeState.K {
    leadingDimensionK = "C"
    leadingBlockDimensionK = 32
  }
  if transposeState.V {
    leadingDimensionV = "C"
    leadingBlockDimensionV = 32
  }
  if transposeState.O {
    leadingDimensionO = "R"
    leadingBlockDimensionO = 32
  }
  
  // Add the function constants.
  do {
    source += """

// Dimensions of each matrix.
constant uint R = \(matrixDimensions.R);
constant uint C = \(matrixDimensions.C);
constant uint D = \(matrixDimensions.D);

// Whether each matrix is transposed.
constant bool Q_trans = \(transposeState.Q);
constant bool K_trans = \(transposeState.K);
constant bool V_trans = \(transposeState.V);
constant bool O_trans = \(transposeState.O);

// Define the memory layout of the matrix block.
constant ushort R_group = 32;
constant ushort C_group = 32;

"""
  }
  
  // Add the setup portion where the addresses are prepared.
  do {
    source += """

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
  
  threadgroup uchar *threadgroup_block [[threadgroup(0)]],
  
  uint gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]]
) {
  ushort2 morton_offset = morton_order(lane_id);
  
  // What registers are needed before the first matmul can be done?
  // - Q (cached)
  // - O accumulator
  // - m accumulator
  // - l accumulator
  // - location where SIMD 0 will async copy from
  
  // Prefetch the Q block.
  auto Q_block = (threadgroup float*)threadgroup_block;
  if (sidx == 0) {
    uint2 Q_offset(0, gid * R_group);
    auto Q_src = simdgroup_matrix_storage<float>::apply_offset(
      Q, \(leadingDimensionQ), Q_offset, Q_trans);
    
    ushort R_tile_dimension = min(uint(R_group), R - Q_offset.y);
    ushort2 Q_tile_src(D, R_tile_dimension);
    ushort2 Q_tile_dst(\(paddedD), R_tile_dimension);
    
    simdgroup_event event;
    event.async_copy(Q_block, \(leadingBlockDimensionQ), Q_tile_dst,
                     Q_src, \(leadingDimensionQ), Q_tile_src, Q_trans);
    simdgroup_event::wait(1, &event);
  }
  
  // Initialize the accumulator.
  float m = -numeric_limits<float>::max();
  float l = numeric_limits<float>::denorm_min();
  simdgroup_matrix_storage<float> O_sram[\(paddedD / 8)];
#pragma clang loop unroll(full)
  for (ushort d = 0; d < \(paddedD); d += 8) {
    O_sram[d / 8] = simdgroup_matrix_storage<float>(0);
  }
  
  // Load the Q block.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_matrix_storage<float> Q_sram[\(paddedD / 8)];
  {
    ushort2 Q_offset = ushort2(0, sidx * 8) + morton_offset;
    auto Q_src = simdgroup_matrix_storage<float>::apply_offset(
      Q_block, \(leadingBlockDimensionQ), Q_offset, Q_trans);
    
    for (ushort d = 0; d < \(paddedD); d += 8) {
      ushort2 origin(d, 0);
      Q_sram[d / 8].load(Q_block, \(leadingBlockDimensionQ), origin, Q_trans);
    }
  }

  for (ushort c = 0; c < C; c += C_group) {

  }
  
  // TODO: Compile the kernel and fix the related issues before moving on.
  // Perhaps write the initialization procedure for the backward kernels, and
  // find where code can be shared.
}

"""
  }
  
  return source
}

func createBackwardQueryAttention() -> String {
  fatalError("Not implemented.")
}

func createBackwardKeyValueAttention() -> String {
  fatalError("Not implemented.")
}
