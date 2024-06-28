//
//  AttentionKernel.swift
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
// - out: L
//
// Kernel 2:
// - in: Q, K, V
// - in: O
// - in: L
// - in: dO
// - out: dQ
// - out: D[i]
//
// Kernel 3:
// - in: Q, K, V
// - in: D[i]
// - in: L
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

struct AttentionKernel {
  // The source code to compile.
  var source: String = ""
  
  private var leadingDimensions: (
    Q: String, K: String, V: String, O: String)
  private var leadingBlockDimensions: (
    Q: UInt16, K: UInt16, V: UInt16, O: UInt16)
  private var memoryPrecisions: (
    Q: AttentionOperandPrecision,
    K: AttentionOperandPrecision,
    V: AttentionOperandPrecision,
    O: AttentionOperandPrecision)
  private var paddedD: UInt16
  
  init(descriptor: AttentionDescriptor) {
    guard let matrixDimensions = descriptor.matrixDimensions,
          let memoryPrecisions = descriptor.memoryPrecisions,
          let transposeState = descriptor.transposeState,
          let type = descriptor.type else {
      fatalError("Descriptor was incomplete.")
    }
    self.memoryPrecisions = memoryPrecisions
    
    // Inject the contents of the headers.
    source += """
  \(createMetalSimdgroupEvent())
  \(createMetalSimdgroupMatrixStorage())
  using namespace metal;

  """
    
    // Declare the size of the register allocation.
    paddedD = (matrixDimensions.D + 8 - 1) / 8 * 8
    
    // Determine the block dimensions from the transpose state.
    leadingDimensions = ("D", "D", "D", "D")
    leadingBlockDimensions = (paddedD, paddedD, paddedD, paddedD)
    if transposeState.Q {
      leadingDimensions.Q = "R"
      leadingBlockDimensions.Q = 32
    }
    if transposeState.K {
      leadingDimensions.K = "C"
      leadingBlockDimensions.K = 32
    }
    if transposeState.V {
      leadingDimensions.V = "C"
      leadingBlockDimensions.V = 32
    }
    if transposeState.O {
      leadingDimensions.O = "R"
      leadingBlockDimensions.O = 32
    }
    
    // Add the function constants.
    do {
      source += """

  // Dimensions of each matrix.
  constant uint R [[function_constant(0)]];
  constant uint C [[function_constant(1)]];
  constant uint D [[function_constant(2)]];

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
    
    source += createSetup(type: type)
  }
}

extension AttentionKernel {
  func createSetup(type: AttentionKernelType) -> String {
    var output: String = ""
    
    struct AttentionOperand {
      var precision: GEMMOperandPrecision
      var bufferBinding: Int
    }
    
    // Index the operands available during the forward pass.
    var operandsMap: [String: AttentionOperand] = [:]
    operandsMap["Q"] = AttentionOperand(
      precision: memoryPrecisions.Q.forwardPrecision, bufferBinding: 0)
    operandsMap["K"] = AttentionOperand(
      precision: memoryPrecisions.K.forwardPrecision, bufferBinding: 1)
    operandsMap["V"] = AttentionOperand(
      precision: memoryPrecisions.V.forwardPrecision, bufferBinding: 2)
    operandsMap["O"] = AttentionOperand(
      precision: memoryPrecisions.O.forwardPrecision, bufferBinding: 3)
    operandsMap["L"] = AttentionOperand(
      precision: memoryPrecisions.O.forwardPrecision, bufferBinding: 4)
    
    // Index the operands available during the backward pass.
    operandsMap["dO"] = AttentionOperand(
      precision: memoryPrecisions.O.backwardPrecision, bufferBinding: 5)
    operandsMap["D_terms"] = AttentionOperand(
      precision: memoryPrecisions.O.backwardPrecision, bufferBinding: 6)
    operandsMap["dV"] = AttentionOperand(
      precision: memoryPrecisions.V.backwardPrecision, bufferBinding: 7)
    operandsMap["dS"] = AttentionOperand(
      // The default kernel doesn't support writing the attention matrix to
      // memory. The purpose of dS is to increase performance when possible. If
      // users wanted to set dS to FP32 for correctness, that would defeat the
      // purpose. In addition, dS serves as a temporary allocation. Its
      // contents should not be visible to any code that would measure
      // numerical correctness.
      //
      // TODO: Find a workable way to set 'L', 'D_terms', and 'dS' terms to
      // reasonable default values. While allowing users to override them for
      // debugging purposes.
      precision: AttentionOperandPrecision.mixed.backwardPrecision,
      bufferBinding: 8)
    
    
    output += """

kernel void forward(
  

"""
    
    output += """
  
  threadgroup uchar *threadgroup_block [[threadgroup(0)]],
  
  uint gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]]
) {
  ushort2 morton_offset = morton_order(lane_id);

"""
    
    output += """
  
  // Prefetch the Q block.
  auto Q_block = (threadgroup float*)threadgroup_block;
  if (sidx == 0) {
    uint2 Q_offset(0, gid * R_group);
    auto Q_src = simdgroup_matrix_storage<float>::apply_offset(
      Q, \(leadingDimensions.Q), Q_offset, Q_trans);
    
    ushort R_tile_dimension = min(uint(R_group), R - Q_offset.y);
    ushort2 Q_tile_src(D, R_tile_dimension);
    ushort2 Q_tile_dst(\(paddedD), R_tile_dimension);
    
    simdgroup_event event;
    event.async_copy(Q_block, \(leadingBlockDimensions.Q), Q_tile_dst,
                     Q_src, \(leadingDimensions.Q), Q_tile_src, Q_trans);
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
      Q_block, \(leadingBlockDimensions.Q), Q_offset, Q_trans);
    
    for (ushort d = 0; d < \(paddedD); d += 8) {
      ushort2 origin(d, 0);
      Q_sram[d / 8].load(
        Q_block, \(leadingBlockDimensions.Q), origin, Q_trans);
    }
  }
}

"""
    
    return output
  }
}
