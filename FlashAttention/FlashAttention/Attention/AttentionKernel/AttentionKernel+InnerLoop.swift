//
//  AttentionKernel+InnerLoop.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/2/24.
//

// MARK: - Simplest Form of the Pseudocode

// Forward
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     (m, l, P) = softmax(m, l, S * scaleFactor)
//     O *= correction
//     load V[c]
//     O += P * V
//   }
//   O /= l
//
// Backward Query (true)
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     P = exp(S - L)
//     load V[c]
//     dP = dO * V^T
//     dS = P * (dP - D) * scaleFactor
//     load K[c]
//     dQ += dS * K
//   }
//
// Backward Key-Value (false)
//   for r in 0..<R {
//     load Q[r]
//     load L[r]
//     S^T = K * Q^T
//     P^T = exp(S^T - L)
//     load dO[r]
//     load D[r]
//     dV += P^T * dO
//     dP^T = V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//     store dS^T[c][r]
//   }
//
// Backward Key-Value (true)
//   for r in 0..<R {
//     load Q[r]
//     load L[r]
//     S^T = K * Q^T
//     P^T = exp(S^T - L)
//     load dO[r]
//     load D[r]
//     dV += P^T * dO
//     dP^T = V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//     load Q[r]
//     dK += dS^T * Q
//   }

// MARK: - Blocking Along the Head Dimension

// The pseudocode assumes the head dimension (D) is 128.
//
// Forward
//   // Setup
//   initialize O[32][128]
//   initialize m[32]
//   initialize l[32]
//
//   // Inner Loop
//   for c in 0..<C {
//     repeat 4 times
//       load Q[r][32]
//       load K[c][32]
//       S += Q * K^T
//     (m, l, P) = softmax(m, l, S * scaleFactor)
//
//     O *= correction
//     repeat 2 times
//       load V[c][64]
//       O[32][64] += P * V
//   }
//
//   // Cleanup
//   O /= l
//   store dO
//   store L
//
// Backward Query (true)
//   // Setup
//   initialize dQ[32][128]
//   load L[32]
//   repeat 4 times
//     load dO[r][32]
//     load O[r][32]
//     D += dO * O
//
//   // Inner Loop
//   for c in 0..<C {
//     repeat 4 times
//       load Q[r][32]
//       load K[c][32]
//       S += Q * K^T
//     P = exp(S - L)
//
//     repeat 4 times
//       load dO[r][32]
//       load V[c][32]
//       dP += dO * V^T
//     dS = P * (dP - D) * scaleFactor
//
//     repeat 2 times
//       load K[c][64]
//       dQ[32][64] += dS * K
//   }
//
//   // Cleanup
//   store dQ
//   store D
//
// Backward Key-Value (true)
//   // Setup
//   initialize dK[32][128]
//   initialize dV[32][128]
//
//   // Inner Loop
//   for r in 0..<R {
//     load L[r]
//     load D[r]
//
//     repeat 4 times
//       load K[c][32]
//       load Q[r][32]
//       S^T += K * Q^T
//     P^T = exp(S^T - L)
//
//     repeat 4 times
//       load V[c][32]
//       load dO[r][32]
//       dV[32][32] += P^T * dO
//       dP^T += V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//
//     repeat 2 times
//       load Q[r][64]
//       dK[32][64] += dS^T * Q
//   }
//
//   // Cleanup
//   store dK
//   store dV


// MARK: - Implementation

extension AttentionKernel {
  func createInnerLoopForward() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "Q"
    outerProductDesc.B = "K"
    outerProductDesc.C = "S"
    outerProductDesc.transposeA = transposeState.Q
    outerProductDesc.transposeB = transposeState.K
    outerProductDesc.leadingDimensionA = leadingDimensions.Q
    outerProductDesc.leadingDimensionB = leadingDimensions.K
    outerProductDesc.matrixDimensions = (M: "R", N: "C")
    outerProductDesc.matrixOffset = (M: "gid * R_group", N: "c")
    let QKT_Descriptor = outerProduct(descriptor: outerProductDesc)
    
    var accessDesc = AttentionHBMAccessDescriptor()
    accessDesc.index = "c"
    accessDesc.leadingBlockDimension = leadingBlockDimensions.V
    accessDesc.leadingDimension = leadingDimensions.V
    accessDesc.name = "V"
    accessDesc.threadgroupAddress = "threadgroup_block"
    accessDesc.transposeState = transposeState.V
    let prefetchV = prefetchColumns(descriptor: accessDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.index = "c"
    accumulateDesc.indexedBlockDimension = blockDimensions.C
    accumulateDesc.leadingBlockDimensionRHS = leadingBlockDimensions.V
    accumulateDesc.names = (accumulator: "O", lhs: "P", rhs: "V")
    accumulateDesc.threadgroupAddress = "threadgroup_block"
    accumulateDesc.transposeStateRHS = transposeState.V
    let accumulateO = accumulate(descriptor: accumulateDesc)
    
    return """
  
  // Iterate over the columns.
  for (uint c = 0; c < C; c += 32) {
    // S = Q * K^T
    simdgroup_matrix_storage<float> S_sram[32 / 8];
    \(twoOperandAccess(descriptor: QKT_Descriptor))
    \(maskAlongColumns(sram: "S_sram"))
    
    // (m, l, P) = softmax(m, l, S * scaleFactor)
    \(onlineSoftmax())
    
    // load V[c]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchV)
    
    // O += P * V
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(accumulateO)
  }
  
  // O /= l
  float l_reciprocal = 1 / l;
#pragma clang loop unroll(full)
  for (ushort d = 0; d < \(paddedD); d += 8) {
   *(O_sram[d / 8].thread_elements()) *= l_reciprocal;
  }
  
"""
  }
  
  func createInnerLoopBackwardQuery() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "Q"
    outerProductDesc.B = "K"
    outerProductDesc.C = "S"
    outerProductDesc.transposeA = transposeState.Q
    outerProductDesc.transposeB = transposeState.K
    outerProductDesc.leadingDimensionA = leadingDimensions.Q
    outerProductDesc.leadingDimensionB = leadingDimensions.K
    outerProductDesc.matrixDimensions = (M: "R", N: "C")
    outerProductDesc.matrixOffset = (M: "gid * R_group", N: "c")
    let QKT_Descriptor = outerProduct(descriptor: outerProductDesc)
    
    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "dO"
    outerProductDesc.B = "V"
    outerProductDesc.C = "dP"
    outerProductDesc.transposeA = transposeState.O
    outerProductDesc.transposeB = transposeState.V
    outerProductDesc.leadingDimensionA = leadingDimensions.O
    outerProductDesc.leadingDimensionB = leadingDimensions.V
    outerProductDesc.matrixDimensions = (M: "R", N: "C")
    outerProductDesc.matrixOffset = (M: "gid * R_group", N: "c")
    let dOVT_Descriptor = outerProduct(descriptor: outerProductDesc)
    
    var accessDesc = AttentionHBMAccessDescriptor()
    accessDesc.index = "c"
    accessDesc.leadingBlockDimension = leadingBlockDimensions.K
    accessDesc.leadingDimension = leadingDimensions.K
    accessDesc.name = "K"
    accessDesc.threadgroupAddress = "threadgroup_block"
    accessDesc.transposeState = transposeState.K
    let prefetchK = prefetchColumns(descriptor: accessDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.index = "c"
    accumulateDesc.indexedBlockDimension = blockDimensions.C
    accumulateDesc.leadingBlockDimensionRHS = leadingBlockDimensions.K
    accumulateDesc.names = (accumulator: "dQ", lhs: "dS", rhs: "K")
    accumulateDesc.threadgroupAddress = "threadgroup_block"
    accumulateDesc.transposeStateRHS = transposeState.K
    let accumulateDerivativeQ = accumulate(descriptor: accumulateDesc)
    
    return """
  
  // Iterate over the columns.
  for (uint c = 0; c < C; c += 32) {
    // S = Q * K^T
    simdgroup_matrix_storage<float> S_sram[32 / 8];
    \(twoOperandAccess(descriptor: QKT_Descriptor))
    
    // P = softmax(S * scaleFactor)
    \(checkpointSoftmax())
    
    // dP = dO * V^T
    simdgroup_matrix_storage<float> dP_sram[32 / 8];
    \(twoOperandAccess(descriptor: dOVT_Descriptor))
    
    // dS = P * (dP - D) * scaleFactor
    \(computeDerivativeSoftmax())
    
    // load K[c]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchK)
    
    // dQ += dS * K
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(accumulateDerivativeQ)
  }
    
"""
  }
  
  func createInnerLoopKeyValue(computeDerivativeK: Bool) -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "K"
    outerProductDesc.B = "Q"
    outerProductDesc.C = "ST"
    outerProductDesc.transposeA = transposeState.K
    outerProductDesc.transposeB = transposeState.Q
    outerProductDesc.leadingDimensionA = leadingDimensions.K
    outerProductDesc.leadingDimensionB = leadingDimensions.Q
    outerProductDesc.matrixDimensions = (M: "C", N: "R")
    outerProductDesc.matrixOffset = (M: "gid * C_group", N: "r")
    
    var KQT_Descriptor = outerProduct(descriptor: outerProductDesc)
    KQT_Descriptor.firstIterationLoading = """
    
    if (sidx == 0) {
      // Locate the L[i] in device and threadgroup memory.
      auto L_terms_src = L_terms + r;
      auto L_terms_dst = \(blockLTerms());

      // Locate the D[i] in device and threadgroup memory.
      auto D_terms_src = D_terms + r;
      auto D_terms_dst = \(blockDTerms());
      
      // Zero-padding for safety, which should harm performance.
      ushort R_src_dimension = min(uint(R_group), R - r);
      ushort R_dst_dimension = 32;
      
      // Issue two async copies.
      simdgroup_event events[2];
      events[0].async_copy(
        L_terms_dst, 1, ushort2(R_dst_dimension, 1),
        L_terms_src, 1, ushort2(R_src_dimension, 1));
      events[1].async_copy(
        D_terms_dst, 1, ushort2(R_dst_dimension, 1),
        D_terms_src, 1, ushort2(R_src_dimension, 1));
      simdgroup_event::wait(2, events);
    }
    
"""
    
    var accessDesc = AttentionHBMAccessDescriptor()
    accessDesc.index = "r"
    accessDesc.leadingBlockDimension = leadingBlockDimensions.Q
    accessDesc.leadingDimension = leadingDimensions.Q
    accessDesc.name = "Q"
    accessDesc.threadgroupAddress = "threadgroup_block"
    accessDesc.transposeState = transposeState.Q
    let prefetchQ = prefetchRows(descriptor: accessDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.index = "r"
    accumulateDesc.indexedBlockDimension = blockDimensions.R
    accumulateDesc.leadingBlockDimensionRHS = leadingBlockDimensions.Q
    accumulateDesc.names = (accumulator: "dK", lhs: "dST", rhs: "Q")
    accumulateDesc.threadgroupAddress = "threadgroup_block"
    accumulateDesc.transposeStateRHS = transposeState.Q
    let accumulateDerivativeK = accumulate(descriptor: accumulateDesc)
    
    var output = """

  // Iterate over the rows.
  for (uint r = 0; r < R; r += 32) {
    // load L[r]
    // load D[r]
    // S^T = K * Q^T
    simdgroup_matrix_storage<float> ST_sram[32 / 8];
    \(twoOperandAccess(descriptor: KQT_Descriptor))
    
    // P^T = exp(S^T - L)
    \(checkpointSoftmaxT())

    // dV += P^T * dO
    // dP^T = V * dO^T
    simdgroup_matrix_storage<float> dPT_sram[32 / 8];
    \(computeDerivativeVDerivativePT())
    
    // dS^T = P^T * (dP^T - D) * scaleFactor
    \(computeDerivativeSoftmaxT())

"""
    
    if computeDerivativeK {
      output += """
  
    // load Q[r]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchQ)

    // dK += dS^T * Q
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(accumulateDerivativeK)
  }
  
"""
    } else {
      output += """

    // store dS^T[c][r]
    {
      uint2 device_origin(r, gid * 32 + sidx * 8);
      device_origin += uint2(morton_offset);
      device bfloat* dst =
      simdgroup_matrix_storage<bfloat>::apply_offset(
        dST, \(leadingDimensionDerivativeST), device_origin, false);
      
#pragma clang loop unroll(full)
      for (ushort c = 0; c < 32; c += 8) {
        ushort2 thread_origin(c, 0);
        dST_sram[c / 8].store_bfloat(
          dst, \(leadingDimensionDerivativeST), thread_origin, false);
      }
    }
  }

"""
    }
    
    return output
  }
}
