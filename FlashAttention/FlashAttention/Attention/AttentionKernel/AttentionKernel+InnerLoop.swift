//
//  AttentionKernel+InnerLoop.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/2/24.
//

// High-level specification of the code structure.

// MARK: - Minimal Pseudocode

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
    outerProductDesc.cacheA = cachedInputs.Q
    outerProductDesc.B = "K"
    outerProductDesc.C = "S"
    outerProductDesc.transposeState = (
      A: transposeState.Q, B: transposeState.K)
    outerProductDesc.leadingDimensions = (
      A: leadingDimensions.Q, B: leadingDimensions.K)
    outerProductDesc.matrixDimensions = (M: "R", N: "C")
    outerProductDesc.matrixOffset = (M: "gid * 32", N: "c")
    let QKT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = "P"
    accumulateDesc.B = "V"
    accumulateDesc.C = "O"
    accumulateDesc.cacheC = cachedOutputs.O
    accumulateDesc.transposeState = (
      B: transposeState.V, C: transposeState.O)
    accumulateDesc.leadingDimensions = (
      B: leadingDimensions.V, C: leadingDimensions.O)
    accumulateDesc.matrixDimensions = (M: "R", K: "C")
    accumulateDesc.matrixOffset = (M: "gid * 32", K: "c")
    accumulateDesc.everyIterationFactor = "correction"
    accumulateDesc.lastIterationFactor = "1 / l"
    
    return """
  
  // Iterate over the columns.
  for (uint c = 0; c < C; c += 32) {
    // S = Q * K^T
    \(QKT)
    
    // (m, l, P) = softmax(m, l, S * scaleFactor)
    \(maskAlongColumns(sram: "S_sram"))
    \(onlineSoftmax())
    
    // O *= correction
    // O += P * V
    // O /= l
    \(accumulate(descriptor: accumulateDesc))
  }
  
"""
  }
  
  func createInnerLoopBackwardQuery() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "Q"
    outerProductDesc.cacheA = cachedInputs.Q
    outerProductDesc.B = "K"
    outerProductDesc.C = "S"
    outerProductDesc.transposeState = (
      A: transposeState.Q, B: transposeState.K)
    outerProductDesc.leadingDimensions = (
      A: leadingDimensions.Q, B: leadingDimensions.K)
    outerProductDesc.matrixDimensions = (M: "R", N: "C")
    outerProductDesc.matrixOffset = (M: "gid * 32", N: "c")
    let QKT = outerProduct(descriptor: outerProductDesc)
    
    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "dO"
    outerProductDesc.cacheA = cachedInputs.dO
    outerProductDesc.B = "V"
    outerProductDesc.C = "dP"
    outerProductDesc.transposeState = (
      A: transposeState.O, B: transposeState.V)
    outerProductDesc.leadingDimensions = (
      A: leadingDimensions.O, B: leadingDimensions.V)
    outerProductDesc.matrixDimensions = (M: "R", N: "C")
    outerProductDesc.matrixOffset = (M: "gid * 32", N: "c")
    let dOVT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = "dS"
    accumulateDesc.B = "K"
    accumulateDesc.C = "dQ"
    accumulateDesc.cacheC = cachedOutputs.dQ
    accumulateDesc.transposeState = (
      B: transposeState.K, C: transposeState.Q)
    accumulateDesc.leadingDimensions = (
      B: leadingDimensions.K, C: leadingDimensions.Q)
    accumulateDesc.matrixDimensions = (M: "R", K: "C")
    accumulateDesc.matrixOffset = (M: "gid * 32", K: "c")
    
    return """
  
  // Iterate over the columns.
  for (uint c = 0; c < C; c += 32) {
    // S = Q * K^T
    \(QKT)
    
    // P = softmax(S * scaleFactor)
    \(checkpointSoftmax())
    
    // dP = dO * V^T
    \(dOVT)
    
    // dS = P * (dP - D) * scaleFactor
    \(computeDerivativeSoftmax())
    
    // dQ += dS * K
    \(accumulate(descriptor: accumulateDesc))
  }
    
"""
  }
  
  func createInnerLoopKeyValue(computeDerivativeK: Bool) -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "K"
    outerProductDesc.cacheA = cachedInputs.K
    outerProductDesc.B = "Q"
    outerProductDesc.C = "ST"
    outerProductDesc.transposeState = (
      A: transposeState.K, B: transposeState.Q)
    outerProductDesc.leadingDimensions = (
      A: leadingDimensions.K, B: leadingDimensions.Q)
    outerProductDesc.matrixDimensions = (M: "C", N: "R")
    outerProductDesc.matrixOffset = (M: "gid * 32", N: "r")
    let KQT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = "PT"
    accumulateDesc.B = "dO"
    accumulateDesc.C = "dV"
    accumulateDesc.cacheC = cachedOutputs.dV
    accumulateDesc.transposeState = (
      B: transposeState.O, C: transposeState.V)
    accumulateDesc.leadingDimensions = (
      B: leadingDimensions.O, C: leadingDimensions.V)
    accumulateDesc.matrixDimensions = (M: "C", K: "R")
    accumulateDesc.matrixOffset = (M: "gid * 32", K: "r")
    
    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "V"
    outerProductDesc.cacheA = cachedInputs.V
    outerProductDesc.B = "dO"
    outerProductDesc.C = "dPT"
    outerProductDesc.transposeState = (
      A: transposeState.V, B: transposeState.O)
    outerProductDesc.leadingDimensions = (
      A: leadingDimensions.V, B: leadingDimensions.O)
    outerProductDesc.matrixDimensions = (M: "C", N: "R")
    outerProductDesc.matrixOffset = (M: "gid * 32", N: "r")
    let VdOT = outerProduct(descriptor: outerProductDesc)
    
    var output = """
  
  // Iterate over the rows.
  for (uint r = 0; r < R; r += 32) {
    // S^T = K * Q^T
    \(KQT)
    
    // P^T = exp(S^T - L)
    \(checkpointSoftmaxT())
    
    // dV += P^T * dO
    \(accumulate(descriptor: accumulateDesc))
    
    // dP^T = V * dO^T
    \(VdOT)
    
    // dS^T = P^T * (dP^T - D) * scaleFactor
    \(computeDerivativeSoftmaxT())
    
"""
    
    if computeDerivativeK {
      var accumulateDesc = AttentionAccumulateDescriptor()
      accumulateDesc.A = "dST"
      accumulateDesc.B = "Q"
      accumulateDesc.C = "dK"
      accumulateDesc.cacheC = cachedOutputs.dK
      accumulateDesc.transposeState = (
        B: transposeState.Q, C: transposeState.K)
      accumulateDesc.leadingDimensions = (
        B: leadingDimensions.Q, C: leadingDimensions.K)
      accumulateDesc.matrixDimensions = (M: "C", K: "R")
      accumulateDesc.matrixOffset = (M: "gid * 32", K: "r")
      
      output += """

// dK += dS^T * Q
\(accumulate(descriptor: accumulateDesc))
  
"""
    } else {
      var leadingDimension = "(C + \(blockDimensions.R) - 1)"
      leadingDimension = "\(leadingDimension) / \(blockDimensions.R)"
      leadingDimension = "\(leadingDimension) * \(blockDimensions.R)"
      
      output += """

// store dS^T[c][r]
{
  uint2 device_origin(r, gid * 32 + sidx * 8);
  device_origin += uint2(morton_offset);
  device float* dst =
  simdgroup_matrix_storage<float>::apply_offset(
    dST, \(leadingDimension), device_origin, false);
  
#pragma clang loop unroll(full)
  for (ushort c = 0; c < 32; c += 8) {
    ushort2 thread_origin(c, 0);
    dST_sram[c / 8].store(
      dst, \(leadingDimension), thread_origin, false);
  }
}

"""
    }
    
    // Add the final closing brace.
    output += """

}

"""
    
    return output
  }
}
