//
//  AttentionKernel+TraversalLoop.swift
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
// Backward Query
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
// Backward Key-Value
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
// Backward Query
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
// Backward Key-Value
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
  func loopForward() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "Q"
    outerProductDesc.B = "K"
    outerProductDesc.C = "S"
    let QKT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = "P"
    accumulateDesc.B = "V"
    accumulateDesc.C = "O"
    accumulateDesc.everyIterationScale = "correction"
    accumulateDesc.lastIterationScale = "fast::divide(1, l)"
    let PV = accumulate(descriptor: accumulateDesc)
    
    return """
  
  // Outer loop over the traversal dimension.
  for (uint c = 0; c < C; c += 32) {
    // S = Q * K^T
    \(QKT)
    
    // (m, l, P) = softmax(m, l, S * scaleFactor)
    \(maskAttentionMatrixEdge())
    \(onlineSoftmax())
    
    // O *= correction
    // O += P * V
    // O /= l
    \(PV)
  }
  
"""
  }
  
  func loopBackwardQuery() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "Q"
    outerProductDesc.B = "K"
    outerProductDesc.C = "S"
    let QKT = outerProduct(descriptor: outerProductDesc)
    
    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "dO"
    outerProductDesc.B = "V"
    outerProductDesc.C = "dP"
    let dOVT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = "dS"
    accumulateDesc.B = "K"
    accumulateDesc.C = "dQ"
    let dSK = accumulate(descriptor: accumulateDesc)
    
    return """
  
  // Outer loop over the traversal dimension.
  for (uint c = 0; c < C; c += 32) {
    // S = Q * K^T
    \(QKT)
    
    // P = softmax(S * scaleFactor)
    \(checkpointSoftmax())
    
    // dP = dO * V^T
    \(dOVT)
    
    // dS = P * (dP - D) * scaleFactor
    \(derivativeSoftmax())
    
    // dQ += dS * K
    \(dSK)
  }
    
"""
  }
  
  func loopBackwardKeyValue(computeDerivativeK: Bool) -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "K"
    outerProductDesc.B = "Q"
    outerProductDesc.C = "ST"
    let KQT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = "PT"
    accumulateDesc.B = "dO"
    accumulateDesc.C = "dV"
    let PTdO = accumulate(descriptor: accumulateDesc)
    
    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = "V"
    outerProductDesc.B = "dO"
    outerProductDesc.C = "dPT"
    let VdOT = outerProduct(descriptor: outerProductDesc)
    
    var dSTQ: String
    if computeDerivativeK {
      var accumulateDesc = AttentionAccumulateDescriptor()
      accumulateDesc.A = "dST"
      accumulateDesc.B = "Q"
      accumulateDesc.C = "dK"
      dSTQ = accumulate(descriptor: accumulateDesc)
    } else {
      dSTQ = ""
    }
    
    return """
    
    // Outer loop over the traversal dimension.
    for (uint r = 0; r < R; r += 32) {
      // S^T = K * Q^T
      \(KQT)
      
      // P^T = exp(S^T - L)
      \(checkpointSoftmaxT())
      
      // dV += P^T * dO
      \(PTdO)
      
      // dP^T = V * dO^T
      \(VdOT)
      
      // dS^T = P^T * (dP^T - D) * scaleFactor
      \(derivativeSoftmaxT())
      
      // dK += dS^T * Q
      \(dSTQ)
    }
    
    """
  }
}
