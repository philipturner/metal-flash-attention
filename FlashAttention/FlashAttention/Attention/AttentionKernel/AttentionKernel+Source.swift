//
//  AttentionKernel+Source.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/2/24.
//

// Top level specification of the code structure.

extension AttentionKernel {
  func createSource() -> String {
    func createLoop() -> String {
      switch type {
      case .forward:
        return loopForward()
      case .backwardQuery:
        return loopBackwardQuery()
      case .backwardKeyValue:
        return loopBackwardKeyValue()
      }
    }
    
    return """
    
    \(createMetalSimdgroupEvent())
    \(createMetalSimdgroupMatrixStorage())
    using namespace metal;
    
    \(createConstants())
    
    // Declare the function.
    kernel void attention(
      \(createBufferBindings())
      threadgroup uchar *threadgroup_block [[threadgroup(0)]],
      
      uint gid [[threadgroup_position_in_grid]],
      ushort sidx [[simdgroup_index_in_threadgroup]],
      ushort lane_id [[thread_index_in_simdgroup]]
    ) {
      ushort2 morton_offset = morton_order(lane_id);
      uint parallelization_group_offset = gid;
      parallelization_group_offset *= \(blockDimensions.parallelization);
      
      // Return early if the entire SIMD is out of bounds.
      if (\(parallelizationGroupOffset) >= \(parallelizationDimension)) {
        return;
      }
      
      \(createSetup())
      \(createLoop())
      \(createCleanup(type: type))
    }
    
    """
  }
}

// MARK: - Function Signature

extension AttentionKernel {
  func createConstants() -> String {
    """
    
    // R = row dimension (output sequence)
    // C = column dimension (input sequence)
    constant uint R [[function_constant(0)]];
    constant uint C [[function_constant(1)]];
    
    """
  }
  
  func createBufferBindings() -> String {
    // What operands does the kernel use?
    var operands: [AttentionOperand] = []
    switch type {
    case .forward:
      // To simplify the implementation, we always compute log-sum-exp in the
      // forward pass. Even when it will never be used (model inference).
      // If this is an issue, clients can change the code to selectively
      // omit the 'L' operand.
      operands += [.Q, .K, .V, .O]
      operands += [.L]
    case .backwardQuery:
      operands += [.Q, .K, .V, .O]
      operands += [.dO, .dQ]
      operands += [.L, .D]
    case .backwardKeyValue:
      operands += [.Q, .K, .V]
      operands += [.dO, .dV, .dK]
      operands += [.L, .D]
    }
    operands.sort {
      $0.bufferBinding! < $1.bufferBinding!
    }
    
    var output: String = ""
    for operand in operands {
      var line = "device \(memoryName(operand))* \(operand) "
      line += "[[buffer(\(operand.bufferBinding!))]],"
      output += "  " + line + "\n"
    }
    return output
  }
}

// MARK: - Outer Loop

// Forward
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     (m, l, P) = softmax(m, l, S * scaleFactor)
//
//     O *= correction
//     load V[c]
//     O += P * V
//   }
//   O /= l
//
//   L = m + logBaseE(l)
//
// Backward Query
//   D = dO * O
//
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     P = exp(S - L)
//
//     load V[c]
//     dP = dO * V^T
//     dS = P * (dP - D) * scaleFactor
//
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
//
//     load dO[r]
//     dV += P^T * dO
//
//     load dO[r]
//     load D[r]
//     dP^T = V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//
//     load Q[r]
//     dK += dS^T * Q
//   }

extension AttentionKernel {
  func loopForward() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .Q
    outerProductDesc.B = .K
    outerProductDesc.C = .S
    let QKT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .P
    accumulateDesc.B = .V
    accumulateDesc.C = .O
    accumulateDesc.everyIterationScale = "correction"
    accumulateDesc.lastIterationScale = "fast::divide(1, l)"
    let PV = accumulate(descriptor: accumulateDesc)
    
    return """
    
    // Outer loop over the traversal dimension.
    for (uint c = 0; c < C; c += \(blockDimensions.traversal)) {
      // S = Q * K^T
      \(QKT)
      \(maskAttentionMatrixEdge())
      
      // m = reduce(m)
      \(onlineReduceMaximum())
      
      // correction = exp(m_old) / exp(m_new)
      \(onlineCorrectO())
      
      // P = softmax(S * scaleFactor)
      \(softmax(derivative: false))
      
      // l = reduce(l)
      \(onlineReduceSum())
      
      // O *= correction
      // O += P * V
      // O /= l
      \(PV)
    }
    
    """
  }
  
  func loopBackwardQuery() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .Q
    outerProductDesc.B = .K
    outerProductDesc.C = .S
    let QKT = outerProduct(descriptor: outerProductDesc)
    
    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .dO
    outerProductDesc.B = .V
    outerProductDesc.C = .dP
    let dOVT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .dS
    accumulateDesc.B = .K
    accumulateDesc.C = .dQ
    let dSK = accumulate(descriptor: accumulateDesc)
    
    return """
    
    // Outer loop over the traversal dimension.
    for (uint c = 0; c < C; c += \(blockDimensions.traversal)) {
      // S = Q * K^T
      \(QKT)
      
      // P = softmax(S * scaleFactor)
      \(softmax(derivative: false))
      
      // dP = dO * V^T
      \(dOVT)
      
      // dS = P * (dP - D) * scaleFactor
      \(softmax(derivative: true))
      
      // dQ += dS * K
      \(dSK)
    }
    
    """
  }
  
  func loopBackwardKeyValue() -> String {
    var outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .K
    outerProductDesc.B = .Q
    outerProductDesc.C = .S // S^T
    let KQT = outerProduct(descriptor: outerProductDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .P // P^T
    accumulateDesc.B = .dO
    accumulateDesc.C = .dV
    let PTdO = accumulate(descriptor: accumulateDesc)
    
    outerProductDesc = AttentionOuterProductDescriptor()
    outerProductDesc.A = .V
    outerProductDesc.B = .dO
    outerProductDesc.C = .dP // dP^T
    let VdOT = outerProduct(descriptor: outerProductDesc)
    
    accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.A = .dS // dS^T
    accumulateDesc.B = .Q
    accumulateDesc.C = .dK
    let dSTQ = accumulate(descriptor: accumulateDesc)
    
    return """
    
    // Outer loop over the traversal dimension.
    for (uint r = 0; r < R; r += \(blockDimensions.traversal)) {
      // S^T = K * Q^T
      \(KQT)
      
      // P^T = exp(S^T - L)
      \(softmax(derivative: false))
      
      // dV += P^T * dO
      \(PTdO)
      
      // dP^T = V * dO^T
      \(VdOT)
      
      // dS^T = P^T * (dP^T - D) * scaleFactor
      \(softmax(derivative: true))
      
      // dK += dS^T * Q
      \(dSTQ)
    }
    
    """
  }
}
