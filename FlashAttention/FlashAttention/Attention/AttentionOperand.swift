//
//  AttentionOperand.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/28/24.
//

// Make an enumeration over the different operands in attention. That
// will make it easier to de-duplicate code across the descriptors.
enum AttentionOperand {
  case Q
  case K
  case V
  case O
  case dO
  case dV
  case dK
  case dQ
  
  // TODO: Try using atomics to accumulate dQ. It may be simpler than
  // materializing the attention matrix in RAM. Of course, it will be complex
  // to code, storing cannot happen through async copies, and gradients will
  // be nondeterministic. But the bandwidth and accuracy issues are already
  // being faced, as we page stuff to RAM and consider reduced precision. We
  // can always revert to the "store dS^T" algorithm if performance remains
  // sub-optimal.
}

enum AttentionOperandPrecision {
  case full
  case mixed
  
  var forwardPrecision: GEMMOperandPrecision {
    switch self {
    case .full: return .FP32
    case .mixed: return .FP16
    }
  }
  
  var backwardPrecision: GEMMOperandPrecision {
    switch self {
    case .full: return .FP32
    case .mixed: return .BF16
    }
  }
}
