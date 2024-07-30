//
//  AttentionOperand.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/30/24.
//

// Make an enumeration over the different operands in attention. That
// will make it easier to de-duplicate code across the descriptors.
enum AttentionOperand {
  case Q
  case K
  case S
  case V
  case O
  
  case dO
  case dV
  case dS
  case dK
  case dQ
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
