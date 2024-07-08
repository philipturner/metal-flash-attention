//
//  AttentionOperandPrecision.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/28/24.
//

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
