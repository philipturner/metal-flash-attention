//
//  AttentionOperand.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/3/24.
//

/// The memory allocations used in attention kernels.
///
/// The raw value is the buffer binding.
enum AttentionOperand: Hashable, Equatable {
  case Q
  case K
  case S
  case P
  case V
  case O
  
  case dO
  case dV
  case dP
  case dS
  case dK
  case dQ
  
  case LTerms
  case DTerms
  
  /// The name in the shader source.
  var name: String {
    switch self {
    case .Q: return "Q"
    case .K: return "K"
    case .S: return "S"
    case .P: return "P"
    case .V: return "V"
    case .O: return "O"
      
    case .dO: return "dO"
    case .dV: return "dV"
    case .dP: return "dP"
    case .dS: return "dS"
    case .dK: return "dK"
    case .dQ: return "dQ"
      
    case .LTerms: return "L_terms"
    case .DTerms: return "D_terms"
    }
  }
  
  var bufferBinding: UInt8? {
    switch self {
    case .Q: return 0
    case .K: return 1
    case .V: return 2
    case .O: return 3
      
    case .dO: return 4
    case .dV: return 5
    case .dK: return 6
    case .dQ: return 7
      
    case .LTerms: return 10
    case .DTerms: return 11
    case .dS: return 12
      
    default: return nil
    }
  }
}
