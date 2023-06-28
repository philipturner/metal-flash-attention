//
//  GEMM.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal

protocol GEMM: Operation {
  var parameters: GEMM_Parameters { get }
}

extension GEMM {
  func equals(_ other: GEMM) -> Bool {
    (type(of: self) == type(of: other)) && (parameters == other.parameters)
  }
}

struct GEMM_Parameters: Hashable, Equatable {
  var is_hgemm: Bool
  var M: Int
  var N: Int
  var K: Int
  var A_trans: Bool
  var B_trans: Bool
  var alpha: Float
  var beta: Float
  var batched: Bool
  var fused_activation: Bool
}

class MFA_GEMM: GEMM, MFA_Operation {
  var parameters: GEMM_Parameters
  
  static var functionConstants: [String: MTLConvertible] = [
    "M_simd": UInt16(16), // 24
    "N_simd": UInt16(16), // 24
    "K_simd": UInt16(32), // 24-32
    "M_splits": UInt16(2),
    "N_splits": UInt16(2),
    "K_splits": UInt16(1),
  ]
  
  init(parameters: GEMM_Parameters) {
    self.parameters = parameters
  }
  
  func makeAsyncPipeline() -> AsyncPipeline {
    fatalError()
  }
}

class MPS_GEMM: GEMM, MPS_Operation {
  var parameters: GEMM_Parameters
  
  init(parameters: GEMM_Parameters) {
    self.parameters = parameters
  }
}

