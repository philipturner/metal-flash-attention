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
