//
//  AttentionKernelType.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/28/24.
//

// Forward, variant 0
// device half *Q
// device half *K
// device half *V
// device half *O

// Forward, variant 1
// device half *Q
// device half *K
// device half *V
// device half *O
// device half *L

// Backward query, variant 0

// Backward query, variant 1

// Backward key-value, variant 0

// Backward key-value, variant 1

enum AttentionKernelType {
  /// Forward attention, computing O and softmax\_logsumexp.
  ///
  /// Variants:
  /// - `false`: compute O
  /// - `true`: compute O and softmax\_logsumexp
  case forward(Bool)
  
  /// Backward attention, computing D[i] and dQ.
  ///
  /// Variants:
  /// - `false`: compute D[i]
  /// - `true`: compute D[i] and dQ
  ///
  /// Depends on: softmax\_logsumexp
  case backwardQuery(Bool)
  
  /// Backward attention, computing dK and dV.
  ///
  /// Variants:
  /// - `false`: compute dV, store the intermediate dS
  /// - `true`: compute dV and dK
  ///
  /// Depends on: softmax\_logsumexp, D[i]
  case backwardKeyValue(Bool)
}

