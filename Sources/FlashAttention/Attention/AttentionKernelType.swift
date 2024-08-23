//
//  AttentionKernelType.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/16/24.
//

/// The three kernels of the FlashAttention algorithm for devices without
/// hardware acceleration for floating-point atomics.
public enum AttentionKernelType {
  /// Forward attention, computing O and L.
  case forward
  
  /// Backward attention, computing D and dQ.
  ///
  /// Depends on L.
  case backwardQuery
  
  /// Backward attention, computing dK and dV.
  ///
  /// Depends on L and D.
  case backwardKeyValue
}
