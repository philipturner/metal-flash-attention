//
//  AttentionDescriptor+BlockSizes.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/19/24.
//

import Metal

// TODO: Get the code to reproduce the parameter specification table exactly.

extension AttentionDescriptor {
  /// The parameters that may vary from kernel to kernel.
  typealias CachingParameters = (
    blockDimensionTraversal: UInt16,
    blockDimensionHead: UInt16,
    cachedOperands: [AttentionOperand])
  
  /// Default parameter set.
  static func defaultParameters() -> CachingParameters {
    var blockDimensionTraversal: UInt16
    var blockDimensionHead: UInt16
    var cachedOperands: [AttentionOperand]
    
    if MTLContext.global.device.supportsFamily(.apple9) {
      blockDimensionTraversal = 128
      blockDimensionHead = 16
    } else {
      blockDimensionTraversal = 64
      blockDimensionHead = 32
    }
    cachedOperands = []
    
    return (blockDimensionTraversal, blockDimensionHead, cachedOperands)
  }
  
  /// Block sizes and cached operands for FP16 forward pass.
  ///
  /// Requires the following statements to be true. If any operand does not
  /// match the specified precision, use the heuristics for FP32 forward.
  ///
  /// ```swift
  /// registerPrecisions[.Q] = .FP16
  /// registerPrecisions[.K] = .FP16
  /// registerPrecisions[.V] = .FP16
  /// registerPrecisions[.S] = .FP16
  /// registerPrecisions[.P] = .FP16
  /// registerPrecisions[.O] = .FP32
  /// ```
  static func forwardHalfPrecision(
    headDimension: UInt16
  ) -> CachingParameters {
    var blockDimensionTraversal: UInt16
    var blockDimensionHead: UInt16
    var cachedOperands: [AttentionOperand]
    
    if MTLContext.global.device.supportsFamily(.apple9) {
      blockDimensionTraversal = 128
      
      if headDimension <= 32 {
        blockDimensionHead = 16
      } else {
        blockDimensionHead = 32
      }
      
      if headDimension <= 96 {
        cachedOperands = [.Q, .O]
      } else if headDimension <= 160 {
        cachedOperands = [.O]
      } else {
        cachedOperands = []
      }
    } else {
      blockDimensionTraversal = 128
      
      if headDimension <= 16 {
        blockDimensionHead = 16
      } else {
        blockDimensionHead = 32
      }
      
      if headDimension <= 64 {
        cachedOperands = [.Q, .O]
      } else if headDimension <= 96 {
        cachedOperands = [.O]
      } else if headDimension <= 160 {
        cachedOperands = [.Q]
      } else {
        cachedOperands = []
      }
    }
    
    return (blockDimensionTraversal, blockDimensionHead, cachedOperands)
  }
  
  /// Block sizes and cached operands for FP32 forward pass.
  static func forwardSinglePrecision(
    headDimension: UInt16
  ) -> CachingParameters {
    
  }
}
