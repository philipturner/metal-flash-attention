//
//  AttentionDescriptor+BlockSizes.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/19/24.
//

import Metal

// Block sizes tuned for a specific kernel or architecture.

#if false

extension AttentionDescriptor {
  func forwardMixedPrecision(_ output: inout AttentionKernelDescriptor) {
    guard let matrixDimensions = self.matrixDimensions else {
      fatalError("Descriptor was incomplete.")
    }
    
    if MTLContext.global.device.supportsFamily(.apple9) {
      
    } else {
      if matrixDimensions.D <= 16 {
        output.blockDimensions!.parallelization = 128
        output.blockDimensions!.head = 16
        output.cacheState[.Q] = true
        output.cacheState[.O] = true
      }
    }
  }
}

#endif
