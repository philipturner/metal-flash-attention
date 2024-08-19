//
//  AttentionDescriptor+BlockSizes.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/19/24.
//

import Metal

// Block sizes tuned for a specific kernel or architecture.

extension AttentionDescriptor {
  // These block sizes were fine-tuned with the following precisions. If any
  // precision (e.g. the Q memory precision while paging to RAM) is changed,
  // the entire parameter set is invalidated. It will be your responsibility
  // to acquire an M1/M2 architecture and an M3+ architecture device, and
  // reoptimize the block sizes on both.
  //
  // memoryPrecisions[.Q] = .FP
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
      } else if matrix
    }
  }
}
