//
//  AttentionDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/8/24.
//

// Design specifications for the attention descriptor:
// - member function to set the function constants as a component of PSO init
// - populates the lists of operands present in each kernel
// - encapsulates the three kernels that make up the attention pass
//   - one set of function constants / buffer bindings should be the same
//     across all of the kernels
//   - user only specifies whether "gradient is required"
//   - member function 'kernelDescriptor(type:)' generates an
//     AttentionKernelDescriptor with the right settings
// - makes the simplification that Q/K/V/O and their gradients have the same
//   transpose state
// - automatically assigns a cache state (default is false for now)
//   - you can intercept and override the results after the
//     AttentionKernelDescriptors are created from the AttentionDescriptor
// - very simple, early heuristics for block sizes
//
// What is not included yet:
// - shader caching (not really needed in this reference impl anyway)
// - mixed precision
// - tuning the block size or caching heuristics
//   - this task should be done simultaneously with mixed precision support
// - whether operands are loaded/stored through async copy
//   - this is the next thing on the TODO list
//
// Taking GEMMDescriptor / GEMMKernelDescriptor as a reference
//
// GEMMDescriptor
// - batchDimension
// - leadingDimensions
// - loadPreviousC
// - matrixDimensions
// - memoryPrecisions
// - transposeState
//
// GEMMKernelDescriptor
// - blockDimensions
// - device
// - leadingBlockDimensions
// - memoryPrecisions
// - preferAsyncLoad
// - preferAsyncStore
// - registerPrecisions
// - splits
// - transposeState

struct AttentionDescriptor {
  var matrixDimensions: (R: UInt32, C: UInt32, D: UInt16)?
  var transposeState: (Q: Bool, K: Bool, V: Bool, O: Bool)?
}

extension AttentionDescriptor {
  func kernelDescriptor(
    type: AttentionKernelType
  ) -> AttentionKernelDescriptor {
    
    
    fatalError("Not implemented.")
  }
}
