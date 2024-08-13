//
//  AttentionDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/8/24.
//

import Metal

// Design specifications for the attention descriptor:
// - member function to set the function constants as a component of PSO init
//   - caller can now create one MTLFunctionConstantValues object for all
//     three kernels [DONE]
// - populates the lists of operands present in each kernel [DONE]
// - encapsulates the three kernels that make up the attention pass
//   - one set of function constants / buffer bindings should be the same
//     across all of the kernels [DONE]
//   - member function 'kernelDescriptor(type:)' generates an
//     AttentionKernelDescriptor with the right settings [DONE]
// - makes the simplification that Q/K/V/O and their gradients have the same
//   transpose state [DONE]
// - automatically assigns a cache state (default is false for now) [DONE]
//   - you can intercept and override the results after the
//     AttentionKernelDescriptors are created from the AttentionDescriptor
// - very simple, early heuristics for block sizes [DONE]
//
// What is not included yet:
// - shader caching
//   - group the three kernels into a single cache query
//   - separate the 1-kernel set for forward from the 3-kernel set for if
//     gradient is requested
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
  /// Initialize the kernel descriptor using another descriptor, which just
  /// specifies the problem size. Then, forget the information about problem
  /// size.
  ///
  /// > WARNING: This function makes high-latency API calls. Avoid calling this
  /// function unless absolutely necessary. A good shader cache design should
  /// avoid calling this function, if the corresponding kernel has already been
  /// cached.
  func kernelDescriptor(
    type: AttentionKernelType
  ) -> AttentionKernelDescriptor {
    guard let matrixDimensions = self.matrixDimensions,
          let transposeState = self.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    // Select the only GPU on an Apple silicon system.
    //
    // NOTE: High-latency API call. See the explanation in GEMMDescriptor.
    let mtlDevice = MTLCreateSystemDefaultDevice()!
    
    var output = AttentionKernelDescriptor()
    output.headDimension = matrixDimensions.D
    output.type = type
    
    // Block sizes for the case where nothing is cached.
    if mtlDevice.supportsFamily(.apple9) {
      output.blockDimensions = (
        parallelization: 16, traversal: 128, head: 16)
    } else {
      output.blockDimensions = (
        parallelization: 32, traversal: 64, head: 32)
    }
    
    // Assign the transpose state.
    output.transposeState[.Q] = transposeState.Q
    output.transposeState[.K] = transposeState.K
    output.transposeState[.V] = transposeState.V
    
    switch type {
    case .forward:
      output.transposeState[.O] = transposeState.O
    case .backwardQuery:
      output.transposeState[.O] = transposeState.O
      output.transposeState[.dO] = transposeState.O
      output.transposeState[.dQ] = transposeState.Q
    case .backwardKeyValue:
      output.transposeState[.dO] = transposeState.O
      output.transposeState[.dV] = transposeState.V
      output.transposeState[.dK] = transposeState.K
    }
    
    // Assign the cache state.
    let cacheInputs = false
    let cacheOutputs = false
    
    switch type {
    case .forward:
      output.cacheState[.Q] = cacheInputs
      output.cacheState[.O] = cacheOutputs
    case .backwardQuery:
      output.cacheState[.Q] = cacheInputs
      output.cacheState[.dO] = cacheInputs
      output.cacheState[.dQ] = cacheOutputs
    case .backwardKeyValue:
      output.cacheState[.K] = cacheInputs
      output.cacheState[.V] = cacheInputs
      output.cacheState[.dV] = cacheOutputs
      output.cacheState[.dK] = cacheOutputs
    }
    
    output.preferAsyncCache = false
    output.preferAsyncLoad = true
    
    return output
  }
}

extension AttentionDescriptor {
  // Specialize the Metal function with this attention descriptor.
  //
  // You can initialize a MTLFunctionConstantValues object once, then recycle
  // it for all three kernels when gradient is requested. This may simplify
  // the code or incrementally reduce the compilation latency.
  func setFunctionConstants(_ constants: MTLFunctionConstantValues) {
    guard let matrixDimensions = self.matrixDimensions else {
      fatalError("Descriptor was incomplete.")
    }
    
    var R = matrixDimensions.R
    var C = matrixDimensions.C
    constants.setConstantValue(&R, type: .uint, index: 0)
    constants.setConstantValue(&C, type: .uint, index: 1)
  }
}
