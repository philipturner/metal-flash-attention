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
  // Q, K, V, dO
  var lowPrecisionInputs: Bool = false
  
  // S, P, L, D, dP, dS
  var lowPrecisionIntermediates: Bool = false
  
  // O, dV, dK, dQ
  var lowPrecisionOutputs: Bool = false
  
  var matrixDimensions: (R: UInt32, C: UInt32, D: UInt16)?
  
  var transposeState: (Q: Bool, K: Bool, V: Bool, O: Bool)?
}

extension AttentionDescriptor {
  /// Initialize the kernel descriptor using another descriptor, which just
  /// specifies the problem size. Then, forget the information about problem
  /// size.
  func kernelDescriptor(
    type: AttentionKernelType
  ) -> AttentionKernelDescriptor {
    guard let matrixDimensions = self.matrixDimensions,
          let transposeState = self.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    // Select the only GPU on an Apple silicon system.
    let mtlDevice = MTLContext.global.device
    
    var output = AttentionKernelDescriptor()
    output.headDimension = matrixDimensions.D
    output.type = type
    
    // Block sizes for the case where nothing is cached.
    if mtlDevice.supportsFamily(.apple9) {
      if matrixDimensions.D % 8 == 0 {
        output.blockDimensions = (
          parallelization: 16, traversal: 128, head: 16)
      } else {
        output.blockDimensions = (
          parallelization: 16, traversal: 128, head: 8)
      }
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
    
    // Access pattern heuristic for when nothing is cached.
    if mtlDevice.supportsFamily(.apple9) {
      output.preferAsyncCache = true
      output.preferAsyncLoad = false
    } else {
      output.preferAsyncCache = false
      output.preferAsyncLoad = true
    }
    
    // Choose the precision for each operand.
    output.memoryPrecisions = self.memoryPrecisions()
    output.registerPrecisions = self.registerPrecisions()
    
    return output
  }
}

extension AttentionDescriptor {
  func memoryPrecisions() -> [AttentionOperand: GEMMOperandPrecision] {
    var memoryPrecisions: [AttentionOperand: GEMMOperandPrecision] = [:]
    
    if lowPrecisionInputs {
      memoryPrecisions[.Q] = .FP16
      memoryPrecisions[.K] = .FP16
      memoryPrecisions[.V] = .FP16
      memoryPrecisions[.dO] = .BF16
    } else {
      memoryPrecisions[.Q] = .FP32
      memoryPrecisions[.K] = .FP32
      memoryPrecisions[.V] = .FP32
      memoryPrecisions[.dO] = .FP32
    }
    
    if lowPrecisionIntermediates {
      memoryPrecisions[.L] = .FP16
      memoryPrecisions[.D] = .BF16
    } else {
      memoryPrecisions[.L] = .FP32
      memoryPrecisions[.D] = .FP32
    }
    
    if lowPrecisionOutputs {
      memoryPrecisions[.O] = .FP16
      memoryPrecisions[.dV] = .BF16
      memoryPrecisions[.dK] = .BF16
      memoryPrecisions[.dQ] = .BF16
    } else {
      memoryPrecisions[.O] = .FP32
      memoryPrecisions[.dV] = .FP32
      memoryPrecisions[.dK] = .FP32
      memoryPrecisions[.dQ] = .FP32
    }
    
    return memoryPrecisions
  }
  
  func registerPrecisions() -> [AttentionOperand: GEMMOperandPrecision] {
    var registerPrecisions: [AttentionOperand: GEMMOperandPrecision] = [:]
    
    // Query whether the hardware fuses the promotion of BF16 to FP32 with
    // the FMA assembly instruction.
    let device = MTLContext.global.device
    let hasNativeBF16Casting = device.supportsFamily(.apple9)
    
    // Inputs have the same register precision across kernels.
    if lowPrecisionInputs {
      registerPrecisions[.Q] = .FP16
      registerPrecisions[.K] = .FP16
      registerPrecisions[.V] = .FP16
      registerPrecisions[.dO] = hasNativeBF16Casting ? .BF16 : .FP32
    } else {
      registerPrecisions[.Q] = .FP32
      registerPrecisions[.K] = .FP32
      registerPrecisions[.V] = .FP32
      registerPrecisions[.dO] = .FP32
    }
    
    // The register precision of L/D only counts for backward key-value.
    if lowPrecisionIntermediates {
      registerPrecisions[.L] = .FP16
      registerPrecisions[.D] = hasNativeBF16Casting ? .BF16 : .FP32
    } else {
      registerPrecisions[.L] = .FP32
      registerPrecisions[.D] = .FP32
    }
    
    // O is the only operand that's an accumulator for one kernel, and input
    // for another. Theoretically, we could optimize M3 by keeping it as
    // BF16 while accumulating dO * O. That would require the kernel type to
    // be an argument for the 'registerPrecisions()' function.
    registerPrecisions[.O] = .FP32
    registerPrecisions[.dV] = .FP32
    registerPrecisions[.dK] = .FP32
    registerPrecisions[.dQ] = .FP32
    
    // The register precision for the attention matrix.
    if lowPrecisionIntermediates {
      // TODO: Utilize the native FP16xFP16->FP16 instruction.
      //
      // S = Q * K is the most often recomputed intermediate (3 out of 9 GEMMs,
      // 2 out of 3 unnecessary GEMMs). If we optimize this, the impact on
      // performance will be greater than for any other multiplication.
      registerPrecisions[.S] = .FP32
      registerPrecisions[.P] = .FP16
      registerPrecisions[.dP] = .FP32
      registerPrecisions[.dS] = hasNativeBF16Casting ? .BF16 : .FP32
    } else {
      registerPrecisions[.S] = .FP32
      registerPrecisions[.P] = .FP32
      registerPrecisions[.dP] = .FP32
      registerPrecisions[.dS] = .FP32
    }
    
    return registerPrecisions
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
