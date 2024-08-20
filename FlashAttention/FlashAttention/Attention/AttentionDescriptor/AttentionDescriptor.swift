//
//  AttentionDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/8/24.
//

import Metal

struct AttentionDescriptor {
  // Q, K, V, dO
  var lowPrecisionInputs: Bool = false
  
  // S, P, L, D, dP, dS
  var lowPrecisionIntermediates: Bool = false
  
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
    guard let matrixDimensions = self.matrixDimensions else {
      fatalError("Descriptor was incomplete.")
    }
    
    // TODO: Refactor this into a bunch of nested functions.
    
    var output = AttentionKernelDescriptor()
    output.headDimension = matrixDimensions.D
    output.type = type
    
    // Fetch the kernel-specific parameters.
    let file = parameterFile(type: type)
    let table = AttentionParameterRow.parseTable(file)
    let row = row(table: table)
    
    // Set the block dimensions.
    do {
      var blockDimensions = row.createBlockDimensions()
      
      // Enforce the rule that head block dimension <= head dimension.
      let paddedHeadDimension = (matrixDimensions.D + 7) / 8 * 8
      blockDimensions[2] = min(blockDimensions[2], paddedHeadDimension)
      
      output.blockDimensions = (
        parallelization: blockDimensions[0],
        traversal: blockDimensions[1],
        head: blockDimensions[2])
    }
    
    // Assign the cache state.
    do {
      switch type {
      case .forward:
        output.cacheState[.Q] = false
        output.cacheState[.O] = false
      case .backwardQuery:
        output.cacheState[.Q] = false
        output.cacheState[.dO] = false
        output.cacheState[.dQ] = false
      case .backwardKeyValue:
        output.cacheState[.K] = false
        output.cacheState[.V] = false
        output.cacheState[.dV] = false
        output.cacheState[.dK] = false
      }
      
      let operands = AttentionParameterRow.parseOperands(row.cachedOperands)
      for operand in operands {
        let previousValue = output.cacheState[operand]
        guard previousValue == false else {
          fatalError("Unexpected operand for \(type) kernel: \(operand)")
        }
        
        output.cacheState[operand] = true
      }
    }
    
    // Assign the transpose state.
    do {
      guard let transposeState = self.transposeState else {
        fatalError("Descriptor was incomplete.")
      }
      
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
    }
    
    // Choose the memory access pattern.
    do {
      output.memoryPrecisions = self.memoryPrecisions()
      output.registerPrecisions = self.registerPrecisions()
      
      if MTLContext.global.device.supportsFamily(.apple9) {
        output.preferAsyncCache = true
        output.preferAsyncLoad = false
      } else {
        output.preferAsyncCache = false
        output.preferAsyncLoad = true
      }
    }
    
    return output
  }
  
  // parallelization, traversal, head
  private func blockDimensions() -> (UInt16, UInt16, UInt16) {
    guard let matrixDimensions = self.matrixDimensions else {
      fatalError("Descriptor was incomplete.")
    }
    
    var output: (parallelization: UInt16, traversal: UInt16, head: UInt16)
    
    // Block sizes for the case where nothing is cached.
    if MTLContext.global.device.supportsFamily(.apple9) {
      output = (
        parallelization: 16, traversal: 128, head: 16)
    } else {
      output = (
        parallelization: 32, traversal: 64, head: 32)
    }
    
    // Enforce the rule that head block dimension <= head dimension.
    let paddedHeadDimension = (matrixDimensions.D + 7) / 8 * 8
    output.head = min(output.head, paddedHeadDimension)
    
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
