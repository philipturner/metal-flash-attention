//
//  AttentionDescriptor.swift
//  FlashAttention
//
//  Created by Philip Turner on 8/8/24.
//

import Metal

public struct AttentionDescriptor {
  // Q, K, V, dO
  public var lowPrecisionInputs: Bool = false
  
  // S, P, L, D, dP, dS
  public var lowPrecisionIntermediates: Bool = false
  
  // row:    Output sequence length; rows of the attention matrix.
  // column: Input sequence length; columns of the attention matrix.
  // head:   Head dimension, typically 32 - 256.
  public var matrixDimensions: (row: UInt32, column: UInt32, head: UInt16)?
  
  public var transposeState: (Q: Bool, K: Bool, V: Bool, O: Bool)?
  
  public init() {
    
  }
}

extension AttentionDescriptor {
  /// Initialize the kernel descriptor using another descriptor, which just
  /// specifies the problem size. Then, forget the information about problem
  /// size.
  public func kernelDescriptor(
    type: AttentionKernelType
  ) -> AttentionKernelDescriptor {
    // Fetch the kernel-specific parameters.
    let file = parameterFile(type: type)
    let table = AttentionParameterRow.parseTable(file)
    let row = row(table: table)
    
    func createBlockDimensions() -> (UInt16, UInt16, UInt16) {
      guard let parallelization = UInt16(row.parallelization),
            let traversal = UInt16(row.traversal),
            let originalHead = UInt16(row.head) else {
        fatalError("Could not decode block dimensions.")
      }
      
      // Enforce the rule that head block dimension <= head dimension.
      let headDimension = createHeadDimension()
      let paddedHeadDimension = (headDimension + 7) / 8 * 8
      let revisedHead = min(originalHead, paddedHeadDimension)
      
      return (parallelization, traversal, revisedHead)
    }
    
    func createCacheState() -> [AttentionOperand: Bool] {
      var expectedOperands: Set<AttentionOperand>
      switch type {
      case .forward:
        expectedOperands = [.Q, .O]
      case .backwardQuery:
        expectedOperands = [.Q, .dO, .dQ]
      case .backwardKeyValue:
        expectedOperands = [.K, .V, .dV, .dK]
      }
      
      // Check for unexpected operands.
      let cachedOperands = AttentionParameterRow
        .parseOperands(row.cachedOperands)
      for operand in cachedOperands {
        guard expectedOperands.contains(operand) else {
          fatalError("Unexpected operand: \(operand)")
        }
      }
      
      // Convert the list into a dictionary.
      var output: [AttentionOperand: Bool] = [:]
      for operand in expectedOperands {
        output[operand] = false
      }
      for operand in cachedOperands {
        output[operand] = true
      }
      
      return output
    }
    
    func createHeadDimension() -> UInt16 {
      guard let matrixDimensions = self.matrixDimensions else {
        fatalError("Descriptor was incomplete.")
      }
      return matrixDimensions.head
    }
    
    func createTransposeState() -> [AttentionOperand: Bool] {
      guard let transposeState = self.transposeState else {
        fatalError("Descriptor was incomplete.")
      }
      
      var output: [AttentionOperand: Bool] = [:]
      output[.Q] = transposeState.Q
      output[.K] = transposeState.K
      output[.V] = transposeState.V
      output[.O] = transposeState.O
      
      output[.dO] = transposeState.O
      output[.dV] = transposeState.V
      output[.dK] = transposeState.K
      output[.dQ] = transposeState.Q
      return output
    }
    
    var output = AttentionKernelDescriptor()
    output.blockDimensions = createBlockDimensions()
    output.cacheState = createCacheState()
    output.headDimension = createHeadDimension()
    output.memoryPrecisions = memoryPrecisions
    if MTLContext.global.device.supportsFamily(.apple9) {
      output.preferAsyncCache = true
      output.preferAsyncLoad = false
    } else {
      output.preferAsyncCache = false
      output.preferAsyncLoad = true
    }
    output.registerPrecisions = registerPrecisions
    output.transposeState = createTransposeState()
    output.type = type
    
    return output
  }
}

extension AttentionDescriptor {
  // Specialize the Metal function with this attention descriptor.
  //
  // You can initialize a MTLFunctionConstantValues object once, then recycle
  // it for all three kernels when gradient is requested. This may simplify
  // the code or incrementally reduce the compilation latency.
  public func setFunctionConstants(_ constants: MTLFunctionConstantValues) {
    guard let matrixDimensions = self.matrixDimensions else {
      fatalError("Descriptor was incomplete.")
    }
    
    var rowDimension = matrixDimensions.row
    var columnDimension = matrixDimensions.column
    constants.setConstantValue(&rowDimension, type: .uint, index: 0)
    constants.setConstantValue(&columnDimension, type: .uint, index: 1)
  }
}
