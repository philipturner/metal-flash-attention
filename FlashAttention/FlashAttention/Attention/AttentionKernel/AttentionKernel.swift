//
//  AttentionKernel.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/27/24.
//

// Declaration of the attention kernel data structure.

struct AttentionKernel {
  var type: AttentionKernelType
  
  // Categorical attributes for each operand.
  var cacheState: [AttentionOperand: Bool]
  var memoryPrecisions: [AttentionOperand: GEMMOperandPrecision]
  var preferAsyncCache: Bool
  var preferAsyncLoad: Bool
  var registerPrecisions: [AttentionOperand: GEMMOperandPrecision]
  var transposeState: [AttentionOperand: Bool]
  
  // Layout of the data in registers and threadgroup memory.
  var blockDimensions: (
    parallelization: UInt16, traversal: UInt16, head: UInt16)
  var headDimension: UInt16
  var threadgroupMemoryAllocation: UInt16
  
  init(descriptor: AttentionKernelDescriptor) {
    guard let blockDimensions = descriptor.blockDimensions,
          let headDimension = descriptor.headDimension,
          let preferAsyncCache = descriptor.preferAsyncCache,
          let preferAsyncLoad = descriptor.preferAsyncLoad,
          let type = descriptor.type else {
      fatalError("Descriptor was incomplete.")
    }
    self.type = type
    
    self.cacheState = descriptor.cacheState
    self.memoryPrecisions = descriptor.memoryPrecisions
    self.preferAsyncCache = preferAsyncCache
    self.preferAsyncLoad = preferAsyncLoad
    self.registerPrecisions = descriptor.registerPrecisions
    self.transposeState = descriptor.transposeState
    
    self.blockDimensions = blockDimensions
    self.headDimension = headDimension
    
    // Pick the threadgroup memory allocation size.
    threadgroupMemoryAllocation = .zero
    threadgroupMemoryAllocation = createThreadgroupMemoryAllocation()
  }
}

// MARK: - Utilities

// Appearances of BF16 -> FP32 conversion functions.
//
// M1
//              FWD | dQ | dK/dV | dK/dV (outputs cached)
// load_bfloat    2   12      24      16
// store_bfloat   2    6      10       8
//
// M3
//              FWD | dQ | dK/dV | dK/dV (outputs cached)
// load_bfloat    2    6      10       2
// store_bfloat   2    6      10       8
extension AttentionKernel {
  func memoryName(_ operand: AttentionOperand) -> String {
    guard let memoryPrecision = memoryPrecisions[operand] else {
      fatalError("Memory precision of \(operand) was not specified.")
    }
    return memoryPrecision.name
  }
  
  func registerName(_ operand: AttentionOperand) -> String {
    guard let registerPrecision = registerPrecisions[operand] else {
      fatalError("Memory precision of \(operand) was not specified.")
    }
    return registerPrecision.name
  }
  
  func loadFunction(_ operand: AttentionOperand) -> String {
    guard let memoryPrecision = memoryPrecisions[operand],
          let registerPrecision = registerPrecisions[operand] else {
      fatalError("Precision of \(operand) was not specified.")
    }
    
    switch (memoryPrecision, registerPrecision) {
    case (.FP16, .FP16):
      return "load"
    case (.FP16, .BF16):
      fatalError("Invalid precisions.")
    case (.FP16, .FP32):
      return "load"
      
    case (.BF16, .FP16):
      fatalError("Invalid precisions.")
    case (.BF16, .BF16):
      return "load"
    case (.BF16, .FP32):
      return "load_bfloat"
      
    case (.FP32, .FP16):
      fatalError("Invalid precisions.")
    case (.FP32, .BF16):
      fatalError("Invalid precisions.")
    case (.FP32, .FP32):
      return "load"
    }
  }
  
  func storeFunction(_ operand: AttentionOperand) -> String {
    guard let memoryPrecision = memoryPrecisions[operand],
          let registerPrecision = registerPrecisions[operand] else {
      fatalError("Precision of \(operand) was not specified.")
    }
    
    switch (memoryPrecision, registerPrecision) {
    case (.FP16, .FP16):
      return "store"
    case (.FP16, .BF16):
      fatalError("Invalid precisions.")
    case (.FP16, .FP32):
      return "store"
      
    case (.BF16, .FP16):
      fatalError("Invalid precisions.")
    case (.BF16, .BF16):
      return "store"
    case (.BF16, .FP32):
      return "store_bfloat"
      
    case (.FP32, .FP16):
      fatalError("Invalid precisions.")
    case (.FP32, .BF16):
      fatalError("Invalid precisions.")
    case (.FP32, .FP32):
      return "store"
    }
  }
  
  func cached(_ operand: AttentionOperand) -> Bool {
    guard let output = cacheState[operand] else {
      fatalError("Cache state of \(operand) was not specified.")
    }
    return output
  }
  
  func transposed(_ operand: AttentionOperand) -> Bool {
    guard let output = transposeState[operand] else {
      fatalError("Transpose state of \(operand) was not specified.")
    }
    return output
  }
}

extension AttentionKernel {
  func sequenceLength(_ operand: AttentionOperand) -> String {
    switch operand {
    case .Q, .dQ: return "R"
    case .K, .dK: return "C"
    case .V, .dV: return "C"
    case .O, .dO: return "R"
    default: fatalError("Unrecognized operand.")
    }
  }
  
  func blockSequenceLength(_ operand: AttentionOperand) -> UInt16 {
    switch type {
    case .forward, .backwardQuery:
      switch operand {
      case .Q, .dQ: return blockDimensions.parallelization
      case .K, .dK: return blockDimensions.traversal
      case .V, .dV: return blockDimensions.traversal
      case .O, .dO: return blockDimensions.parallelization
      default: fatalError("Unrecognized operand.")
      }
      
    case .backwardKeyValue:
      switch operand {
      case .Q, .dQ: return blockDimensions.traversal
      case .K, .dK: return blockDimensions.parallelization
      case .V, .dV: return blockDimensions.parallelization
      case .O, .dO: return blockDimensions.traversal
      default: fatalError("Unrecognized operand.")
      }
    }
  }
  
  func leadingDimension(_ operand: AttentionOperand) -> String {
    if transposed(operand) {
      return sequenceLength(operand)
    } else {
      return "\(headDimension)"
    }
  }
  
  func leadingBlockDimension(_ operand: AttentionOperand) -> UInt16 {
    if transposed(operand) {
      return blockSequenceLength(operand)
    } else {
      return blockDimensions.head
    }
  }
}

extension AttentionKernel {
  var parallelizationDimension: String {
    switch type {
    case .forward, .backwardQuery:
      return "R"
    case .backwardKeyValue:
      return "C"
    }
  }
  
  var parallelizationGroupOffset: String {
    "parallelization_group_offset"
  }
  
  var unsafeParallelizationThreadOffset: String {
    "\(parallelizationGroupOffset) + sidx * 8 + morton_offset.y"
  }
  
  var clampedParallelizationThreadOffset: String {
    "min(\(unsafeParallelizationThreadOffset), \(parallelizationDimension) - 1)"
  }
  
  var traversalDimension: String {
    switch type {
    case .forward, .backwardQuery:
      return "C"
    case .backwardKeyValue:
      return "R"
    }
  }
  
  var traversalOffset: String {
    switch type {
    case .forward, .backwardQuery:
      return "c"
    case .backwardKeyValue:
      return "r"
    }
  }
  
  var paddedTraversalEdge: String {
    let blockDim = blockDimensions.traversal
    let remainder = "\(traversalDimension) % \(blockDim)"
    
    var output = "(\(remainder) == 0) ? \(blockDim) : \(remainder)"
    output = "((\(output)) + 7) / 8 * 8"
    return output
  }
  
  var paddedHeadDimension: UInt16 {
    (headDimension + 8 - 1) / 8 * 8
  }
  
  var paddedHeadEdge: UInt16 {
    let blockDim = blockDimensions.head
    let remainder = (headDimension) % (blockDim)
    
    var output = (remainder) == 0 ? (blockDim) : (remainder)
    output = (((output)) + 7) / 8 * 8
    return output
  }
  
  var threadgroupSize: UInt16 {
    32 * (blockDimensions.parallelization / 8)
  }
  
  private func createThreadgroupMemoryAllocation() -> UInt16 {
    var output: UInt16 = .zero
    
    // Sets the allocation to the maximum of this and the previous allocated
    // size.
    func allocateParallelization(_ operand: AttentionOperand) {
      guard let memoryPrecision = memoryPrecisions[operand] else {
        fatalError("Precision of \(operand) was not specified.")
      }
      
      var blockBytes: UInt16 = 1
      blockBytes *= blockDimensions.parallelization
      blockBytes *= blockDimensions.head
      blockBytes *= UInt16(memoryPrecision.size)
      
      output = max(output, blockBytes)
    }
    func allocateTraversal(_ operand: AttentionOperand) {
      guard let memoryPrecision = memoryPrecisions[operand] else {
        fatalError("Precision of \(operand) was not specified.")
      }
      
      var blockBytes: UInt16 = 1
      blockBytes *= blockDimensions.traversal
      blockBytes *= blockDimensions.head
      blockBytes *= UInt16(memoryPrecision.size)
      
      output = max(output, blockBytes)
    }
    
    // Allocate memory for the GEMM operands.
    switch type {
    case .forward:
      // S = Q * K^T
      allocateParallelization(.Q)
      allocateTraversal(.K)
      
      // O += P * V
      allocateParallelization(.O)
      allocateTraversal(.V)
      
    case .backwardQuery:
      // S = Q * K^T
      allocateParallelization(.Q)
      allocateTraversal(.K)
      
      // dP = dO * V^T
      allocateParallelization(.dO)
      allocateTraversal(.V)
      
      // dQ += dS * K
      allocateParallelization(.dQ)
      allocateTraversal(.K)
      
    case .backwardKeyValue:
      // S^T = K * Q^T
      allocateParallelization(.K)
      allocateTraversal(.Q)
      
      // dV += P^T * dO
      allocateParallelization(.dV)
      allocateTraversal(.dO)
      
      // dP^T = V * dO^T
      allocateParallelization(.V)
      allocateTraversal(.dO)
      
      // dK += dS^T * Q
      allocateParallelization(.dK)
      allocateTraversal(.Q)
    }
    
    // dO * O
    //
    // Will never exceed 4 KB (128 threads/group), 8 KB (256 threads/group).
    if case .backwardQuery = type {
      output = max(
        output,
        2 * blockDimensions.parallelization * 8 * 4)
    }
    
    // L or D
    //
    // Will never exceed ~512 bytes.
    if case .backwardKeyValue = type {
      output = max(
        output,
        blockDimensions.traversal * 4)
    }
    
    return output
  }
}
