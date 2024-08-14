//
//  AttentionKernel.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/27/24.
//

// Declaration of the attention kernel data structure.

// MARK: - Attention Kernel

// Design a set of simple kernels for forward and backward FlashAttention:
// - FP32 (hardcoded data type keyword)
// - 32x32 block, 4 splits (hardcoded block size)
// - all GEMM operands accessed like with standard GEMM + M1
//   - use async copies liberally (no origin shifting for M3)
//   - transposes are supported
// - no masking, dropout, etc.
//
// Within this constrained design space, reach the greatest performance
// physically possible. Compare to standard and semi-standard attention
// kernels with the same data type constraints. Prove the efficacy of each
// design choice before fine-tuning block sizes.

struct AttentionKernel {
  var type: AttentionKernelType
  
  // Categorical attributes for each operand.
  var cacheState: [AttentionOperand: Bool]
  var memoryPrecisions: [AttentionOperand: GEMMOperandPrecision]
  var preferAsyncCache: Bool
  var preferAsyncLoad: Bool
  var transposeState: [AttentionOperand: Bool]
  
  // Layout of the data in registers and threadgroup memory.
  var blockDimensions: (
    parallelization: UInt16, traversal: UInt16, head: UInt16)
  var headDimension: UInt16
  
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
    self.transposeState = descriptor.transposeState
    
    self.blockDimensions = blockDimensions
    self.headDimension = headDimension
  }
}

// MARK: - Utilities

extension AttentionKernel {
  func memoryName(_ operand: AttentionOperand) -> String {
    guard let memoryPrecision = memoryPrecisions[operand] else {
      fatalError("Memory precision of \(operand) was not specified.")
    }
    return memoryPrecision.name
  }
  
  func loadFunction(_ operand: AttentionOperand) -> String {
    guard let memoryPrecision = memoryPrecisions[operand] else {
      fatalError("Precision of \(operand) was not specified.")
    }
    if memoryPrecision == .BF16 {
      return "load_bfloat"
    } else {
      return "load"
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
  
  var threadgroupMemoryAllocation: UInt16 {
    // TODO: Make a higher-level abstraction like "blockBytes", which lets
    // the allocation shrink when using mixed precision. This will only be
    // effective when both inputs and outputs are stored as 16-bit types.
    var output = max(
      blockDimensions.parallelization * blockDimensions.head * 4,
      blockDimensions.traversal * blockDimensions.head * 4)
    
    if case .backwardQuery = type {
      // D[i] = dO * O
      output = max(
        output,
        2 * blockDimensions.parallelization * 8 * 4)
    }
    if case .backwardKeyValue = type {
      // load L or D[i]
      output = max(
        output,
        blockDimensions.traversal * 4)
    }
    
    return output
  }
}
