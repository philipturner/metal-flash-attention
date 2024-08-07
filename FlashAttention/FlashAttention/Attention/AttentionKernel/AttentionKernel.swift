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
  var blockDimensions: (
    parallelization: UInt16,
    traversal: UInt16,
    head: UInt16)
  var cachedInputs: (Q: Bool, K: Bool, V: Bool, dO: Bool)
  var cachedOutputs: (dQ: Bool, dK: Bool, dV: Bool, O: Bool)
  var headDimension: UInt16
  var transposeState: (Q: Bool, K: Bool, V: Bool, O: Bool)
  var type: AttentionKernelType
  
  // The source code to compile.
  var source: String = ""
  
  // The number of threads per group.
  var threadgroupSize: UInt16
  
  // If you allocate threadgroup memory after compiling the kernel, the code
  // has higher performance.
  var threadgroupMemoryAllocation: UInt16
  
  init(descriptor: AttentionDescriptor) {
    guard let blockDimensions = descriptor.blockDimensions,
          let cachedInputs = descriptor.cachedInputs,
          let cachedOutputs = descriptor.cachedOutputs,
          let headDimension = descriptor.headDimension,
          let transposeState = descriptor.transposeState,
          let type = descriptor.type else {
      fatalError("Descriptor was incomplete.")
    }
    self.blockDimensions = blockDimensions
    self.cachedInputs = cachedInputs
    self.cachedOutputs = cachedOutputs
    self.headDimension = headDimension
    self.transposeState = transposeState
    self.type = type
    
    threadgroupSize = 32 * (blockDimensions.parallelization / 8)
    threadgroupMemoryAllocation = max(
      blockDimensions.parallelization * blockDimensions.head * 4,
      blockDimensions.traversal * blockDimensions.head * 4)
    
    if case .backwardQuery = type {
      // D[i] = dO * O
      threadgroupMemoryAllocation = max(
        threadgroupMemoryAllocation,
        2 * blockDimensions.parallelization * 8 * 4)
    }
    if case .backwardKeyValue = type {
      // load L[i] or D[i]
      threadgroupMemoryAllocation = max(
        threadgroupMemoryAllocation,
        blockDimensions.traversal * 4)
    }
    
    // Add the contents of the headers.
    source += """
    
    \(createMetalSimdgroupEvent())
    \(createMetalSimdgroupMatrixStorage())
    using namespace metal;
    
    
    
    """
    
    // Add the contents of the function.
    source += createFunctionSignature()
    source += createSetup()
    
    switch type {
    case .forward:
      source += loopForward()
    case .backwardQuery(let computeDerivativeQ):
      if computeDerivativeQ {
        source += loopBackwardQuery()
      }
    case .backwardKeyValue(let computeDerivativeK):
      source += loopBackwardKeyValue(
        computeDerivativeK: computeDerivativeK)
    }
    
    source += createCleanup(type: type)
    source += """
    
    }
    
    """
  }
}

// MARK: - Utilities

extension AttentionKernel {
  func cached(_ operand: String) -> Bool {
    switch operand {
    case "Q": return cachedInputs.Q
    case "K": return cachedInputs.K
    case "V": return cachedInputs.V
    case "O": return cachedOutputs.O
      
    case "dQ": return cachedOutputs.dQ
    case "dK": return cachedOutputs.dK
    case "dV": return cachedOutputs.dV
    case "dO": return cachedInputs.dO
      
    default: fatalError("Unrecognized operand.")
    }
  }
  
  func transposed(_ operand: String) -> Bool {
    switch operand {
    case "Q", "dQ": return transposeState.Q
    case "K", "dK": return transposeState.K
    case "V", "dV": return transposeState.V
    case "O", "dO": return transposeState.O
    default: fatalError("Unrecognized operand.")
    }
  }
  
  func sequenceLength(_ operand: String) -> String {
    switch operand {
    case "Q", "dQ": return "R"
    case "K", "dK": return "C"
    case "V", "dV": return "C"
    case "O", "dO": return "R"
    default: fatalError("Unrecognized operand.")
    }
  }
  
  func blockSequenceLength(_ operand: String) -> UInt16 {
    switch type {
    case .forward, .backwardQuery:
      switch operand {
      case "Q", "dQ": return blockDimensions.parallelization
      case "K", "dK": return blockDimensions.traversal
      case "V", "dV": return blockDimensions.traversal
      case "O", "dO": return blockDimensions.parallelization
      default: fatalError("Unrecognized operand.")
      }
      
    case .backwardKeyValue:
      switch operand {
      case "Q", "dQ": return blockDimensions.traversal
      case "K", "dK": return blockDimensions.parallelization
      case "V", "dV": return blockDimensions.parallelization
      case "O", "dO": return blockDimensions.traversal
      default: fatalError("Unrecognized operand.")
      }
    }
  }
  
  func leadingDimension(_ operand: String) -> String {
    if transposed(operand) {
      return sequenceLength(operand)
    } else {
      return "\(headDimension)"
    }
  }
  
  func leadingBlockDimension(_ operand: String) -> UInt16 {
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
  
  var parallelizationOffset: String {
    "parallelization_group_offset"
  }
  
  var parallelizationThreadOffset: String {
    "parallelization_thread_offset"
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
}

// MARK: - Function Signature

extension AttentionKernel {
  func createFunctionSignature() -> String {
    // What operands does the kernel use?
    var operands: [AttentionOperand] = [
      .LTerms, .DTerms,
    ]
    switch type {
    case .forward:
      operands += [.Q, .K, .V, .O]
    case .backwardQuery(let computeDerivativeQ):
      operands += [.O, .dO]
      if computeDerivativeQ {
        operands += [.Q, .K, .V, .dQ]
      }
    case .backwardKeyValue(let computeDerivativeK):
      operands += [.Q, .K, .V, .dO, .dV]
      if computeDerivativeK {
        operands += [.dK]
      }
    }
    operands.sort {
      $0.bufferBinding! < $1.bufferBinding!
    }
    
    // Declare the buffer binding for each operand.
    func createBufferBindings() -> String {
      var output: String = ""
      for key in operands {
        var line = "device float* \(key.name) "
        line += "[[buffer(\(key.bufferBinding!))]],"
        output += "  " + line + "\n"
      }
      return output
    }
    
    // Declare the memory offsets.
    func declareOffsets() -> String {
      """
      
      // Base address for async copies.
      uint parallelization_group_offset = gid;
      parallelization_group_offset *= \(blockDimensions.parallelization);
      
      // Base address for directly accessing RAM.
      ushort2 morton_offset = morton_order(lane_id);
      uint parallelization_thread_offset = parallelization_group_offset;
      parallelization_thread_offset += sidx * 8 + morton_offset.y;
      
      """
    }
    
    // Generate the full signature.
    return """
    
    // R = row dimension (output sequence)
    // C = column dimension (input sequence)
    constant uint R [[function_constant(0)]];
    constant uint C [[function_constant(1)]];
    
    // Declare the function.
    kernel void attention(
      \(createBufferBindings())
      threadgroup uchar *threadgroup_block [[threadgroup(0)]],
      
      uint gid [[threadgroup_position_in_grid]],
      ushort sidx [[simdgroup_index_in_threadgroup]],
      ushort lane_id [[thread_index_in_simdgroup]]
    ) {
      \(declareOffsets())
    
    """
  }
}
