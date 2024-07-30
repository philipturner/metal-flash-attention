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
  var cachedInputs: (Q: Bool, K: Bool, V: Bool, dO: Bool)
  var cachedOutputs: (dQ: Bool, dK: Bool, dV: Bool, O: Bool)
  var transposeState: (Q: Bool, K: Bool, V: Bool, O: Bool)
  
  // [TODO: Document] ... Alternatively, change the definition
  // of "R" and "C", so they flip during the backward pass. This decision
  // will remove a lot of ambiguity in intermediate variable names. We
  // currently resort to GEMM M/N/K, which causes a name conflict with the
  // sequence length dimension N).
  var blockDimensions: (R: UInt16, C: UInt16, D: UInt16)
  var headDimension: UInt16
  var leadingDimensions: (Q: String, K: String, V: String, O: String)
  
  // The source code to compile.
  var source: String = ""
  
  // The number of threads per group.
  var threadgroupSize: UInt16
  
  // If you allocate threadgroup memory after compiling the kernel, the code
  // has higher performance.
  var threadgroupMemoryAllocation: UInt16
  
  init(descriptor: AttentionDescriptor) {
    guard let cachedInputs = descriptor.cachedInputs,
          let cachedOutputs = descriptor.cachedOutputs,
          let matrixDimensions = descriptor.matrixDimensions,
          let transposeState = descriptor.transposeState,
          let type = descriptor.type else {
      fatalError("Descriptor was incomplete.")
    }
    self.cachedInputs = cachedInputs
    self.cachedOutputs = cachedOutputs
    self.transposeState = transposeState
    
    // Declare the block sizes.
    blockDimensions = (R: 32, C: 32, D: 64)
    headDimension = matrixDimensions.D
    leadingDimensions = (
      transposeState.Q ? "R" : "\(headDimension)",
      transposeState.K ? "C" : "\(headDimension)",
      transposeState.V ? "C" : "\(headDimension)",
      transposeState.O ? "R" : "\(headDimension)")
    
    threadgroupSize = 32 * (blockDimensions.R / 8)
    threadgroupMemoryAllocation = 32 * blockDimensions.D * 4
    
    // Inject the contents of the headers.
    source += """
\(createMetalSimdgroupEvent())
\(createMetalSimdgroupMatrixStorage())
using namespace metal;

// Currently, the attention matrix row and column dimensions must be the same.
// This is to simplify many ambiguities.
constant uint N [[function_constant(0)]];

// Declare the function.
kernel void attention(

"""
    
    source += createArguments(type: type)
    source += createSetup(type: type)
    switch type {
    case .forward:
      source += createInnerLoopForward()
    case .backwardQuery(let computeDerivativeQ):
      if computeDerivativeQ {
        source += createInnerLoopBackwardQuery()
      }
    case .backwardKeyValue(let computeDerivativeK):
      source += createInnerLoopKeyValue(
        computeDerivativeK: computeDerivativeK)
    }
    
    source += createCleanup(type: type)
    source += """

}

"""
  }
}

// MARK: - Arguments

extension AttentionKernel {
  func createArguments(type: AttentionKernelType) -> String {
    struct AttentionOperand {
      var precision: GEMMOperandPrecision
      var bufferBinding: Int
    }
    
    // Index the operands available during the forward pass.
    var operandsMap: [String: AttentionOperand] = [:]
    operandsMap["Q"] = AttentionOperand(
      precision: .FP32, bufferBinding: 0)
    operandsMap["K"] = AttentionOperand(
      precision: .FP32, bufferBinding: 1)
    operandsMap["V"] = AttentionOperand(
      precision: .FP32, bufferBinding: 2)
    operandsMap["O"] = AttentionOperand(
      precision: .FP32, bufferBinding: 3)
    operandsMap["L_terms"] = AttentionOperand(
      precision: .FP32, bufferBinding: 4)
    
    // Index the operands available during the backward pass.
    operandsMap["dO"] = AttentionOperand(
      precision: .FP32, bufferBinding: 5)
    operandsMap["D_terms"] = AttentionOperand(
      precision: .FP32, bufferBinding: 6)
    operandsMap["dV"] = AttentionOperand(
      precision: .FP32, bufferBinding: 7)
    operandsMap["dST"] = AttentionOperand(
      // This is an intermediate allocation, managed internally by the MFA
      // backend. We can impose constraints on it that wouldn't typically be
      // feasible. For example, we can force the row stride to be divisible by
      // the block size (~32). This simplifies the code; we don't need to run
      // async copies to safeguard against corrupted memory accesses.
      //
      // If the matrix rows are noncontiguous, we must modify the in-tree
      // GEMM kernel to support custom leading dimensions. This can be
      // something modified explicitly by the user - an option to override the
      // default leading dimension. The leading dimension is specified after
      // the 'GEMMKernelDescriptor' is created from the 'GEMMDescriptor', and
      // before the 'GEMMKernel' is created from the 'GEMMKernelDescriptor'.
      precision: .FP32, bufferBinding: 8)
    operandsMap["dK"] = AttentionOperand(
      precision: .FP32, bufferBinding: 8)
    operandsMap["dQ"] = AttentionOperand(
      precision: .FP32, bufferBinding: 9)
    
    // Select the operands used by this variant.
    var operandKeys: [String]
    switch type {
    case .forward(let computeL):
      operandKeys = [
        "Q", "K", "V", "O"
      ]
      if computeL {
        operandKeys.append("L_terms")
      }
    case .backwardQuery(let computeDerivativeQ):
      if computeDerivativeQ {
        operandKeys = [
          "Q", "K", "V", "O",
          "L_terms", "dO", "D_terms", "dQ"
        ]
      } else {
        operandKeys = [
          "O", "dO", "D_terms"
        ]
      }
    case .backwardKeyValue(let computeDerivativeK):
      operandKeys = [
        "Q", "K", "V",
        "L_terms", "dO", "D_terms", "dV"
      ]
      if computeDerivativeK {
        operandKeys.append("dK")
      } else {
        operandKeys.append("dST")
      }
    }
    
    // Collect the operands into a single string.
    var output: String = ""
    for key in operandKeys {
      let operand = operandsMap[key]!
      
      var line = "  "
      line += "device "
      line += operand.precision.name + " "
      line += "*" + key + " "
      line += "[[buffer(\(operand.bufferBinding))]]"
      line += ",\n"
      output += line
    }
    
    // Add the arguments that define the thread's position.
    output += """
  
  threadgroup uchar *threadgroup_block [[threadgroup(0)]],
  
  uint gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]]
) {
  ushort2 morton_offset = morton_order(lane_id);

"""
    
    // The thread's array slot in the row or column dimension (whichever the
    // kernel is parallelized over). Used for indexing into 1D arrays.
    switch type {
    case .forward, .backwardQuery:
      output += """

  uint linear_array_slot = gid * 32 + sidx * 8 + morton_offset.y;

"""
    default:
      break
    }
    
    return output
  }
}
