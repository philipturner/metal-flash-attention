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
//   - use async copies
//   - transposes are supported
// - no masking, dropout, etc.

struct AttentionKernel {
  // The source code to compile.
  var source: String = ""
  
  var blockDimensions: (R: UInt16, C: UInt16)
  var leadingDimensions: (
    Q: String, K: String, V: String, O: String)
  var matrixDimensionD: UInt16
  var memoryPrecisions: (
    Q: AttentionOperandPrecision,
    K: AttentionOperandPrecision,
    V: AttentionOperandPrecision,
    O: AttentionOperandPrecision)
  var paddedD: UInt16
  var transposeState: (
    Q: Bool, K: Bool, V: Bool, O: Bool)
  
  // The row stride of the intermediate attention matrix.
  var leadingDimensionDerivativeST: UInt32
  
  // The number of threads per group.
  var threadgroupSize: UInt16
  
  // If you allocate threadgroup memory after compiling the kernel, the code
  // has higher performance.
  var threadgroupMemoryAllocation: UInt16
  
  init(descriptor: AttentionDescriptor) {
    guard let matrixDimensions = descriptor.matrixDimensions,
          let memoryPrecisions = descriptor.memoryPrecisions,
          let transposeState = descriptor.transposeState,
          let type = descriptor.type else {
      fatalError("Descriptor was incomplete.")
    }
    self.matrixDimensionD = matrixDimensions.D
    self.memoryPrecisions = memoryPrecisions
    self.transposeState = transposeState
    
    // Inject the contents of the headers.
    source += """
\(createMetalSimdgroupEvent())
\(createMetalSimdgroupMatrixStorage())
using namespace metal;

"""
    
    // Declare the size of the register allocation.
    paddedD = (matrixDimensions.D + 8 - 1) / 8 * 8
    
    blockDimensions = (R: 32, C: 32)
    leadingDimensions = ("D", "D", "D", "D")
    leadingDimensionDerivativeST = matrixDimensions.C + 32 - 1
    leadingDimensionDerivativeST = leadingDimensionDerivativeST / 32 * 32
    threadgroupSize = 128
    
    source += """

// Dimensions of each matrix.
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant ushort D [[function_constant(2)]];

// Declare the function.
kernel void attention(

"""
    
    threadgroupMemoryAllocation = (32 * 32 + 32 * 32) * 4
    
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
      threadgroupMemoryAllocation += (32 + 32) * 4
    }
    
    source += createCleanup(type: type)
    source += """

}

"""
  }
}
