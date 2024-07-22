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
  
  // These variables should be 'private', but we need to split the code into
  // multiple files. Swift treats 'private' as a synonym for 'fileprivate'.
  var leadingDimensions: (
    Q: String, K: String, V: String, O: String)
  var leadingBlockDimensions: (
    Q: UInt16, K: UInt16, V: UInt16, O: UInt16)
  var matrixDimensionD: UInt16
  var memoryPrecisions: (
    Q: AttentionOperandPrecision,
    K: AttentionOperandPrecision,
    V: AttentionOperandPrecision,
    O: AttentionOperandPrecision)
  var paddedD: UInt16
  var transposeState: (
    Q: Bool, K: Bool, V: Bool, O: Bool)
  
  // Reads of very large K/V operands may be read in small chunks along 'D',
  // to minimize register pressure. Therefore, there can be a block dimension
  // for D.
  var blockDimensions: (R: UInt16, C: UInt16, D: UInt16)
  
  // The row stride of the intermediate attention matrix.
  var leadingDimensionDerivativeST: UInt32
  
  // If you allocate threadgroup memory after compiling the kernel, the code
  // has higher performance.
  var threadgroupMemoryAllocation: UInt16
  
  // The number of threads per group.
  var threadgroupSize: UInt16
  
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
    
    // Determine the block dimensions from the transpose state.
    leadingDimensions = ("D", "D", "D", "D")
    leadingBlockDimensions = (paddedD, paddedD, paddedD, paddedD)
    if transposeState.Q {
      leadingDimensions.Q = "R"
      leadingBlockDimensions.Q = 32
    }
    if transposeState.K {
      leadingDimensions.K = "C"
      leadingBlockDimensions.K = 32
    }
    if transposeState.V {
      leadingDimensions.V = "C"
      leadingBlockDimensions.V = 32
    }
    if transposeState.O {
      leadingDimensions.O = "R"
      leadingBlockDimensions.O = 32
    }
    leadingDimensionDerivativeST = matrixDimensions.C + 32 - 1
    leadingDimensionDerivativeST = leadingDimensionDerivativeST / 32 * 32
    
    blockDimensions = (R: 32, C: 32, D: paddedD)
    threadgroupMemoryAllocation = .zero
    threadgroupSize = 128
    
    source += """

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-const-variable"

// Dimensions of each matrix.
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant ushort D [[function_constant(2)]];

// Define the memory layout of the matrix block.
constant ushort R_group = 32;
constant ushort C_group = 32;

#pragma clang diagnostic pop

// Declare the function.
kernel void attention(

"""
    
    // R/C_group * D * sizeof(float)
    //
    // Temporary patch: paddedD -> max(paddedD, 64)
    threadgroupMemoryAllocation += 32 * max(paddedD, 64) * 4
    
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
      // R_group * sizeof(float)
      threadgroupMemoryAllocation += 32 * 4
      
      source += createInnerLoopKeyValue(
        computeDerivativeK: computeDerivativeK)
    }
    
    // Temporary patch, until the new versions of the kernels are finished.
    threadgroupMemoryAllocation *= 2
    
    source += createCleanup(type: type)
    source += """

}

"""
  }
}
