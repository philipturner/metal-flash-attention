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
  
  var blockDimensions: (R: UInt16, C: UInt16)
  var matrixDimensionD: UInt16
  var paddedD: UInt16
  
  // The row stride of the intermediate attention matrix.
  var leadingDimensionDerivativeST: UInt32
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
    
    // Declare the size of the register allocation.
    matrixDimensionD = matrixDimensions.D
    paddedD = (matrixDimensions.D + 8 - 1) / 8 * 8
    blockDimensions = (R: 32, C: 32)
    
    leadingDimensions = ("D", "D", "D", "D")
    leadingDimensionDerivativeST = matrixDimensions.C + 32 - 1
    leadingDimensionDerivativeST = leadingDimensionDerivativeST / 32 * 32
    
    threadgroupSize = 128
    threadgroupMemoryAllocation = (32 * 32 + 32 * 32) * 4
    
    // Inject the contents of the headers.
    source += """
\(createMetalSimdgroupEvent())
\(createMetalSimdgroupMatrixStorage())
using namespace metal;

// Dimensions of each matrix.
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant ushort D [[function_constant(2)]];

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
      threadgroupMemoryAllocation += (32 + 32) * 4
    }
    
    source += createCleanup(type: type)
    source += """

}

"""
  }
}
