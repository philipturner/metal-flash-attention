//
//  Softmax.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/25/24.
//

// A configuration for a softmax kernel.
struct SoftmaxDescriptor {
  // The precision of the attention matrix in memory.
  var memoryPrecision: GEMMOperandPrecision?
  
  // Only working with square attention matrices for simplicity. Also, this
  // kernel will fail when N is larger than 16 bits, because you can't fit
  // that many registers on a GPU core with reasonable performance.
  var matrixDimensions: (N: UInt16, D: UInt16)?
  
  // The number of threads per threadgroup.
  var threadgroupSize: UInt16?
}

// Performs scaled softmax on the attention matrix, in-place.
// - Problem size is baked into shader source.
// - Must handle unaligned matrices without sacrificing performance or reading
//   out of bounds.
// - Allow threadgroup sizes of 64, 128, and 256.
struct SoftmaxKernel {
  var source: String = ""
  
  var threadgroupSize: UInt16
  
  init(descriptor: SoftmaxDescriptor) {
    guard let threadgroupSize = descriptor.threadgroupSize,
          let matrixDimensions = descriptor.matrixDimensions,
          let memoryPrecision = descriptor.memoryPrecision else {
      fatalError("Descriptor was incomplete.")
    }
    self.threadgroupSize = threadgroupSize
    
    // Find the MSL keyword corresponding to the precision.
    var precision: String
    switch memoryPrecision {
    case .FP32:
      precision = "float"
    case .FP16:
      precision = "half"
    default:
      fatalError("Unsupported precision.")
    }
    
    // Allocate enough registers to cache the entire matrix row.
    guard threadgroupSize % 32 == 0,
          threadgroupSize >= 64 else {
      fatalError("Invalid group size.")
    }
    var paddedProblemSize = matrixDimensions.N + threadgroupSize - 1
    paddedProblemSize = (paddedProblemSize / threadgroupSize) * threadgroupSize
    
    source = """
  kernel void softmax(
    device \(precision) *attentionMatrix [[buffer(0)]],
    
    uint gid [[threadgroup_position_in_grid]],
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]])
  {
    \(precision) elements[\(paddedProblemSize / threadgroupSize)];

    // Initial proof of concept:
    // - Allocate the number of registers required.
    // - Run a kernel that doesn't do anything
    // - Run the dumb softmax that reads from RAM multiple times.
    // - Run an alternative dumb softmax that writes to TG memory.
    // - Prove that my softmax kernel has better performance.

    attentionMatrix[0] = 1;
  }

  """
  }
}
