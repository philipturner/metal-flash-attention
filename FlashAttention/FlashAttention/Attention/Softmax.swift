//
//  Softmax.swift
//  FlashAttention
//
//  Created by Philip Turner on 6/25/24.
//

// A configuration for a softmax kernel.
struct SoftmaxDescriptor {
  // The number of threads per threadgroup.
  var groupSize: UInt16?
  
  // The precision of the attention matrix in memory.
  var memoryPrecision: GEMMOperandPrecision?
  
  // Only working with square attention matrices for simplicity. Also, this
  // kernel will fail when N is larger than 16 bits, because you can't fit
  // that many registers on a GPU core with reasonable performance.
  var matrixDimensions: (N: UInt16, D: UInt16)?
}

// Performs scaled softmax on the attention matrix, in-place.
// - Problem size is baked into shader source.
// - Must handle unaligned matrices without sacrificing performance or reading
//   out of bounds.
// - Allow threadgroup sizes of 64, 128, and 256.
func createSoftmaxKernel(descriptor: SoftmaxDescriptor) -> String {
  guard let groupSize = descriptor.groupSize,
        let matrixDimensions = descriptor.matrixDimensions,
        let memoryPrecision = descriptor.memoryPrecision else {
    fatalError("Descriptor was incomplete.")
  }
  
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
  guard groupSize % 32 == 0,
        groupSize >= 64 else {
    fatalError("Invalid group size.")
  }
  var paddedProblemSize = matrixDimensions.N + groupSize - 1
  paddedProblemSize = (paddedProblemSize / groupSize) * groupSize
  
  return """
kernel void softmax(
  device \(precision) *attentionMatrix [[buffer(0)]],
  
  uint gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  \(precision) elements[\(paddedProblemSize / groupSize)];

  // Initial proof of concept:
  // - Allocate the number of registers required.
  // - Run a kernel that doesn't do anything
  // - Run the dumb softmax that reads from RAM multiple times.
  // - Run an alternative dumb softmax that writes to TG memory.
  // - Prove that my softmax kernel has better performance.
}

"""
}
