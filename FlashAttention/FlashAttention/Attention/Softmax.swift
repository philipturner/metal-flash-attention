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
    
    // Check that the threadgroup size is compatible with the shader.
    guard threadgroupSize % 32 == 0,
          threadgroupSize >= 64,
          threadgroupSize.nonzeroBitCount == 1 else {
      fatalError("Invalid group size.")
    }
    

    
    // Allocate enough registers to cache the entire matrix row.
    let C = matrixDimensions.N
    var paddedC = C + threadgroupSize - 1
    paddedC = (paddedC / threadgroupSize) * threadgroupSize
    
    // Apply the "scale" in scaled dot product attention.
    let scaleFactor = 1 / Float(matrixDimensions.D).squareRoot()
    

    
    source = """
#include <metal_stdlib>
using namespace metal;

kernel void softmax(
  device \(precision) *attentionMatrix [[buffer(0)]],
  
  uint gid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  ushort lane_id [[thread_index_in_simdgroup]])
{
  \(precision) elements[\(paddedC / threadgroupSize)];
  threadgroup float simd_messages[\(threadgroupSize / 32)];

  // Initial proof of concept:
  // - Allocate the number of registers required.
  // - Run a kernel that doesn't do anything
  // - Run the dumb softmax that reads from RAM multiple times.
  // - Run an alternative dumb softmax that writes to TG memory.
  // - Prove that my softmax kernel has better performance.

  ushort thread_id = sidx * 32 + lane_id;
  auto baseAddress = attentionMatrix + gid * \(C);

  // Accumulate the maximum.
  float m = -numeric_limits<float>::max();
  for (uint c = thread_id; c < \(C); c += \(threadgroupSize)) {
    \(precision) value = \(scaleFactor) * baseAddress[c];
    m = max(m, value);
  }
  m = simd_max(m);
  \(createReduction("m") { "max(\($0), \($1))" })
  
  // Accumulate the sum.

  // Write the output.
  baseAddress[thread_id] = m;
  
}

"""
  }
  
  // Utility function for reducing across simds.
  private func createReduction(
    _ registerName: String,
    _ operation: (String, String) -> String
  ) -> String {
    var output = """
simd_messages[sidx] = \(registerName);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane_id < \(threadgroupSize / 32)) {
  \(registerName) = simd_messages[lane_id];
} else {
  \(registerName) = 0;
}
"""
    
    var shiftAmount = 1
    while shiftAmount < threadgroupSize / 32 {
      let lhs = registerName
      let rhs = "simd_shuffle_xor(\(registerName), \(shiftAmount))"
      output += "\(registerName) = \(operation(lhs, rhs));"
      output += "\n"
      
      shiftAmount *= 2
    }
    
    return output
  }
}
