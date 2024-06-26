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
    
    // Shorten some names to keep the code within 80 spaces.
    let precision = memoryPrecision.name
    let C = matrixDimensions.N
    
    // Check that the threadgroup size is compatible with the shader.
    guard threadgroupSize % 32 == 0,
          threadgroupSize >= 64,
          threadgroupSize.nonzeroBitCount == 1 else {
      fatalError("Invalid group size.")
    }
    
    // Allocate enough registers to cache the entire matrix row.
    var paddedC = C + threadgroupSize - 1
    paddedC = (paddedC / threadgroupSize) * threadgroupSize
    
    // Apply the "scale" in scaled dot product attention.
    let scaleFactorValue = 1 / Float(matrixDimensions.D).squareRoot()
    let scaleFactor = "\(precision)(\(scaleFactorValue))"
    
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
  
  ushort thread_id = sidx * 32 + lane_id;
  auto baseAddress = attentionMatrix + gid * \(C) + thread_id;
  
#pragma clang loop unroll(full)
  for (uint c = 0; c < \(C / threadgroupSize); c += 1) {
    elements[c] = baseAddress[c * \(threadgroupSize)];
  }
  if (thread_id < \(threadgroupSize - (paddedC - C))) {
    uint c = \(paddedC / threadgroupSize - 1);
    elements[c] = baseAddress[c * \(threadgroupSize)];
  } else {
    elements[\(paddedC / threadgroupSize - 1)] = -numeric_limits<float>::max();
  }

"""
    
    source += """
  
  // Accumulate the maximum.
  float m = -numeric_limits<float>::max();
#pragma clang loop unroll(full)
  for (uint c = 0; c < \(paddedC / threadgroupSize); c += 1) {
    \(precision) value = elements[c];
    m = max(m, float(value));
  }
  m = simd_max(m);
  \(createReduction("m") { "max(\($0), \($1))" })
  m *= \(scaleFactor);
  
  // Accumulate the sum.
  float l = 0;
#pragma clang loop unroll(full)
  for (uint c = 0; c < \(paddedC / threadgroupSize); c += 1) {
    \(precision) value = elements[c];
    float exp_term = fast::exp2(
      fma(float(value), \(scaleFactor) * M_LOG2E_F, -m));
    l += exp_term;
   
    elements[c] = \(precision)(exp_term);
  }
  l = simd_sum(l);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  \(createReduction("l") { "\($0) + \($1)" })
  
  // Write the output.
  \(precision) l_recip = \(precision)(1 / l);
#pragma clang loop unroll(full)
  for (uint c = 0; c < \(C / threadgroupSize); c += 1) {
    \(precision) value = elements[c];
    baseAddress[c * \(threadgroupSize)] = value * l_recip;
  }
  if (thread_id < \(threadgroupSize - (paddedC - C))) {
    uint c = \(paddedC / threadgroupSize - 1);
    \(precision) value = elements[c];
    baseAddress[c * \(threadgroupSize)] = value * l_recip;
  }
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
\(registerName) = simd_messages[lane_id & \(threadgroupSize / 32 - 1)];
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
