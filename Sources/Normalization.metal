//
//  Normalization.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/26/23.
//

#include <metal_stdlib>
using namespace metal;

// Slow reference implementation of group normalization, with the same IO
// complexity as having separate shaders. A future implementation can coalesce
// memory accesses and cache the data inside registers for a drastic speedup.
//
// Although much slower than theoretically possible, this should be much
// faster than the LLaMA.cpp implementation or an MPSGraph implementation.
//
// With this shader, dispatch 256 threads/threadgroup and 1 threadgroup/row.
template <typename T>
void _normalization_impl(device T *source [[buffer(0)]],
                         device T *destination [[buffer(1)]],
                         constant uint &row_size [[buffer(2)]],
                         threadgroup float *partials,
                         
                         uint tgid [[threadgroup_position_in_grid]],
                         ushort sidx [[simdgroup_index_in_threadgroup]],
                         ushort lid [[thread_position_in_threadgroup]])
{
  source += tgid * row_size;
  destination += tgid * row_size;
  
  float sum = 0;
  for (uint i = lid; i < row_size; i += 256) {
    sum += source[i];
  }
  partials[sidx] = simd_sum(sum);
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
  float scale_factor;
  if (lid < 8) {
    sum = partials[lid];
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 4);
    
    scale_factor = 1 / float(row_size);
    partials[lid] = sum * scale_factor;
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mean = partials[sidx];
  float variance = 0;
  for (uint i = lid; i < row_size; i += 256) {
    float deviation = source[i] - mean;
    variance = fma(deviation, deviation, variance);
  }
  partials[sidx] = simd_sum(variance);
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < 8) {
    variance = partials[lid];
    variance += simd_shuffle_xor(variance, 1);
    variance += simd_shuffle_xor(variance, 2);
    variance += simd_shuffle_xor(variance, 4);
    variance = fma(variance, scale_factor, 1e-8);
    
    partials[lid] = rsqrt(variance);
  }
#pragma clang diagnostic pop
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float standard_deviation_reciprocal = partials[sidx];
  float scaled_mean = mean * standard_deviation_reciprocal;
  for (uint i = lid; i < row_size; i += 256) {
    half source_value = source[i];
    destination[i] = half(fma(float(source_value), standard_deviation_reciprocal, -scaled_mean));
  }
}

// Only support half-precision for now; a future implementation will use
// function constants. Then, you can cache multiple pipeline variants, including
// ones that accept single-precision.
kernel void normalization(device half *source [[buffer(0)]],
                          device half *destination [[buffer(1)]],
                          constant uint &row_size [[buffer(2)]],
                          
                          uint tgid [[threadgroup_position_in_grid]],
                          ushort sidx [[simdgroup_index_in_threadgroup]],
                          ushort lid [[thread_position_in_threadgroup]])
{
  threadgroup float partials[8];
  _normalization_impl(source, destination, row_size, partials, tgid, sidx, lid);
}
