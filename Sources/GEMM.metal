//
//  GEMM.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/23/23.
//

#include <metal_stdlib>
#include "metal_simdgroup_event"
#include "metal_simdgroup_matrix_storage"
using namespace metal;

constant bool batched [[function_constant(100)]];
constant bool fused_activation [[function_constant(101)]];
constant bool batched_fused_activation = batched && fused_activation;

template <typename T>
struct activation_functor {
  using function = void(threadgroup T *C,
                        device void *D,
                        uint grid_index_in_batch,
                        uint2 origin,
                        ushort2 tile_dimensions,
                        ushort lane_id);
  
  typedef visible_function_table<function> function_table;
};

// TODO: For K-splits, always modulo the sidx by 4.
template <typename T>
void _gemm_impl(device T *A [[buffer(0)]],
                device T *B [[buffer(1)]],
                device T *C [[buffer(2)]],
                device void *D [[buffer(3), function_constant(fused_activation)]],
                
                threadgroup T *threadgroup_block [[threadgroup(0)]],
                constant ulong3 *matrix_offsets [[buffer(10), function_constant(batched)]],
                typename activation_functor<T>::function_table table [[buffer(11), function_constant(fused_activation)]],
                constant uint *activation_function_offsets [[buffer(12), function_constant(batched_fused_activation)]],
                
                uint3 gid [[threadgroup_position_in_grid]],
                ushort sidx [[simdgroup_index_in_threadgroup]],
                ushort lane_id [[thread_index_in_simdgroup]])
{
  
}

kernel void hgemm(device half *A [[buffer(0)]],
                  device half *B [[buffer(1)]],
                  device half *C [[buffer(2)]],
                  device void *D [[buffer(3), function_constant(fused_activation)]],
                                   
                  threadgroup half *threadgroup_block [[threadgroup(0)]],
                  constant ulong3 *matrix_offsets [[buffer(10), function_constant(batched)]],
                  typename activation_functor<half>::function_table table [[buffer(11), function_constant(fused_activation)]],
                  constant uint *activation_function_offsets [[buffer(12), function_constant(batched_fused_activation)]],
                                   
                  uint3 gid [[threadgroup_position_in_grid]],
                  ushort sidx [[simdgroup_index_in_threadgroup]],
                  ushort lane_id [[thread_index_in_simdgroup]])
{
  _gemm_impl<half>(A, B, C, D, threadgroup_block, matrix_offsets, table, activation_function_offsets, gid, sidx, lane_id);
}

kernel void sgemm(device float *A [[buffer(0)]],
                  device float *B [[buffer(1)]],
                  device float *C [[buffer(2)]],
                  device void *D [[buffer(3), function_constant(fused_activation)]],
                                   
                  threadgroup float *threadgroup_block [[threadgroup(0)]],
                  constant ulong3 *matrix_offsets [[buffer(10), function_constant(batched)]],
                  typename activation_functor<float>::function_table table [[buffer(11), function_constant(fused_activation)]],
                  constant uint *activation_function_offsets [[buffer(12), function_constant(batched_fused_activation)]],
                                   
                  uint3 gid [[threadgroup_position_in_grid]],
                  ushort sidx [[simdgroup_index_in_threadgroup]],
                  ushort lane_id [[thread_index_in_simdgroup]])
{
  _gemm_impl<float>(A, B, C, D, threadgroup_block, matrix_offsets, table, activation_function_offsets, gid, sidx, lane_id);
}
