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

// Dimensions of each matrix.
constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

// Whether each matrix is transposed.
constant bool A_trans [[function_constant(10)]];
constant bool B_trans [[function_constant(11)]];
constant bool C_trans [[function_constant(12)]];
constant uint A_leading_dim = (A_trans ? M : K);
constant uint B_leading_dim = (B_trans ? K : N);
constant uint C_leading_dim = (C_trans ? M : N);

// Alpha and beta constants from BLAS.
constant float alpha [[function_constant(20)]];
constant float beta [[function_constant(21)]];

constant bool batched [[function_constant(100)]];
constant bool fused_activation [[function_constant(101)]];
constant bool batched_fused_activation = batched && fused_activation;

constant ushort M_simd [[function_constant(200)]];
constant ushort N_simd [[function_constant(201)]];
constant ushort K_simd [[function_constant(202)]];

constant ushort M_splits [[function_constant(210)]];
constant ushort N_splits [[function_constant(211)]];
constant ushort K_splits [[function_constant(212)]];

constant ushort M_group = M_simd * M_splits;
constant ushort N_group = N_simd * N_splits;
constant ushort K_group = K_simd * K_splits;
constant ushort A_block_leading_dim = (A_trans ? M_group : K_group);
constant ushort B_block_leading_dim = (B_trans ? K_group : N_group);

// There is no padding for M reads/writes.
// There is no padding for N reads/writes.
constant ushort K_group_padded = (K % K_group == 0) ? K_group : ~(K_group - 1) & (K % K_group + K_group - 1);
constant ushort K_simd_padded = (K % K_simd == 0) ? K_simd : ~7 & (K % K_simd + 7);

constant ushort A_sram_length = (M_simd / 8) * 1;
constant ushort B_sram_length = 1 * (N_simd / 8);
constant ushort A_block_length = (M_simd * M_splits) * (K_simd * K_splits);
constant ushort B_block_length = (K_simd * K_splits) * (N_simd * N_splits);

constant ushort A_sram_offset = 0;
constant ushort B_sram_offset = A_sram_offset + A_sram_length;
constant ushort C_sram_offset = B_sram_offset + B_sram_length;
constant ushort A_block_offset = 0;
constant ushort B_block_offset = A_block_offset + A_block_length;
constant ushort threadgroup_memory_length = B_block_offset + B_block_length;

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* A_sram(thread simdgroup_matrix_storage<T> *sram, ushort2 matrix_origin) {
  // A_sram[M_simd][8]
  return sram + A_sram_offset + matrix_origin.y * (8 / 8) + matrix_origin.x;
}

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* B_sram(thread simdgroup_matrix_storage<T> *sram, ushort2 matrix_origin) {
  // A_sram[8][N_simd]
  return sram + B_sram_offset + matrix_origin.y * (N_simd / 8) + matrix_origin.x;
}

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* C_sram(thread simdgroup_matrix_storage<T> *sram, ushort2 matrix_origin) {
  // C_sram[M_simd][N_simd]
  return sram + C_sram_offset + matrix_origin.y * (N_simd / 8) + matrix_origin.x;
}

template <typename T>
METAL_FUNC void prefetch(threadgroup T *A_block, device T *A,
                         ushort2 A_tile_src, uint2 A_offset,
                         threadgroup T *B_block, device T *B,
                         ushort2 B_tile_src, uint2 B_offset, uint k)
{
  A_tile_src.x = min(uint(K_group), K - k);
  B_tile_src.y = min(uint(K_group), K - k);
  auto A_src = simdgroup_matrix_storage<T>::apply_offset(A, A_leading_dim, A_offset, A_trans);
  auto B_src = simdgroup_matrix_storage<T>::apply_offset(B, B_leading_dim, B_offset, B_trans);
  
  const uint K_edge_floor = K - K_group_padded;
  const uint K_edge_ceil = K_edge_floor + K_group_padded;
  ushort K_padded;
  if (K_edge_floor == K_group) {
    K_padded = K_group;
  } else {
    K_padded = min(uint(K_group), K_edge_ceil - k);
  }
  ushort2 A_tile_dst(K_padded, A_tile_src.y);
  ushort2 B_tile_dst(B_tile_src.x, K_padded);
  
  simdgroup_event events[2];
  events[0].async_copy(A_block, A_block_leading_dim, 1, ulong2(A_tile_dst), A_src, A_leading_dim, 1, ulong2(A_tile_src), long2(0), simdgroup_async_copy_clamp_mode::clamp_to_zero, A_trans);
  events[1].async_copy(B_block, B_block_leading_dim, 1, ulong2(B_tile_dst), B_src, B_leading_dim, 1, ulong2(B_tile_src), long2(0), simdgroup_async_copy_clamp_mode::clamp_to_zero, B_trans);
  simdgroup_event::wait(2, events);
}

// One iteration of the MACC loop, effectively k=8 iterations.
template <typename T>
METAL_FUNC void multiply_accumulate(thread simdgroup_matrix_storage<T> *sram,
                                    const threadgroup T *A_block,
                                    const threadgroup T *B_block,
                                    bool accumulate = true)
{
  // Elide multiplications on the edge if any matrix dimension is smaller than
  // the block dimension.
  const ushort M_modulo = (M % M_simd == 0) ? M_simd : (M % M_simd);
  const ushort N_modulo = (N % N_simd == 0) ? N_simd : (N % N_simd);
  const ushort M_padded = (M < M_simd) ? (M_modulo + 7) / 8 * 8 : M_simd;
  const ushort N_padded = (N < N_simd) ? (N_modulo + 7) / 8 * 8 : N_simd;
  
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
    ushort2 origin(m, 0);
    auto A = A_sram(sram, origin);
    A->load(A_block, A_block_leading_dim, origin, A_trans);
  }
#pragma clang loop unroll(full)
  for (ushort n = 0; n < N_padded; n += 8) {
    ushort2 origin(0, n);
    auto B = B_sram(sram, origin);
    B->load(B_block, B_block_leading_dim, origin, B_trans);
  }
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
    auto A = A_sram(sram, ushort2(m, 0));
#pragma clang loop unroll(full)
    for (ushort n = 0; n < N_padded; n += 8) {
      auto B = B_sram(sram, ushort2(0, n));
      auto C = C_sram(sram, ushort2(m, n));
      C->multiply(*A, *B, accumulate);
    }
  }
}

template <typename T>
struct activation_functor {
  using function = void(threadgroup T *C,
                        device void *D,
                        uint grid_index_in_batch,
                        uint2 matrix_origin,
                        ushort2 tile_dimensions,
                        ushort lane_id);
  
  typedef visible_function_table<function> function_table;
};

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
  if (batched) {
    ulong3 offsets = matrix_offsets[gid.z];
    A += offsets[0];
    B += offsets[1];
    C += offsets[2];
  }
  
  simdgroup_matrix_storage<T> sram[1024];
  auto A_block = threadgroup_block + A_block_offset;
  auto B_block = threadgroup_block + B_block_offset;
  ushort3 sid(sidx % N_splits,
              (sidx % (M_splits * N_splits) / N_splits),
              sidx / (M_splits * N_splits));
  ushort2 offset_in_simd = simdgroup_matrix_storage<T>::offset(lane_id);
  
  uint2 A_offset(0, gid.y * M_group);
  uint2 B_offset(gid.x * N_group);
  {
    uint C_base_offset_x = B_offset.x + sid.x * N_simd;
    uint C_base_offset_y = A_offset.y + sid.y * M_simd;
    if (C_base_offset_x >= N || C_base_offset_y >= M) {
      return;
    }
  }
  
  // If there are no K splits, do not access sid.z.
  ushort3 offset_in_group(sid.x * N_simd + offset_in_simd.x,
                          sid.y * M_simd + offset_in_simd.y, 0);
  if (K_splits > 1) {
    offset_in_group.z = sid.z * K_simd;
  }
  
  ushort2 A_tile_src;
  ushort2 B_tile_src;
  const uint K_edge_floor = K - K_group_padded;
  const uint K_edge_ceil = K_edge_floor + K_group_padded;
  if (sidx == 0) {
    A_tile_src.y = min(uint(M_group), M - A_offset.y);
    B_tile_src.x = min(uint(N_group), N - B_offset.x);
    prefetch(A_block, A, A_tile_src, A_offset, B_block, B, B_tile_src, B_offset, 0);
  }
  
  if (K > K_simd) {
#pragma clang loop unroll(full)
    for (int m = 0; m < M_simd; m += K_simd) {
#pragma clang loop unroll(full)
      for (int n = 0; n < N_simd; n += K_simd) {
        *C_sram(sram, ushort2(n, m)) = simdgroup_matrix_storage<T>(0);
      }
    }
  }
  
  for (uint K_floor = 0; K_floor < K; K_floor += K_group) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (K_splits > 1 && K_floor + offset_in_group.z >= K) {
      break;
    }
    
    ushort2 A_block_offset(offset_in_group.z, offset_in_group.y);
    ushort2 B_block_offset(offset_in_group.x, offset_in_group.z);
    auto A_block_src = simdgroup_matrix_storage<T>::apply_offset(A_block, A_block_leading_dim, A_block_offset);
    auto B_block_src = simdgroup_matrix_storage<T>::apply_offset(B_block, B_block_leading_dim, B_block_offset);
    
#pragma clang loop unroll(full)
    for (ushort k = 0; k < K_simd_padded; k += 8) {
      bool accumulate = !(k == 0 && K <= K_simd);
      multiply_accumulate(sram, A_block_src, B_block_src, accumulate);
      A_block_src += A_trans ? 8 * M_group : 8;
      B_block_src += B_trans ? 8 : 8 * N_group;
    }
    
    if (K_floor + K_group < K) {
#pragma clang loop unroll(full)
      for (ushort k = K_simd_padded; k < K_simd; k += 8) {
        multiply_accumulate(sram, A_block_src, B_block_src);
        A_block_src += A_trans ? 8 * M_group : 8;
        B_block_src += B_trans ? 8 : 8 * N_group;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      if (sidx == 0) {
        uint K_next = K_floor + K_group;
        prefetch(A_block, A, A_tile_src, A_offset, B_block, B, B_tile_src, B_offset, K_next);
      }
    }
  }
  
  // To properly implement beta in the presence of alpha, fetch the previous C
  // value at the end when storing the accumulator.
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
