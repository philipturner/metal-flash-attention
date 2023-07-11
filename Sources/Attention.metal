//
//  Attention.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/26/23.
//

#include <metal_stdlib>
#include "metal_data_type"
#include "metal_simdgroup_event"
#include "metal_simdgroup_matrix_storage"
using namespace metal;

// MARK: - Function Constants

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused"

// Dimensions of each matrix.
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant uint H [[function_constant(2)]];
constant uint D [[function_constant(3)]];

// Whether each matrix is transposed.
constant bool Q_trans [[function_constant(10)]];
constant bool K_trans [[function_constant(11)]];
constant bool V_trans [[function_constant(12)]];
constant bool O_trans [[function_constant(13)]];

// Leading dimension for the operand and its gradient.
constant uint Q_leading_dim = Q_trans ? R : H * D;
constant uint K_leading_dim = K_trans ? H * D : C;
constant uint V_leading_dim = V_trans ? C : H * D;
constant uint O_leading_dim = O_trans ? R : H * D;
constant uint lm_leading_dim = R;

constant uint Q_data_type [[function_constant(20)]];

constant bool batched [[function_constant(100)]];
constant bool forward [[function_constant(50000)]]; // 101
constant bool backward [[function_constant(102)]];
constant bool gradient [[function_constant(103)]];
constant bool have_lm = gradient;

// Masking and sparsity only supported in the forward kernel.
constant bool masked [[function_constant(110)]];
constant bool block_sparse [[function_constant(111)]];
constant bool generate_block_mask = masked && block_sparse && !forward && !backward;

constant ushort R_simd [[function_constant(200)]];
constant ushort C_simd [[function_constant(201)]];
constant ushort D_simd [[function_constant(202)]];

constant ushort R_splits [[function_constant(210)]];
constant ushort R_group = R_simd * R_splits;
constant ushort D_group = (Q_data_type == MTLDataTypeFloat) ? 16 : 32;

constant ushort Q_block_leading_dim = (Q_trans ? R_group : D);
constant ushort K_block_leading_dim = (K_trans ? D : C_simd);
constant ushort V_block_leading_dim = (V_trans ? C_simd : D);
constant ushort O_block_leading_dim = (O_trans ? R_group : D);

#pragma clang diagnostic pop

// MARK: - Utilities

template <typename T>
METAL_FUNC device T* apply_batch_offset(device T *pointer, ulong offset) {
  if (batched) {
    auto casted = (device uchar*)pointer;
    casted += offset;
    return (device T*)casted;
  } else {
    return pointer;
  }
}

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(thread simdgroup_matrix_storage<T> *sram, ushort sram_leading_dim, ushort2 matrix_origin) {
  return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}

template <typename A_data_type, typename B_data_type, typename C_data_type>
void gemm(ushort M, ushort N, ushort K, bool accumulate,
          thread simdgroup_matrix_storage<A_data_type>* A,
          thread simdgroup_matrix_storage<B_data_type>* B,
          thread simdgroup_matrix_storage<C_data_type>* C)
{
#pragma clang loop unroll(full)
  for (ushort k = 0; k < K; k += 8) {
#pragma clang loop unroll(full)
    for (ushort m = 0; m < K; m += 8) {
      ushort2 a_origin(k, m);
#pragma clang loop unroll(full)
      for (ushort n = 0; n < N; n += 8) {
        ushort2 b_origin(n, k);
        auto a = get_sram(A, K, a_origin);
        auto b = get_sram(B, N, b_origin);
        
        ushort2 c_origin(n, m);
        auto c = get_sram(C, N, c_origin);
        c->multiply(*a, *b, accumulate);
      }
    }
  }
}

// MARK: - Kernels

// The mask cannot be padded using simple zero-padding. Consider writing
// directly to threadgroup memory and setting to -INF. Then, guarantee that the
// other operand is zero, not NAN.
template <typename T>
void _generate_block_mask_impl(threadgroup T *threadgroup_block [[threadgroup(0)]],
                               device T *mask [[buffer(11), function_constant(masked)]],
                               device uchar *block_mask [[buffer(12), function_constant(block_sparse)]],
                               
                               uint3 gid [[threadgroup_position_in_grid]],
                               ushort sidx [[simdgroup_index_in_threadgroup]],
                               ushort lane_id [[thread_index_in_simdgroup]])
{
  uint2 mask_offset(gid.x * C_simd, gid.y * R_group + sidx * R_simd);
  if (sidx == 0) {
    ushort2 src_tile(min(uint(C_simd), C - mask_offset.x),
                     min(uint(R_group), R - mask_offset.y));
    ushort2 dst_tile(C_simd, R_group);
    auto mask_src = simdgroup_matrix_storage<T>::apply_offset(mask, C, mask_offset);
    
    simdgroup_event events[1];
    events[0].async_copy(threadgroup_block, C_simd, dst_tile, mask_src, C, src_tile, false, simdgroup_async_copy_clamp_mode::clamp_to_edge);
    simdgroup_event::wait(1, events);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  bool all_off = true;
  bool all_on = true;
  if (mask_offset.x < C && mask_offset.y < R) {
    ushort2 offset_in_simd = simdgroup_matrix_storage<T>::offset(lane_id);
    threadgroup T *mask_block = threadgroup_block + sidx * R_simd * C_simd;
    mask_block = simdgroup_matrix_storage<T>::apply_offset(mask_block, C_simd, offset_in_simd);
    
#pragma clang loop unroll(full)
    for (ushort r = 0; r < R_simd; r += 8) {
#pragma clang loop unroll(full)
      for (ushort c = 0; c < C_simd; c += 8) {
        ushort2 origin(c, r);
        simdgroup_matrix_storage<T> mask;
        mask.load(mask_block, C_simd, origin);
        vec<T, 2> elements = mask.thread_elements()[0];
        
        T off = -numeric_limits<T>::max();
        if (!(elements[0] <= off)) { all_off = false; }
        if (!(elements[1] <= off)) { all_off = false; }
        if (!(elements[0] == 0)) { all_on = false; }
        if (!(elements[1] == 0)) { all_on = false; }
      }
    }
  }
  all_off = simd_ballot(all_off).all();
  all_on = simd_ballot(all_on).all();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  auto results_block = (threadgroup ushort2*)threadgroup_block;
  if (lane_id == 0) {
    results_block[sidx] = ushort2(all_off, all_on);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  if (sidx == 0) {
    if (lane_id < R_splits) {
      ushort2 results = results_block[lane_id];
      all_off = results[0];
      all_on = results[1];
    }
    all_off = simd_ballot(all_off).all();
    all_on = simd_ballot(all_on).all();
    
    ushort block_mask_element;
    if (all_off && all_on) {
      block_mask_element = 2;
    } else if (all_off) {
      block_mask_element = 0;
    } else if (all_on) {
      block_mask_element = 1;
    } else {
      block_mask_element = 2;
    }
    uint block_mask_leading_dim = (C + C_simd - 1) / C_simd;
    block_mask[gid.y * block_mask_leading_dim + gid.x] = block_mask_element;
  }
}

template <typename T, typename dT, bool is_bfloat>
void _attention_impl(device T *Q [[buffer(0)]],
                     device T *K [[buffer(1)]],
                     device T *V [[buffer(2)]],
                     device T *O [[buffer(3)]],
                     
                     device dT *dQ [[buffer(5), function_constant(backward)]],
                     device dT *dK [[buffer(6), function_constant(backward)]],
                     device dT *dV [[buffer(7), function_constant(backward)]],
                     device dT *dO [[buffer(8), function_constant(backward)]],
                     
                     threadgroup T *threadgroup_block [[threadgroup(0)]],
                     device T *mask [[buffer(11), function_constant(masked)]],
                     device uchar *block_mask [[buffer(12), function_constant(block_sparse)]],
                     device float2 *lm [[buffer(13), function_constant(have_lm)]],
                     
                     uint3 gid [[threadgroup_position_in_grid]],
                     ushort sidx [[simdgroup_index_in_threadgroup]],
                     ushort lane_id [[thread_index_in_simdgroup]])
{
  ushort2 offset_in_simd = simdgroup_matrix_storage<T>::offset(lane_id);
  
  // TODO: Rewrite the entire file, forget about elision of zero padding.
  
  // Threadgroup size is 128.
  if (forward) {
    simdgroup_matrix_storage<T> Q_sram[128];
    simdgroup_matrix_storage<float> O_sram[128];
    
    uint i = gid.x * R_group + sidx * R_simd;
    if (sidx == 0) {
      simdgroup_event events[1];
      prefetch_q(threadgroup_block, Q, gid, i, events);
      simdgroup_event::wait(1, events);
    }
    zero_init_o(O_sram);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_q(Q_sram, threadgroup_block, D, false, i, offset_in_simd);
    
    // Inner loop.
    for (uint j = 0; j < C; j += C_group) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        simdgroup_event events[1];
        prefetch_k(threadgroup_block, K, gid, j, events);
        simdgroup_event::wait(1, events);
      }
      
      simdgroup_matrix_storage<T> K_sram[128];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_k(K_sram, threadgroup_block, D, false, j, offset_in_simd);
      
      simdgroup_matrix_storage<T> attention_matrix[128];
      gemm(R_simd, C_simd, D_simd_padded, false, Q_sram, K_sram, attention_matrix);
      
      // TODO: Scale by rsqrt(D) and softmax
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        simdgroup_event events[1];
        prefetch_v(threadgroup_block, V, gid, j, events);
        simdgroup_event::wait(1, events);
      }
      
      simdgroup_matrix_storage<T> V_sram[128];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_v(V_sram, threadgroup_block, D, false, j, offset_in_simd);
      
      gemm(R_simd, D_simd_padded, C_simd, true, attention_matrix, V_sram, O_sram);
    }
  }
  
  // During the backward pass, data flows both toward Q and toward K. A single
  // threadgroup must tackle an entire NxN attention matrix. However, unlike
  // inference, there's typically several batches. Make threadgroups as large as
  // possible (384 threads) and set the X grid dimension to 1.
  if (backward) {
    // Stream all of Q/O/K/V/dQ/dO/dK/dV directly from RAM, only temporarily
    // cached in D=8 matrices.
    for (uint j = 0; j < C; j += C_group) {
      for (uint i = 0; i < R; i += R_group) {
        simdgroup_matrix_storage<T> attention_matrix[128];
        zero_init_attention(attention_matrix);
        
        for (ushort d = 0; d < D; d += D_group) {
          auto Q_block = threadgroup_block;
          auto K_block = threadgroup_block + R_group * D_group;
          if (sidx == 0) {
            simdgroup_event events[2];
            prefetch_q(Q_block, Q, gid, i, events + 0);
            prefetch_k(K_block, K, gid, j, events + 1);
            simdgroup_event::wait(2, events);
          }
          
          simdgroup_matrix_storage<T> Q_sram[128];
          simdgroup_matrix_storage<T> K_sram[128];
#pragma clang loop unroll(full)
          for (ushort k = 0; k < D_group; k += 8) {
            load_q(Q_sram, Q_block, 8, false, i, offset_in_simd);
            load_k(K_sram, K_block, 8, false, j, offset_in_simd);
            gemm(R_simd, C_simd, 8, true, Q_sram, K_sram, attention_matrix);
            Q_block += R_group * 8;
            K_block += 8 * C_group;
          }
        }
        
        // TODO: Scale by rsqrt(D) and virtually softmax
        
        for (ushort d = 0; d < D; d += D_group) {
          auto dV_block = threadgroup_block;
          auto dO_block = threadgroup_block + ;
        }
      }
    }
  }
}

#if 0
simdgroup_matrix_storage<T> K_sram[128];
simdgroup_matrix_storage<T> V_sram[128];
simdgroup_matrix_storage<float> dK_sram[128];
simdgroup_matrix_storage<float> dV_sram[128];

uint sequence_position = gid.x * C_group + sidx * C_simd;
auto K_block = threadgroup_block;
auto V_block = K_block + C_group * D_simd_padded;
if (sidx == 0) {
  simdgroup_event events[2];
  prefetch_k(K_block, K, gid, sequence_position, events + 0);
  prefetch_v(V_block, V, gid, sequence_position, events + 1);
  simdgroup_event::wait(2, events);
}
zero_init_k(dK_sram);
zero_init_v(dV_sram);

threadgroup_barrier(mem_flags::mem_threadgroup);
load_k(K_sram, K_block, sequence_position, offset_in_simd);
load_v(V_sram, V_block, sequence_position, offset_in_simd);
threadgroup_barrier(mem_flags::mem_threadgroup);

// Materialize a 192x256 attention matrix (FP16), 192x128 (FP32).
for (uint i = 0; i < R; i += R_group) {
  
}
#endif

kernel void attention(device void *Q [[buffer(0)]],
                      device void *K [[buffer(1)]],
                      device void *V [[buffer(2)]],
                      device void *O [[buffer(3)]],
                      
                      device void *dQ [[buffer(5), function_constant(backward)]],
                      device void *dK [[buffer(6), function_constant(backward)]],
                      device void *dV [[buffer(7), function_constant(backward)]],
                      device void *dO [[buffer(8), function_constant(backward)]],
                      
                      threadgroup void *threadgroup_block [[threadgroup(0)]],
                      constant ulong4 *matrix_offsets [[buffer(10), function_constant(batched)]],
                      device void *mask [[buffer(11), function_constant(masked)]],
                      device uchar *block_mask [[buffer(12), function_constant(block_sparse)]],
                      device float2 *lm [[buffer(13), function_constant(have_lm)]],
                      
                      uint3 gid [[threadgroup_position_in_grid]],
                      ushort sidx [[simdgroup_index_in_threadgroup]],
                      ushort lane_id [[thread_index_in_simdgroup]])
{
  if (batched) {
    if (masked) {
      mask = ((device uchar*)mask) + matrix_offsets[gid.z].x;
    }
    if (block_sparse) {
      block_mask += matrix_offsets[gid.z].y;
    }
  }
  
  // Don't do anything when function constants are invalid.
  if (forward && backward) {
    
  } else if (backward && !gradient) {
    
  } else if (gradient && !forward && !backward) {
    
  } else if (generate_block_mask) {
    if (mask_data_type == MTLDataTypeFloat) {
      _generate_block_mask_impl<float>((threadgroup float*)threadgroup_block,
                                       (device float*)mask, block_mask,
                                       gid, sidx, lane_id);
    } else if (mask_data_type == MTLDataTypeHalf) {
      _generate_block_mask_impl<half>((threadgroup half*)threadgroup_block,
                                      (device half*)mask, block_mask,
                                      gid, sidx, lane_id);
    }
  } else {
    if (Q_data_type == MTLDataTypeFloat) {
      _attention_impl<
      float, float, false
      >((device float*)Q, (device float*)K,
        (device float*)V, (device float*)O,
        (device float*)dQ, (device float*)dK,
        (device float*)dV, (device float*)dO,
        (threadgroup float*)threadgroup_block,
        (device float*)mask, block_mask, lm,
        gid, sidx, lane_id);
    } else if (Q_data_type == MTLDataTypeHalf) {
      _attention_impl
      <half, ushort, true
      >((device half*)Q, (device half*)K,
        (device half*)V, (device half*)O,
        (device ushort*)dQ, (device ushort*)dK,
        (device ushort*)dV, (device ushort*)dO,
        (threadgroup half*)threadgroup_block,
        (device half*)mask, block_mask, lm,
        gid, sidx, lane_id);
    }
  }
}

