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

// TODO: (Q/O/dQ/dO), (K/V/dK/dV), (lm) should use a pre-determined offset.
// These are batch offsets, not offsets caused by H having strides.

// Whether each matrix is transposed.
constant bool Q_trans [[function_constant(10)]];
constant bool K_trans [[function_constant(11)]];
constant bool V_trans [[function_constant(12)]];
constant bool O_trans [[function_constant(13)]];
constant bool mask_trans = false;
constant bool lm_trans = true;

// Leading dimension for the operand and its gradient.
constant uint Q_leading_dim = Q_trans ? R : H * D;
constant uint K_leading_dim = K_trans ? H * D : C;
constant uint V_leading_dim = V_trans ? C : H * D;
constant uint O_leading_dim = O_trans ? R : H * D;
constant uint mask_leading_dim = mask_trans ? R : C;
constant uint lm_leading_dim = lm_trans ? R : H * D;

// Data type of each matrix.
constant uint Q_data_type [[function_constant(20)]];
constant uint K_data_type = Q_data_type;
constant uint V_data_type = Q_data_type;
constant uint O_data_type = Q_data_type;
constant uint mask_data_type = Q_data_type;

constant uint dQ_data_type = (Q_data_type == MTLDataTypeFloat) ? MTLDataTypeFloat : MTLDataTypeBFloat;
constant uint dK_data_type = dQ_data_type;
constant uint dV_data_type = dQ_data_type;
constant uint dO_data_type = dQ_data_type;
constant uint lm_data_type = MTLDataTypeFloat;

constant bool batched [[function_constant(100)]];
constant bool forward [[function_constant(50000)]]; // 101
constant bool backward [[function_constant(102)]];
constant bool gradient [[function_constant(103)]];

// TODO: Allow `block_sparse` without `masked` by just not checking whether the flag applies the mask.
constant bool masked [[function_constant(110)]];
constant bool block_sparse [[function_constant(111)]];
constant bool generate_block_mask = masked && block_sparse && !forward && !backward;

constant ushort R_simd [[function_constant(200)]];
constant ushort C_simd [[function_constant(201)]];

constant ushort R_modulo = (R % R_simd == 0) ? R_simd : (R % R_simd);
constant ushort C_modulo = (C % C_simd == 0) ? C_simd : (C % C_simd);
constant ushort R_padded = (R_modulo + 7) / 8 * 8;
constant ushort C_padded = (C_modulo + 7) / 8 * 8;

constant ushort R_splits [[function_constant(210)]];
constant ushort C_splits [[function_constant(50001)]]; // 211
constant bool have_lm = gradient;

constant ushort R_group = R_simd * R_splits;
constant ushort C_group = C_simd;
constant uint block_mask_leading_dim = mask_trans ? (R + R_group - 1) / R_group : (C + C_group - 1) / C_group;

constant ushort R_simd_padded = (R + 7) / 8 * 8;
constant ushort C_simd_padded = (C + 7) / 8 * 8;
constant ushort D_thread_padded = (D + 1) / 2 * 2;
constant ushort D_simd_padded = (D + 7) / 8 * 8;

constant ushort Q_block_leading_dim = (Q_trans ? R_group : D);
constant ushort K_block_leading_dim = (K_trans ? D : C_group);
constant ushort V_block_leading_dim = (V_trans ? C_group : D);
constant ushort O_block_leading_dim = (O_trans ? R_group : D);
constant ushort mask_block_leading_dim = (mask_trans ? R_group : C_group);

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

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_mask(thread simdgroup_matrix_storage<T> *sram, ushort2 matrix_origin) {
  return get_sram(sram, C_simd, matrix_origin);
}

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_qvo(thread simdgroup_matrix_storage<T> *sram, ushort2 matrix_origin) {
  return get_sram(sram, D_simd_padded, matrix_origin);
}

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_k(thread simdgroup_matrix_storage<T> *sram, ushort2 matrix_origin) {
  return get_sram(sram, C_simd, matrix_origin);
}

template <typename T>
void load_block_mask(thread simdgroup_matrix_storage<T> *mask_sram,
                     threadgroup T *mask_block,
                     ushort r_start, ushort r_end,
                     ushort c_start, ushort c_end,
                     bool broadcast)
{
  if (broadcast) {
#pragma clang loop unroll(full)
    for (ushort r = r_start; r < r_end; r += 8) {
#pragma clang loop unroll(full)
      for (ushort c = c_start; c < c_end; c += 8) {
        ushort2 origin(c, r);
        auto mask = get_mask(mask_sram, origin);
        *mask = mask_sram[0];
      }
    }
  } else {
#pragma clang loop unroll(full)
    for (ushort r = r_start; r < r_end; r += 8) {
#pragma clang loop unroll(full)
      for (ushort c = c_start; c < c_end; c += 8) {
        ushort2 origin(c, r);
        auto mask = get_mask(mask_sram, origin);
        mask->load(mask_block, mask_block_leading_dim, origin, mask_trans);
      }
    }
  }
}

template <typename T, bool is_q, bool is_o, bool is_k, bool is_v>
void load_sram_iter(thread simdgroup_matrix_storage<T> *sram,
                    threadgroup void *block, ushort d)
{
#define LOAD_SRAM(MATRIX, LEADING_DIM, TRANS) \
MATRIX->load((threadgroup T*)block, LEADING_DIM, origin, TRANS); \
  
  if (is_q || is_o) {
#pragma clang loop unroll(full)
    for (ushort r = 0; r < R_simd; r += 8) {
      ushort2 origin(d, r);
      auto qo = get_sram(sram, D_simd_padded, origin);
      LOAD_SRAM(qo, Q_block_leading_dim, Q_trans)
    }
  } else if (is_k || is_v) {
#pragma clang loop unroll(full)
    for (ushort c = 0; c < C_simd; c += 8) {
      if (is_k) {
        ushort2 origin(c, d);
        auto k = get_sram(sram, C_simd, origin);
        LOAD_SRAM(k, K_block_leading_dim, K_trans)
      } else if (is_v) {
        ushort2 origin(d, c);
        auto v = get_sram(sram, D_simd_padded, origin);
        LOAD_SRAM(v, V_block_leading_dim, V_trans)
      }
    }
  }
#undef LOAD_SRAM
}

template <bool upcast_bfloat, bool is_q, bool is_o, bool is_k, bool is_v>
void load_sram_iter_bfloat(thread simdgroup_matrix_storage<float> *sram,
                    threadgroup void *block, ushort d)
{
#define LOAD_SRAM(MATRIX, LEADING_DIM, TRANS) \
if (upcast_bfloat) { \
MATRIX->load_bfloat((threadgroup ushort*)block, LEADING_DIM, origin, TRANS); \
} else { \
MATRIX->load((threadgroup float*)block, LEADING_DIM, origin, TRANS); \
} \
  
  if (is_q || is_o) {
#pragma clang loop unroll(full)
    for (ushort r = 0; r < R_simd; r += 8) {
      ushort2 origin(d, r);
      auto qo = get_sram(sram, D_simd_padded, origin);
      LOAD_SRAM(qo, Q_block_leading_dim, Q_trans)
    }
  } else if (is_k || is_v) {
#pragma clang loop unroll(full)
    for (ushort c = 0; c < C_simd; c += 8) {
      if (is_k) {
        ushort2 origin(c, d);
        auto k = get_sram(sram, C_simd, origin);
        LOAD_SRAM(k, K_block_leading_dim, K_trans)
      } else if (is_v) {
        ushort2 origin(d, c);
        auto v = get_sram(sram, D_simd_padded, origin);
        LOAD_SRAM(v, V_block_leading_dim, V_trans)
      }
    }
  }
#undef LOAD_SRAM
}

template <typename T, bool is_q, bool is_o>
void zero_sram_iter_qo(thread simdgroup_matrix_storage<T> *sram,
                       ushort d, ushort r_start)
{
#pragma clang loop unroll(full)
  for (ushort r = r_start; r < R_simd; r += 8) {
    ushort2 origin(d, r);
    auto qo = get_sram(sram, D_simd_padded, origin);
    *qo = simdgroup_matrix_storage<T>(0);
  }
}

template <typename T, bool is_k, bool is_v>
void zero_sram_iter_kv(thread simdgroup_matrix_storage<T> *sram,
                       ushort d, ushort c_start)
{
#pragma clang loop unroll(full)
  for (ushort c = c_start; c < C_simd; c += 8) {
    if (is_k) {
      ushort2 origin(c, d);
      auto k = get_sram(sram, C_simd, origin);
      *k = simdgroup_matrix_storage<T>(0);
    } else if (is_v) {
      ushort2 origin(d, c);
      auto v = get_sram(sram, D_simd_padded, origin);
      *v = simdgroup_matrix_storage<T>(0);
    }
  }
}

// `sequence_position` is the position at the start of the simd.
template <typename T, bool is_q, bool is_o, bool is_k, bool is_v>
void load_sram(thread simdgroup_matrix_storage<T> *sram,
               threadgroup void *block,
               uint sequence_position, ushort2 offset_in_simd)
{
  if ((is_q || is_o) && (R_modulo < R_simd)) {
    if (sequence_position >= R) {
      return;
    }
  } else if ((is_k || is_v) && (C_modulo < C_simd)) {
    if (sequence_position >= C) {
      return;
    }
  }
  
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D_simd_padded - 8; d += 8) {
    load_sram_iter<
    T, is_q, is_o, is_k, is_v
    >(sram, block, d);
  }
  
  bool do_load;
  if (D_thread_padded == D_simd_padded) {
    do_load = true;
  } else if (is_q || is_o || is_v) {
    do_load = (offset_in_simd.x >= D_thread_padded % 8);
  } else if (is_k) {
    do_load = (offset_in_simd.y >= D_thread_padded % 8);
  } else {
    do_load = false;
  }
  if (do_load) {
    load_sram_iter<
    T, is_q, is_o, is_k, is_v
    >(sram, block, D_simd_padded - 8);
  } else {
    ushort d = D_simd_padded - 8;
    
    if (is_q || is_o) {
      zero_sram_iter_qo<T, is_q, is_o>(sram, d, 0);
    } else if (is_k || is_v) {
      zero_sram_iter_kv<T, is_k, is_v>(sram, d, 0);
    }
  }
  
  if ((is_q || is_o) && (R_simd_padded < R_simd) && (backward)) {
    if (sequence_position + R_simd > R) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < D_simd_padded; d += 8) {
        zero_sram_iter_qo<T, is_q, is_o>(sram, d, R_simd_padded);
      }
    }
  } else if ((is_k || is_v) && (C_simd_padded < C_simd) && (forward)) {
    if (sequence_position + C_simd > C) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < D_simd_padded; d += 8) {
        zero_sram_iter_kv<T, is_k, is_v>(sram, d, C_simd_padded);
      }
    }
  }
}

template <bool upcast_bfloat, bool is_q, bool is_o, bool is_k, bool is_v>
void load_sram_bfloat(thread simdgroup_matrix_storage<float> *sram,
                      threadgroup void *block,
                      uint sequence_position, ushort2 offset_in_simd)
{
  if ((is_q || is_o) && (R_modulo < R_simd)) {
    if (sequence_position >= R) {
      return;
    }
  } else if ((is_k || is_v) && (C_modulo < C_simd)) {
    if (sequence_position >= C) {
      return;
    }
  }
  
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D_simd_padded - 8; d += 8) {
    load_sram_iter_bfloat<
    upcast_bfloat, is_q, is_o, is_k, is_v
    >(sram, block, d);
  }
  
  bool do_load;
  if (D_thread_padded == D_simd_padded) {
    do_load = true;
  } else if (is_q || is_o || is_v) {
    do_load = (offset_in_simd.x >= D_thread_padded % 8);
  } else if (is_k) {
    do_load = (offset_in_simd.y >= D_thread_padded % 8);
  } else {
    do_load = false;
  }
  if (do_load) {
    load_sram_iter_bfloat<
    upcast_bfloat, is_q, is_o, is_k, is_v
    >(sram, block, D_simd_padded - 8);
  } else {
    ushort d = D_simd_padded - 8;
    
    if (is_q || is_o) {
      zero_sram_iter_qo<float, is_q, is_o>(sram, d, 0);
    } else if (is_k || is_v) {
      zero_sram_iter_kv<float, is_k, is_v>(sram, d, 0);
    }
  }
  
  if ((is_q || is_o) && (R_simd_padded < R_simd) && (backward)) {
    if (sequence_position + R_simd > R) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < D_simd_padded; d += 8) {
        zero_sram_iter_qo<float, is_q, is_o>(sram, d, R_simd_padded);
      }
    }
  } else if ((is_k || is_v) && (C_simd_padded < C_simd) && (forward)) {
    if (sequence_position + C_simd > C) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < D_simd_padded; d += 8) {
        zero_sram_iter_kv<float, is_k, is_v>(sram, d, C_simd_padded);
      }
    }
  }
}

template <typename T>
void load_q(thread simdgroup_matrix_storage<T> *sram, threadgroup void *block,
            uint sequence_position, ushort2 offset_in_simd) {
  load_sram<T, true, false, false, false>(sram, block, sequence_position, offset_in_simd);
}

template <typename T>
void load_o(thread simdgroup_matrix_storage<T> *sram, threadgroup void *block,
            uint sequence_position, ushort2 offset_in_simd) {
  load_sram<T, false, true, false, false>(sram, block, sequence_position, offset_in_simd);
}

template <typename T>
void load_k(thread simdgroup_matrix_storage<T> *sram, threadgroup void *block,
            uint sequence_position, ushort2 offset_in_simd) {
  load_sram<T, false, false, true, false>(sram, block, sequence_position, offset_in_simd);
}

template <typename T>
void load_v(thread simdgroup_matrix_storage<T> *sram, threadgroup void *block,
            uint sequence_position, ushort2 offset_in_simd) {
  load_sram<T, false, false, false, true>(sram, block, sequence_position, offset_in_simd);
}

template <bool upcast_bfloat>
void load_q_bfloat(thread simdgroup_matrix_storage<float> *sram, threadgroup void *block,
            uint sequence_position, ushort2 offset_in_simd) {
  load_sram_bfloat<upcast_bfloat, true, false, false, false>(sram, block, sequence_position, offset_in_simd);
}

template <bool upcast_bfloat>
void load_o_bfloat(thread simdgroup_matrix_storage<float> *sram, threadgroup void *block,
            uint sequence_position, ushort2 offset_in_simd) {
  load_sram_bfloat<upcast_bfloat, false, true, false, false>(sram, block, sequence_position, offset_in_simd);
}

template <bool upcast_bfloat>
void load_k_bfloat(thread simdgroup_matrix_storage<float> *sram, threadgroup void *block,
            uint sequence_position, ushort2 offset_in_simd) {
  load_sram_bfloat<upcast_bfloat, false, false, true, false>(sram, block, sequence_position, offset_in_simd);
}

template <bool upcast_bfloat>
void load_v_bfloat(thread simdgroup_matrix_storage<float> *sram, threadgroup void *block,
            uint sequence_position, ushort2 offset_in_simd) {
  load_sram_bfloat<upcast_bfloat, false, false, false, true>(sequence_position, offset_in_simd);
}

// The mask cannot be padded using simple zero-padding or -INF padding. Instead,
// the code for the softmax must explicitly elide computations on the edge.
template <typename T, bool is_generate_block_mask>
void load_mask(thread simdgroup_matrix_storage<T> *sram,
               threadgroup T *block, uint2 mask_offset)
{
  if (is_generate_block_mask) {
    load_block_mask(sram, block, 0, R_padded, 0, C_padded, false);
    
    load_block_mask(sram, block, 0, R_padded, C_padded, C_simd,
                    mask_offset.x + C_simd > C);
    
    load_block_mask(sram, block, R_padded, R_simd, 0, C_padded,
                    mask_offset.y + R_simd > R);
    
    load_block_mask(sram, block, R_padded, R_simd, C_padded, C_simd,
                    (mask_offset.y + R_simd > R) || (mask_offset.x + C_simd > C));
  } else {
    load_block_mask(sram, block, 0, R_simd, 0, C_simd, false);
  }
}

template <typename T, bool is_generate_block_mask>
void prefetch_mask(threadgroup T *dst, device T *src, uint2 mask_offset)
{
  ushort2 src_tile(min(uint(C_group), C - mask_offset.x),
                   min(uint(R_group), R - mask_offset.y));
  ushort2 dst_tile = src_tile;
  if (is_generate_block_mask) {
    dst_tile = (~7 & (src_tile + 7));
  }
  auto mask_src = simdgroup_matrix_storage<T>::apply_offset(src, mask_leading_dim, mask_offset, mask_trans);
  
  simdgroup_event events[1];
  events[0].async_copy(dst, mask_block_leading_dim, dst_tile, mask_src, mask_leading_dim, src_tile, mask_trans, simdgroup_async_copy_clamp_mode::clamp_to_edge);
  simdgroup_event::wait(1, events);
}

// `sequence_position` is the position at the start of the threadgroup.
template <typename T, bool is_q, bool is_o, bool is_k, bool is_v>
void prefetch(threadgroup T *dst, device T *src,
              uint3 gid, uint sequence_position,
              thread simdgroup_event* event)
{
  // TODO: Incorporate the H dimension (gid.y) when fetching data.
  // TODO: Incorporate the batch dimension (gid.z).
  
  // TODO: Finish the function body.
}

template <typename T>
void prefetch_q(threadgroup T *dst, device T *src, uint3 gid, uint sequence_position,
                thread simdgroup_event* event) {
  prefetch<T, true, false, false, false>(dst, src, gid, sequence_position, event);
}

template <typename T>
void prefetch_o(threadgroup T *dst, device T *src, uint3 gid, uint sequence_position,
                thread simdgroup_event* event) {
  prefetch<T, false, true, false, false>(dst, src, gid, sequence_position, event);
}

template <typename T>
void prefetch_k(threadgroup T *dst, device T *src, uint3 gid, uint sequence_position,
                thread simdgroup_event* event) {
  prefetch<T, false, false, true, false>(dst, src, gid, sequence_position, event);
}

template <typename T>
void prefetch_v(threadgroup T *dst, device T *src, uint3 gid, uint sequence_position,
                thread simdgroup_event* event) {
  prefetch<T, false, false, false, true>(dst, src, gid, sequence_position, event);
}

template <typename T>
void zero_init(thread simdgroup_matrix_storage<T>* sram, ushort rows, ushort cols) {
  for (ushort r = 0; r < rows; r += 8) {
    for (ushort c = 0; c < rows; c += 8) {
      ushort2 origin(c, r);
      auto value = get_sram(sram, cols, origin);
      *value = simdgroup_matrix_storage<T>(0);
    }
  }
}

template <typename T>
void zero_init_q(thread simdgroup_matrix_storage<T>* sram) {
  zero_init(sram, R_simd, D_simd_padded);
}

template <typename T>
void zero_init_o(thread simdgroup_matrix_storage<T>* sram) {
  zero_init(sram, R_simd, D_simd_padded);
}

template <typename T>
void zero_init_k(thread simdgroup_matrix_storage<T>* sram) {
  zero_init(sram, D_simd_padded, C_simd);
}

template <typename T>
void zero_init_v(thread simdgroup_matrix_storage<T>* sram) {
  zero_init(sram, C_simd, D_simd_padded);
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

template <typename T>
void _generate_block_mask_impl(threadgroup T *threadgroup_block [[threadgroup(0)]],
                               device T *mask [[buffer(11), function_constant(masked)]],
                               device uchar *block_mask [[buffer(12), function_constant(block_sparse)]],
                               
                               uint3 gid [[threadgroup_position_in_grid]],
                               ushort sidx [[simdgroup_index_in_threadgroup]],
                               ushort lane_id [[thread_index_in_simdgroup]])
{
  uint2 mask_offset(gid.x * C_group, gid.y * R_group + sidx * R_simd);
  if (sidx == 0) {
    prefetch_mask<T, true>(threadgroup_block, mask, mask_offset);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  bool all_off = true;
  bool all_on = true;
  if (mask_offset.x < C && mask_offset.y < R) {
    // Only fetch data from threadgroup memory if mask_sram[0] is valid. The
    // data in that matrix broadcasts to all other registers.
    simdgroup_matrix_storage<T> mask_sram[1024];
    
    ushort2 offset_in_simd = simdgroup_matrix_storage<T>::offset(lane_id);
    threadgroup T *mask_block = threadgroup_block + sidx * R_simd * C_group;
    mask_block = simdgroup_matrix_storage<T>::apply_offset(mask_block, mask_block_leading_dim, offset_in_simd);
    
    load_mask<T, true>(mask_sram, mask_block, mask_offset);
    
#pragma clang loop unroll(full)
    for (ushort r = 0; r < R_simd; r += 8) {
#pragma clang loop unroll(full)
      for (ushort c = 0; c < C_simd; c += 8) {
        ushort2 origin(c, r);
        auto mask = get_mask(mask_sram, origin);
        vec<T, 2> elements = mask->thread_elements()[0];
        
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
    block_mask[gid.y * block_mask_leading_dim + gid.x] = block_mask_element;
  }
}

// TODO:
// Grid X = sequence dimension (R for forward, C for backward)
// Grid Y = heads dimension (H)
// Grid Z = batch dimension
// Threadgroup size = 128
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
    load_q(Q_sram, threadgroup_block, i, offset_in_simd);
    
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
      load_k(K_sram, threadgroup_block, j, offset_in_simd);
      
      simdgroup_matrix_storage<T> attention_matrix[128];
      gemm(R_simd, C_simd, D_simd_padded, false, Q_sram, K_sram, attention_matrix);
      
      // TODO: Softmax
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        simdgroup_event events[1];
        prefetch_v(threadgroup_block, V, gid, j, events);
        simdgroup_event::wait(1, events);
      }
      
      simdgroup_matrix_storage<T> V_sram[128];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      load_v(V_sram, threadgroup_block, j, offset_in_simd);
      
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
    simdgroup_matrix_storage<T> attention_matrix[128];
    
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
  }
}

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

