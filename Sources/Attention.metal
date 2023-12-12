//
//  Attention.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/26/23.
//

#include <metal_stdlib>
#include "metal_data_type"
#include "metal_fault_counter"
#include "metal_simdgroup_event"
#include "metal_simdgroup_matrix_storage"
using namespace metal;

// MARK: - Function Constants

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
constant uint Q_leading_dim = Q_trans ? R : H * D;
constant uint K_leading_dim = K_trans ? H * D : C;
constant uint V_leading_dim = V_trans ? C : H * D;
constant uint O_leading_dim = O_trans ? R : H * D;

// Value of `rsqrt(float(D))`.
constant float alpha [[function_constant(20)]];

constant uint Q_data_type [[function_constant(30)]];

constant bool batched [[function_constant(100)]];
constant bool masked [[function_constant(50000)]]; // 101
constant bool block_sparse [[function_constant(102)]];
constant bool block_sparse_masked = masked && block_sparse;
constant bool triangular [[function_constant(103)]];

constant bool forward [[function_constant(110)]];
constant bool backward [[function_constant(111)]];
constant bool generate_block_mask [[function_constant(112)]];
constant bool grouped_query [[function_constant(113)]];
constant bool float_accumulator [[function_constant(114)]];

constant ushort R_simd [[function_constant(200)]];
constant ushort C_simd [[function_constant(201)]];
constant ushort D_simd = (D + 7) / 8 * 8;

constant ushort R_splits [[function_constant(210)]];
constant ushort R_group = R_simd * R_splits;
constant bool fuse_async_loads [[function_constant(213)]];
constant bool _fuse_async_loads = is_function_constant_defined(fuse_async_loads) ? fuse_async_loads : false;

constant ushort R_bank_offset [[function_constant(220)]];
constant ushort C_bank_offset [[function_constant(221)]];
constant ushort D_bank_offset [[function_constant(222)]];
constant bool R_bank_offset_defined = is_function_constant_defined(R_bank_offset);
constant bool C_bank_offset_defined = is_function_constant_defined(C_bank_offset);
constant bool D_bank_offset_defined = is_function_constant_defined(D_bank_offset);

constant ushort R_block_dim = R_group + (R_bank_offset_defined ? R_bank_offset : 0);
constant ushort C_block_dim = C_simd + (C_bank_offset_defined ? C_bank_offset : 0);
constant ushort D_block_dim = D_simd + (D_bank_offset_defined ? D_bank_offset : 0);
constant ushort V_block_offset = (K_trans ? C_simd * D_block_dim : D_simd * C_block_dim);

constant ushort Q_block_leading_dim = (Q_trans ? R_block_dim : D_block_dim);
constant ushort K_block_leading_dim = (K_trans ? D_block_dim : C_block_dim);
constant ushort V_block_leading_dim = (V_trans ? C_block_dim : D_block_dim);
constant ushort O_block_leading_dim = (O_trans ? R_block_dim : D_block_dim);

// MARK: - Utilities

template <typename T>
METAL_FUNC device T* apply_batch_offset(device T *pointer, ulong offset) {
  if (batched) {
    return pointer + offset;
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
          ushort A_leading_dim, thread simdgroup_matrix_storage<A_data_type>* A,
          ushort B_leading_dim, thread simdgroup_matrix_storage<B_data_type>* B,
          ushort C_leading_dim, thread simdgroup_matrix_storage<C_data_type>* C)
{
#pragma clang loop unroll(full)
  for (ushort k = 0; k < K; k += 8) {
#pragma clang loop unroll(full)
    for (ushort m = 0; m < M; m += 8) {
      ushort2 a_origin(k, m);
#pragma clang loop unroll(full)
      for (ushort n = 0; n < N; n += 8) {
        ushort2 b_origin(n, k);
        auto a = get_sram(A, A_leading_dim, a_origin);
        auto b = get_sram(B, B_leading_dim, b_origin);
        
        ushort2 c_origin(n, m);
        auto c = get_sram(C, C_leading_dim, c_origin);
        c->multiply(*a, *b, accumulate);
      }
    }
  }
}

template <typename T, typename U, typename TtoU>
void apply_mask(ushort r, uint j_next, ushort2 offset_in_simd,
                device T *mask_src,
                thread simdgroup_matrix_storage<U>* attention_matrix)
{
#define LOAD_MASK \
ushort2 origin(c, r); \
simdgroup_matrix_storage<T> mask; \
\
if (C % 2 == 0) { \
mask.load(mask_src, C, origin); \
} else { \
mask.load_first(mask_src, C, origin); \
origin.x += 1; \
mask.load_second(mask_src, C, origin); \
} \
\
auto s = get_sram(attention_matrix, C_simd, origin); \
if (generate_block_mask) { \
*(s->thread_elements()) = TtoU(*(mask.thread_elements())); \
} else { \
*(s->thread_elements()) += TtoU(*(mask.thread_elements())); \
} \

  ushort C_edge = (C % C_simd == 0) ? C_simd : (C % C_simd);
  ushort C_floor = C_edge / 8 * 8;
  short C_modulo = C_edge - C_floor;
  
  if (C % 8 == 0) {
#pragma clang loop unroll(full)
    for (ushort c = 0; c < C_floor; c += 8) {
      LOAD_MASK
    }
    
    if (C_edge > 0) {
      if (j_next <= C) {
#pragma clang loop unroll(full)
        for (ushort c = C_floor; c < C_simd; c += 8) {
          LOAD_MASK
        }
      }
    }
  } else if (j_next <= C) {
#pragma clang loop unroll(full)
    for (ushort c = 0; c < C_simd; c += 8) {
      LOAD_MASK
    }
  } else {
#pragma clang loop unroll(full)
    for (ushort c = 0; c < C_floor; c += 8) {
      LOAD_MASK
    }
    ushort2 origin(C_floor, r);
    simdgroup_matrix_storage<T> mask;
    
    if (C % 2 == 0) {
      if (short(offset_in_simd.x) <= short(C_modulo - 2)) {
        mask.load(mask_src, C, origin);
      }
    } else {
      if (short(offset_in_simd.x) <= short(C_modulo - 1)) {
        mask.load_first(mask_src, C, origin);
      }
      
      if (generate_block_mask) {
        if (short(offset_in_simd.x) <= short(C_modulo - 2)) {
          origin.x += 1;
        }
        mask.load_second(mask_src, C, origin);
      } else {
        origin.x += 1;
        if (short(offset_in_simd.x) <= short(C_modulo - 2)) {
          mask.load_second(mask_src, C, origin);
        }
      }
    }
    
    auto s = get_sram(attention_matrix, C_simd, origin);
    if (generate_block_mask) {
      *(s->thread_elements()) = TtoU(*(mask.thread_elements()));
    } else {
      *(s->thread_elements()) += TtoU(*(mask.thread_elements()));
    }
  }
}

class triangular_pass {
public:
  static constant ushort none = 0;
  static constant ushort lower = 1;
  static constant ushort upper = 2;
  
  ushort pass_id;
  bool can_store_o;
  
  uint i_block;
  uint j_block;
  uint j_block_end;
  
  METAL_FUNC triangular_pass(ushort pass_id, uint i_block) {
    this->pass_id = pass_id;
    
    uint grid_x = (R + R_group - 1) / R_group;
    uint complete_blocks = R / R_group;
    uint upper_blocks = complete_blocks / 2;
    uint lower_blocks = complete_blocks - upper_blocks;
    uint edge_blocks = grid_x - complete_blocks;
    uint split_blocks = lower_blocks + edge_blocks;
    
    uint split_i_block = i_block % split_blocks;
    uint j_block_max = (C + C_simd - 1) / C_simd;
    
    switch (pass_id) {
      case none: {
        can_store_o = true;
        
        this->i_block = i_block;
        j_block = 0;
        j_block_end = j_block_max;
        break;
      }
      case lower: {
        can_store_o = (i_block != split_i_block);
        
        this->i_block = split_i_block;
        if (i_block != split_i_block) {
          j_block = 0;
          j_block_end = (j_block_max + 1) / 2;
        } else {
          j_block = (j_block_max + 1) / 2;
          j_block_end = j_block_max;
        }
        break;
      }
      case upper: {
        can_store_o = (i_block < upper_blocks);
        
        this->i_block = upper_blocks - 1 - i_block;
        j_block = j_block_max - 1;
        j_block_end = 0;
        break;
      }
    }
  }
  
  METAL_FUNC bool continue_j_block() const {
    if (pass_id == upper) {
      return int(j_block) >= 0;
    } else {
      return j_block < j_block_end;
    }
  }
  
  METAL_FUNC void iterate_j_block() {
    if (pass_id == upper) {
      j_block -= 1;
    } else {
      j_block += 1;
    }
  }
  
  METAL_FUNC void encode(device float *object, float2 value) const {
    uint2 bitpattern = as_type<uint2>(value);
    bitpattern = (bitpattern & 0xFFFFFFFE) | 1;
    atomic_store_explicit((device atomic_uint*)object, bitpattern[0], memory_order_relaxed);
    atomic_store_explicit((device atomic_uint*)object + 1, bitpattern[1], memory_order_relaxed);
  }
  
  METAL_FUNC bool decode(device float *object, thread float2 *value) {
    *value = as_type<float2>(uint2(
      atomic_load_explicit((device atomic_uint*)object, memory_order_relaxed),
      atomic_load_explicit((device atomic_uint*)object + 1, memory_order_relaxed)
    ));
    return all(bool2(as_type<uint2>(*value) & 1));
  }
};

// MARK: - Kernels

template <typename T>
void _generate_block_mask_impl(threadgroup T *threadgroup_block [[threadgroup(0)]],
                               device T *mask [[buffer(21), function_constant(masked)]],
                               device uchar *block_mask [[buffer(13), function_constant(block_sparse_masked)]],
                               
                               uint3 gid [[threadgroup_position_in_grid]],
                               ushort sidx [[simdgroup_index_in_threadgroup]],
                               ushort lane_id [[thread_index_in_simdgroup]])
{
  auto results_block = (threadgroup ushort2*)threadgroup_block;
  uint i = gid.x * R_group + sidx * R_simd;
  if (R % R_group > 0) {
    if (i >= R) {
      results_block[sidx] = ushort2(1, 1);
      return;
    }
  }
  
  ushort2 offset_in_simd = simdgroup_matrix_storage<T>::offset(lane_id);
  ushort R_edge = (R % R_simd == 0) ? R_simd : (R % R_simd);
  ushort R_floor = R_edge / 8 * 8;
  ushort R_modulo = R_edge - R_floor;
  bool i_in_bounds = i + R_simd <= R;
  
  uint j_block = gid.y;
  uint j_next = j_block * C_simd + C_simd;
  simdgroup_matrix_storage<T> attention_matrix[128];
  
  uint2 mask_offset(j_block * C_simd, gid.x * R_group + sidx * R_simd);
  mask_offset += uint2(offset_in_simd);
  auto mask_src = simdgroup_matrix_storage<T>::apply_offset(mask, C, mask_offset);
  
  // Apply explicit mask.
#pragma clang loop unroll(full)
  for (ushort r = 0; r < R_floor; r += 8) {
    apply_mask<T, T, vec<T, 2>>(r, j_next, offset_in_simd, mask_src, attention_matrix);
  }
  
  if (R_edge > 0) {
    if (i_in_bounds) {
#pragma clang loop unroll(full)
      for (ushort r = R_floor; r < R_simd; r += 8) {
        apply_mask<T, T, vec<T, 2>>(r, j_next, offset_in_simd, mask_src, attention_matrix);
      }
    } else {
      if (offset_in_simd.y < R_modulo) {
        apply_mask<T, T, vec<T, 2>>(R_floor, j_next, offset_in_simd, mask_src, attention_matrix);
      }
      T placeholder = attention_matrix[0].thread_elements()[0][0];
      placeholder = simd_broadcast_first(placeholder);
      
      if (offset_in_simd.y >= R_modulo) {
#pragma clang loop unroll(full)
        for (ushort c = 0; c < C_simd; c += 8) {
          ushort2 origin(c, R_floor);
          auto s = get_sram(attention_matrix, C_simd, origin);
          *s = simdgroup_matrix_storage<T>(placeholder);
        }
      }
#pragma clang loop unroll(full)
      for (ushort r = R_floor + 8; r < R_simd; r += 8) {
#pragma clang loop unroll(full)
        for (ushort c = 0; c < C_simd; c += 8) {
          ushort2 origin(c, r);
          auto s = get_sram(attention_matrix, C_simd, origin);
          *s = simdgroup_matrix_storage<T>(placeholder);
        }
      }
    }
  }
  
  // Apply edge mask.
  if ((C % C_simd > 0) && (j_next > C)) {
    ushort C_modulo = C % C_simd;
    ushort C_floor = C_modulo - (C_modulo % 8);
    ushort C_edge_thread = C_modulo - C_floor;
    ushort c = C_floor;
    T placeholder = attention_matrix[0].thread_elements()[0][0];
    placeholder = simd_broadcast_first(placeholder);
    
    if (offset_in_simd.x >= C_edge_thread) {
#pragma clang loop unroll(full)
      for (ushort r = 0; r < R_simd; r += 8) {
        auto s = get_sram(attention_matrix, C_simd, ushort2(c, r));
        *s = simdgroup_matrix_storage<T>(placeholder);
      }
    }
    
#pragma clang loop unroll(full)
    for (c += 8; c < C_simd; c += 8) {
#pragma clang loop unroll(full)
      for (ushort r = 0; r < R_simd; r += 8) {
        auto s = get_sram(attention_matrix, C_simd, ushort2(c, r));
        *s = simdgroup_matrix_storage<T>(placeholder);
      }
    }
  }
  
  T minimum = 0;
  T maximum = -numeric_limits<T>::max();
#pragma clang loop unroll(full)
  for (ushort r = 0; r < R_simd; r += 8) {
#pragma clang loop unroll(full)
    for (ushort c = 0; c < C_simd; c += 8) {
      auto s = get_sram(attention_matrix, C_simd, ushort2(c, r));
      vec<T, 2> elements = s->thread_elements()[0];
      minimum = min3(elements[0], elements[1], minimum);
      maximum = max3(elements[0], elements[1], maximum);
    }
  }
  ushort all_zero = false;
  ushort all_masked = false;
  if (minimum == maximum) {
    if (minimum == 0) {
      all_zero = true;
    } else if (minimum == -numeric_limits<T>::max()) {
      all_masked = true;
    }
  }
  all_zero = simd_ballot(all_zero).all();
  all_masked = simd_ballot(all_masked).all();
  results_block[sidx] = ushort2(all_zero, all_masked);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  if (sidx == 0) {
    all_zero = true;
    all_masked = true;
    if (lane_id < R_splits) {
      ushort2 result = results_block[lane_id];
      all_zero = result[0];
      all_masked = result[1];
    }
    all_zero = simd_ballot(all_zero).all();
    all_masked = simd_ballot(all_masked).all();
    
    ushort block_mask_element;
    if (all_masked) {
      block_mask_element = 0;
    } else if (all_zero) {
      block_mask_element = 1;
    } else {
      block_mask_element = 2;
    }
    uint block_mask_leading_dim = (C + C_simd - 1) / C_simd;
    block_mask[gid.x * block_mask_leading_dim + gid.y] = block_mask_element;
  }
}

template <typename T, typename U, typename TtoU>
void _attention_impl(device T *Q [[buffer(0)]],
                     device T *K [[buffer(1)]],
                     device T *V [[buffer(2)]],
                     device T *O [[buffer(3)]],
                     
                     threadgroup T *threadgroup_block [[threadgroup(0)]],
                     constant uint4 *query_offsets [[buffer(11), function_constant(grouped_query)]],
                     device T *mask [[buffer(12), function_constant(masked)]],
                     device uchar *block_mask [[buffer(13), function_constant(block_sparse_masked)]],
                     device atomic_uint *locks [[buffer(14), function_constant(triangular)]],
                     device float *partials [[buffer(15), function_constant(triangular)]],
                     
                     triangular_pass pass,
                     uint3 gid [[threadgroup_position_in_grid]],
                     ushort sidx [[simdgroup_index_in_threadgroup]],
                     ushort lane_id [[thread_index_in_simdgroup]])
{
  uint i = pass.i_block * R_group + sidx * R_simd;
  if ((R % R_group > 0) && (pass.pass_id == triangular_pass::none)) {
    if (i >= R) {
      return;
    }
  }
  
  ushort2 offset_in_simd = simdgroup_matrix_storage<T>::offset(lane_id);
  uint4 head_offsets = gid.y;
  if (grouped_query) {
    head_offsets = query_offsets[gid.y];
  }
  ushort R_edge = (R % R_simd == 0) ? R_simd : (R % R_simd);
  ushort R_floor = R_edge / 8 * 8;
  ushort R_modulo = R_edge - R_floor;
  bool i_in_bounds = i + R_simd <= R;
  
  simdgroup_matrix_storage<T> Q_sram[128];
  simdgroup_matrix_storage<float> O_sram[128];
  float l_sram[128];
  float m_sram[128];
  
  // Load Q block.
  if (sidx == 0) {
    uint2 Q_offset(0, i);
    ushort2 src_tile(D, min(uint(R_group), R - Q_offset.y));
    ushort2 dst_tile(D_simd, src_tile.y);
    
    auto Q_src = simdgroup_matrix_storage<T>::apply_offset(Q, Q_leading_dim, Q_offset, Q_trans);
    Q_src += head_offsets[0] * (Q_trans ? D * R : D);
    Q_src = apply_batch_offset(Q_src, gid.z * R * H * D);
    
    simdgroup_event event;
    event.async_copy(threadgroup_block, Q_block_leading_dim, dst_tile, Q_src, Q_leading_dim, src_tile, Q_trans);
    simdgroup_event::wait(1, &event);
  }
  
  // Initialize O, l, m.
#pragma clang loop unroll(full)
  for (ushort r = 0; r < R_simd; r += 8) {
#pragma clang loop unroll(full)
    for (ushort d = 0; d < D_simd; d += 8) {
      auto o = get_sram(O_sram, D_simd, ushort2(d, r));
      *o = simdgroup_matrix_storage<float>(0);
    }
    l_sram[r / 8] = numeric_limits<float>::denorm_min();
    m_sram[r / 8] = -numeric_limits<float>::max();
  }
  auto Q_offset = ushort2(0, sidx * R_simd) + offset_in_simd;
  auto Q_block = simdgroup_matrix_storage<T>::apply_offset(threadgroup_block, Q_block_leading_dim, Q_offset, Q_trans);
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
  for (ushort r = 0; r < R_simd; r += 8) {
#pragma clang loop unroll(full)
    for (ushort d = 0; d < D_simd; d += 8) {
      ushort2 origin(d, r);
      auto q = get_sram(Q_sram, D_simd, origin);
      q->load(Q_block, Q_block_leading_dim, origin, Q_trans);
    }
  }
  
  for (; pass.continue_j_block(); pass.iterate_j_block()) {
    uint j_block = pass.j_block;
    ushort flags = masked ? 2 : 1;
    if (block_sparse_masked) {
      uint j_block_stride = (C + C_simd - 1) / C_simd;
      flags = block_mask[pass.i_block * j_block_stride + j_block];
      if (flags == 0) {
        continue;
      }
    }
    
    // Load K block.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      uint2 K_offset(j_block * C_simd, 0);
      ushort2 K_src_tile(min(uint(C_simd), C - K_offset.x), D);
      ushort2 K_dst_tile(K_src_tile.x, D);
      
      auto K_src = simdgroup_matrix_storage<T>::apply_offset(K, K_leading_dim, K_offset, K_trans);
      K_src += head_offsets[1] * (K_trans ? D : D * C);
      K_src = apply_batch_offset(K_src, gid.z * C * H * D);
      
      if (_fuse_async_loads) {
        uint2 V_offset(0, j_block * C_simd);
        uint C_ceil = (C + 7) / 8 * 8;
        ushort2 V_src_tile(D, min(uint(C_simd), C - V_offset.y));
        ushort2 V_dst_tile(D, (C >= 2560) ? C_simd : min(uint(C_simd), C_ceil - V_offset.y));
        
        auto V_src = simdgroup_matrix_storage<T>::apply_offset(V, V_leading_dim, V_offset, V_trans);
        V_src += head_offsets[2] * (V_trans ? D * C : D);
        V_src = apply_batch_offset(V_src, gid.z * C * H * D);
        
        simdgroup_event events[2];
        events[0].async_copy(threadgroup_block, K_block_leading_dim, K_dst_tile, K_src, K_leading_dim, K_src_tile, K_trans);
        events[1].async_copy(threadgroup_block + V_block_offset, V_block_leading_dim, V_dst_tile, V_src, V_leading_dim, V_src_tile, V_trans);
        simdgroup_event::wait(2, events);
      } else {
        simdgroup_event event;
        event.async_copy(threadgroup_block, K_block_leading_dim, K_dst_tile, K_src, K_leading_dim, K_src_tile, K_trans);
        simdgroup_event::wait(1, &event);
      }
    }
    auto K_block = simdgroup_matrix_storage<T>::apply_offset(threadgroup_block, K_block_leading_dim, offset_in_simd, K_trans);
    
    // Multiply Q * K.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix_storage<U> attention_matrix[128];
#pragma clang loop unroll(full)
    for (ushort d = 0; d < D_simd; d += 8) {
      simdgroup_matrix_storage<T> K_sram[128];
      if ((D != D_simd) && (d + 8 == D_simd) && (offset_in_simd.y >= D % 8)) {
#pragma clang loop unroll(full)
        for (ushort c = 0; c < C_simd; c += 8) {
          auto k = get_sram(K_sram, C_simd, ushort2(c, 0));
          *k = simdgroup_matrix_storage<T>(0);
        }
      } else {
#pragma clang loop unroll(full)
        for (ushort c = 0; c < C_simd; c += 8) {
          auto k = get_sram(K_sram, C_simd, ushort2(c, 0));
          k->load(K_block, K_block_leading_dim, ushort2(c, d), K_trans);
        }
      }
      gemm(R_simd, C_simd, 8, d > 0,
           D_simd, Q_sram + d / 8,
           C_simd, K_sram,
           C_simd, attention_matrix);
    }
    
    // Apply explicit mask.
    uint j_next = j_block * C_simd + C_simd;
    if (masked && flags == 2) {
      uint2 mask_offset(j_block * C_simd, pass.i_block * R_group + sidx * R_simd);
      mask_offset += uint2(offset_in_simd);
      auto mask_src = simdgroup_matrix_storage<T>::apply_offset(mask, C, mask_offset);
      
#pragma clang loop unroll(full)
      for (ushort r = 0; r < R_floor; r += 8) {
        apply_mask<T, U, TtoU>(r, j_next, offset_in_simd, mask_src, attention_matrix);
      }
      
      if (R_edge > 0) {
        if (i_in_bounds) {
#pragma clang loop unroll(full)
          for (ushort r = R_floor; r < R_simd; r += 8) {
            apply_mask<T, U, TtoU>(r, j_next, offset_in_simd, mask_src, attention_matrix);
          }
        } else {
          if (offset_in_simd.y < R_modulo) {
            apply_mask<T, U, TtoU>(R_floor, j_next, offset_in_simd, mask_src, attention_matrix);
          }
        }
      }
    }
    
    // Apply edge mask.
    if ((C % C_simd > 0) && (j_next > C)) {
      ushort C_modulo = C % C_simd;
      ushort C_floor = C_modulo - (C_modulo % 8);
      ushort C_edge_thread = C_modulo - C_floor;
      ushort c = C_floor;
      
#pragma clang loop unroll(full)
      for (ushort index = 0; index < 2; index += 1) {
        if (offset_in_simd.x + index >= C_edge_thread) {
#pragma clang loop unroll(full)
          for (ushort r = 0; r < R_simd; r += 8) {
            auto s = get_sram(attention_matrix, C_simd, ushort2(c, r));
            (s->thread_elements()[0])[index] = -numeric_limits<U>::max();
          }
        }
      }
      
#pragma clang loop unroll(full)
      for (c += 8; c < C_simd; c += 8) {
#pragma clang loop unroll(full)
        for (ushort r = 0; r < R_simd; r += 8) {
          auto s = get_sram(attention_matrix, C_simd, ushort2(c, r));
          *(s->thread_elements()) = -numeric_limits<U>::max();
        }
      }
    }
    
    // Compute softmax.
#pragma clang loop unroll(full)
    for (ushort r = 0; r < R_simd; r += 8) {
      float2 _m;
#pragma clang loop unroll(full)
      for (ushort c = 0; c < C_simd; c += 8) {
        auto s = get_sram(attention_matrix, C_simd, ushort2(c, r));
        if (c == 0) {
          _m = float2(*(s->thread_elements()));
        } else {
          _m = max(_m, float2(*(s->thread_elements())));
        }
      }
      float m = max(_m[0], _m[1]);
      m = max(m, simd_shuffle_xor(m, 1));
      m = max(m, simd_shuffle_xor(m, 8));
      
      constexpr T threshold = -numeric_limits<T>::max() / 2;
      if (masked && m <= threshold) {
        for (ushort c = 0; c < C_simd; c += 8) {
          auto s = get_sram(attention_matrix, C_simd, ushort2(c, r));
          *(s->thread_elements()) = vec<U, 2>(0);
        }
      } else {
        m *= alpha;
        m = max(m, m_sram[r / 8]);
        
        float correction = exp2(M_LOG2E_F * (m_sram[r / 8] - m));
        if (m > m_sram[r / 8]) {
#pragma clang loop unroll(full)
          for (ushort d = 0; d < D_simd; d += 8) {
            auto o = get_sram(O_sram, D_simd, ushort2(d, r));
            *(o->thread_elements()) *= correction;
          }
        }
        m_sram[r / 8] = m;
        float subtrahend = m * M_LOG2E_F;
        
        float2 _l = 0;
#pragma clang loop unroll(full)
        for (ushort c = 0; c < C_simd; c += 8) {
          auto s = get_sram(attention_matrix, C_simd, ushort2(c, r));
          float2 p = float2(*s->thread_elements());
          p = exp2(fma(p, alpha * M_LOG2E_F, -subtrahend));
          *(s->thread_elements()) = vec<U, 2>(p);
          _l += float2(*(s->thread_elements()));
        }
        float l = _l[0] + _l[1];
        l += simd_shuffle_xor(l, 1);
        l += simd_shuffle_xor(l, 8);
        l_sram[r / 8] = fma(l_sram[r / 8], correction, l);
      }
    }
    
    // Load V block.
    if (!_fuse_async_loads) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 V_offset(0, j_block * C_simd);
        uint C_ceil = (C + 7) / 8 * 8;
        ushort2 src_tile(D, min(uint(C_simd), C - V_offset.y));
        ushort2 dst_tile(D, (C >= 2560) ? C_simd : min(uint(C_simd), C_ceil - V_offset.y));
        
        auto V_src = simdgroup_matrix_storage<T>::apply_offset(V, V_leading_dim, V_offset, V_trans);
        V_src += head_offsets[2] * (V_trans ? D * C : D);
        V_src = apply_batch_offset(V_src, gid.z * C * H * D);
        
        simdgroup_event event;
        event.async_copy(threadgroup_block, V_block_leading_dim, dst_tile, V_src, V_leading_dim, src_tile, V_trans);
        simdgroup_event::wait(1, &event);
      }
    }
    auto V_block = simdgroup_matrix_storage<T>::apply_offset(threadgroup_block, V_block_leading_dim, offset_in_simd, V_trans);
    
    // Multiply P * V.
    if (_fuse_async_loads) {
      V_block += V_block_offset;
    } else {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    ushort C_edge = (C % C_simd == 0) ? C_simd : (C % C_simd);
    ushort C_ceil = (C >= 2560) ? C_simd : (C_edge + 7) / 8 * 8;
#pragma clang loop unroll(full)
    for (ushort c = 0; c < C_ceil; c += 8) {
      simdgroup_matrix_storage<T> V_sram[128];
#pragma clang loop unroll(full)
      for (ushort d = 0; d < D_simd; d += 8) {
        auto v = get_sram(V_sram, 8, ushort2(d, 0));
        v->load(V_block, V_block_leading_dim, ushort2(d, c), V_trans);
      }
      gemm(R_simd, D_simd, 8, true,
           C_simd, attention_matrix + c / 8,
           D_simd, V_sram,
           D_simd, O_sram);
    }
    
    if ((C % C_simd > 0) && (j_next < C)) {
#pragma clang loop unroll(full)
      for (ushort c = C_ceil; c < C_simd; c += 8) {
        simdgroup_matrix_storage<T> V_sram[128];
#pragma clang loop unroll(full)
        for (ushort d = 0; d < D_simd; d += 8) {
          auto v = get_sram(V_sram, 8, ushort2(d, 0));
          v->load(V_block, V_block_leading_dim, ushort2(d, c), V_trans);
        }
        gemm(R_simd, D_simd, 8, true,
             C_simd, attention_matrix + c / 8,
             D_simd, V_sram,
             D_simd, O_sram);
      }
    }
  }
  
  // Combine partial sums.
  if (pass.pass_id == triangular_pass::lower) {
    ushort num_lm_elements = (2 * R_group + 31) / 32 * 32;
    ushort num_O_elements = (R_group * D_simd + 31) / 32 * 32;
    uint head_O_blocks = (R + R_group - 1) / R_group;
    uint O_block_index = gid.y * head_O_blocks + pass.i_block;
    O_block_index += gid.z * H * head_O_blocks;
    
    ulong address = O_block_index * (num_lm_elements + num_O_elements);
    auto partial = partials + address;
    ushort r_base = offset_in_simd.x + sidx * R_simd;
    ushort base = num_lm_elements + offset_in_simd.x;
    base += (sidx * R_simd + offset_in_simd.y) * D_simd;
    
    if (!pass.can_store_o) {
      if (offset_in_simd.x == 0) {
#pragma clang loop unroll(full)
        for (ushort r = 0; r < R_simd; r += 8) {
          auto object = partial + 2 * (r + r_base);
          pass.encode(object, { l_sram[r / 8], m_sram[r / 8] });
        }
      }
      partial += base;
      
#pragma clang loop unroll(full)
      for (ushort r = 0; r < R_simd; r += 8) {
#pragma clang loop unroll(full)
        for (ushort d = 0; d < D_simd; d += 8) {
          auto object = partial + (r * D_simd + d);
          auto o = get_sram(O_sram, D_simd, ushort2(d, r));
          pass.encode(object, *o->thread_elements());
        }
      }
      threadgroup_barrier(mem_flags::mem_device);
      
      if (sidx == 0 && lane_id == 0) {
        atomic_store_explicit(locks + O_block_index, 1, memory_order_relaxed);
      }
    } else {
      if (sidx == 0 && lane_id == 0) {
        fault_counter counter(1000);
        bool succeeded = false;
        while (!succeeded) {
          if (counter.quit()) {
            return;
          }
          
          uint expected = 1;
          succeeded = atomic_compare_exchange_weak_explicit(locks + O_block_index, &expected, 0, memory_order_relaxed, memory_order_relaxed);
        }
        ((device uint*)locks)[O_block_index] = 0;
      }
      threadgroup_barrier(mem_flags::mem_device);
      
      float2 lm[128];
      fault_counter counter(100);
      bool failed = true;
      while (failed) {
        if (counter.quit()) {
          return;
        }
        failed = false;
        
#pragma clang loop unroll(full)
        for (ushort r = 0; r < R_simd; r += 8) {
          auto object = partial + 2 * (r + r_base);
          if (!pass.decode(object, lm + r / 8)) {
            failed = true;
          }
        }
      }
      
      float correction_send[128];
#pragma clang loop unroll(full)
      for (ushort r = 0; r < R_simd; r += 8) {
        float l = lm[r / 8][0];
        float m = lm[r / 8][1];
        m = max(m_sram[r / 8], m);
        float correction_recv = exp2(M_LOG2E_F * (m_sram[r / 8] - m));
        ((device float2*)partial)[r + r_base] = 0;
        
        constexpr T threshold = -numeric_limits<T>::max() / 2;
        if (m <= threshold) {
          correction_send[r / 8] = 0;
        } else if (m > m_sram[r / 8]) {
          correction_send[r / 8] = 1;
#pragma clang loop unroll(full)
          for (ushort d = 0; d < D_simd; d += 8) {
            auto o = get_sram(O_sram, D_simd, ushort2(d, r));
            *(o->thread_elements()) *= correction_recv;
          }
          l_sram[r / 8] *= correction_recv;
        } else {
          correction_send[r / 8] = 1 / correction_recv;
        }
        l_sram[r / 8] = fma(l, correction_send[r / 8], l_sram[r / 8]);
      }
      partial += base;
      
#pragma clang loop unroll(full)
      for (ushort r = 0; r < R_simd; r += 8) {
#pragma clang loop unroll(full)
        for (ushort d = 0; d < C_simd; d += 8) {
          bool failed = true;
          
          fault_counter counter(10);
          while (failed) {
            if (counter.quit()) {
              return;
            }
            
            auto object = partial + (r * D_simd + d);
            float2 o_send;
            if (pass.decode(object, &o_send)) {
              *((device float2*)object) = 0;
              failed = false;
              
              auto o = get_sram(O_sram, D_simd, ushort2(d, r));
              auto o_elements = o->thread_elements();
              auto correction = correction_send[r / 8];
              *o_elements = fma(o_send, correction, *o_elements);
            }
          }
        }
      }
    }
  }
  
  // Write O block.
  if (pass.can_store_o) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto O_offset = ushort2(0, sidx * R_simd) + offset_in_simd;
    auto O_block = simdgroup_matrix_storage<T>::apply_offset(threadgroup_block, O_block_leading_dim, O_offset, O_trans);
    
#pragma clang loop unroll(full)
    for (ushort r = 0; r < R_simd; r += 8) {
      float l = 1 / l_sram[r / 8];
#pragma clang loop unroll(full)
      for (ushort d = 0; d < D_simd; d += 8) {
        ushort2 origin(d, r);
        auto o_elements = get_sram(O_sram, D_simd, origin)->thread_elements();
        simdgroup_matrix_storage<T> o(vec<T, 2>(*o_elements * l));
        o.store(O_block, O_block_leading_dim, origin, O_trans);
      }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      uint2 O_offset(0, pass.i_block * R_group);
      ushort2 tile(D, min(uint(R_group), R - O_offset.y));
      
      auto O_dst = simdgroup_matrix_storage<T>::apply_offset(O, O_leading_dim, O_offset, O_trans);
      O_dst += head_offsets[3] * (O_trans ? D * R : D);
      O_dst = apply_batch_offset(O_dst, gid.z * R * H * D);
      
      simdgroup_event event;
      event.async_copy(O_dst, O_leading_dim, tile, threadgroup_block, O_block_leading_dim, tile, O_trans);
    }
  }
}

template <typename T, typename U, typename TtoU>
void _triangular_impl(device T *Q [[buffer(0)]],
                      device T *K [[buffer(1)]],
                      device T *V [[buffer(2)]],
                      device T *O [[buffer(3)]],
                      
                      threadgroup T *threadgroup_block [[threadgroup(0)]],
                      constant uint4 *query_offsets [[buffer(11), function_constant(grouped_query)]],
                      device T *mask [[buffer(12), function_constant(masked)]],
                      device uchar *block_mask [[buffer(13), function_constant(block_sparse_masked)]],
                      device atomic_uint *locks [[buffer(14), function_constant(triangular)]],
                      device float *partials [[buffer(15), function_constant(triangular)]],
                      
                      uint3 gid [[threadgroup_position_in_grid]],
                      ushort sidx [[simdgroup_index_in_threadgroup]],
                      ushort lane_id [[thread_index_in_simdgroup]])
{
  triangular_pass lower_pass(triangular_pass::lower, gid.x);
  uint i = lower_pass.i_block * R_group + sidx * R_simd;
  if (R % R_group > 0) {
    if (i >= R) {
      return;
    }
  }
  
  _attention_impl<T, U, TtoU>(Q, K, V, O, threadgroup_block, query_offsets, mask, block_mask, locks, partials, lower_pass, gid, sidx, lane_id);
  
  triangular_pass upper_pass(triangular_pass::upper, gid.x);
  if (upper_pass.can_store_o) {
    _attention_impl<T, U, TtoU>(Q, K, V, O, threadgroup_block, query_offsets, mask, block_mask, locks, partials, upper_pass, gid, sidx, lane_id);
  }
}

kernel void attention(device void *Q [[buffer(0)]],
                      device void *K [[buffer(1)]],
                      device void *V [[buffer(2)]],
                      device void *O [[buffer(3)]],
                      
                      threadgroup void *threadgroup_block [[threadgroup(0)]],
                      constant ulong4 *matrix_offsets [[buffer(10), function_constant(batched)]],
                      constant uint4 *query_offsets [[buffer(11), function_constant(grouped_query)]],
                      device void *mask [[buffer(12), function_constant(masked)]],
                      device uchar *block_mask [[buffer(13), function_constant(block_sparse_masked)]],
                      device atomic_uint *locks [[buffer(14), function_constant(triangular)]],
                      device float *partials [[buffer(15), function_constant(triangular)]],
                      
                      uint3 gid [[threadgroup_position_in_grid]],
                      ushort sidx [[simdgroup_index_in_threadgroup]],
                      ushort lane_id [[thread_index_in_simdgroup]])
{
  if (batched) {
    if (masked) {
      mask = ((device uchar*)mask) + matrix_offsets[gid.z][0];
    }
    if (block_sparse_masked) {
      block_mask += matrix_offsets[gid.z][1];
    }
  }
  
  if (generate_block_mask) {
    if (Q_data_type == MTLDataTypeFloat) {
      _generate_block_mask_impl((threadgroup float*)threadgroup_block, (device float*)mask, block_mask, gid, sidx, lane_id);
    } else if (Q_data_type == MTLDataTypeHalf) {
      _generate_block_mask_impl((threadgroup half*)threadgroup_block, (device half*)mask, block_mask, gid, sidx, lane_id);
    }
  } else if (forward) {
    triangular_pass pass(triangular_pass::none, gid.x);
    if (Q_data_type == MTLDataTypeFloat) {
      if (triangular) {
        _triangular_impl<float, float, vec<float, 2>>((device float*)Q, (device float*)K, (device float*)V, (device float*)O, (threadgroup float*)threadgroup_block, query_offsets, (device float*)mask, block_mask, locks, partials, gid, sidx, lane_id);
      } else {
        _attention_impl<float, float, vec<float, 2>>((device float*)Q, (device float*)K, (device float*)V, (device float*)O, (threadgroup float*)threadgroup_block, query_offsets, (device float*)mask, block_mask, locks, partials, pass, gid, sidx, lane_id);
      }
    } else if (Q_data_type == MTLDataTypeHalf) {
      if (float_accumulator) {
        if (triangular) {
          _triangular_impl<half, float, float2>((device half*)Q, (device half*)K, (device half*)V, (device half*)O, (threadgroup half*)threadgroup_block, query_offsets, (device half*)mask, block_mask, locks, partials, gid, sidx, lane_id);
        } else {
          _attention_impl<half, float, float2>((device half*)Q, (device half*)K, (device half*)V, (device half*)O, (threadgroup half*)threadgroup_block, query_offsets, (device half*)mask, block_mask, locks, partials, pass, gid, sidx, lane_id);
        }
      } else {
        if (triangular) {
          _triangular_impl<half, half, vec<half, 2>>((device half*)Q, (device half*)K, (device half*)V, (device half*)O, (threadgroup half*)threadgroup_block, query_offsets, (device half*)mask, block_mask, locks, partials, gid, sidx, lane_id);
        } else {
          _attention_impl<half, half, vec<half, 2>>((device half*)Q, (device half*)K, (device half*)V, (device half*)O, (threadgroup half*)threadgroup_block, query_offsets, (device half*)mask, block_mask, locks, partials, pass, gid, sidx, lane_id);
        }
      }
    }
  }
}
