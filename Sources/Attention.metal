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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused"

// Dimensions of each matrix.
constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant uint H [[function_constant(2)]];
constant uint D [[function_constant(3)]];

// TODO: Matrix offsets are (Q/O/dQ/dO), (K/V/dK/dV), (mask), (block mask), (lm).
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
constant uint mask_data_type [[function_constant(24)]];

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
constant bool triangular [[function_constant(112)]];
constant bool store_lm = (gradient && forward) || triangular;
constant bool generate_block_mask = masked && block_sparse && !forward && !backward;
// TODO: Change l -> inv(l) after combining partial sums for triangular.

constant ushort R_simd [[function_constant(200)]];
constant ushort C_simd [[function_constant(201)]];

constant ushort R_modulo = (R % R_simd == 0) ? R_simd : (R % R_simd);
constant ushort C_modulo = (C % C_simd == 0) ? C_simd : (C % C_simd);
constant ushort R_padded = (R_modulo + 7) / 8 * 8;
constant ushort C_padded = (C_modulo + 7) / 8 * 8;

constant ushort R_splits [[function_constant(210)]];
constant ushort C_splits = 1;

constant ushort R_group = R_simd * R_splits;
constant ushort C_group = C_simd * C_splits;
constant uint block_mask_leading_dim = mask_trans ? (R + R_group - 1) / R_group : (C + C_group - 1) / C_group;

// TODO: To avoid zero-padding for D, create a single SIMD matrix in registers,
// which masks the tail elements. Apply it immediately after loading Q/K/etc
// from threadgroup.
constant ushort D_padded = (D + 7) / 8 * 8;

constant ushort Q_block_leading_dim = (Q_trans ? R_group : D);
constant ushort K_block_leading_dim = (K_trans ? D : C_group);
constant ushort V_block_leading_dim = (V_trans ? C_group : D);
constant ushort O_block_leading_dim = (O_trans ? R_group : D);
constant ushort mask_block_leading_dim = (mask_trans ? R_group : C_group);

#pragma clang diagnostic pop

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(thread simdgroup_matrix_storage<T> *sram, ushort sram_leading_dim, ushort2 matrix_origin) {
  return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_mask(thread simdgroup_matrix_storage<T> *sram, ushort2 matrix_origin) {
  return get_sram(sram, C_simd, matrix_origin);
}

template <typename T>
METAL_FUNC device T* apply_batch_offset(device T *pointer, constant vec<ulong, 8> *matrix_offsets, uint index) {
  if (batched) {
    auto casted = (device uchar*)pointer;
    casted += (*matrix_offsets)[index];
    return (device T*)casted;
  } else {
    return pointer;
  }
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

template <typename T>
void _generate_block_mask_impl(threadgroup T *threadgroup_block [[threadgroup(0)]],
                               constant vec<ulong, 8> *matrix_offsets [[buffer(10), function_constant(batched)]],
                               device T *mask [[buffer(11), function_constant(masked)]],
                               device uchar *block_mask [[buffer(12), function_constant(block_sparse)]],
                               
                               uint3 gid [[threadgroup_position_in_grid]],
                               ushort sidx [[simdgroup_index_in_threadgroup]],
                               ushort lane_id [[thread_index_in_simdgroup]])
{
  mask = apply_batch_offset(mask, matrix_offsets, 2);
  
  uint2 mask_offset(gid.x * C_group, gid.y * R_group + sidx * R_simd);
  if (sidx == 0) {
    ushort2 src_tile(min(uint(C_group), C - mask_offset.x),
                     min(uint(R_group), R - mask_offset.y));
    ushort2 dst_tile(~7 & (src_tile + 7));
    auto mask_src = simdgroup_matrix_storage<T>::apply_offset(mask, mask_leading_dim, mask_offset, mask_trans);
    
    simdgroup_event events[1];
    events[0].async_copy(threadgroup_block, mask_block_leading_dim, dst_tile, mask_src, mask_leading_dim, src_tile, mask_trans, simdgroup_async_copy_clamp_mode::clamp_to_edge);
    simdgroup_event::wait(1, events);
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
    
    load_block_mask(mask_sram, mask_block, 0, R_padded, 0, C_padded,
                    false);
    
    load_block_mask(mask_sram, mask_block, 0, R_padded, C_padded, C_simd,
                    mask_offset.x + C_simd > C);
    
    load_block_mask(mask_sram, mask_block, R_padded, R_simd, 0, C_padded,
                    mask_offset.y + R_simd > R);
    
    load_block_mask(mask_sram, mask_block, R_padded, R_simd, C_padded, C_simd,
                    (mask_offset.y + R_simd > R) || (mask_offset.x + C_simd > C));
    
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
    
    block_mask = apply_batch_offset(block_mask, matrix_offsets, 3);
    block_mask += gid.y * block_mask_leading_dim;
    block_mask += gid.x;
    block_mask[0] = block_mask_element;
  }
}


// TODO:
// Grid X = sequence dimension (R for forward, C for backward)
// Grid Y = heads dimension (H)
// Grid Z = batch dimension
// Threadgroup size = 128
template <typename T, typename dT>
void _attention_impl()
{
  // TODO: Do not copy the Q/K/V/O/dQ/dK/dV/dO pointer to match the batch
  // offset. Recompute that upon every async access inside SIMD 0.
  
  // For every element in threadgroup memory, there's 4-6 in registers.
//  simdgroup_matrix_storage<T> Q_sram[64][64];
//  simdgroup_matrix_storage<float> O_sram[64][64];
//  simdgroup_matrix_storage<float> dK_sram[64][64];
//  simdgroup_matrix_storage<float> dV_sram[64][64];
}

kernel void attention(threadgroup void *threadgroup_block [[threadgroup(0)]],
                      constant vec<ulong, 8> *matrix_offsets [[buffer(10), function_constant(batched)]],
                      device void *mask [[buffer(11), function_constant(masked)]],
                      device uchar *block_mask [[buffer(12), function_constant(block_sparse)]],
                      
                      uint3 gid [[threadgroup_position_in_grid]],
                      ushort sidx [[simdgroup_index_in_threadgroup]],
                      ushort lane_id [[thread_index_in_simdgroup]])
{
  matrix_offsets += gid.z;
  
  if (generate_block_mask) {
    if (mask_data_type == MTLDataTypeFloat) {
      _generate_block_mask_impl<float>((threadgroup float*)threadgroup_block, matrix_offsets, (device float*)mask, block_mask, gid, sidx, lane_id);
    } else if (mask_data_type == MTLDataTypeHalf) {
      _generate_block_mask_impl<half>((threadgroup half*)threadgroup_block, matrix_offsets, (device half*)mask, block_mask, gid, sidx, lane_id);
    }
  } else {
    if (Q_data_type == MTLDataTypeFloat) {
      _attention_impl<float, float>();
    } else {
      _attention_impl<half, ushort>();
    }
  }
}

