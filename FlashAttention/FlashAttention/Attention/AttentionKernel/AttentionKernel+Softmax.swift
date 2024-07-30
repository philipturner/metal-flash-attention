//
//  AttentionKernel+Softmax.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Elementwise operations on the attention matrix.

// MARK: - D[i] Computation

extension AttentionKernel {
  func computeDTerm() -> String {
    let leadingBlockDimensionO: UInt16 = transposeState.O ? 32 : 8
    
    return """

float2 D_term_accumulator(0);
{
  // Threads outside of the matrix along the row dimension, have their origin
  // shifted in-bounds.
  uint D_offset = morton_offset.x;
  uint R_offset = min(R, gid * 32 + sidx * 8 + morton_offset.y);
  uint2 offset_src(D_offset, R_offset);
  
  // Find where the dO and O data will be read from.
  auto dO_src = simdgroup_matrix_storage<float>::apply_offset(
    dO, \(leadingDimensions.O), offset_src, \(transposeState.O));
  auto O_src = simdgroup_matrix_storage<float>::apply_offset(
    O, \(leadingDimensions.O), offset_src, \(transposeState.O));
  
  // Going to use async copy to handle the matrix edge.
#pragma clang loop unroll(disable)
  for (ushort d = 0; d < \(headDimension) - (\(headDimension) % 8); d += 8) {
    ushort2 origin(d, 0);
    simdgroup_matrix_storage<float> dO;
    simdgroup_matrix_storage<float> O;
    dO.load(dO_src, \(leadingDimensions.O), origin, \(transposeState.O));
    O.load(O_src, \(leadingDimensions.O), origin, \(transposeState.O));
    
    // Perform the pointwise multiplication.
    float2 dO_value = *(dO.thread_elements());
    float2 O_value = *(O.thread_elements());
    D_term_accumulator += dO_value * O_value;
  }
}

if (\(headDimension) % 8 != 0) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sidx == 0) {
    uint D_offset = \(headDimension) - (\(headDimension) % 8);
    uint R_offset = gid * 32;
    uint2 offset_src(D_offset, R_offset);
    
    auto dO_src = simdgroup_matrix_storage<float>::apply_offset(
      dO, \(leadingDimensions.O), offset_src, \(transposeState.O));
    auto O_src = simdgroup_matrix_storage<float>::apply_offset(
      O, \(leadingDimensions.O), offset_src, \(transposeState.O));
    auto dO_dst = (threadgroup float*)(threadgroup_block);
    auto O_dst = (threadgroup float*)(threadgroup_block) + \(32 * 8);
    
    ushort D_src_dimension = \(headDimension) % 8;
    ushort D_dst_dimension = 8;
    ushort R_src_dimension = min(uint(32), R - gid * 32);
    ushort2 tile_src(D_src_dimension, R_src_dimension);
    ushort2 tile_dst(D_dst_dimension, R_src_dimension);
    
    // Issue two async copies.
    simdgroup_event events[2];
    events[0].async_copy(
      dO_dst, \(leadingBlockDimensionO), tile_dst,
      dO_src, \(leadingDimensions.O), tile_src, \(transposeState.O));
    events[1].async_copy(
      O_dst, \(leadingBlockDimensionO), tile_dst,
      O_src, \(leadingDimensions.O), tile_src, \(transposeState.O));
    simdgroup_event::wait(2, events);
  }

  // Find where the dO and O data will be read from.
  ushort2 offset_src(morton_offset.x, morton_offset.y + sidx * 8);
  auto dO_block = (threadgroup float*)(threadgroup_block);
  auto O_block = (threadgroup float*)(threadgroup_block) + \(32 * 8);
  dO_block = simdgroup_matrix_storage<float>::apply_offset(
    dO_block, \(leadingBlockDimensionO), offset_src, \(transposeState.O));
  O_block = simdgroup_matrix_storage<float>::apply_offset(
    O_block, \(leadingBlockDimensionO), offset_src, \(transposeState.O));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Load the zero-padded edge data.
  ushort2 origin(0, 0);
  simdgroup_matrix_storage<float> dO;
  simdgroup_matrix_storage<float> O;
  dO.load(dO_block, \(leadingBlockDimensionO), origin, \(transposeState.O));
  O.load(O_block, \(leadingBlockDimensionO), origin, \(transposeState.O));
  
  // Perform the pointwise multiplication.
  float2 dO_value = *(dO.thread_elements());
  float2 O_value = *(O.thread_elements());
  D_term_accumulator += dO_value * O_value;
}

float D_term = D_term_accumulator[0] + D_term_accumulator[1];
D_term += simd_shuffle_xor(D_term, 1);
D_term += simd_shuffle_xor(D_term, 8);
D_term *= 1 / sqrt(float(\(headDimension)));

"""
  }
}

// MARK: - Masking the Matrix Edge

extension AttentionKernel {
  // Prevent the zero padding from changing the values of 'm' and 'l'.
  func maskAlongColumns(sram: String) -> String {
    return """
    
    if ((C % 32 != 0) && (c + 32 > C)) {
      const ushort remainder32 = uint(C % 32);
      const ushort remainder32_floor = remainder32 - ushort(remainder32 % 8);
      
      // Prevent the value from becoming -INF during the FMA before the
      // exponentiation. If the multiplication during FMA returns -INF,
      // subtracting a positive 'm' value will turn it into zero. We don't want
      // that. exp(0) evaluates to 1.00 and corrupts the value of 'l'.
      const float mask_value =
      (0.875 / M_LOG2E_F) * -numeric_limits<float>::max();
      
#pragma clang loop unroll(full)
      for (ushort index = 0; index < 2; ++index) {
        if (morton_offset.x + index >= remainder32 - remainder32_floor) {
          auto S_elements = \(sram)[remainder32_floor / 8].thread_elements();
          (*S_elements)[index] = mask_value;
        }
      }
#pragma clang loop unroll(full)
      for (ushort c = remainder32_floor + 8; c < 32; c += 8) {
        auto S_elements = \(sram)[c / 8].thread_elements();
        *S_elements = mask_value;
      }
    }
    
"""
  }
}

// MARK: - Softmax

extension AttentionKernel {
  func onlineSoftmax() -> String {
    let scaleFactor = "(M_LOG2E_F / sqrt(float(\(headDimension))))"
    
    return """

    // update 'm'
    float2 m_new_accumulator;
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
      auto S_elements = S_sram[c / 8].thread_elements();
      if (c == 0) {
        m_new_accumulator = *S_elements;
      } else {
        m_new_accumulator = max(m_new_accumulator, *S_elements);
      }
    }
    float m_new = max(m_new_accumulator[0], m_new_accumulator[1]);
    m_new = max(m_new, simd_shuffle_xor(m_new, 1));
    m_new = max(m_new, simd_shuffle_xor(m_new, 8));
    m_new *= \(scaleFactor);
    
    // update 'm'
    float correction = 1;
    if (m_new > m) {
      correction = fast::exp2(m - m_new);
      m = m_new;
    }
    
    // P = softmax(S * scaleFactor)
    simdgroup_matrix_storage<float> P_sram[32 / 8];
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
      float2 S_elements = float2(*(S_sram[c / 8].thread_elements()));
      float2 P_elements = fast::exp2(S_elements * \(scaleFactor) - m);
      *(P_sram[c / 8].thread_elements()) = P_elements;
    }
    
    // update 'l'
    float2 l_new_accumulator;
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
      auto P_elements = P_sram[c / 8].thread_elements();
      if (c == 0) {
        l_new_accumulator = *P_elements;
      } else {
        l_new_accumulator += *P_elements;
      }
    }
    float l_new = l_new_accumulator[0] + l_new_accumulator[1];
    l_new += simd_shuffle_xor(l_new, 1);
    l_new += simd_shuffle_xor(l_new, 8);
    l = l * correction + l_new;

"""
  }
  
  func checkpointSoftmax() -> String {
    let scaleFactor = "(M_LOG2E_F / sqrt(float(\(headDimension))))"
    
    return """

    simdgroup_matrix_storage<float> P_sram[32 / 8];
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
      float2 S_elements = float2(*(S_sram[c / 8].thread_elements()));
      float2 P_elements = fast::exp2(S_elements * \(scaleFactor) - L_term);
      *(P_sram[c / 8].thread_elements()) = P_elements;
    }

"""
  }
  
  func checkpointSoftmaxT() -> String {
    let scaleFactor = "(M_LOG2E_F / sqrt(float(\(headDimension))))"
    
    return """

simdgroup_matrix_storage<float> PT_sram[32 / 8];
{
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sidx == 0) {
    // Locate the L[i] in device and threadgroup memory.
    auto L_terms_src = L_terms + r;
    auto L_terms_dst = (threadgroup float*)(threadgroup_block);
    
    // Declare the remainder of the row dimension.
    const ushort R_remainder = (R % 32 == 0) ? 32 : R % 32;
    const ushort R_remainder_padded = (R_remainder + 7) / 8 * 8;
    ushort R_src_dimension = min(uint(32), R - r);
    ushort R_dst_dimension = (r + 32 < R) ? ushort(32) : R_remainder_padded;
    
    // Issue an async copy.
    simdgroup_event event;
    event.async_copy(
      L_terms_dst, 1, ushort2(R_dst_dimension, 1),
      L_terms_src, 1, ushort2(R_src_dimension, 1));
    simdgroup_event::wait(1, &event);
  }
  
  // Find where the L data will be read from.
  auto L_terms_block = (threadgroup float*)(threadgroup_block);
  L_terms_block += morton_offset.x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Compute the softmax.
  #pragma clang loop unroll(full)
  for (ushort r = 0; r < 32; r += 8) {
    ushort2 origin(r, 0);
    simdgroup_matrix_storage<float> L_terms;
    L_terms.load(L_terms_block, 1, origin, false);
    float2 L_term = *(L_terms.thread_elements());
    
    float2 ST_elements = float2(*(ST_sram[r / 8].thread_elements()));
    float2 PT_elements = fast::exp2(ST_elements * \(scaleFactor) - L_term);
    *(PT_sram[r / 8].thread_elements()) = PT_elements;
  }
}

"""
  }
}

// MARK: - Softmax Derivative

extension AttentionKernel {
  func computeDerivativeSoftmax() -> String {
    let scaleFactor = "(1 / sqrt(float(\(headDimension))))"
    
    return """

    simdgroup_matrix_storage<float> dS_sram[32 / 8];
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
      float2 P_elements = float2(*(P_sram[c / 8].thread_elements()));
      float2 dP_elements = float2(*(dP_sram[c / 8].thread_elements()));
      float2 dS_elements = dP_elements * \(scaleFactor) - D_term;
      dS_elements *= P_elements;
      *(dS_sram[c / 8].thread_elements()) = dS_elements;
    }

"""
  }
  
  func computeDerivativeSoftmaxT() -> String {
    let scaleFactor = "(1 / sqrt(float(\(headDimension))))"
    
    return """

simdgroup_matrix_storage<float> dST_sram[32 / 8];
{
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sidx == 0) {
    // Locate the D[i] in device and threadgroup memory.
    auto D_terms_src = D_terms + r;
    auto D_terms_dst = (threadgroup float*)(threadgroup_block);
    
    // Declare the remainder of the row dimension.
    const ushort R_remainder = (R % 32 == 0) ? 32 : R % 32;
    const ushort R_remainder_padded = (R_remainder + 7) / 8 * 8;
    ushort R_src_dimension = min(uint(32), R - r);
    ushort R_dst_dimension = (r + 32 < R) ? ushort(32) : R_remainder_padded;
    
    // Issue an async copy.
    simdgroup_event event;
    event.async_copy(
      D_terms_dst, 1, ushort2(R_dst_dimension, 1),
      D_terms_src, 1, ushort2(R_src_dimension, 1));
    simdgroup_event::wait(1, &event);
  }
  
  // Find where the D data will be read from.
  auto D_terms_block = (threadgroup float*)(threadgroup_block);
  D_terms_block += morton_offset.x;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Compute the softmax derivative.
  #pragma clang loop unroll(full)
  for (ushort r = 0; r < 32; r += 8) {
    ushort2 origin(r, 0);
    simdgroup_matrix_storage<float> D_terms;
    D_terms.load(D_terms_block, 1, origin, false);
    float2 D_term = *(D_terms.thread_elements());
    
    float2 PT_elements = float2(*(PT_sram[r / 8].thread_elements()));
    float2 dPT_elements = float2(*(dPT_sram[r / 8].thread_elements()));
    float2 dST_elements = dPT_elements * \(scaleFactor) - D_term;
    dST_elements *= PT_elements;
    *(dST_sram[r / 8].thread_elements()) = dST_elements;
  }
}

"""
  }
}
