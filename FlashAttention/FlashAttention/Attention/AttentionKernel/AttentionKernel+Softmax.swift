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
    let loopEndFloor = headDimension - headDimension % 8
    
    var output = """

float2 D_term_accumulator(0);
{
  // Threads outside of the matrix along the row dimension, have their origin
  // shifted in-bounds.
  uint D_offset = morton_offset.x;
  uint R_offset = min(R, \(parallelizationThreadOffset));
  uint2 offset_src(D_offset, R_offset);
  
  // Where the dO and O data will be read from.
  auto dO_src = simdgroup_matrix_storage<float>::apply_offset(
    dO, \(leadingDimension("dO")), offset_src, \(transposed("dO")));
  auto O_src = simdgroup_matrix_storage<float>::apply_offset(
    O, \(leadingDimension("O")), offset_src, \(transposed("O")));
  
  // Going to use async copy to handle the matrix edge.
#pragma clang loop unroll(disable)
  for (ushort d = 0; d < \(loopEndFloor); d += 8) {
    ushort2 origin(d, 0);
    simdgroup_matrix_storage<float> dO;
    simdgroup_matrix_storage<float> O;
    dO.load(dO_src, \(leadingDimension("dO")), origin, \(transposed("dO")));
    O.load(O_src, \(leadingDimension("O")), origin, \(transposed("O")));
    
    // Perform the pointwise multiplication.
    float2 dO_value = *(dO.thread_elements());
    float2 O_value = *(O.thread_elements());
    D_term_accumulator += dO_value * O_value;
  }
}
"""
    
    if headDimension % 8 != 0 {
      output += """

{
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sidx == 0) {
    uint D_offset = \(loopEndFloor);
    uint R_offset = \(parallelizationOffset);
    uint2 offset_src(D_offset, R_offset);
    
    auto dO_src = simdgroup_matrix_storage<float>::apply_offset(
      dO, \(leadingDimension("dO")), offset_src, \(transposed("dO")));
    auto O_src = simdgroup_matrix_storage<float>::apply_offset(
      O, \(leadingDimension("O")), offset_src, \(transposed("O")));
    auto dO_dst = (threadgroup float*)(threadgroup_block);
    auto O_dst = (threadgroup float*)(threadgroup_block);
    O_dst += \(blockDimensions.parallelization * 8);
    
    ushort D_src_dimension = \(headDimension) % 8;
    ushort D_dst_dimension = 8;
    ushort R_dimension = min(
      uint(\(blockDimensions.parallelization)),
      uint(\(parallelizationDimension) - \(parallelizationOffset)));
    ushort2 tile_src(D_src_dimension, R_dimension);
    ushort2 tile_dst(D_dst_dimension, R_dimension);
    
    // Issue two async copies.
    simdgroup_event events[2];
    events[0].async_copy(
      dO_dst, \(leadingBlockDimension("dO")), tile_dst,
      dO_src, \(leadingDimension("dO")), tile_src, \(transposed("dO")));
    events[1].async_copy(
      O_dst, \(leadingBlockDimension("O")), tile_dst,
      O_src, \(leadingDimension("O")), tile_src, \(transposed("O")));
    simdgroup_event::wait(2, events);
  }

  // Where the dO and O data will be read from.
  ushort2 offset_src(morton_offset.x, morton_offset.y + sidx * 8);
  auto dO_block = (threadgroup float*)(threadgroup_block);
  auto O_block = (threadgroup float*)(threadgroup_block);
  O_block += \(blockDimensions.parallelization * 8);

  dO_block = simdgroup_matrix_storage<float>::apply_offset(
    dO_block, \(leadingBlockDimension("dO")), offset_src, \(transposed("dO")));
  O_block = simdgroup_matrix_storage<float>::apply_offset(
    O_block, \(leadingBlockDimension("O")), offset_src, \(transposed("O")));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  // Load the zero-padded edge data.
  ushort2 origin(0, 0);
  simdgroup_matrix_storage<float> dO;
  simdgroup_matrix_storage<float> O;
  dO.load(
    dO_block, \(leadingBlockDimension("dO")), origin, \(transposed("dO")));
  O.load(
    O_block, \(leadingBlockDimension("O")), origin, \(transposed("O")));
  
  // Perform the pointwise multiplication.
  float2 dO_value = *(dO.thread_elements());
  float2 O_value = *(O.thread_elements());
  D_term_accumulator += dO_value * O_value;
}
"""
    }
    
    output += """

float D_term = D_term_accumulator[0] + D_term_accumulator[1];
D_term += simd_shuffle_xor(D_term, 1);
D_term += simd_shuffle_xor(D_term, 8);
D_term *= \(backwardScale);

"""
    
    return output
  }
  
  // Load a term when parallelizing over columns.
  func loadTerm(name: String) -> String {
    guard parallelizationDimension == "C" else {
      fatalError("Not allowed to call this function.")
    }
    return """
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      // Locate the \(name)[i] in device and threadgroup memory.
      auto \(name)_terms_src = \(name)_terms + r;
      auto \(name)_terms_dst = (threadgroup float*)(threadgroup_block);
      
      ushort R_src_dimension = min(
        uint(\(blockDimensions.traversal)),
        uint(\(traversalDimension) - \(traversalOffset)));
      ushort R_dst_dimension = max(
        ushort(\(paddedTraversalBlockDimension)),
        ushort(R_src_dimension));
      
      // Issue an async copy.
      simdgroup_event event;
      event.async_copy(
        \(name)_terms_dst, 1, ushort2(R_dst_dimension, 1),
        \(name)_terms_src, 1, ushort2(R_src_dimension, 1));
      simdgroup_event::wait(1, &event);
    }
    
    // Where the \(name) data will be read from.
    auto \(name)_terms_block = (threadgroup float*)(threadgroup_block);
    \(name)_terms_block += morton_offset.x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    """
  }
}

// MARK: - Masking the Matrix Edge

extension AttentionKernel {
  // Prevent the zero padding from changing the values of 'm' and 'l'.
  func maskAttentionMatrixEdge() -> String {
    let blockDim = blockDimensions.traversal
    let remainder = "(\(traversalDimension) % \(blockDim))"
    let remainderFloor = "(\(remainder) - (\(remainder) % 8))";
    
    return """
    
    if ((\(remainder) != 0) &&
        (\(traversalOffset) + \(blockDim) > \(traversalDimension))) {
      // Prevent the value from becoming -INF during the FMA before the
      // exponentiation. If the multiplication during FMA returns -INF,
      // subtracting a positive 'm' value will turn it into zero. We don't want
      // that. exp(0) evaluates to 1.00 and corrupts the value of 'l'.
      const float mask_value =
      (0.875 / M_LOG2E_F) * -numeric_limits<float>::max();
      
#pragma clang loop unroll(full)
      for (ushort index = 0; index < 2; ++index) {
        if (morton_offset.x + index >= \(remainder) - \(remainderFloor)) {
          auto S_elements = S_sram[\(remainderFloor) / 8].thread_elements();
          (*S_elements)[index] = mask_value;
        }
      }
#pragma clang loop unroll(full)
      for (ushort c = \(remainderFloor) + 8; c < \(blockDim); c += 8) {
        auto S_elements = S_sram[c / 8].thread_elements();
        *S_elements = mask_value;
      }
    }
    
"""
  }
}

// MARK: - Softmax

extension AttentionKernel {
  fileprivate var forwardScale: Float {
    // M_LOG2E_F / sqrt(D)
    return 1.442695041 / Float(headDimension).squareRoot()
  }
  
  func onlineSoftmax() -> String {
    """

    // update 'm'
    float2 m_new_accumulator;
#pragma clang loop unroll(full)
    for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
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
    m_new *= \(forwardScale);
    
    // update 'm'
    float correction = 1;
    if (m_new > m) {
      correction = fast::exp2(m - m_new);
      m = m_new;
    }
    
    // P = softmax(S * scaleFactor)
    simdgroup_matrix_storage<float> P_sram[\(blockDimensions.traversal) / 8];
#pragma clang loop unroll(full)
    for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
      float2 S_elements = float2(*(S_sram[c / 8].thread_elements()));
      float2 P_elements = fast::exp2(S_elements * \(forwardScale) - m);
      *(P_sram[c / 8].thread_elements()) = P_elements;
    }
    
    // update 'l'
    float2 l_new_accumulator;
#pragma clang loop unroll(full)
    for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
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
    """

    simdgroup_matrix_storage<float> P_sram[\(blockDimensions.traversal) / 8];
#pragma clang loop unroll(full)
    for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
      float2 S_elements = float2(*(S_sram[c / 8].thread_elements()));
      float2 P_elements = fast::exp2(S_elements * \(forwardScale) - L_term);
      *(P_sram[c / 8].thread_elements()) = P_elements;
    }

"""
  }
  
  func checkpointSoftmaxT() -> String {
    """

    simdgroup_matrix_storage<float> PT_sram[\(blockDimensions.traversal) / 8];
    {
      \(loadTerm(name: "L"))
      
      // Compute the softmax.
      #pragma clang loop unroll(full)
      for (ushort r = 0; r < \(blockDimensions.traversal); r += 8) {
        ushort2 origin(r, 0);
        simdgroup_matrix_storage<float> L_terms;
        L_terms.load(L_terms_block, 1, origin, false);
        float2 L_term = *(L_terms.thread_elements());
        
        float2 ST_elements = float2(*(ST_sram[r / 8].thread_elements()));
        float2 PT_elements = fast::exp2(ST_elements * \(forwardScale) - L_term);
        *(PT_sram[r / 8].thread_elements()) = PT_elements;
      }
    }

    """
  }
}

// MARK: - Softmax Derivative

extension AttentionKernel {
  fileprivate var backwardScale: Float {
    1 / Float(headDimension).squareRoot()
  }
  
  func derivativeSoftmax() -> String {
    """

    simdgroup_matrix_storage<float> dS_sram[\(blockDimensions.traversal) / 8];
#pragma clang loop unroll(full)
    for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
      float2 P_elements = float2(*(P_sram[c / 8].thread_elements()));
      float2 dP_elements = float2(*(dP_sram[c / 8].thread_elements()));
      float2 dS_elements = dP_elements * \(backwardScale) - D_term;
      dS_elements *= P_elements;
      *(dS_sram[c / 8].thread_elements()) = dS_elements;
    }

"""
  }
  
  func derivativeSoftmaxT() -> String {
    """
    
    simdgroup_matrix_storage<float> dST_sram[\(blockDimensions.traversal) / 8];
    {
      \(loadTerm(name: "D"))
      
      // Compute the softmax derivative.
      #pragma clang loop unroll(full)
      for (ushort r = 0; r < \(blockDimensions.traversal); r += 8) {
        ushort2 origin(r, 0);
        simdgroup_matrix_storage<float> D_terms;
        D_terms.load(D_terms_block, 1, origin, false);
        float2 D_term = *(D_terms.thread_elements());
        
        float2 PT_elements = float2(*(PT_sram[r / 8].thread_elements()));
        float2 dPT_elements = float2(*(dPT_sram[r / 8].thread_elements()));
        float2 dST_elements = dPT_elements * \(backwardScale) - D_term;
        dST_elements *= PT_elements;
        *(dST_sram[r / 8].thread_elements()) = dST_elements;
      }
    }

    """
  }
}
