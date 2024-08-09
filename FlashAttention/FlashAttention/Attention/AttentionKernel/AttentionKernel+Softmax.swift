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
    func bulkContributions(truncatedHeadDimension: UInt16) -> String {
      // Recycle most of the cached values for dO.
      func declareDerivativeOLocation() -> String {
        if cached(.dO) {
          return ""
        } else {
          return """
          
          // Where the dO data will be read from.
          auto dO_src = simdgroup_matrix_storage<float>::apply_offset(
            dO, \(leadingDimension(.dO)), offset_src, \(transposed(.dO)));
          
          """
        }
      }
      func loadDerivativeO() -> String {
        if cached(.dO) {
          return """
          
          auto dO = dO_sram[d / 8];
          
          """
        } else {
          return """
          
          simdgroup_matrix_storage<float> dO;
          dO.load(
            dO_src, \(leadingDimension(.dO)),
            ushort2(d, 0), \(transposed(.dO)));
          
          """
        }
      }
      
      return """
      
      // Threads outside of the matrix along the row dimension,
      // have their origin shifted in-bounds.
      uint D_offset = morton_offset.x;
      uint R_offset = min(R, \(parallelizationThreadOffset));
      uint2 offset_src(D_offset, R_offset);
      
      \(declareDerivativeOLocation())
      
      // Where the O data will be read from.
      auto O_src = simdgroup_matrix_storage<float>::apply_offset(
        O, \(leadingDimension(.O)), offset_src, \(transposed(.O)));
      
      // Going to use async copy to handle the matrix edge.
      #pragma clang loop unroll(disable)
      for (ushort d = 0; d < \(truncatedHeadDimension); d += 8) {
        \(loadDerivativeO())
        
        simdgroup_matrix_storage<float> O;
        O.load(
          O_src, \(leadingDimension(.O)),
          ushort2(d, 0), \(transposed(.O)));
        
        // Perform the pointwise multiplication.
        float2 dO_value = *(dO.thread_elements());
        float2 O_value = *(O.thread_elements());
        D_term_accumulator += dO_value * O_value;
      }

      """
    }
    
    func edgeContributions(truncatedHeadDimension: UInt16) -> String {
      guard headDimension % 8 != 0 else {
        return ""
      }
      
      // Abbreviated block, only covers the last 8 elements.
      func leadingBlockDimension(_ operand: AttentionOperand) -> UInt16 {
        if transposed(operand) {
          return blockSequenceLength(operand)
        } else {
          return 8
        }
      }
      
      return """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint D_offset = \(truncatedHeadDimension);
        uint R_offset = \(parallelizationOffset);
        uint2 offset_src(D_offset, R_offset);
        
        auto dO_src = simdgroup_matrix_storage<float>::apply_offset(
          dO, \(leadingDimension(.dO)), offset_src, \(transposed(.dO)));
        auto O_src = simdgroup_matrix_storage<float>::apply_offset(
          O, \(leadingDimension(.O)), offset_src, \(transposed(.O)));
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
          dO_dst, \(leadingBlockDimension(.dO)), tile_dst,
          dO_src, \(leadingDimension(.dO)), tile_src, \(transposed(.dO)));
        events[1].async_copy(
          O_dst, \(leadingBlockDimension(.O)), tile_dst,
          O_src, \(leadingDimension(.O)), tile_src, \(transposed(.O)));
        simdgroup_event::wait(2, events);
      }
      
      // Where the dO and O data will be read from.
      ushort2 offset_src(morton_offset.x, morton_offset.y + sidx * 8);
      auto dO_block = (threadgroup float*)(threadgroup_block);
      auto O_block = (threadgroup float*)(threadgroup_block);
      O_block += \(blockDimensions.parallelization * 8);
      
      dO_block = simdgroup_matrix_storage<float>::apply_offset(
        dO_block, \(leadingBlockDimension(.dO)),
        offset_src, \(transposed(.dO)));
      O_block = simdgroup_matrix_storage<float>::apply_offset(
        O_block, \(leadingBlockDimension(.O)),
        offset_src, \(transposed(.O)));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Load the zero-padded edge data.
      ushort2 origin(0, 0);
      simdgroup_matrix_storage<float> dO;
      simdgroup_matrix_storage<float> O;
      dO.load(
        dO_block, \(leadingBlockDimension(.dO)), origin, \(transposed(.dO)));
      O.load(
        O_block, \(leadingBlockDimension(.O)), origin, \(transposed(.O)));
      
      // Perform the pointwise multiplication.
      float2 dO_value = *(dO.thread_elements());
      float2 O_value = *(O.thread_elements());
      D_term_accumulator += dO_value * O_value;
      
      """
    }
    
    let loopEndFloor = headDimension - headDimension % 8
    return """
    
    float2 D_term_accumulator(0);
    {
      \(bulkContributions(truncatedHeadDimension: loopEndFloor))
    }
    {
      \(edgeContributions(truncatedHeadDimension: loopEndFloor))
    }
    
    D_term = D_term_accumulator[0] + D_term_accumulator[1];
    D_term += simd_shuffle_xor(D_term, 1);
    D_term += simd_shuffle_xor(D_term, 8);
    D_term *= \(backwardScale);
    
    """
  }
}

// MARK: - Mask the Matrix Edge

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

// MARK: - Parallelized Along Rows

extension AttentionKernel {
  // M_LOG2E_F / sqrt(D)
  fileprivate var forwardScale: Float {
    return 1.442695041 / Float(headDimension).squareRoot()
  }
  
  // 1 / sqrt(D)
  fileprivate var backwardScale: Float {
    1 / Float(headDimension).squareRoot()
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
    
    // update 'O'
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
}

// MARK: - Parallelized Along Columns

extension AttentionKernel {
  func checkpointSoftmaxT() -> String {
    return checkpointSoftmax(derivative: false)
  }
  
  func derivativeSoftmaxT() -> String {
    return checkpointSoftmax(derivative: true)
  }
  
  // MARK: - Utilities that will soon become nested functions.
  
  // Whether the L/D_terms can be read directly from RAM.
  fileprivate var directLoadCondition: String {
    if preferAsyncLoad {
      return "false"
    } else {
      let blockDim = blockDimensions.traversal
      return "\(traversalOffset) + \(blockDim) <= \(traversalDimension)"
    }
  }
  
  // Load a vector where each entry corresponds to a different row.
  fileprivate func loadAsync(term: AttentionOperand) -> String {
    guard case .backwardKeyValue = type else {
      fatalError("This function should not have been called.")
    }
    
    return """
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      // Locate the \(term)[i] in device and threadgroup memory.
      auto \(term)_src = \(term) + r;
      auto \(term)_dst = (threadgroup float*)(threadgroup_block);
      
      ushort R_src_dimension = min(
        uint(\(blockDimensions.traversal)),
        uint(\(traversalDimension) - \(traversalOffset)));
      ushort R_dst_dimension = max(
        ushort(\(paddedTraversalEdge)),
        ushort(R_src_dimension));
      
      // Issue an async copy.
      simdgroup_event event;
      event.async_copy(
        \(term)_dst, 1, ushort2(R_dst_dimension, 1),
        \(term)_src, 1, ushort2(R_src_dimension, 1));
      simdgroup_event::wait(1, &event);
    }
    
    """
  }
  
  fileprivate func declareAsyncLocation(term: AttentionOperand) -> String {
    guard case .backwardKeyValue = type else {
      fatalError("This function should not have been called.")
    }
    
    return """
    
    // Where the \(term) data will be read from.
    auto \(term)_block = (threadgroup float*)(threadgroup_block);
    \(term)_block += morton_offset.x;
    
    """
  }
  
  fileprivate func declareDirectLocation(term: AttentionOperand) -> String {
    guard case .backwardKeyValue = type else {
      fatalError("This function should not have been called.")
    }
    
    return """
    
    // Where the \(term) data will be read from.
    auto \(term)_src = \(term) + \(traversalOffset) + morton_offset.x;
    
    """
  }
}

extension AttentionKernel {
  // A softmax where the per-row statistics have been reduced beforehand.
  //
  // Parameters:
  // - derivative: Whether this is the derivative softmax.
  func checkpointSoftmax(derivative: Bool) -> String {
    func allocateOutput() -> String {
      let name = derivative ? "dS" : "P"
      let blockDim = blockDimensions.traversal
      return """
      
      simdgroup_matrix_storage<float> \(name)_sram[\(blockDim) / 8];
      
      """
    }
    
    func computeSoftmax() -> String {
      if derivative {
        return """
        
        float2 P_elements = float2(*(P_sram[c / 8].thread_elements()));
        float2 dP_elements = float2(*(dP_sram[c / 8].thread_elements()));
        float2 dS_elements = dP_elements * \(backwardScale) - D_terms_elements;
        dS_elements *= P_elements;
        *(dS_sram[c / 8].thread_elements()) = dS_elements;
        
        """
      } else {
        return """
        
        float2 S_elements = float2(*(S_sram[c / 8].thread_elements()));
        float2 P_elements = fast::exp2(
          S_elements * \(forwardScale) - L_terms_elements);
        *(P_sram[c / 8].thread_elements()) = P_elements;
        
        """
      }
    }
    
    func directBranch() -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
        \(computeSoftmax())
      }
      
      """
    }
    
    func asyncBranch() -> String {
      let term: AttentionOperand = derivative ? .DTerms : .LTerms
      
      return """
      
      \(loadAsync(term: term))
      \(declareAsyncLocation(term: term))
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      #pragma clang loop unroll(full)
      for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
        ushort2 origin(c, 0);
        simdgroup_matrix_storage<float> \(term);
        \(term).load(\(term)_block, 1, origin, false);
        float2 \(term)_elements = *(\(term).thread_elements());
        
        \(computeSoftmax())
      }
      
      """
      
    }
    
    return """
    
    \(allocateOutput())
    {
      \(asyncBranch())
    }
    
    """
  }
}
