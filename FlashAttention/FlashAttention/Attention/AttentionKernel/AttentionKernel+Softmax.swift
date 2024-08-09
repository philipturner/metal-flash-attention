//
//  AttentionKernel+Softmax.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Elementwise operations on the attention matrix.

// MARK: - Scale Factor

extension AttentionKernel {
  // The scale factor in scaled dot product attention.
  //
  // Parameters:
  // - derivative: Whether this is the derivative softmax.
  func dotProductScale(derivative: Bool) -> Float {
    let logBase2E: Float = 1.442695041
    let rsqrtD = 1 / Float(headDimension).squareRoot()
    
    if !derivative {
      return logBase2E * rsqrtD
    } else {
      return rsqrtD
    }
  }
}

// MARK: - Compute D (dO * O)

extension AttentionKernel {
  func computeD() -> String {
    // Parts of the dO * O reduction that fall within block bounds.
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
        D_accumulator += dO_value * O_value;
      }

      """
    }
    
    // Parts of the dO * O reduction that fall on an indivisible edge.
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
      D_accumulator += dO_value * O_value;
      
      """
    }
    
    // Outer loop over the head dimension.
    let loopEndFloor = headDimension - headDimension % 8
    return """
    
    float2 D_accumulator(0);
    {
      \(bulkContributions(truncatedHeadDimension: loopEndFloor))
    }
    {
      \(edgeContributions(truncatedHeadDimension: loopEndFloor))
    }
    
    float D_sram = D_accumulator[0] + D_accumulator[1];
    D_sram += simd_shuffle_xor(D_sram, 1);
    D_sram += simd_shuffle_xor(D_sram, 8);
    D_sram *= \(dotProductScale(derivative: true));
    
    """
  }
}

// MARK: - Mask

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

// MARK: - Reduce

extension AttentionKernel {
  // Reduce maximum during the online softmax.
  func onlineReduceMaximum() -> String {
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
    m_new *= \(dotProductScale(derivative: false));
    
    """
  }
  
  // Rescale 'O' to reflect the new maximum.
  func onlineCorrectO() -> String {
    """
    
    // update 'O'
    float correction = 1;
    if (m_new > m) {
      correction = fast::exp2(m - m_new);
      m = m_new;
    }
    
    """
  }
  
  // Reduce sum during the online softmax.
  func onlineReduceSum() -> String {
    """
    
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
}

// MARK: - Softmax

/*

extension AttentionKernel {
  // Whether the L/D can be read directly from RAM.
  fileprivate var directLoadCondition: String {
    if preferAsyncLoad {
      return "false"
    } else {
      let blockDim = blockDimensions.traversal
      return "\(traversalOffset) + \(blockDim) <= \(traversalDimension)"
    }
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
 
 */

extension AttentionKernel {
  // A softmax where the per-row statistics have been reduced beforehand.
  //
  // Parameters:
  // - derivative: Whether this is the derivative softmax.
  func softmax(derivative: Bool) -> String {
    let operand: AttentionOperand = derivative ? .D : .L
    
    func allocateOutput() -> String {
      let blockDim = blockDimensions.traversal
      if !derivative {
        return """
        
        simdgroup_matrix_storage<float> P_sram[\(blockDim) / 8];
        
        """
      } else {
        return """
        
        simdgroup_matrix_storage<float> dS_sram[\(blockDim) / 8];
        
        """
      }
    }
    
    func innerLoop() -> String {
      let scale = dotProductScale(derivative: derivative)
      
      if !derivative {
        return """
        
        float2 S = float2(*(S_sram[c / 8].thread_elements()));
        float2 P = fast::exp2(S * \(scale) - L);
        *(P_sram[c / 8].thread_elements()) = P;
        
        """
      } else {
        return """
        
        float2 P = float2(*(P_sram[c / 8].thread_elements()));
        float2 dP = float2(*(dP_sram[c / 8].thread_elements()));
        float2 dS = P * (dP * \(scale) - D);
        *(dS_sram[c / 8].thread_elements()) = dS;
        
        """
      }
    }
    
    func directBranch() -> String {
      switch type {
      case .forward:
        return """
        
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          auto L = m;
          \(innerLoop())
        }
        
        """
      case .backwardQuery:
        return """
        
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          auto \(operand) = \(operand)_sram;
          \(innerLoop())
        }
        
        """
      case .backwardKeyValue:
        return """
        
        auto \(operand)_src = \(operand);
        \(operand)_src += \(traversalThreadOffset);
        
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          ushort2 origin(c, 0);
          simdgroup_matrix_storage<float> \(operand)_sram;
          \(operand)_sram.load(\(operand)_src, 1, origin, false);
          float2 \(operand) = *(\(operand)_sram.thread_elements());
          
          \(innerLoop())
        }
        
        """
      }
    }
    
    func loadOperand() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        // Locate the \(operand)[i] in device and threadgroup memory.
        auto \(operand)_src = \(operand) + \(traversalOffset);
        auto \(operand)_dst = (threadgroup float*)(threadgroup_block);
        
        ushort R_src_dimension = min(
          uint(\(blockDimensions.traversal)),
          uint(\(traversalDimension) - \(traversalOffset)));
        ushort R_dst_dimension = max(
          ushort(\(paddedTraversalEdge)),
          ushort(R_src_dimension));
        
        // Issue an async copy.
        simdgroup_event event;
        event.async_copy(
          \(operand)_dst, 1, ushort2(R_dst_dimension, 1),
          \(operand)_src, 1, ushort2(R_src_dimension, 1));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    func asyncBranch() -> String {
      """
      
      auto \(operand)_block = (threadgroup float*)(threadgroup_block);
      \(operand)_block += morton_offset.x;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      #pragma clang loop unroll(full)
      for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
        ushort2 origin(c, 0);
        simdgroup_matrix_storage<float> \(operand)_sram;
        \(operand)_sram.load(\(operand)_block, 1, origin, false);
        float2 \(operand) = *(\(operand)_sram.thread_elements());
        
        \(innerLoop())
      }
      
      """
    }
    
    switch type {
    case .forward, .backwardQuery:
      return """
      
      \(allocateOutput())
      {
        \(directBranch())
      }
      
      """
    case .backwardKeyValue:
      if preferAsyncLoad {
        return """
        
        \(allocateOutput())
        {
          \(loadOperand())
          \(asyncBranch())
        }
        
        """
      } else {
        let blockDim = blockDimensions.traversal
        #if true
        
        let condition = """
        (\(traversalDimension) % \(blockDim) != 0) &&
        (\(traversalOffset) + \(blockDim) <= \(traversalDimension))
        """
        #else
        
        let condition = """
        (\(traversalOffset) + \(blockDim) <= \(traversalDimension))
        """
        #endif
        
        return """
        
        \(allocateOutput())
        if (\(condition)) {
          \(directBranch())
        } else {
          \(loadOperand())
          \(asyncBranch())
        }
        
        """
      }
    }
  }
}
