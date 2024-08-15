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
          auto dO_src = simdgroup_matrix_storage<\(memoryName(.dO))>
          ::apply_offset(
            dO, \(leadingDimension(.dO)), 
            offset_src, \(transposed(.dO)));
          
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
          
          simdgroup_matrix_storage<\(registerName(.dO))> dO;
          dO.\(loadFunction(.dO))(
            dO_src, \(leadingDimension(.dO)),
            ushort2(d, 0), \(transposed(.dO)));
          
          """
        }
      }
      
      return """
      
      // Threads outside of the matrix along the row dimension,
      // have their origin shifted in-bounds.
      uint D_offset = morton_offset.x;
      uint R_offset = \(clampedParallelizationThreadOffset);
      uint2 offset_src(D_offset, R_offset);
      
      \(declareDerivativeOLocation())
      
      // Where the O data will be read from.
      auto O_src = simdgroup_matrix_storage<\(memoryName(.O))>
      ::apply_offset(
        O, \(leadingDimension(.O)),
        offset_src, \(transposed(.O)));
      
      // Going to use async copy to handle the matrix edge.
      #pragma clang loop unroll(disable)
      for (ushort d = 0; d < \(truncatedHeadDimension); d += 8) {
        \(loadDerivativeO())
        
        simdgroup_matrix_storage<\(registerName(.O))> O;
        O.\(loadFunction(.O))(
          O_src, \(leadingDimension(.O)),
          ushort2(d, 0), \(transposed(.O)));
        
        // Perform the pointwise multiplication.
        auto dO_value = *(dO.thread_elements());
        auto O_value = *(O.thread_elements());
        D_accumulator += float2(dO_value) * float2(O_value);
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
      
      // Distinct from the block bytes that would be used to calculate
      // the threadgroup memory allocation.
      func blockBytesDerivativeO() -> UInt16 {
        let memoryPrecision = memoryPrecisions[.dO]!
        let size = UInt16(memoryPrecision.size)
        return blockDimensions.parallelization * 8 * size
      }
      
      return """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint D_offset = \(truncatedHeadDimension);
        uint R_offset = \(parallelizationGroupOffset);
        uint2 offset_src(D_offset, R_offset);
        
        auto dO_src = simdgroup_matrix_storage<\(memoryName(.dO))>
        ::apply_offset(
          dO, \(leadingDimension(.dO)), 
          offset_src, \(transposed(.dO)));
        auto O_src = simdgroup_matrix_storage<\(memoryName(.O))>
        ::apply_offset(
          O, \(leadingDimension(.O)), 
          offset_src, \(transposed(.O)));
        
        auto dO_dst = (threadgroup \(memoryName(.dO))*)(threadgroup_block);
        auto O_dst = (threadgroup \(memoryName(.O))*)(
          threadgroup_block + \(blockBytesDerivativeO()));
        
        ushort D_src_dimension = \(headDimension) % 8;
        ushort D_dst_dimension = 8;
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
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
      auto dO_block = (threadgroup \(memoryName(.dO))*)(threadgroup_block);
      auto O_block = (threadgroup \(memoryName(.O))*)(
        threadgroup_block + \(blockBytesDerivativeO()));
      
      dO_block = simdgroup_matrix_storage<\(memoryName(.dO))>
      ::apply_offset(
        dO_block, \(leadingBlockDimension(.dO)),
        offset_src, \(transposed(.dO)));
      O_block = simdgroup_matrix_storage<\(memoryName(.O))>
      ::apply_offset(
        O_block, \(leadingBlockDimension(.O)),
        offset_src, \(transposed(.O)));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Load the zero-padded edge data.
      ushort2 origin(0, 0);
      simdgroup_matrix_storage<\(registerName(.dO))> dO;
      simdgroup_matrix_storage<\(registerName(.O))> O;
      dO.\(loadFunction(.dO))(
        dO_block, \(leadingBlockDimension(.dO)),
        origin, \(transposed(.dO)));
      O.\(loadFunction(.O))(
        O_block, \(leadingBlockDimension(.O)),
        origin, \(transposed(.O)));
      
      // Perform the pointwise multiplication.
      auto dO_value = *(dO.thread_elements());
      auto O_value = *(O.thread_elements());
      D_accumulator += float2(dO_value) * float2(O_value);
      
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
    let logBase2E: Float = 1.442695041
    
    return """
    
    if ((\(remainder) != 0) &&
        (\(traversalOffset) + \(blockDim) > \(traversalDimension))) {
      // Prevent the value from becoming -INF during the FMA before the
      // exponentiation. If the multiplication during FMA returns -INF,
      // subtracting a positive 'm' value will turn it into zero. We don't want
      // that. exp(0) evaluates to 1.00 and corrupts the value of 'l'.
      const \(registerName(.S)) mask_value =
      (0.875 / \(logBase2E)) * -numeric_limits<\(registerName(.S))>::max();
      
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
    vec<\(registerName(.S)), 2> m_new_accumulator;
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
    // I don't think keeping the sum in FP16 here, should negatively affect
    // numerical accuracy too much. It is typically 8-16 summations. If I am
    // wrong, clients - please just fix this ('float2 l_new_accumulator')
    // instead of promoting the entire P matrix to FP32. Having P in low
    // precision is critical for performance on M1, which typically will not
    // receive high coverage in your performance tests.
    //
    // TODO: When debugging mixed precision, revert this to float2. Examine
    // the impact it has on numerical correctness, with every other variable
    // in FP32.
    // - Casting to float2 may cause the Metal compiler to allocate a bunch
    //   of extra registers for the FP16 attention matrix.
    """
    
    // update 'l'
    vec<\(registerName(.P)), 2> l_new_accumulator;
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
        
        simdgroup_matrix_storage<\(registerName(.P))> \
        P_sram[\(blockDim) / 8];
        
        """
      } else {
        return """
        
        simdgroup_matrix_storage<\(registerName(.dS))> \
        dS_sram[\(blockDim) / 8];
        
        """
      }
    }
    
    func loadOperand() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        auto \(operand)_src = \(operand) + \(traversalOffset);
        auto \(operand)_dst =
        (threadgroup \(memoryName(operand))*)(threadgroup_block);
        
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
    
    // Declares the source of L or D.
    //
    // Also guards against unsafe accesses to the declared pointer (barrier).
    func declareOperandLocation(addressSpace: MTLAddressSpace) -> String {
      if addressSpace == .device {
        return """
        
        auto \(operand)_src = \(operand);
        \(operand)_src += \(traversalOffset) + morton_offset.x;
        
        """
      } else {
        return """
        
        auto \(operand)_src =
        (threadgroup \(memoryName(operand))*)(threadgroup_block);
        \(operand)_src += morton_offset.x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        """
      }
    }
    
    func overwriteAttentionMatrixElements() -> String {
      let scale = dotProductScale(derivative: derivative)
      
      if !derivative {
        return """
        
        auto S = *(S_sram[c / 8].thread_elements());
        auto P = vec<\(registerName(.P)), 2>(
          fast::exp2(float2(S) * \(scale) - float2(L_elements)));
        *(P_sram[c / 8].thread_elements()) = P;
        
        """
      } else {
        return """
        
        auto P = *(P_sram[c / 8].thread_elements());
        auto dP = *(dP_sram[c / 8].thread_elements());
        auto dS = vec<\(registerName(.dS)), 2>(
          float2(P) * (float2(dP) * \(scale) - float2(D_elements)));
        *(dS_sram[c / 8].thread_elements()) = dS;
        
        """
      }
    }
    
    func innerLoop() -> String {
      switch type {
      case .forward:
        return """
        
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          auto L_elements = m;
          \(overwriteAttentionMatrixElements())
        }
        
        """
      case .backwardQuery:
        return """
        
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          auto \(operand)_elements = \(operand)_sram;
          \(overwriteAttentionMatrixElements())
        }
        
        """
      case .backwardKeyValue:
        return """
        
        #pragma clang loop unroll(full)
        for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
          ushort2 \(operand)_origin(c, 0);
          simdgroup_matrix_storage<\(registerName(operand))> \(operand);
          \(operand).\(loadFunction(operand))(
            \(operand)_src, 1,
            \(operand)_origin, false);
          auto \(operand)_elements = *(\(operand).thread_elements());
          
          \(overwriteAttentionMatrixElements())
        }
        
        """
      }
    }
    
    switch type {
    case .forward, .backwardQuery:
      return """
      
      \(allocateOutput())
      {
        \(innerLoop())
      }
      
      """
    case .backwardKeyValue:
      let blockDim = blockDimensions.traversal
      let condition = """
      \(!preferAsyncLoad) && (
        (\(traversalDimension) % \(blockDim) == 0) ||
        (\(traversalOffset) + \(blockDim) <= \(traversalDimension))
      )
      """
      
      return """
      
      \(allocateOutput())
      if (\(condition)) {
        \(declareOperandLocation(addressSpace: .device))
        \(innerLoop())
      } else {
        \(loadOperand())
        \(declareOperandLocation(addressSpace: .threadgroup))
        \(innerLoop())
      }
      
      """
    }
  }
}
