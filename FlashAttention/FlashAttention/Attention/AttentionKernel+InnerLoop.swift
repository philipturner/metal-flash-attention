//
//  AttentionKernel+InnerLoop.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/2/24.
//

// MARK: - Inner Loop

// Forward
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     (m, l, P) = softmax(m, l, S * scaleFactor)
//     O *= correction
//     load V[c]
//     O += P * V
//   }
//   O /= l
//
// Backward Query (true)
//   for c in 0..<C {
//     load K[c]
//     S = Q * K^T
//     P = exp(S - L)
//     load V[c]
//     dP = dO * V^T
//     dS = P * (dP - D) * scaleFactor
//     load K[c]
//     dQ += dS * K
//   }
//
// Backward Key-Value (false)
//   for r in 0..<R {
//     load Q[r]
//     load L[r]
//     S^T = K * Q^T
//     P^T = exp(S^T - L)
//     load dO[r]
//     load D[r]
//     dV += P^T * dO
//     dP^T = V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//     store dS^T[c][r]
//   }
//
// Backward Key-Value (true)
//   for r in 0..<R {
//     load Q[r]
//     load L[r]
//     S^T = K * Q^T
//     P^T = exp(S^T - L)
//     load dO[r]
//     load D[r]
//     dV += P^T * dO
//     dP^T = V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//     load Q[r]
//     dK += dS^T * Q
//   }

extension AttentionKernel {
  func createInnerLoopForward() -> String {
    var accessDesc = AttentionHBMAccessDescriptor()
    accessDesc.index = "c"
    accessDesc.leadingBlockDimension = leadingBlockDimensions.V
    accessDesc.leadingDimension = leadingDimensions.V
    accessDesc.name = "V"
    accessDesc.threadgroupAddress = "threadgroup_block"
    accessDesc.transposeState = transposeState.V
    let prefetchV = prefetchColumns(descriptor: accessDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.index = "c"
    accumulateDesc.indexedBlockDimension = blockDimensions.C
    accumulateDesc.leadingBlockDimensionRHS = leadingBlockDimensions.V
    accumulateDesc.names = (accumulator: "O", lhs: "P", rhs: "V")
    accumulateDesc.threadgroupAddress = "threadgroup_block"
    accumulateDesc.transposeStateRHS = transposeState.V
    let accumulateO = accumulate(descriptor: accumulateDesc)
    
    return """
  
  // Iterate over the columns.
  for (uint c = 0; c < C; c += 32) {
    // S = Q * K^T
    \(computeS())
    \(maskAlongColumns(sram: "S_sram"))
    
    // (m, l, P) = softmax(m, l, S * scaleFactor)
    \(onlineSoftmax())
    
    // load V[c]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchV)
    
    // O += P * V
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(accumulateO)
  }
  
  // O /= l
  float l_reciprocal = 1 / l;
#pragma clang loop unroll(full)
  for (ushort d = 0; d < \(paddedD); d += 8) {
   *(O_sram[d / 8].thread_elements()) *= l_reciprocal;
  }
  
"""
  }
  
  func createInnerLoopBackwardQuery() -> String {
    var accessDesc = AttentionHBMAccessDescriptor()
    accessDesc.index = "c"
    accessDesc.leadingBlockDimension = leadingBlockDimensions.K
    accessDesc.leadingDimension = leadingDimensions.K
    accessDesc.name = "K"
    accessDesc.threadgroupAddress = "threadgroup_block"
    accessDesc.transposeState = transposeState.K
    let prefetchK = prefetchColumns(descriptor: accessDesc)
    
    accessDesc = AttentionHBMAccessDescriptor()
    accessDesc.index = "c"
    accessDesc.leadingBlockDimension = leadingBlockDimensions.V
    accessDesc.leadingDimension = leadingDimensions.V
    accessDesc.name = "V"
    accessDesc.threadgroupAddress = "threadgroup_block"
    accessDesc.transposeState = transposeState.V
    let prefetchV = prefetchColumns(descriptor: accessDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.index = "c"
    accumulateDesc.indexedBlockDimension = blockDimensions.C
    accumulateDesc.leadingBlockDimensionRHS = leadingBlockDimensions.K
    accumulateDesc.names = (accumulator: "dQ", lhs: "dS", rhs: "K")
    accumulateDesc.threadgroupAddress = "threadgroup_block"
    accumulateDesc.transposeStateRHS = transposeState.K
    let accumulateDerivativeQ = accumulate(descriptor: accumulateDesc)
    
    return """
  
  // Iterate over the columns.
  for (uint c = 0; c < C; c += 32) {
    // load K[c]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchK)
    
    // S = Q * K^T
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(computeS())
    
    // P = softmax(S * scaleFactor)
    \(checkpointSoftmax())
    
    // load V[c]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchV)
    
    // dP = dO * V^T
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(computeDerivativeP())
    
    // dS = P * (dP - D) * scaleFactor
    \(computeDerivativeSoftmax())
    
    // load K[c]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchK)
    
    // dQ += dS * K
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(accumulateDerivativeQ)
  }
    
"""
  }
  
  func createInnerLoopKeyValue(computeDerivativeK: Bool) -> String {
    var accessDesc = AttentionHBMAccessDescriptor()
    accessDesc.index = "r"
    accessDesc.leadingBlockDimension = leadingBlockDimensions.Q
    accessDesc.leadingDimension = leadingDimensions.Q
    accessDesc.name = "Q"
    accessDesc.threadgroupAddress = "threadgroup_block"
    accessDesc.transposeState = transposeState.Q
    let prefetchQ = prefetchRows(descriptor: accessDesc)
    
    var accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.index = "r"
    accumulateDesc.indexedBlockDimension = blockDimensions.R
    accumulateDesc.leadingBlockDimensionRHS = leadingBlockDimensions.O
    accumulateDesc.names = (accumulator: "dV", lhs: "PT", rhs: "dO")
    accumulateDesc.threadgroupAddress = "threadgroup_block"
    accumulateDesc.transposeStateRHS = transposeState.O
    let accumulateDerivativeV = accumulate(descriptor: accumulateDesc)
    
    accumulateDesc = AttentionAccumulateDescriptor()
    accumulateDesc.index = "r"
    accumulateDesc.indexedBlockDimension = blockDimensions.R
    accumulateDesc.leadingBlockDimensionRHS = leadingBlockDimensions.Q
    accumulateDesc.names = (accumulator: "dK", lhs: "dST", rhs: "Q")
    accumulateDesc.threadgroupAddress = "threadgroup_block"
    accumulateDesc.transposeStateRHS = transposeState.Q
    let accumulateDerivativeK = accumulate(descriptor: accumulateDesc)
    
    var output = """

  // Iterate over the rows.
  for (uint r = 0; r < R; r += 32) {
    // load Q[r]
    // load L[r]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchQLTerms())
    
    // S^T = K * Q^T
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(computeST())
    
    // P^T = exp(S^T - L)
    \(checkpointSoftmaxT())
    
    // load dO[r]
    // load D[r]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchDerivativeODTerms())
    
    // dV += P^T * dO
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(accumulateDerivativeV)
    
    // dP^T = V * dO^T
    \(computeDerivativePT())
    
    // dS^T = P^T * (dP^T - D) * scaleFactor
    \(computeDerivativeSoftmaxT())

"""
    
    if computeDerivativeK {
      output += """
  
    // load Q[r]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchQ)

    // dK += dS^T * Q
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(accumulateDerivativeK)
  }
  
"""
    } else {
      output += """

    // store dS^T[c][r]
    {
      uint2 device_origin(r, gid * 32 + sidx * 8);
      device_origin += uint2(morton_offset);
      device bfloat* dst =
      simdgroup_matrix_storage<bfloat>::apply_offset(
        dST, \(leadingDimensionDerivativeST), device_origin, false);
      
#pragma clang loop unroll(full)
      for (ushort c = 0; c < 32; c += 8) {
        ushort2 thread_origin(c, 0);
        dST_sram[c / 8].store_bfloat(
          dst, \(leadingDimensionDerivativeST), thread_origin, false);
      }
    }
  }

"""
    }
    
    return output
  }
}

// MARK: - Prefetching Operations

extension AttentionKernel {
  func blockQ() -> String {
    "(threadgroup float*)(threadgroup_block)"
  }
  
  func blockLTerms() -> String {
    if transposeState.Q {
      // D x R, where R is the row stride.
      return "\(blockQ()) + \(paddedD) * \(leadingBlockDimensions.Q)"
    } else {
      // R x D, where D is the row stride.
      return "\(blockQ()) + R_group * \(leadingBlockDimensions.Q)"
    }
  }
  
  func prefetchQLTerms() -> String {
    return """
    
    if (sidx == 0) {
      uint2 device_origin(0, r);
      auto Q_src = simdgroup_matrix_storage<float>::apply_offset(
        Q, \(leadingDimensions.Q), device_origin, \(transposeState.Q));
      auto Q_dst = \(blockQ());
      auto L_terms_src = L_terms + r;
      auto L_terms_dst = \(blockLTerms());
      
      // Zero-padding for safety, which should harm performance.
      ushort R_tile_dimension = min(uint(R_group), R - r);
      ushort2 tile_src(D, R_tile_dimension);
      ushort2 tile_dst(\(paddedD), R_group);
      
      // Issue two async copies.
      simdgroup_event events[2];
      events[0].async_copy(
        Q_dst, \(leadingBlockDimensions.Q), tile_dst,
        Q_src, \(leadingDimensions.Q), tile_src, \(transposeState.Q));
      events[1].async_copy(
        L_terms_dst, 1, ushort2(tile_dst.y, 1),
        L_terms_src, 1, ushort2(tile_src.y, 1));
      simdgroup_event::wait(2, events);
    }

"""
  }
  
  func blockDerivativeO() -> String {
    "(threadgroup float*)(threadgroup_block)"
  }
  
  func blockDTerms() -> String {
    if transposeState.O {
      // D x R, where R is the row stride.
      return "\(blockDerivativeO()) + \(paddedD) * \(leadingBlockDimensions.O)"
    } else {
      // R x D, where D is the row stride.
      return "\(blockDerivativeO()) + R_group * \(leadingBlockDimensions.O)"
    }
  }
  
  func prefetchDerivativeODTerms() -> String {
    return """
    
    if (sidx == 0) {
      uint2 device_origin(0, r);
      auto dO_src = simdgroup_matrix_storage<float>::apply_offset(
        dO, \(leadingDimensions.O), device_origin, \(transposeState.O));
      auto dO_dst = \(blockDerivativeO());
      auto D_terms_src = D_terms + r;
      auto D_terms_dst = \(blockDTerms());
      
      // Zero-padding for safety, which should harm performance.
      ushort R_tile_dimension = min(uint(R_group), R - r);
      ushort2 tile_src(D, R_tile_dimension);
      ushort2 tile_dst(\(paddedD), R_group);
      
      // Issue two async copies.
      simdgroup_event events[2];
      events[0].async_copy(
        dO_dst, \(leadingBlockDimensions.O), tile_dst,
        dO_src, \(leadingDimensions.O), tile_src, \(transposeState.O));
      events[1].async_copy(
        D_terms_dst, 1, ushort2(tile_dst.y, 1),
        D_terms_src, 1, ushort2(tile_src.y, 1));
      simdgroup_event::wait(2, events);
    }

"""
  }
}

// MARK: - Attention Matrix

extension AttentionKernel {
  // We may need to recycle this code across similarly shaped matrix
  // multiplications.
  func computeS() -> String {
    return """
    // Zero-initialize the accumulator (we'll optimize this away later).
    simdgroup_matrix_storage<float> S_sram[32 / 8];
  #pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
      S_sram[c / 8] = simdgroup_matrix_storage<float>(0);
    }

    // Find where the Q data will be read from.
    auto Q_block = (threadgroup float*)(threadgroup_block);
    Q_block = simdgroup_matrix_storage<float>::apply_offset(
      Q_block, \(leadingBlockDimensions.Q),
      ushort2(0, sidx * 8) + morton_offset, \(transposeState.Q));

    // Find where the K data will be read from.
    auto KT_block = (threadgroup float*)(threadgroup_block) + \(32 * 32);
    KT_block = simdgroup_matrix_storage<float>::apply_offset(
      KT_block, \(leadingBlockDimensions.K),
      morton_offset, \(!transposeState.K));

    // Outer loop over D.
#pragma clang loop unroll(full)
    for (ushort d = 0; d < D; d += 32) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      if (sidx == 0) {
        ushort D_src_dimension = min(ushort(32), ushort(D - d));
        ushort D_dst_dimension = min(ushort(32), ushort(\(paddedD) - d));
        
        // load Q[r]
        simdgroup_event events[2];
        {
          uint2 device_origin(d, gid * R_group);
          auto src = simdgroup_matrix_storage<float>::apply_offset(
            Q, \(leadingDimensions.Q), device_origin, \(transposeState.Q));
          auto dst = (threadgroup float*)(threadgroup_block);
          
          ushort R_src_dimension = min(uint(R_group), R - gid * R_group);
          ushort2 tile_src(D_src_dimension, R_src_dimension);
          ushort2 tile_dst(D_dst_dimension, R_group); // excessive R padding
          
          events[0].async_copy(
            dst, \(leadingBlockDimensions.Q), tile_dst,
            src, \(leadingDimensions.Q), tile_src, \(transposeState.Q));
        }
        
        // load K[c]
        {
          uint2 device_origin(d, c);
          auto src = simdgroup_matrix_storage<float>::apply_offset(
            K, \(leadingDimensions.K), device_origin, \(transposeState.K));
          auto dst = (threadgroup float*)(threadgroup_block) + \(32 * 32);
          
          ushort C_src_dimension = min(uint(C_group), C - c);
          ushort2 tile_src(D_src_dimension, C_src_dimension);
          ushort2 tile_dst(D_dst_dimension, C_group); // excessive C padding
          
          events[1].async_copy(
          dst, \(leadingBlockDimensions.K), tile_dst,
          src, \(leadingDimensions.K), tile_src, \(transposeState.K));
        }
        simdgroup_event::wait(2, events);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Inner loop over D.
      ushort d_outer = d;
#pragma clang loop unroll(full)
      for (ushort d = 0; d < min(32, \(paddedD) - d_outer); d += 8) {
        simdgroup_matrix_storage<float> Q;
        Q.load(
          Q_block, \(leadingBlockDimensions.Q),
          ushort2(d, 0), \(transposeState.Q));
#pragma clang loop unroll(full)
        for (ushort c = 0; c < 32; c += 8) {
          simdgroup_matrix_storage<float> KT;
          KT.load(
            KT_block, \(leadingBlockDimensions.K),
            ushort2(c, d), \(!transposeState.K));
          S_sram[c / 8]
            .multiply(Q, KT, true);
        }
      }
    }

"""
  }
  
  func computeST() -> String {
    return """

    auto QT_block = \(blockQ());
    QT_block = simdgroup_matrix_storage<float>::apply_offset(
      QT_block, \(leadingBlockDimensions.Q), morton_offset,
      \(!transposeState.Q));

    simdgroup_matrix_storage<float> ST_sram[32 / 8];
#pragma clang loop unroll(full)
    for (ushort d = 0; d < \(paddedD); d += 8) {
#pragma clang loop unroll(full)
      for (ushort r = 0; r < 32; r += 8) {
        ushort2 origin(r, d);
        simdgroup_matrix_storage<float> QT;
        QT.load(
          QT_block, \(leadingBlockDimensions.Q), origin, \(!transposeState.Q));
        ST_sram[r / 8]
          .multiply(K_sram[d / 8], QT, d > 0);
      }
    }

"""
  }
  
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
    let scaleFactor = "(M_LOG2E_F / sqrt(float(D)))"
    
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

    // update the previous value of 'O'
    float correction = 1;
    if (m_new > m) {
      correction = fast::exp2(m - m_new);
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD); d += 8) {
        *(O_sram[d / 8].thread_elements()) *= correction;
      }
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
    let scaleFactor = "(M_LOG2E_F / sqrt(float(D)))"
    
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
    let scaleFactor = "(M_LOG2E_F / sqrt(float(D)))"
    
    return """

    auto L_terms_block = \(blockLTerms());
    L_terms_block += morton_offset.x;
    
    simdgroup_matrix_storage<float> PT_sram[32 / 8];
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

"""
  }
}

// MARK: - Softmax Derivative

extension AttentionKernel {
  func computeDerivativeSoftmax() -> String {
    let scaleFactor = "(1 / sqrt(float(D)))"
    
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
    let scaleFactor = "(1 / sqrt(float(D)))"
    
    return """

    auto D_terms_block = \(blockDTerms());
    D_terms_block += morton_offset.x;

    simdgroup_matrix_storage<float> dST_sram[32 / 8];
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

"""
  }
}

// MARK: - Attention Matrix Derivative

extension AttentionKernel {
  func computeDerivativeP() -> String {
    return """

    auto VT_block = (threadgroup float*)(threadgroup_block);
    VT_block = simdgroup_matrix_storage<float>::apply_offset(
      VT_block, \(leadingBlockDimensions.V), morton_offset,
      \(!transposeState.V));
    
    simdgroup_matrix_storage<float> dP_sram[32 / 8];
#pragma clang loop unroll(full)
    for (ushort d = 0; d < \(paddedD); d += 8) {
#pragma clang loop unroll(full)
      for (ushort c = 0; c < 32; c += 8) {
        ushort2 origin(c, d);
        simdgroup_matrix_storage<float> VT;
        VT.load(
          VT_block, \(leadingBlockDimensions.V), origin, \(!transposeState.V));
        dP_sram[c / 8]
          .multiply(dO_sram[d / 8], VT, d > 0);
      }
    }
    
"""
  }
  
  func computeDerivativePT() -> String {
    return """

    auto dOT_block = \(blockDerivativeO());
    dOT_block = simdgroup_matrix_storage<float>::apply_offset(
      dOT_block, \(leadingBlockDimensions.O), morton_offset,
      \(!transposeState.O));
    
    simdgroup_matrix_storage<float> dPT_sram[32 / 8];
#pragma clang loop unroll(full)
    for (ushort d = 0; d < \(paddedD); d += 8) {
#pragma clang loop unroll(full)
      for (ushort r = 0; r < 32; r += 8) {
        ushort2 origin(r, d);
        simdgroup_matrix_storage<float> dOT;
        dOT.load(
          dOT_block, \(leadingBlockDimensions.O), origin, \(!transposeState.O));
        dPT_sram[r / 8]
          .multiply(V_sram[d / 8], dOT, d > 0);
      }
    }

"""
  }
}

// MARK: - Accumulate

struct AttentionAccumulateDescriptor {
  var index: String?
  var indexedBlockDimension: UInt16?
  var leadingBlockDimensionRHS: UInt16?
  var names: (accumulator: String, lhs: String, rhs: String)?
  var threadgroupAddress: String?
  var transposeStateRHS: Bool?
}

extension AttentionKernel {
  func accumulate(descriptor: AttentionAccumulateDescriptor) -> String {
    guard let index = descriptor.index,
          let indexedBlockDimension = descriptor.indexedBlockDimension,
          let leadingBlockDimensionRHS = descriptor.leadingBlockDimensionRHS,
          let names = descriptor.names,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeStateRHS = descriptor.transposeStateRHS else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """
    
    // Where the did the async copy put the RHS?
    auto \(names.rhs)_block = (threadgroup float*)(\(threadgroupAddress));
    \(names.rhs)_block = simdgroup_matrix_storage<float>::apply_offset(
      \(names.rhs)_block, 
      \(leadingBlockDimensionRHS),
      morton_offset,
      \(transposeStateRHS));
    
    // Iterate over the row/column dimension.
#pragma clang loop unroll(full)
    for (
      ushort \(index) = 0;
      \(index) < \(indexedBlockDimension);
      \(index) += 8
    ) {

      // Iterate over the head dimension.
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD); d += 8) {
        ushort2 origin(d, \(index));
        simdgroup_matrix_storage<float> \(names.rhs);
        
        // Load the RHS from threadgroup memory.
        \(names.rhs).load(
          \(names.rhs)_block, 
          \(leadingBlockDimensionRHS),
          origin,
          \(transposeStateRHS));
        
        // Add the contributions from the c-th/r-th element of the attention
        // matrix row/column.
        \(names.accumulator)_sram[d / 8].multiply(
          \(names.lhs)_sram[\(index) / 8],
          \(names.rhs),
          /*accumulate=*/true);
      }
    }

"""
  }
}

// MARK: - Blocking Along Head Dimension

// The pseudocode assumes the head dimension (D) is 128.
//
// Forward
//   // Setup
//   initialize O[32][128]
//   initialize m[32]
//   initialize l[32]
//
//   // Inner Loop
//   for c in 0..<C {
//     repeat 4 times
//       load Q[r][32]
//       load K[c][32]
//       S += Q * K^T
//     (m, l, P) = softmax(m, l, S * scaleFactor)
//
//     O *= correction
//     repeat 2 times
//       load V[c][64]
//       O[32][64] += P * V
//   }
//
//   // Cleanup
//   O /= l
//   store dO
//   store L
//
// Backward Query (true)
//   // Setup
//   initialize dQ[32][128]
//   repeat 4 times
//     load dO[r][32]
//     load O[r][32]
//     D += dO * O
//   load L[32]
//
//   // Inner Loop
//   for c in 0..<C {
//     repeat 4 times
//       load Q[r][32]
//       load K[c][32]
//       S += Q * K^T
//     P = exp(S - L)
//
//     repeat 4 times
//       load dO[r][32]
//       load V[c][32]
//       dP += dO * V^T
//     dS = P * (dP - D) * scaleFactor
//
//     repeat 2 times
//       load K[c][64]
//       dQ[32][64] += dS * K
//   }
//
//   // Cleanup
//   store dQ
//   store D
//
// Backward Key-Value (true)
//   // Setup
//   initialize dK[32][128]
//   initialize dV[32][128]
//
//   // Inner Loop
//   for r in 0..<R {
//     load L[r]
//     load D[r]
//
//     repeat 4 times
//       load K[c][32]
//       load Q[r][32]
//       S^T += K * Q^T
//     P^T = exp(S^T - L)
//
//     repeat 4 times
//       load V[c][32]
//       load dO[r][32]
//       dV[32][32] += P^T * dO
//       dP^T += V * dO^T
//     dS^T = P^T * (dP^T - D) * scaleFactor
//
//     repeat 2 times
//       load Q[r][64]
//       dK[32][64] += dS^T * Q
//   }
//
//   // Cleanup
//   store dK
//   store dV
