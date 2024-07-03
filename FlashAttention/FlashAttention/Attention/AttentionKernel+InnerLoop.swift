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
    
    return """
  
  // Iterate over the columns.
  for (uint c = 0; c < C; c += 32) {
    // load K[c]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchK)
    
    // S = Q * K^T
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(computeS())
    \(maskAlongColumns(sram: "S_sram"))
    
    // (m, l, P) = softmax(m, l, S * scaleFactor)
    \(onlineSoftmax())

    // load V[c]
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(prefetchV)
    
    // O += P * V
    threadgroup_barrier(mem_flags::mem_threadgroup);
    \(accumulateO())
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
    \(accumulateDerivativeQ())
  }
    
"""
  }
  
  func createInnerLoopValue() -> String {
    return """
  
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
    \(accumulateDerivativeV())
    
    // dP^T = V * dO^T
    \(computeDerivativePT())
    
    // dS^T = P^T * (dP^T - D) * scaleFactor
    \(computeDerivativeSoftmaxT())
    
    if (r == 0) {
      dK_sram[0] = dST_sram[0];
    }
  }
  
"""
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
        L_terms_dst, 1, ushort2(tile_dst.y, 0),
        L_terms_src, 1, ushort2(tile_src.y, 0));
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
        D_terms_dst, 1, ushort2(tile_dst.y, 0),
        D_terms_src, 1, ushort2(tile_src.y, 0));
      simdgroup_event::wait(2, events);
    }

"""
  }
}

// MARK: - Attention Matrix

extension AttentionKernel {
  func computeS() -> String {
    return """

    auto KT_block = (threadgroup float*)(threadgroup_block);
    KT_block = simdgroup_matrix_storage<float>::apply_offset(
      KT_block, \(leadingBlockDimensions.K), morton_offset,
      \(!transposeState.K));

    simdgroup_matrix_storage<float> S_sram[32 / 8];
#pragma clang loop unroll(full)
    for (ushort d = 0; d < \(paddedD); d += 8) {
#pragma clang loop unroll(full)
      for (ushort c = 0; c < 32; c += 8) {
        ushort2 origin(c, d);
        simdgroup_matrix_storage<float> KT;
        KT.load(
          KT_block, \(leadingBlockDimensions.K), origin, \(!transposeState.K));
        S_sram[c / 8]
          .multiply(Q_sram[d / 8], KT, d > 0);
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
      
#pragma clang loop unroll(full)
      for (ushort index = 0; index < 2; ++index) {
        if (morton_offset.x + index >= remainder32 - remainder32_floor) {
          auto S_elements = \(sram)[remainder32_floor / 8].thread_elements();
          (*S_elements)[index] = -numeric_limits<float>::max();
        }
      }
#pragma clang loop unroll(full)
      for (ushort c = remainder32_floor + 8; c < 32; c += 8) {
        auto S_elements = \(sram)[c / 8].thread_elements();
        *S_elements = -numeric_limits<float>::max();
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

extension AttentionKernel {
  func accumulateO() -> String {
    return """

    auto V_block = (threadgroup float*)(threadgroup_block);
    V_block = simdgroup_matrix_storage<float>::apply_offset(
      V_block, \(leadingBlockDimensions.V), morton_offset, \(transposeState.V));

#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD); d += 8) {
        ushort2 origin(d, c);
        simdgroup_matrix_storage<float> V;
        V.load(
          V_block, \(leadingBlockDimensions.V), origin, \(transposeState.V));
        O_sram[d / 8]
          .multiply(P_sram[c / 8], V, true);
      }
    }

"""
  }
  
  func accumulateDerivativeV() -> String {
    return """
    
    auto dO_block = \(blockDerivativeO());
    dO_block = simdgroup_matrix_storage<float>::apply_offset(
      dO_block, \(leadingBlockDimensions.O), morton_offset,
      \(transposeState.O));
    
#pragma clang loop unroll(full)
    for (ushort r = 0; r < 32; r += 8) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD); d += 8) {
        ushort2 origin(d, r);
        simdgroup_matrix_storage<float> dO;
        dO.load(
          dO_block, \(leadingBlockDimensions.O), origin, \(transposeState.O));
        dV_sram[d / 8]
          .multiply(PT_sram[r / 8], dO, true);
      }
    }
    
"""
  }
  
  func accumulateDerivativeQ() -> String {
    return """

    auto K_block = (threadgroup float*)(threadgroup_block);
    K_block = simdgroup_matrix_storage<float>::apply_offset(
      K_block, \(leadingBlockDimensions.K), morton_offset, \(transposeState.K));

#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD); d += 8) {
        ushort2 origin(d, c);
        simdgroup_matrix_storage<float> K;
        K.load(
          K_block, \(leadingBlockDimensions.K), origin, \(transposeState.K));
        dQ_sram[d / 8]
          .multiply(dS_sram[c / 8], K, true);
      }
    }
    
"""
  }
}
