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
    
    let scaleFactor = "(1 / sqrt(float(D)))"
    
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
    simdgroup_matrix_storage<float> dS_sram[32 / 8];
#pragma clang loop unroll(full)
    for (ushort c = 0; c < 32; c += 8) {
      float2 P_elements = float2(*(P_sram[c / 8].thread_elements()));
      float2 dP_elements = float2(*(dP_sram[c / 8].thread_elements()));
      float2 dS_elements = dP_elements * \(scaleFactor) - D_term;
      dS_elements *= P_elements;
      *(dS_sram[c / 8].thread_elements()) = dS_elements;
    }

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
    \(prefetchQL())
    
    // S^T = K * Q^T
  }

"""
  }
}

// MARK: - Prefetching Operations

extension AttentionKernel {
  func prefetchQL() -> String {
    return """

    if (sidx == 0) {
      // Declare the Q source and destination.
      uint2 device_origin(0, r);
      auto Q_src = simdgroup_matrix_storage<float>::apply_offset(
        Q, \(leadingDimensions.Q), device_origin, \(transposeState.Q));
      auto Q_dst = (threadgroup float*)(threadgroup_block);
      
      // Locate the L destination, relative to the Q destination.
      auto Q_dst_end = Q_dst;
      if (\(transposeState.Q)) {
        // D x R, where R is the row stride.
        Q_dst_end += \(paddedD) * \(leadingBlockDimensions.Q);
      } else {
        // R x D, where D is the row stride.
        Q_dst_end += R_group * \(leadingDimensions.Q);
      }
      
      // Declare the L source and destination.
      auto L_src = L_terms + r;
      auto L_dst = (threadgroup float*)(Q_dst_end);
      
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
        L_dst, 1, ushort2(tile_dst.y, 0),
        L_src, 1, ushort2(tile_src.y, 0));
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

    // Prevent the zero padding from changing the values of 'm' and 'l'.
    if ((C % 32 != 0) && (c + 32 > C)) {
      const ushort remainder32 = uint(C % 32);
      const ushort remainder32_floor = remainder32 - ushort(remainder32 % 8);
      
#pragma clang loop unroll(full)
      for (ushort index = 0; index < 2; ++index) {
        if (morton_offset.x + index >= remainder32 - remainder32_floor) {
          auto S_elements = S_sram[remainder32_floor / 8].thread_elements();
          (*S_elements)[index] = -numeric_limits<float>::max();
        }
      }
#pragma clang loop unroll(full)
      for (ushort c = remainder32_floor + 8; c < 32; c += 8) {
        auto S_elements = S_sram[c / 8].thread_elements();
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
