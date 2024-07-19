//
//  AttentionKernel+Softmax.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// MARK: - L Terms and D Terms

// TODO: Refactor the D terms computation to use the blocked algorithm.

extension AttentionKernel {
  func computeLTerm() -> String {
    return """

  // Premultiplied by M_LOG2E_F.
  float L_term = m + fast::log2(l);
  
"""
  }
  
  func computeDTerm() -> String {
    return """

  float D_term = 0;
#pragma clang loop unroll(full)
  for (ushort d = 0; d < \(paddedD); d += 8) {
    float2 O_value = *(O_sram[d / 8].thread_elements());
    float2 dO_value = *(dO_sram[d / 8].thread_elements());
    D_term += O_value[0] * dO_value[0];
    D_term += O_value[1] * dO_value[1];
  }
  D_term += simd_shuffle_xor(D_term, 1);
  D_term += simd_shuffle_xor(D_term, 8);
  D_term *= 1 / sqrt(float(D));

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
