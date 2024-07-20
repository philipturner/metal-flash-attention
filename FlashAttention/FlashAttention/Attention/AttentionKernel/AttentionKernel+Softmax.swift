//
//  AttentionKernel+Softmax.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// MARK: - L Terms and D Terms

extension AttentionKernel {
  func computeLTerm() -> String {
    return """

  // Premultiplied by M_LOG2E_F.
  float L_term = m + fast::log2(l);
  
"""
  }
  
  func computeDTerm() -> String {
    var accessDesc = AttentionTwoOperandAccessDescriptor()
    accessDesc.A = "dO"
    accessDesc.B = "O"
    accessDesc.transposeA = transposeState.O
    accessDesc.transposeB = transposeState.O
    accessDesc.leadingDimensionA = leadingDimensions.O
    accessDesc.leadingDimensionB = leadingDimensions.O
    accessDesc.matrixDimensions = (M: "R", N: "R")
    accessDesc.matrixOffset = (M: "gid * R_group", N: "gid * R_group")
    
    accessDesc.reservePointers = """

      // Find where the dO data will be read from.
      ushort2 A_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto dO_block = (threadgroup float*)(threadgroup_block);
      dO_block = simdgroup_matrix_storage<float>::apply_offset(
        dO_block, 32, A_block_offset, \(transposeState.O));
      
      // Find where the O data will be read from.
      ushort2 B_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto O_block = (threadgroup float*)(threadgroup_block) + \(32 * 32);
      O_block = simdgroup_matrix_storage<float>::apply_offset(
        O_block, 32, B_block_offset, \(transposeState.O));

"""
    
    accessDesc.innerLoop = """

        // Inner loop over D.
        ushort d_outer = d;
#pragma clang loop unroll(full)
        for (ushort d = 0; d < min(32, \(paddedD) - d_outer); d += 8) {
          simdgroup_matrix_storage<float> dO;
          simdgroup_matrix_storage<float> O;
          dO.load(dO_block, 32, ushort2(d, 0), \(transposeState.O));
          O.load(O_block, 32, ushort2(d, 0), \(transposeState.O));

          float2 dO_value = *(dO.thread_elements());
          float2 O_value = *(O.thread_elements());
          D_term_accumulator += dO_value * O_value;
        }

"""
    
    let dOO = twoOperandAccess(descriptor: accessDesc)
    
    return """

  float2 D_term_accumulator(0);
  \(dOO)
  
  float D_term = D_term_accumulator[0] + D_term_accumulator[1];
  D_term += simd_shuffle_xor(D_term, 1);
  D_term += simd_shuffle_xor(D_term, 8);
  D_term *= 1 / sqrt(float(D));

"""
  }
  
  func blockLTerms() -> String {
    let offset = 2 * 32 * 32
    return "(threadgroup float*)(threadgroup_block) + (\(offset))"
  }
  
  func blockDTerms() -> String {
    let offset = 2 * 32 * 32 + 1 * 32
    return "(threadgroup float*)(threadgroup_block) + (\(offset))"
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
