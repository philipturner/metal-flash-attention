//
//  AttentionKernel+Accumulate.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Operations where one argument is the attention matrix, the other argument is
// read from RAM.

// MARK: - Accumulate

struct AttentionAccumulateDescriptor {
  /// Name of left-hand side register allocation (32 x 32).
  var A: String?
  
  /// Name of right-hand side, source of a 32 x D block.
  var B: String?
  
  /// Name of product register allocation (32 x D).
  var C: String?
  
  var transposeB: Bool?
  var leadingDimensionB: String?
  
  // M = 32 (assuming four SIMDs)
  // N = D
  // K = specified by caller
  var matrixDimensionK: String?
  var matrixOffsetK: String?
}

extension AttentionKernel {
  func accumulate(descriptor: AttentionAccumulateDescriptor) -> String {
    guard let A = descriptor.A,
          let B = descriptor.B,
          let C = descriptor.C,
          let transposeB = descriptor.transposeB,
          let leadingDimensionB = descriptor.leadingDimensionB,
          let matrixDimensionK = descriptor.matrixDimensionK,
          let matrixOffsetK = descriptor.matrixOffsetK else {
      fatalError("Descriptor was incomplete.")
    }
    
    // 32 x 64 allocation in threadgroup memory
    // leading dimension = transposeB ? 32 : 64
    let leadingBlockDimensionB = transposeB ? UInt16(32) : UInt16(64)
    
    let loopBodyAB = """

ushort2 origin(d, k);

// Load the RHS from threadgroup memory.
simdgroup_matrix_storage<float> \(B);
\(B).load(
  \(B)_block, \(leadingBlockDimensionB), origin, \(transposeB));

// Add the contributions from the c-th/r-th element of the
// attention matrix row/column.
\(C)_sram[(d_outer + d) / 8].multiply(
  \(A)_sram[k / 8], \(B), /*accumulate=*/true);

"""
    
    let innerLoopAB = """

// Iterate over the head dimension.
if (D - d_outer >= 64) {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < 64; d += 8) {
    \(loopBodyAB)
  }
} else {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D % 64; d += 8) {
    \(loopBodyAB)
  }
}

"""
    
    return """
    {
      // Find where the \(B) data will be read from.
      ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
      auto \(B)_block = (threadgroup float*)(threadgroup_block);
      \(B)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(B)_block, \(leadingBlockDimensionB),
        \(B)_block_offset, \(transposeB));
      
      // Outer loop over D.
#pragma clang loop unroll(full)
      for (ushort d_outer = 0; d_outer < D; d_outer += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Declare the remainder of the row/column dimension.
        ushort K_remainder = (\(matrixDimensionK) % 32 == 0)
          ? 32 : \(matrixDimensionK) % 32;
        ushort K_remainder_padded = (K_remainder + 7) / 8 * 8;
        
        if (sidx == 0) {
          uint2 B_offset(d_outer, \(matrixOffsetK));
          auto src = simdgroup_matrix_storage<float>::apply_offset(
            \(B), \(leadingDimensionB), B_offset, \(transposeB));
          auto dst = (threadgroup float*)(threadgroup_block);
          
          ushort K_src_dimension = min(
            uint(32), \(matrixDimensionK) - \(matrixOffsetK));
          ushort K_dst_dimension = max(K_remainder_padded, K_src_dimension);
          ushort D_src_dimension = min(ushort(64), ushort(D - d_outer));
          ushort2 tile_src(D_src_dimension, K_src_dimension);
          ushort2 tile_dst(D_src_dimension, K_dst_dimension);
          
          simdgroup_event event;
          event.async_copy(
            dst, \(leadingBlockDimensionB), tile_dst,
            src, \(leadingDimensionB), tile_src, \(transposeB));
          simdgroup_event::wait(1, &event);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Iterate over the row/column dimension.
  #pragma clang loop unroll(full)
        for (ushort k = 0; k < K_remainder_padded; k += 8) {
          \(innerLoopAB)
        }
        if ((K_remainder_padded == 32) ||
            (\(matrixOffsetK) + 32 <= \(matrixDimensionK))) {
  #pragma clang loop unroll(full)
          for (ushort k = K_remainder_padded; k < 32; k += 8) {
            \(innerLoopAB)
          }
        }
      }
    }

"""
  }
}

// MARK: - Attention Matrix Derivative

// A hybrid between 'accumulate' and 'outerProduct'. To more uniformly
// distribute code between 'Accumulate' and 'OuterProduct', the function
// is placed here.

extension AttentionKernel {
  func computeDerivativeVDerivativePT() -> String {
    var accessDesc = AttentionTwoOperandAccessDescriptor()
    accessDesc.A = "V"
    accessDesc.B = "dO"
    accessDesc.transposeA = transposeState.V
    accessDesc.transposeB = transposeState.O
    accessDesc.leadingDimensionA = leadingDimensions.V
    accessDesc.leadingDimensionB = leadingDimensions.O
    accessDesc.matrixDimensions = (M: "C", N: "R")
    accessDesc.matrixOffset = (M: "gid * 32", N: "r")
    
    accessDesc.reservePointers = """

      // Find where the V data will be read from.
      ushort2 V_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto V_block = (threadgroup float*)(threadgroup_block);
      V_block = simdgroup_matrix_storage<float>::apply_offset(
        V_block, 32, V_block_offset, \(transposeState.V));
      
      // Find where the dO data will be read from.
      ushort2 dO_block_offset(morton_offset.x, morton_offset.y);
      auto dO_block = (threadgroup float*)(threadgroup_block) + \(32 * 32);
      dO_block = simdgroup_matrix_storage<float>::apply_offset(
        dO_block, 32, dO_block_offset, \(transposeState.O));
      
      // Find where the dO^T data will be read from.
      ushort2 dOT_block_offset(morton_offset.x, morton_offset.y);
      auto dOT_block = (threadgroup float*)(threadgroup_block) + \(32 * 32);
      dOT_block = simdgroup_matrix_storage<float>::apply_offset(
        dOT_block, 32, dOT_block_offset, \(!transposeState.O));
    
"""
    
    let innerLoopDerivativeV = """

// Load the RHS from threadgroup memory.
ushort2 origin(d, r);
simdgroup_matrix_storage<float> dO;
dO.load(dO_block, 32, origin, \(transposeState.O));

// Add the contributions from the r-th element of the attention
// matrix column.
dV_sram[(d_outer + d) / 8].multiply(
  PT_sram[r / 8], dO, /*accumulate=*/true);

"""
    
    func innerLoopDerivativeP(startN: String, endN: String) -> String {
      let loopBody = """

// Load the LHS from threadgroup memory.
ushort2 origin(d, 0);
simdgroup_matrix_storage<float> V;
V.load(V_block, 32, origin, \(transposeState.V));

// Inner loop over the row dimension.
#pragma clang loop unroll(full)
for (ushort r = 0; r < 32; r += 8) {
  // Load the RHS from threadgroup memory.
  ushort2 origin(r, d);
  simdgroup_matrix_storage<float> dOT;
  dOT.load(dOT_block, 32, origin, \(!transposeState.O));

  // Mask out the first accumulate at compile-time.
  bool accumulate = (d_outer > 0) || (d > 0);
  dPT_sram[r / 8].multiply(V, dOT, accumulate);
}

"""
      
      return """

if (D - d_outer >= 32) {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < 32; d += 8) {
    \(loopBody)
  }
} else {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D % 32; d += 8) {
    \(loopBody)
  }
}

"""
    }
    
    accessDesc.innerLoop = """

// First multiplication: dV += P^T * dO
//
// Inner loop over the row dimension.
#pragma clang loop unroll(full)
for (ushort r = 0; r < 32; r += 8) {
  // Inner loop over the head dimension.
  if (D - d_outer >= 32) {
  #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      \(innerLoopDerivativeV)
    }
  } else {
  #pragma clang loop unroll(full)
    for (ushort d = 0; d < D % 32; d += 8) {
      \(innerLoopDerivativeV)
    }
  }
}

// Second multiplication: dP = V * dO^T
//
// Inner loop over the head dimension.
\(innerLoopDerivativeP(startN: "0", endN: "32"))
"""
    
    return twoOperandAccess(descriptor: accessDesc)
  }
}
