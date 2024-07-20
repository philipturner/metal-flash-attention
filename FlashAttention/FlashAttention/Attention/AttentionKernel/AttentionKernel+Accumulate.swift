//
//  AttentionKernel+Accumulate.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Operations where one operand is the attention matrix, the other operand is
// read from RAM.

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
      for (ushort d = 0; d < D; d += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (sidx == 0) {
          uint2 B_offset(d, \(matrixOffsetK));
          auto src = simdgroup_matrix_storage<float>::apply_offset(
            \(B), \(leadingDimensionB), B_offset, \(transposeB));
          auto dst = (threadgroup float*)(threadgroup_block);
          
          ushort RC_src_dimension = min(
            uint(32), \(matrixDimensionK) - \(matrixOffsetK));
          ushort D_src_dimension = min(ushort(64), ushort(D - d));
          ushort D_dst_dimension = min(ushort(64), ushort(\(paddedD) - d));
          ushort2 tile_src(D_src_dimension, RC_src_dimension);
          ushort2 tile_dst(D_dst_dimension, 32); // excessive padding
          
          simdgroup_event event;
          event.async_copy(
            dst, \(leadingBlockDimensionB), tile_dst,
            src, \(leadingDimensionB), tile_src, \(transposeB));
          simdgroup_event::wait(1, &event);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Iterate over the row/column dimension.
#pragma clang loop unroll(full)
        for (ushort k = 0; k < 32; k += 8) {
          // Iterate over the head dimension.
          ushort d_outer = d;
          if (\(paddedD) - d_outer >= 64) {
#pragma clang loop unroll(full)
            for (ushort d = 0; d < 64; d += 8) {
              \(loopBodyAB)
            }
          } else {
#pragma clang loop unroll(full)
            for (ushort d = 0; d < \(paddedD) % 64; d += 8) {
              \(loopBodyAB)
            }
          }
        }
      }
    }

"""
  }
}

// MARK: - Attention Matrix Derivative

// A hybrid between 'accumulate' and 'outerProduct'. To more uniformly
// distribute code between 'Accumulate' and 'TwoOperandAccess', the function
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
    accessDesc.matrixOffset = (M: "gid * C_group", N: "r")
    
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
    
    let innerLoopDerivativeP = """

// Load the LHS from threadgroup memory.
ushort2 origin(d, 0);
simdgroup_matrix_storage<float> V;
V.load(V_block, 32, origin, \(transposeState.V));

// Inner loop over the column dimension.
#pragma clang loop unroll(full)
for (ushort c = 0; c < 32; c += 8) {
  // Load the RHS from threadgroup memory.
  ushort2 origin(c, d);
  simdgroup_matrix_storage<float> dOT;
  dOT.load(dOT_block, 32, origin, \(!transposeState.O));

  // Mask out the first accumulate at compile-time.
  bool accumulate = (d_outer > 0) || (d > 0);
  dPT_sram[c / 8].multiply(V, dOT, accumulate);
}

"""
    
    accessDesc.innerLoop = """

ushort d_outer = d;

// First multiplication: dV += P^T * dO
//
// Inner loop over the column dimension.
#pragma clang loop unroll(full)
for (ushort r = 0; r < 32; r += 8) {
  // Inner loop over the head dimension.
  if (\(paddedD) - d_outer >= 32) {
  #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      \(innerLoopDerivativeV)
    }
  } else {
  #pragma clang loop unroll(full)
    for (ushort d = 0; d < \(paddedD) % 32; d += 8) {
      \(innerLoopDerivativeV)
    }
  }
}

// Second multiplication: dP = V * dO^T
//
// Inner loop over the head dimension.
if (\(paddedD) - d_outer >= 32) {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < 32; d += 8) {
    \(innerLoopDerivativeP)
  }
} else {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < \(paddedD) % 32; d += 8) {
    \(innerLoopDerivativeP)
  }
}

"""
    
    return twoOperandAccess(descriptor: accessDesc)
  }
}
