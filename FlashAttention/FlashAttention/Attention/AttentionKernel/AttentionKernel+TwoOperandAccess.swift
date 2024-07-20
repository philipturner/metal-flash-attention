//
//  AttentionKernel+TwoOperandAccess.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Operations where both operands are read from RAM.

struct AttentionTwoOperandAccessDescriptor {
  /// Name of left-hand side, source of a 32 x D block.
  var A: String?
  
  /// Name of right-hand side, source of a 32 x D block.
  var B: String?
  
  var transposeA: Bool?
  var transposeB: Bool?
  var leadingDimensionA: String?
  var leadingDimensionB: String?
  var matrixDimensions: (M: String, N: String)?
  var matrixOffset: (M: String, N: String)?
  
  /// Required. Code that sets the various pointers into threadgroup memory.
  var reservePointers: String?
  
  /// Optional. Loading code to only execute on the first D iteration.
  var firstIterationLoading: String?
  
  /// Required. Code for the inner loop, scoped over D.
  var innerLoop: String?
}

extension AttentionKernel {
  func twoOperandAccess(
    descriptor: AttentionTwoOperandAccessDescriptor
  ) -> String {
    guard let A = descriptor.A,
          let B = descriptor.B,
          let transposeA = descriptor.transposeA,
          let transposeB = descriptor.transposeB,
          let leadingDimensionA = descriptor.leadingDimensionA,
          let leadingDimensionB = descriptor.leadingDimensionB,
          let matrixDimensions = descriptor.matrixDimensions,
          let matrixOffset = descriptor.matrixOffset,
          let reservePointers = descriptor.reservePointers,
          let innerLoop = descriptor.innerLoop else {
      fatalError("Descriptor was incomplete.")
    }
    
    let firstIterationLoading = descriptor.firstIterationLoading ?? ""
    
    return """
    {
      uint M_offset = \(matrixOffset.M);
      uint N_offset = \(matrixOffset.N);
      ushort M_src_dimension = min(uint(32), \(matrixDimensions.M) - M_offset);
      ushort N_src_dimension = min(uint(32), \(matrixDimensions.N) - N_offset);
      
      \(reservePointers)
      
      // Outer loop over D.
#pragma clang loop unroll(full)
      for (ushort d = 0; d < D; d += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (d == 0) {
          \(firstIterationLoading)
        }
        
        if (sidx == 0) {
          ushort D_src_dimension = min(ushort(32), ushort(D - d));
          ushort D_dst_dimension = min(ushort(32), ushort(\(paddedD) - d));
          
          // load \(A)[m]
          simdgroup_event events[2];
          {
            uint2 A_offset(d, M_offset);
            auto src = simdgroup_matrix_storage<float>::apply_offset(
              \(A), \(leadingDimensionA), A_offset, \(transposeA));
            auto dst = (threadgroup float*)(threadgroup_block);
            
            ushort2 tile_src(D_src_dimension, M_src_dimension);
            ushort2 tile_dst(D_dst_dimension, 32); // excessive padding
            events[0].async_copy(
              dst, 32, tile_dst,
              src, \(leadingDimensionA), tile_src, \(transposeA));
          }
          
          // load \(B)[n]
          {
            uint2 B_offset(d, N_offset);
            auto src = simdgroup_matrix_storage<float>::apply_offset(
              \(B), \(leadingDimensionB), B_offset, \(transposeB));
            auto dst = (threadgroup float*)(threadgroup_block) + \(32 * 32);
            
            ushort2 tile_src(D_src_dimension, N_src_dimension);
            ushort2 tile_dst(D_dst_dimension, 32); // excessive padding
            events[1].async_copy(
              dst, 32, tile_dst,
              src, \(leadingDimensionB), tile_src, \(transposeB));
          }
          simdgroup_event::wait(2, events);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        \(innerLoop)
      }
    }

"""
  }
}

// MARK: - Outer Product

struct AttentionOuterProductDescriptor {
  /// Name of left-hand side, source of a 32 x D block.
  var A: String?
  
  /// Name of right-hand side, source of a 32 x D block.
  var B: String?
  
  /// Name of product register allocation (32 x 32).
  var C: String?
  
  var transposeA: Bool?
  var transposeB: Bool?
  var leadingDimensionA: String?
  var leadingDimensionB: String?
  var matrixDimensions: (M: String, N: String)?
  var matrixOffset: (M: String, N: String)?
}

extension AttentionKernel {
  // Accepts the operands A and B, then performs the multiplication A * B^T.
  //
  // A and C are divided along four SIMDs in the M dimension. Each SIMD carries
  // out an (8 x D) x (D x 32) matrix multiplication. The product has
  // dimensions 8 (M dimension) x 32 (N dimension). The caller specifies which
  // attention dimension (R, C) corresponds to N.
  //
  // Returns: Another descriptor, which you can intercept before materializing
  // the final source code.
  func outerProduct(
    descriptor: AttentionOuterProductDescriptor
  ) -> AttentionTwoOperandAccessDescriptor {
    guard let A = descriptor.A,
          let B = descriptor.B,
          let C = descriptor.C,
          let transposeA = descriptor.transposeA,
          let transposeB = descriptor.transposeB,
          let leadingDimensionA = descriptor.leadingDimensionA,
          let leadingDimensionB = descriptor.leadingDimensionB,
          let matrixDimensions = descriptor.matrixDimensions,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    var accessDesc = AttentionTwoOperandAccessDescriptor()
    accessDesc.A = A
    accessDesc.B = B
    accessDesc.transposeA = transposeA
    accessDesc.transposeB = transposeB
    accessDesc.leadingDimensionA = leadingDimensionA
    accessDesc.leadingDimensionB = leadingDimensionB
    accessDesc.matrixDimensions = matrixDimensions
    accessDesc.matrixOffset = matrixOffset
    
    accessDesc.reservePointers = """

      // Find where the \(A) data will be read from.
      ushort2 A_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto \(A)_block = (threadgroup float*)(threadgroup_block);
      \(A)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(A)_block, 32, A_block_offset, \(transposeA));
      
      // Find where the \(B) data will be read from.
      ushort2 B_block_offset(morton_offset.x, morton_offset.y);
      auto \(B)T_block = (threadgroup float*)(threadgroup_block) + \(32 * 32);
      \(B)T_block = simdgroup_matrix_storage<float>::apply_offset(
        \(B)T_block, 32, B_block_offset, \(!transposeB));

"""
    
    accessDesc.innerLoop = """

        // Inner loop over D.
        ushort d_outer = d;
#pragma clang loop unroll(full)
        for (ushort d = 0; d < min(32, \(paddedD) - d_outer); d += 8) {
          simdgroup_matrix_storage<float> \(A);
          \(A).load(\(A)_block, 32, ushort2(d, 0), \(transposeA));
          
          // Inner loop over N.
#pragma clang loop unroll(full)
          for (ushort n = 0; n < 32; n += 8) {
            simdgroup_matrix_storage<float> \(B)T;
            \(B)T.load(\(B)T_block, 32, ushort2(n, d), \(!transposeB));

            // Mask out the first accumulate at compile-time.
            bool accumulate = (d_outer > 0) || (d > 0);
            \(C)_sram[n / 8].multiply(\(A), \(B)T, accumulate);
          }
        }

"""
    
    return accessDesc
  }
}

// MARK: - Attention Matrix Derivative

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
    
    accessDesc.innerLoop = """
        
        ushort d_outer = d;
        
        // First multiplication: dV += P^T * dO
        //
        // Inner loop over the column dimension.
#pragma clang loop unroll(full)
        for (ushort r = 0; r < 32; r += 8) {
          // Inner loop over the head dimension.
#pragma clang loop unroll(full)
          for (ushort d = 0; d < min(32, \(paddedD) - d_outer); d += 8) {
            // Load the RHS from threadgroup memory.
            simdgroup_matrix_storage<float> dO;
            dO.load(
              dO_block, 32, ushort2(d, r), \(transposeState.O));
            
            // Add the contributions from the r-th element of the attention
            // matrix column.
            dV_sram[(d_outer + d) / 8].multiply(
              PT_sram[r / 8], dO, /*accumulate=*/true);
          }
        }
        
        // Second multiplication: dP = V * dO^T
        //
        // Inner loop over the head dimension.
#pragma clang loop unroll(full)
        for (ushort d = 0; d < min(32, \(paddedD) - d_outer); d += 8) {
          simdgroup_matrix_storage<float> V;
          V.load(V_block, 32, ushort2(d, 0), \(transposeState.V));
          
          // Inner loop over the column dimension.
#pragma clang loop unroll(full)
          for (ushort c = 0; c < 32; c += 8) {
            simdgroup_matrix_storage<float> dOT;
            dOT.load(dOT_block, 32, ushort2(c, d), \(!transposeState.O));

            // Mask out the first accumulate at compile-time.
            bool accumulate = (d_outer > 0) || (d > 0);
            dPT_sram[c / 8].multiply(V, dOT, accumulate);
          }
        }

"""
    
    return twoOperandAccess(descriptor: accessDesc)
  }
}
