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
  
  /// Code that sets the various pointers into threadgroup memory.
  var reservePointers: String?
  
  /// Code for the inner loop, scoped over D.
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

        // Inner loop over D.
        ushort d_outer = d;
  #pragma clang loop unroll(full)
        for (ushort d = 0; d < min(32, \(paddedD) - d_outer); d += 8) {
          \(innerLoop)
        }
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
  func outerProduct(descriptor: AttentionOuterProductDescriptor) -> String {
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

"""
    
    return twoOperandAccess(descriptor: accessDesc)
  }
}

// MARK: - Attention Matrix Derivative

// TODO: Refactor 'dPT' to use the blocked algorithm.

extension AttentionKernel {
  func computeDerivativePT() -> String {
    return """
    
    auto dOT_block = \(blockDerivativeO());
    dOT_block = simdgroup_matrix_storage<float>::apply_offset(
      dOT_block, \(leadingBlockDimensions.O), morton_offset,
      \(!transposeState.O));
    
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
  
  func computeDerivativePT2() -> String {
    /*
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
     
     var accessDesc = AttentionTwoOperandAccessDescriptor()
     accessDesc.A = "dO"
     accessDesc.B = "O"
     accessDesc.transposeA = transposeState.O
     accessDesc.transposeB = transposeState.O
     accessDesc.leadingDimensionA = leadingDimensions.O
     accessDesc.leadingDimensionB = leadingDimensions.O
     accessDesc.matrixDimensions = (M: "R", N: "R")
     accessDesc.matrixOffset = (M: "gid * R_group", N: "gid * R_group")
     
     var accessDesc = AttentionHBMAccessDescriptor()
     accessDesc.index = "gid * C_group"
     accessDesc.leadingBlockDimension = leadingBlockDimensions.V
     accessDesc.leadingDimension = leadingDimensions.V
     accessDesc.name = "V"
     accessDesc.threadgroupAddress = "threadgroup_block"
     accessDesc.transposeState = transposeState.V
     */
    
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
      ushort2 A_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto V_block = (threadgroup float*)(threadgroup_block);
      V_block = simdgroup_matrix_storage<float>::apply_offset(
        V_block, 32, A_block_offset, \(transposeState.V));
      
      // Find where the dO data will be read from.
      ushort2 B_block_offset(morton_offset.x, morton_offset.y);
      auto dO_block = (threadgroup float*)(threadgroup_block) + \(32 * 32);
      dO_block = simdgroup_matrix_storage<float>::apply_offset(
        dO_block, 32, B_block_offset, \(!transposeState.O));

"""
    
    accessDesc.innerLoop = """

          simdgroup_matrix_storage<float> V;
          V.load(V_block, 32, ushort2(d, 0), \(transposeState.V));
          
          // Inner loop over N.
  #pragma clang loop unroll(full)
          for (ushort n = 0; n < 32; n += 8) {
            simdgroup_matrix_storage<float> dOT;
            dOT.load(dO_block, 32, ushort2(n, d), \(!transposeState.O));

            // Mask out the first accumulate at compile-time.
            bool accumulate = (d_outer > 0) || (d > 0);
            dPT_sram[n / 8].multiply(V, dOT, accumulate);
          }

"""
    
    return twoOperandAccess(descriptor: accessDesc)
  }
}
