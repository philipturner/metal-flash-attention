//
//  AttentionKernel+OuterProduct.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Operations where both arguments are read from RAM.

// MARK: - Two-Operand Access

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
      // Declare the remainder of the row/column dimension.
      ushort N_remainder = (\(matrixDimensions.N) % 32 == 0)
        ? 32 : \(matrixDimensions.N) % 32;
      ushort N_remainder_padded = (N_remainder + 7) / 8 * 8;
      
      uint M_offset = \(matrixOffset.M);
      uint N_offset = \(matrixOffset.N);
      ushort M_src_dimension = min(uint(32), \(matrixDimensions.M) - M_offset);
      ushort N_src_dimension = min(uint(32), \(matrixDimensions.N) - N_offset);
      ushort N_dst_dimension = max(N_remainder_padded, N_src_dimension);
      
      \(reservePointers)
      
      // Outer loop over D.
#pragma clang loop unroll(full)
      for (ushort d_outer = 0; d_outer < D; d_outer += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (d_outer == 0) {
          \(firstIterationLoading)
        }
        
        if (sidx == 0) {
          ushort D_src_dimension = min(ushort(32), ushort(D - d_outer));
          ushort D_dst_dimension = min(
            ushort(32), ushort(\(paddedD) - d_outer));
          
          // load \(A)[m]
          simdgroup_event events[2];
          {
            uint2 \(A)_offset(d_outer, M_offset);
            auto src = simdgroup_matrix_storage<float>::apply_offset(
              \(A), \(leadingDimensionA), \(A)_offset, \(transposeA));
            auto dst = (threadgroup float*)(threadgroup_block);
            
            ushort2 tile_src(D_src_dimension, M_src_dimension);
            ushort2 tile_dst(D_dst_dimension, M_src_dimension);
            events[0].async_copy(
              dst, 32, tile_dst,
              src, \(leadingDimensionA), tile_src, \(transposeA));
          }
          
          // load \(B)[n]
          {
            uint2 \(B)_offset(d_outer, N_offset);
            auto src = simdgroup_matrix_storage<float>::apply_offset(
              \(B), \(leadingDimensionB), \(B)_offset, \(transposeB));
            auto dst = (threadgroup float*)(threadgroup_block) + \(32 * 32);
            
            ushort2 tile_src(D_src_dimension, N_src_dimension);
            ushort2 tile_dst(D_dst_dimension, N_dst_dimension);
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
      ushort2 \(A)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto \(A)_block = (threadgroup float*)(threadgroup_block);
      \(A)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(A)_block, 32, \(A)_block_offset, \(transposeA));
      
      // Find where the \(B) data will be read from.
      ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
      auto \(B)T_block = (threadgroup float*)(threadgroup_block) + \(32 * 32);
      \(B)T_block = simdgroup_matrix_storage<float>::apply_offset(
        \(B)T_block, 32, \(B)_block_offset, \(!transposeB));

"""
    
    func innerLoopAB(startN: String, endN: String) -> String {
      let loopBody = """

// Load the LHS from threadgroup memory.
ushort2 origin(d, 0);
simdgroup_matrix_storage<float> \(A);
\(A).load(\(A)_block, 32, origin, \(transposeA));

// Inner loop over N.
#pragma clang loop unroll(full)
for (ushort n = \(startN); n < \(endN); n += 8) {
  // Load the RHS from threadgroup memory.
  ushort2 origin(n, d);
  simdgroup_matrix_storage<float> \(B)T;
  \(B)T.load(\(B)T_block, 32, origin, \(!transposeB));
  
  // Mask out the first accumulate at compile-time.
  bool accumulate = (d_outer > 0) || (d > 0);
  \(C)_sram[n / 8].multiply(\(A), \(B)T, accumulate);
}

"""
      
      return """

// Inner loop over D.
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

// Iterate over the row/column dimension.
\(innerLoopAB(startN: "0", endN: "N_remainder_padded"))
if (\(matrixOffset.N) + 32 < \(matrixDimensions.N)) {
  \(innerLoopAB(startN: "N_remainder_padded", endN: "32"))
}

"""
    
    return accessDesc
  }
}
