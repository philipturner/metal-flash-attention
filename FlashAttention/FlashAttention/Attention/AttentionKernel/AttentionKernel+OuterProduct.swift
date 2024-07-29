//
//  AttentionKernel+OuterProduct.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Operations where the accumulator is the attention matrix.

// MARK: - Outer Product

struct AttentionOuterProductDescriptor {
  /// Name of left-hand side, source of a 32 x D block.
  var A: String?
  var cacheA: Bool?
  
  /// Name of right-hand side, source of a 32 x D block.
  var B: String?
  
  /// Name of product register allocation (32 x 32).
  var C: String?
  
  var transposeState: (A: Bool, B: Bool)?
  var leadingDimensions: (A: String, B: String)?
  var matrixDimensions: (M: String, N: String)?
  var matrixOffset: (M: String, N: String)?
}

extension AttentionKernel {
  // Accepts the operands A and B, then performs the multiplication A * B^T.
  //
  // A and C are divided along four SIMDs in the M dimension. Each SIMD carries
  // out an (8 x D) x (D x 32) matrix multiplication. The product has
  // dimensions 8 (M dimension) x 32 (N dimension). The caller specifies which
  // attention matrix dimension (R, C) corresponds to N.
  func outerProduct(
    descriptor: AttentionOuterProductDescriptor
  ) -> String {
    guard let A = descriptor.A,
          let cacheA = descriptor.cacheA,
          let B = descriptor.B,
          let C = descriptor.C,
          let transposeState = descriptor.transposeState,
          let leadingDimensions = descriptor.leadingDimensions,
          let matrixDimensions = descriptor.matrixDimensions,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    // How do I incrementally transform this function into one very similar to
    // 'accumulate'?
    // - Start by extracting constants that are currently named in the shader
    //   source. This action breaks some dependencies between the code modules.
    // - Recompose the existing algorithm into modules + DSL.
    // - Migrate to a different algorithm, where the LHS is read into registers
    //   before the RHS is accessed.
    
    // Declare the block size along the D dimension.
    let blockDimensionD: UInt16 = 32
    let leadingBlockDimensionA = transposeState.A ? 32 : blockDimensionD
    let leadingBlockDimensionB = transposeState.B ? 32 : blockDimensionD
    
    // MARK: - LHS
    
    func declareLHSLocation() -> String {
      guard !cacheA else {
        return ""
      }
      return """
      
      // Find where the \(A) data will be read from.
      ushort2 \(A)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto \(A)_block = (threadgroup float*)(threadgroup_block);
      \(A)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(A)_block, \(leadingBlockDimensionA),
        \(A)_block_offset, \(transposeState.A));
      
      """
    }
    
    func loadLHS() -> String {
      guard !cacheA else {
        return ""
      }
      return """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(A)_offset(d_outer, \(matrixOffset.M));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(A), \(leadingDimensions.A), \(A)_offset, \(transposeState.A));
        auto dst = (threadgroup float*)(threadgroup_block);
        
        ushort D_src_dimension = min(
          ushort(\(blockDimensionD)), ushort(D - d_outer));
        ushort D_dst_dimension = min(
          ushort(\(blockDimensionD)), ushort(\(paddedD) - d_outer));
        ushort M_src_dimension = min(
          uint(32), \(matrixDimensions.M) - \(matrixOffset.M));
        ushort2 tile_src(D_src_dimension, M_src_dimension);
        ushort2 tile_dst(D_dst_dimension, M_src_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimensionA), tile_dst,
          src, \(leadingDimensions.A), tile_src, \(transposeState.A));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    // MARK: - RHS
    
    // 'offset' - The offset in threadgroup memory. Accomodates for the fact
    // that the current algorithm reads both operands at once.
    
    func declareRHSLocation() -> String {
      let offset: UInt16 = cacheA ? 0 : 32 * 32
      
      return """
      
      // Find where the \(B) data will be read from.
      ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
      auto \(B)T_block = (threadgroup float*)(threadgroup_block) + \(offset);
      \(B)T_block = simdgroup_matrix_storage<float>::apply_offset(
        \(B)T_block, \(leadingBlockDimensionB),
        \(B)_block_offset, \(!transposeState.B));
      
      """
    }
    
    func loadRHS() -> String {
      let offset: UInt16 = cacheA ? 0 : 32 * 32
      
      return """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(B)_offset(d_outer, \(matrixOffset.N));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(B), \(leadingDimensions.B), \(B)_offset, \(transposeState.B));
        auto dst = (threadgroup float*)(threadgroup_block) + \(offset);
        
        ushort D_src_dimension = min(
          ushort(\(blockDimensionD)), ushort(D - d_outer));
        ushort D_dst_dimension = min(
          ushort(\(blockDimensionD)), ushort(\(paddedD) - d_outer));
        ushort N_src_dimension = min(
          uint(32), \(matrixDimensions.N) - \(matrixOffset.N));
        ushort N_dst_dimension = max(
          N_remainder_padded, N_src_dimension);
        ushort2 tile_src(D_src_dimension, N_src_dimension);
        ushort2 tile_dst(D_dst_dimension, N_dst_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimensionB), tile_dst,
          src, \(leadingDimensions.B), tile_src, \(transposeState.B));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    func innerLoopAB(startN: String, endN: String) -> String {
      var loopBody: String
      if cacheA {
        loopBody = """

// Inner loop over N.
#pragma clang loop unroll(full)
for (ushort n = \(startN); n < \(endN); n += 8) {
  // Load the RHS from threadgroup memory.
  ushort2 origin(n, d);
  simdgroup_matrix_storage<float> \(B)T;
  \(B)T.load(
    \(B)T_block, \(leadingBlockDimensionB), origin, \(!transposeState.B));
  
  // Mask out the first accumulate at compile-time.
  bool accumulate = (d_outer > 0) || (d > 0);
  \(C)_sram[n / 8].multiply(
    \(A)_sram[(d_outer + d) / 8], \(B)T, accumulate);
}

"""
      } else {
        loopBody = """

// Load the LHS from threadgroup memory.
ushort2 origin(d, 0);
simdgroup_matrix_storage<float> \(A);
\(A).load(\(A)_block, 32, origin, \(transposeState.A));

// Inner loop over N.
#pragma clang loop unroll(full)
for (ushort n = \(startN); n < \(endN); n += 8) {
  // Load the RHS from threadgroup memory.
  ushort2 origin(n, d);
  simdgroup_matrix_storage<float> \(B)T;
  \(B)T.load(\(B)T_block, 32, origin, \(!transposeState.B));
  
  // Mask out the first accumulate at compile-time.
  bool accumulate = (d_outer > 0) || (d > 0);
  \(C)_sram[n / 8].multiply(\(A), \(B)T, accumulate);
}

"""
      }
      
      return """

// Inner loop over D.
if (D - d_outer >= \(blockDimensionD)) {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < \(blockDimensionD); d += 8) {
    \(loopBody)
  }
} else {
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D % \(blockDimensionD); d += 8) {
    \(loopBody)
  }
}

"""
    }
    
    // MARK: - Loop Over Head Dimension
    
    return """
    
    // Outer loop over D.
    #pragma clang loop unroll(\(cacheA ? "full" : "disable"))
    for (ushort d_outer = 0; d_outer < D; d_outer += \(blockDimensionD)) {
      \(loadLHS())
      \(declareLHSLocation())
      
      // Declare the remainder of the row/column dimension.
      ushort N_remainder = (\(matrixDimensions.N) % 32 == 0)
        ? 32 : \(matrixDimensions.N) % 32;
      ushort N_remainder_padded = (N_remainder + 7) / 8 * 8;
      \(loadRHS())
      \(declareRHSLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Inner loop over D.
      \(innerLoopAB(startN: "0", endN: "N_remainder_padded"))
      if (\(matrixOffset.N) + 32 < \(matrixDimensions.N)) {
        \(innerLoopAB(startN: "N_remainder_padded", endN: "32"))
      }
    }
    
    """
  }
}
