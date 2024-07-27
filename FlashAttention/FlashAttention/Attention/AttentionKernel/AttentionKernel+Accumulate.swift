//
//  AttentionKernel+Accumulate.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// Operations where the LHS is the attention matrix.

// MARK: - Accumulate

struct AttentionAccumulateDescriptor {
  /// Name of left-hand side register allocation (32 x 32).
  var A: String?
  
  /// Name of right-hand side, source of a 32 x D block.
  var B: String?
  
  /// Name of product register allocation (32 x D).
  var C: String?
  var cacheC: Bool?
  
  var transposeState: (B: Bool, C: Bool)?
  var leadingDimensions: (B: String, C: String)?
  var matrixDimensions: (M: String, K: String)?
  var matrixOffset: (M: String, K: String)?
  
  /// Optional. Factor to multiply every time the accumulator is loaded.
  var everyIterationFactor: String?
  
  /// Optional. Factor to multiply, on the last iteration of the K dimension.
  var lastIterationFactor: String?
}

extension AttentionKernel {
  func accumulate(descriptor: AttentionAccumulateDescriptor) -> String {
    guard let A = descriptor.A,
          let B = descriptor.B,
          let C = descriptor.C,
          let cacheC = descriptor.cacheC,
          let transposeState = descriptor.transposeState,
          let leadingDimensions = descriptor.leadingDimensions,
          let matrixDimensions = descriptor.matrixDimensions,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    // Declare the block size along the D dimension.
    let blockDimensionD: UInt16 = 64
    let leadingBlockDimensionB = transposeState.B ? 32 : blockDimensionD
    let leadingBlockDimensionC = transposeState.C ? 32 : blockDimensionD
    
    // MARK: - Accumulator
    
    func allocateAccumulator() -> String {
      guard !cacheC else {
        return ""
      }
      return """
      
      // Where the \(C) data will be written to.
      simdgroup_matrix_storage<float> \(C)_sram[\(blockDimensionD) / 8];
      
      """
    }
    
    func initializeAccumulator() -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(blockDimensionD); d += 8) {
        \(C)_sram[(d_register_start + d) / 8] =
        simdgroup_matrix_storage<float>(0);
      }
      
      """
    }
    
    func declareAccumulatorLocation() -> String {
      guard !cacheC else {
        return ""
      }
      return """
      
      // Where the \(C) data will be read from.
      ushort2 \(C)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto \(C)_block = (threadgroup float*)(threadgroup_block);
      \(C)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(C)_block, \(leadingBlockDimensionC),
        \(C)_block_offset, \(transposeState.C));
      
      """
    }
    
    func loadAccumulator() -> String {
      guard !cacheC else {
        return ""
      }
      return """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 C_offset(d_outer, \(matrixOffset.M));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(C), \(leadingDimensions.C), C_offset, \(transposeState.C));
        auto dst = (threadgroup float*)(threadgroup_block);
        
        // It doesn't matter if the rows below the matrix edge are garbage.
        ushort D_dimension = min(
          ushort(\(blockDimensionD)), ushort(D - d_outer));
        ushort M_dimension = min(
          uint(32), \(matrixDimensions.M) - \(matrixOffset.M));
        ushort2 tile(D_dimension, M_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimensionC), tile,
          src, \(leadingDimensions.C), tile, \(transposeState.C));
        simdgroup_event::wait(1, &event);
      }
      
      \(declareAccumulatorLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Iterate over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(blockDimensionD); d += 8) {
        ushort2 origin(d, 0);
        \(C)_sram[(d_register_start + d) / 8].load(
          \(C)_block, \(leadingBlockDimensionC),
          origin, \(transposeState.C));
      }
      
      """
    }
    
    func multiplyAccumulator(factor: String?) -> String {
      guard let factor else {
        return ""
      }
      return """
      
      // Iterate over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(blockDimensionD); d += 8) {
        *(O_sram[(d_register_start + d) / 8].thread_elements()) *= \(factor);
      }
      
      """
    }
    
    func storeAccumulator() -> String {
      guard !cacheC else {
        return ""
      }
      return """
      
      \(declareAccumulatorLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Iterate over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(blockDimensionD); d += 8) {
        ushort2 origin(d, 0);
        \(C)_sram[(d_register_start + d) / 8].store(
          \(C)_block, \(leadingBlockDimensionC), origin, \(transposeState.C));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      if (sidx == 0) {
        uint2 \(C)_offset(d_outer, \(matrixOffset.M));
        auto src = (threadgroup float*)(threadgroup_block);
        auto dst = simdgroup_matrix_storage<float>::apply_offset(
          \(C), \(leadingDimensions.C), \(C)_offset, \(transposeState.C));
        
        ushort D_dimension = min(
          ushort(\(blockDimensionD)), ushort(D - d_outer));
        ushort M_dimension = min(
          uint(32), \(matrixDimensions.M) - \(matrixOffset.M));
        ushort2 tile(D_dimension, M_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingDimensions.C), tile,
          src, \(leadingBlockDimensionC), tile, \(transposeState.C));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    // MARK: - RHS
    
    func declareRHSLocation() -> String {
      """
      
      // Find where the \(B) data will be read from.
      ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
      auto \(B)_block = (threadgroup float*)(threadgroup_block);
      \(B)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(B)_block, \(leadingBlockDimensionB),
        \(B)_block_offset, \(transposeState.B));
      
      """
    }
    
    func loadRHS() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(B)_offset(d_outer, \(matrixOffset.K));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(B), \(leadingDimensions.B), \(B)_offset, \(transposeState.B));
        auto dst = (threadgroup float*)(threadgroup_block);
        
        ushort D_dimension = min(
          ushort(\(blockDimensionD)), ushort(D - d_outer));
        ushort K_src_dimension = min(
          uint(32), \(matrixDimensions.K) - \(matrixOffset.K));
        ushort K_dst_dimension = max(K_remainder_padded, K_src_dimension);
        ushort2 tile_src(D_dimension, K_src_dimension);
        ushort2 tile_dst(D_dimension, K_dst_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimensionB), tile_dst,
          src, \(leadingDimensions.B), tile_src, \(transposeState.B));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    func multiplyAB(startK: String, endK: String) -> String {
      """
      // Iterate over the row/column dimension.
      #pragma clang loop unroll(full)
      for (ushort k = \(startK); k < \(endK); k += 8) {
        // Iterate over the head dimension.
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(blockDimensionD); d += 8) {
          // Load the RHS from threadgroup memory.
          ushort2 origin(d, k);
          simdgroup_matrix_storage<float> \(B);
          \(B).load(
            \(B)_block, \(leadingBlockDimensionB),
            origin, \(transposeState.B));
          
          // Add the contributions from the c-th/r-th element of the
          // attention matrix row/column.
          \(C)_sram[(d_register_start + d) / 8].multiply(
            \(A)_sram[k / 8], \(B), /*accumulate=*/true);
        }
      }
      """
    }
    
    // An iteration over the D dimension.
    let loopBodyD = """

\(allocateAccumulator())
if (\(matrixOffset.K) == 0) {
  \(initializeAccumulator())
} else {
  \(loadAccumulator())
  \(multiplyAccumulator(factor: descriptor.everyIterationFactor))
}

\(loadRHS())
\(declareRHSLocation())
threadgroup_barrier(mem_flags::mem_threadgroup);

\(multiplyAB(startK: "0", endK: "K_remainder_padded"))
if (\(matrixOffset.K) + 32 < \(matrixDimensions.K)) {
  \(multiplyAB(startK: "K_remainder_padded", endK: "32"))
} else {
  \(multiplyAccumulator(factor: descriptor.lastIterationFactor))
}

\(storeAccumulator())

"""
    
    return """

{
  // Declare the remainder of the row/column dimension.
  ushort K_remainder = (\(matrixDimensions.K) % 32 == 0)
    ? 32 : \(matrixDimensions.K) % 32;
  ushort K_remainder_padded = (K_remainder + 7) / 8 * 8;

  // Outer loop over D.
  #pragma clang loop unroll(\(cacheC ? "full" : "disable"))
  for (ushort d_outer = 0; d_outer < D; d_outer += \(blockDimensionD)) {
    ushort d_register_start = \(cacheC ? "d_outer" : "0");
    
    \(loopBodyD)
  }
}

"""
  }
}
