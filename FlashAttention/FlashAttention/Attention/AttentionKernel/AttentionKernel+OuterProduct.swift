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
    descriptor outerProductDesc: AttentionOuterProductDescriptor
  ) -> String {
    guard let A = outerProductDesc.A,
          let cacheA = outerProductDesc.cacheA,
          let B = outerProductDesc.B,
          let C = outerProductDesc.C,
          let transposeState = outerProductDesc.transposeState,
          let leadingDimensions = outerProductDesc.leadingDimensions,
          let matrixDimensions = outerProductDesc.matrixDimensions,
          let matrixOffset = outerProductDesc.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    // How do I incrementally transform this function into one very similar to
    // 'accumulate'?
    // - Start by extracting constants that are currently named in the shader
    //   source. This action breaks some dependencies between the code modules.
    // - Recompose the existing algorithm into modules + DSL.
    // - Migrate to a different algorithm, where the LHS is read into registers
    //   before the RHS is accessed.
    
    // Declare the block dimensions.
    let leadingBlockDimensionA = transposeState.A ? 32 : blockDimensionD
    let leadingBlockDimensionB = transposeState.B ? 32 : blockDimensionD
    
    // MARK: - Accumulator
    
    func allocateAccumulator() -> String {
      """
      
      // Where the \(C) data will be written to.
      simdgroup_matrix_storage<float> \(C)_sram[32 / 8];
      
      """
    }
    
    // Future optimization: when the loop is unrolled, fuse the zero
    // initialization with a non-accumulating SIMD matmul. This optimization
    // may apply to the case where A isn't cached. Provided, it doesn't harm
    // the register pressure.
    func initializeAccumulator() -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort n = 0; n < 32; n += 8) {
        \(C)_sram[n / 8] = simdgroup_matrix_storage<float>(0);
      }
      
      """
    }
    
    // MARK: - LHS
    
    func allocateLHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cacheA else {
        return ""
      }
      return """
      
      // Where the \(A) data will be written to.
      simdgroup_matrix_storage<float>
      \(A)_sram[\(descriptor.registerSize) / 8];
      
      """
    }
    
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
    
    func loadLHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
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
        ushort D_dst_dimension = \(descriptor.registerSize);
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
      
      \(declareLHSLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Iterate over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        ushort2 origin(d, 0);
        \(A)_sram[(\(descriptor.registerOffset) + d) / 8].load(
          \(A)_block, \(leadingBlockDimensionA), origin, \(transposeState.A));
      }
      
      """
    }
    
    // MARK: - RHS
    
    // 'offset' - The offset in threadgroup memory. Accomodates for the fact
    // that the current algorithm reads both operands at once.
    
    func declareRHSLocation() -> String {
      """
      
      // Find where the \(B) data will be read from.
      ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
      auto \(B)T_block = (threadgroup float*)(threadgroup_block);
      \(B)T_block = simdgroup_matrix_storage<float>::apply_offset(
        \(B)T_block, \(leadingBlockDimensionB),
        \(B)_block_offset, \(!transposeState.B));
      
      """
    }
    
    func loadRHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(B)_offset(d_outer, \(matrixOffset.N));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(B), \(leadingDimensions.B), \(B)_offset, \(transposeState.B));
        auto dst = (threadgroup float*)(threadgroup_block);
        
        ushort D_src_dimension = min(
          ushort(\(blockDimensionD)), ushort(D - d_outer));
        ushort D_dst_dimension = \(descriptor.registerSize);
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
    
    // MARK: - Matrix Multiplication
    
    func multiplyAB(
      startN: String,
      endN: String,
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      // Inner loop over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        // Inner loop over the row/column dimension.
        #pragma clang loop unroll(full)
        for (ushort n = \(startN); n < \(endN); n += 8) {
          // Load the RHS from threadgroup memory.
          ushort2 origin(n, d);
          simdgroup_matrix_storage<float> \(B)T;
          \(B)T.load(
            \(B)T_block, \(leadingBlockDimensionB),
            origin, \(!transposeState.B));
          
          // Issue one SIMD matmul instruction.
          \(C)_sram[n / 8].multiply(
            \(A)_sram[(\(descriptor.registerOffset) + d) / 8],
            \(B)T, /*accumulate=*/\(descriptor.accumulateConditional));
        }
      }
      
      """
    }
    
    // MARK: - Loop Over Head Dimension
    
    struct LoopIterationDescriptor {
      // Whether to accumulate in the SIMD matmul.
      var accumulateConditional: String = ""
      var registerOffset: String = ""
      var registerSize: UInt16 = .zero
    }
    
    func loopIteration(
      descriptor iterationDesc: LoopIterationDescriptor
    ) -> String {
      """
      
      // Load the left-hand side.
      \(allocateLHS(descriptor: iterationDesc))
      \(loadLHS(descriptor: iterationDesc))
      
      // Declare the remainder of the row/column dimension.
      ushort N_remainder = (\(matrixDimensions.N) % 32 == 0)
        ? 32 : \(matrixDimensions.N) % 32;
      ushort N_remainder_padded = (N_remainder + 7) / 8 * 8;
      
      // Load the right-hand side.
      \(loadRHS(descriptor: iterationDesc))
      \(declareRHSLocation())
      
      // Inner loop over D, the accumulation dimension.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      \(multiplyAB(
          startN: "0",
          endN: "N_remainder_padded",
          descriptor: iterationDesc))
      if (\(matrixOffset.N) + 32 < \(matrixDimensions.N)) {
        \(multiplyAB(
            startN: "N_remainder_padded",
            endN: "32",
            descriptor: iterationDesc))
      }
      
      """
    }
    
    var output = allocateAccumulator()
    var descriptor = LoopIterationDescriptor()
    if cacheA {
      descriptor.accumulateConditional = "true"
      descriptor.registerOffset = "d_outer"
      descriptor.registerSize = blockDimensionD
      
      // Add the first iterations.
      let paddedD = (matrixDimensionD + 8 - 1) / 8 * 8
      let loopEndFloor = paddedD - paddedD % blockDimensionD
      output += """
      
      \(initializeAccumulator())
      
      #pragma clang loop unroll(full)
      for (
        ushort d_outer = 0;
        d_outer < \(loopEndFloor);
        d_outer += \(blockDimensionD)
      ) {
        \(loopIteration(descriptor: descriptor))
      }
      
      """
      
      // Add the last iteration, if unaligned.
      if loopEndFloor < paddedD {
        descriptor.registerOffset = "\(loopEndFloor)"
        descriptor.registerSize = paddedD - loopEndFloor
        
        output += """
        {
          ushort d_outer = \(loopEndFloor);
          \(loopIteration(descriptor: descriptor))
        }
        """
      }
      
      return output
    } else {
      descriptor.accumulateConditional = "true"
      descriptor.registerOffset = "0"
      
      // Future optimization: shorten the last loop iteration, if doing so
      // doesn't increase the register pressure.
      descriptor.registerSize = blockDimensionD
      
      output += """
      
      \(initializeAccumulator())
      
      #pragma clang loop unroll(disable)
      for (ushort d_outer = 0; d_outer < D; d_outer += \(blockDimensionD)) {
        \(loopIteration(descriptor: descriptor))
      }
      
      """
    }
    return output
  }
}
