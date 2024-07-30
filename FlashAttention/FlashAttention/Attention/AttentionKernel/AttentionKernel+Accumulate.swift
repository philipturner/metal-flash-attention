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
  func accumulate(
    descriptor accumulateDesc: AttentionAccumulateDescriptor
  ) -> String {
    guard let A = accumulateDesc.A,
          let B = accumulateDesc.B,
          let C = accumulateDesc.C,
          let cacheC = accumulateDesc.cacheC,
          let transposeState = accumulateDesc.transposeState,
          let leadingDimensions = accumulateDesc.leadingDimensions,
          let matrixDimensions = accumulateDesc.matrixDimensions,
          let matrixOffset = accumulateDesc.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    // Declare the block size along the D dimension.
    let leadingBlockDimensionB = transposeState.B ? 32 : blockDimensionD
    let leadingBlockDimensionC = transposeState.C ? 32 : blockDimensionD
    
    // MARK: - Accumulator
    
    func allocateAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cacheC else {
        return ""
      }
      return """
      
      // Where the \(C) data will be written to.
      simdgroup_matrix_storage<float>
      \(C)_sram[\(descriptor.registerSize) / 8];
      
      """
    }
    
    func initializeAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        \(C)_sram[(\(descriptor.registerOffset) + d) / 8] =
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
    
    func loadAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
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
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        ushort2 origin(d, 0);
        \(C)_sram[(\(descriptor.registerOffset) + d) / 8].load(
          \(C)_block, \(leadingBlockDimensionC), origin, \(transposeState.C));
      }
      
      """
    }
    
    func multiplyAccumulator(
      by factor: String?,
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard let factor else {
        return ""
      }
      return """
      
      // Iterate over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        *(O_sram[(\(descriptor.registerOffset) + d) / 8]
          .thread_elements()) *= \(factor);
      }
      
      """
    }
    
    func storeAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cacheC else {
        return ""
      }
      return """
      
      \(declareAccumulatorLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Iterate over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        ushort2 origin(d, 0);
        \(C)_sram[(\(descriptor.registerOffset) + d) / 8].store(
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
    
    // MARK: - Matrix Multiplication
    
    func multiplyAB(
      startK: String,
      endK: String,
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      // Iterate over the row/column dimension.
      #pragma clang loop unroll(full)
      for (ushort k = \(startK); k < \(endK); k += 8) {
        // Iterate over the head dimension.
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          // Load the RHS from threadgroup memory.
          ushort2 origin(d, k);
          simdgroup_matrix_storage<float> \(B);
          \(B).load(
            \(B)_block, \(leadingBlockDimensionB),
            origin, \(transposeState.B));
          
          // Issue one SIMD matmul instruction.
          \(C)_sram[(\(descriptor.registerOffset) + d) / 8].multiply(
            \(A)_sram[k / 8], \(B), /*accumulate=*/true);
        }
      }
      """
    }
    
    // MARK: - Loop Over Head Dimension
    
    struct LoopIterationDescriptor {
      var registerOffset: String = ""
      var registerSize: UInt16 = .zero
    }
    
    func loopIteration(
      descriptor iterationDesc: LoopIterationDescriptor
    ) -> String {
      """
      
      // Load the accumulator.
      \(allocateAccumulator(descriptor: iterationDesc))
      if (\(matrixOffset.K) == 0) {
        \(initializeAccumulator(descriptor: iterationDesc))
      } else {
        \(loadAccumulator(descriptor: iterationDesc))
        \(multiplyAccumulator(
            by: accumulateDesc.everyIterationFactor,
            descriptor: iterationDesc))
      }
      
      // Declare the remainder of the row/column dimension.
      const ushort K_remainder = (\(matrixDimensions.K) % 32 == 0)
        ? 32 : \(matrixDimensions.K) % 32;
      const ushort K_remainder_padded = (K_remainder + 7) / 8 * 8;
      
      // Load the right-hand side.
      \(loadRHS())
      \(declareRHSLocation())
      
      // Inner loop over K, the accumulation dimension.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      \(multiplyAB(
          startK: "0",
          endK: "K_remainder_padded",
          descriptor: iterationDesc))
      if (\(matrixOffset.K) + 32 < \(matrixDimensions.K)) {
        \(multiplyAB(
            startK: "K_remainder_padded", 
            endK: "32",
            descriptor: iterationDesc))
      } else {
        \(multiplyAccumulator(
            by: accumulateDesc.lastIterationFactor,
            descriptor: iterationDesc))
      }
      
      // Store the accumulator.
      \(storeAccumulator(descriptor: iterationDesc))
      
      """
    }
    
    // Branch on which form the loop should take.
    var descriptor = LoopIterationDescriptor()
    if cacheC {
      descriptor.registerOffset = "d_outer"
      descriptor.registerSize = blockDimensionD
      
      // Add the first iterations.
      let loopEndFloor = paddedD - paddedD % blockDimensionD
      var output = """
      
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
      descriptor.registerOffset = "0"
      
      // Future optimization: shorten the last loop iteration, if doing so
      // doesn't increase the register pressure.
      descriptor.registerSize = blockDimensionD
      
      return """
      
      #pragma clang loop unroll(disable)
      for (ushort d_outer = 0; d_outer < D; d_outer += \(blockDimensionD)) {
        \(loopIteration(descriptor: descriptor))
      }

      """
    }
  }
}
