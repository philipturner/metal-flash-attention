//
//  AttentionKernel+Accumulate.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// M x K x N
// parallelization x traversal x head

struct AttentionAccumulateDescriptor {
  var A: String?
  var B: String?
  var C: String?
  
  /// Optional. Factor to multiply every time the accumulator is loaded.
  var everyIterationScale: String?
  
  /// Optional. Factor to multiply, on the last iteration of the K dimension.
  var lastIterationScale: String?
}

extension AttentionKernel {
  func accumulate(
    descriptor accumulateDesc: AttentionAccumulateDescriptor
  ) -> String {
    guard let A = accumulateDesc.A,
          let B = accumulateDesc.B,
          let C = accumulateDesc.C else {
      fatalError("Descriptor was incomplete.")
    }
    
    // MARK: - Accumulator
    
    func allocateAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cached(C) else {
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
      for (ushort c = 0; c < \(descriptor.registerSize); c += 8) {
        \(C)_sram[(\(descriptor.registerOffset) + c) / 8] =
        simdgroup_matrix_storage<float>(0);
      }
      
      """
    }
    
    func declareAccumulatorLocation() -> String {
      guard !cached(C) else {
        return ""
      }
      return """
      
      // Where the \(C) data will be read from.
      ushort2 \(C)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto \(C)_block = (threadgroup float*)(threadgroup_block);
      \(C)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(C)_block, \(leadingBlockDimension(C)),
        \(C)_block_offset, \(transposed(C)));
      
      """
    }
    
    func loadAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cached(C) else {
        return ""
      }
      return """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 C_offset(d_outer, \(parallelizationOffset));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(C), \(leadingDimension(C)), C_offset, \(transposed(C)));
        auto dst = (threadgroup float*)(threadgroup_block);
      
        ushort D_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationOffset)));
        ushort2 tile(D_dimension, R_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimension(C)), tile,
          src, \(leadingDimension(C)), tile, \(transposed(C)));
        simdgroup_event::wait(1, &event);
      }
      
      \(declareAccumulatorLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Inner loop over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        ushort2 origin(d, 0);
        \(C)_sram[(\(descriptor.registerOffset) + d) / 8].load(
          \(C)_block, \(leadingBlockDimension(C)), origin, \(transposed(C)));
      }
      
      """
    }
    
    func scaleAccumulator(
      by scale: String?,
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard let scale else {
        return ""
      }
      return """
      
      // Inner loop over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        *(\(C)_sram[(\(descriptor.registerOffset) + d) / 8]
          .thread_elements()) *= \(scale);
      }
      
      """
    }
    
    func storeAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cached(C) else {
        return ""
      }
      return """
      
      \(declareAccumulatorLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Inner loop over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        ushort2 origin(d, 0);
        \(C)_sram[(\(descriptor.registerOffset) + d) / 8].store(
          \(C)_block, \(leadingBlockDimension(C)), origin, \(transposed(C)));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      if (sidx == 0) {
        uint2 \(C)_offset(d_outer, \(parallelizationOffset));
        auto src = (threadgroup float*)(threadgroup_block);
        auto dst = simdgroup_matrix_storage<float>::apply_offset(
          \(C), \(leadingDimension(C)), \(C)_offset, \(transposed(C)));
        
        ushort D_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationOffset)));
        ushort2 tile(D_dimension, R_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingDimension(C)), tile,
          src, \(leadingBlockDimension(C)), tile, \(transposed(C)));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    // MARK: - RHS
    
    func declareRHSLocation() -> String {
      """
      
      // Where the \(B) data will be read from.
      ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
      auto \(B)_block = (threadgroup float*)(threadgroup_block);
      \(B)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(B)_block, \(leadingBlockDimension(B)),
        \(B)_block_offset, \(transposed(B)));
      
      """
    }
    
    func loadRHS() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(B)_offset(d_outer, \(traversalOffset));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(B), \(leadingDimension(B)), \(B)_offset, \(transposed(B)));
        auto dst = (threadgroup float*)(threadgroup_block);
        
        ushort D_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort C_src_dimension = min(
          uint(\(blockDimensions.traversal)),
          uint(\(traversalDimension) - \(traversalOffset)));
        ushort C_dst_dimension = max(
          ushort(\(paddedTraversalBlockDimension)),
          ushort(C_src_dimension));
        ushort2 tile_src(D_dimension, C_src_dimension);
        ushort2 tile_dst(D_dimension, C_dst_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimension(B)), tile_dst,
          src, \(leadingDimension(B)), tile_src, \(transposed(B)));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    // MARK: - Matrix Multiplication
    
    func multiplyAB(
      traversalStart: String,
      traversalEnd: String,
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      // Inner loop over the traversal dimension.
      #pragma clang loop unroll(full)
      for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
        // Inner loop over the head dimension.
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          // Load the RHS from threadgroup memory.
          ushort2 origin(d, c);
          simdgroup_matrix_storage<float> \(B);
          \(B).load(
            \(B)_block, \(leadingBlockDimension(B)),
            origin, \(transposed(B)));
          
          // Issue one SIMD matmul instruction.
          \(C)_sram[(\(descriptor.registerOffset) + d) / 8].multiply(
            \(A)_sram[c / 8], \(B), /*accumulate=*/true);
        }
      }
      """
    }
    
    // MARK: - Outer Loop over Head Dimension
    
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
      if (\(traversalOffset) == 0) {
        \(initializeAccumulator(descriptor: iterationDesc))
      } else {
        \(loadAccumulator(descriptor: iterationDesc))
        \(scaleAccumulator(
            by: accumulateDesc.everyIterationScale,
            descriptor: iterationDesc))
      }
      
      // Load the right-hand side.
      \(loadRHS())
      \(declareRHSLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      \(multiplyAB(
          traversalStart: "0",
        traversalEnd: paddedTraversalBlockDimension,
          descriptor: iterationDesc))
      if (\(traversalOffset) + \(blockDimensions.traversal)
          < \(traversalDimension)) {
        \(multiplyAB(
            traversalStart: paddedTraversalBlockDimension,
            traversalEnd: "\(blockDimensions.traversal)",
            descriptor: iterationDesc))
      } else {
        \(scaleAccumulator(
            by: accumulateDesc.lastIterationScale,
            descriptor: iterationDesc))
      }
      
      // Store the accumulator.
      \(storeAccumulator(descriptor: iterationDesc))
      
      """
    }
    
    // Outer loop over the head dimension.
    var descriptor = LoopIterationDescriptor()
    if true {
      let loopEnd = paddedHeadDimension
      let loopEndFloor = loopEnd - loopEnd % blockDimensions.head
      descriptor.registerOffset = cached(C) ? "d_outer" : "0"
      
      // Add the first iterations.
      descriptor.registerSize = blockDimensions.head
      var output = """
      
      #pragma clang loop unroll(\(cached(C) ? "full" : "disable"))
      for (
        ushort d_outer = 0;
        d_outer < \(loopEndFloor);
        d_outer += \(blockDimensions.head)
      ) {
        \(loopIteration(descriptor: descriptor))
      }
      
      """
      
      // Add the last iteration, if unaligned.
      if loopEndFloor < loopEnd {
        descriptor.registerSize = loopEnd - loopEndFloor
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
      descriptor.registerSize = blockDimensions.head
      return """
      
      #pragma clang loop unroll(disable)
      for (
        ushort d_outer = 0;
        d_outer < \(headDimension);
        d_outer += \(blockDimensions.head)
      ) {
        \(loopIteration(descriptor: descriptor))
      }

      """
    }
  }
}
