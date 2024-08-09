//
//  AttentionKernel+Accumulate.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// M x K x N
// parallelization x traversal x head

struct AttentionAccumulateDescriptor {
  var A: AttentionOperand?
  var B: AttentionOperand?
  var C: AttentionOperand?
  
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
      \(C)_sram[\(blockDimensions.head) / 8];
      
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
      for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
        ushort2 origin(d, 0);
        \(C)_sram[d / 8].load(
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
      for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
        ushort2 origin(d, 0);
        \(C)_sram[d / 8].store(
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
          ushort(\(paddedTraversalEdge)),
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
    
    // MARK: - Loop
    
    struct LoopIterationDescriptor {
      var registerOffset: String = ""
      var registerSize: UInt16 = .zero
    }
    
    func innerLoopHead(
      headStart: UInt16,
      headEnd: UInt16,
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = \(headStart); d < \(headEnd); d += 8) {
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
      
      """
    }
    
    func innerLoopTraversal(
      traversalStart: String,
      traversalEnd: String,
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
        \(innerLoopHead(
            headStart: 0,
            headEnd: paddedHeadEdge,
            descriptor: descriptor))
        if (d_outer + \(blockDimensions.head) < \(headDimension)) {
          \(innerLoopHead(
              headStart: paddedHeadEdge,
              headEnd: blockDimensions.head,
              descriptor: descriptor))
        }
      }
      
      """
    }
    
    func loopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      // Load the accumulator.
      \(allocateAccumulator(descriptor: descriptor))
      if (\(traversalOffset) == 0) {
        \(initializeAccumulator(descriptor: descriptor))
      } else {
        \(loadAccumulator(descriptor: descriptor))
        \(scaleAccumulator(
            by: accumulateDesc.everyIterationScale,
            descriptor: descriptor))
      }
      
      // Load the right-hand side.
      \(loadRHS())
      \(declareRHSLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      \(innerLoopTraversal(
          traversalStart: "0",
          traversalEnd: paddedTraversalEdge,
          descriptor: descriptor))
      if (\(traversalOffset) + \(blockDimensions.traversal)
          < \(traversalDimension)) {
        \(innerLoopTraversal(
            traversalStart: paddedTraversalEdge,
            traversalEnd: "\(blockDimensions.traversal)",
            descriptor: descriptor))
      } else {
        \(scaleAccumulator(
            by: accumulateDesc.lastIterationScale,
            descriptor: descriptor))
      }
      
      // Store the accumulator.
      \(storeAccumulator(descriptor: descriptor))
      
      """
    }
    
    // Outer loop over the head dimension.
    var outerIterationDesc = LoopIterationDescriptor()
    if cached(C) {
      let loopEnd = paddedHeadDimension
      let loopEndFloor = loopEnd - loopEnd % blockDimensions.head
      outerIterationDesc.registerOffset = "d_outer"
      outerIterationDesc.registerSize = blockDimensions.head
      
      // Add the first iterations.
      var output = """
      
      #pragma clang loop unroll(full)
      for (
        ushort d_outer = 0;
        d_outer < \(loopEndFloor);
        d_outer += \(blockDimensions.head)
      ) {
        \(loopIteration(descriptor: outerIterationDesc))
      }
      
      """
      
      // Add the last iteration, if unaligned.
      if loopEndFloor < loopEnd {
        outerIterationDesc.registerSize = paddedHeadEdge
        output += """
        {
          ushort d_outer = \(loopEndFloor);
          \(loopIteration(descriptor: outerIterationDesc))
        }
        """
      }
      
      return output
    } else {
      outerIterationDesc.registerOffset = "0"
      outerIterationDesc.registerSize = blockDimensions.head
      
      // Add all of the iterations.
      return """
      
      #pragma clang loop unroll(disable)
      for (
        ushort d_outer = 0;
        d_outer < \(headDimension);
        d_outer += \(blockDimensions.head)
      ) {
        \(loopIteration(descriptor: outerIterationDesc))
      }

      """
    }
  }
}
