//
//  AttentionKernel+OuterProduct.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

// M x K x N
// parallelization x head x traversal

struct AttentionOuterProductDescriptor {
  var A: AttentionOperand?
  var B: AttentionOperand?
  var C: AttentionOperand?
}

extension AttentionKernel {
  func outerProduct(
    descriptor outerProductDesc: AttentionOuterProductDescriptor
  ) -> String {
    guard let A = outerProductDesc.A,
          let B = outerProductDesc.B,
          let C = outerProductDesc.C else {
      fatalError("Descriptor was incomplete.")
    }
    
    // MARK: - Accumulator
    
    func allocateAccumulator() -> String {
      """
      
      // Where the \(C) data will be written to.
      simdgroup_matrix_storage<float> \
      \(C)_sram[\(blockDimensions.traversal) / 8];
      
      """
    }
    
    func initializeAccumulator() -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort c = 0; c < \(blockDimensions.traversal); c += 8) {
        \(C)_sram[c / 8] = simdgroup_matrix_storage<float>(0);
      }
      
      """
    }
    
    // MARK: - LHS
    
    func allocateLHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cached(A) else {
        return ""
      }
      return """
      
      // Where the \(A) data will be written to.
      simdgroup_matrix_storage<float>
      \(A)_sram[\(blockDimensions.head) / 8];
      
      """
    }
    
    func declareLHSLocation() -> String {
      guard !cached(A) else {
        return ""
      }
      return """
      
      // Where the \(A) data will be read from.
      ushort2 \(A)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto \(A)_block = (threadgroup float*)(threadgroup_block);
      \(A)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(A)_block, \(leadingBlockDimension(A)),
        \(A)_block_offset, \(transposed(A)));
      
      """
    }
    
    func loadLHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cached(A) else {
        return ""
      }
      return """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(A)_offset(d_outer, \(parallelizationOffset));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(A), \(leadingDimension(A)), \(A)_offset, \(transposed(A)));
        auto dst = (threadgroup float*)(threadgroup_block);
        
        ushort D_src_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort D_dst_dimension = max(
          ushort(\(paddedHeadEdge)),
          ushort(D_src_dimension));
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationOffset)));
        ushort2 tile_src(D_src_dimension, R_dimension);
        ushort2 tile_dst(D_dst_dimension, R_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimension(A)), tile_dst,
          src, \(leadingDimension(A)), tile_src, \(transposed(A)));
        simdgroup_event::wait(1, &event);
      }
      
      \(declareLHSLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      // Inner loop over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
        ushort2 origin(d, 0);
        \(A)_sram[d / 8].load(
          \(A)_block, \(leadingBlockDimension(A)), origin, \(transposed(A)));
      }
      
      """
    }
    
    // MARK: - RHS
    
    // When does the code fall into each branch?
    // - direct
    //   - sequence length is in-bounds
    //   - head dimension is in-bounds
    // - async
    //   - any other situation
    //
    // Where can these checks be performed?
    // - before any of the code in this loop
    //   - if r/c falls out of bounds, or d_outer falls out of bounds
    // - duplicate the entire piece of code, just to handle the case
    //   with threadgroup memory instead of device memory
    //   - that means the access type should be an argument to every function
    //   - or part of the descriptor
    //
    // First step:
    // - if/else branch that just dives into the same exact code
    
    // NOTE: Affected by preferAsyncLoad
    // - omitted completely
    func loadRHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard descriptor.addressSpace == .threadgroup else {
        return ""
      }
      
      return """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(B)_offset(d_outer, \(traversalOffset));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(B), \(leadingDimension(B)), \(B)_offset, \(transposed(B)));
        auto dst = (threadgroup float*)(threadgroup_block);
        
        ushort D_src_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort D_dst_dimension = max(
          ushort(\(paddedHeadEdge)),
          ushort(D_src_dimension));
        ushort C_src_dimension = min(
          uint(\(blockDimensions.traversal)),
          uint(\(traversalDimension) - \(traversalOffset)));
        ushort C_dst_dimension = max(
          ushort(\(paddedTraversalEdge)),
          ushort(C_src_dimension));
        ushort2 tile_src(D_src_dimension, C_src_dimension);
        ushort2 tile_dst(D_dst_dimension, C_dst_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimension(B)), tile_dst,
          src, \(leadingDimension(B)), tile_src, \(transposed(B)));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    func leadingDimensionRHS(
      _ descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpace == .device {
        return leadingDimension(B)
      } else {
        return "\(leadingBlockDimension(B))"
      }
    }
    
    // Declares the source of the B matrix.
    //
    // Also guards against unsafe accesses to the declared pointer (barrier).
    func declareRHSLocation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpace == .device {
        return """
        
        uint2 \(B)_src_offset(
          morton_offset.y + d_outer,
          morton_offset.x + \(traversalOffset));
        auto \(B)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(B), \(leadingDimension(B)), \(B)_src_offset, \(transposed(B)));
        
        """
      } else {
        return """
        
        ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
        auto \(B)_src = (threadgroup float*)(threadgroup_block);
        \(B)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(B)_src, \(leadingBlockDimension(B)),
          \(B)_block_offset, \(!transposed(B)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        """
      }
    }
    
    // MARK: - Loop
    
    // NOTE: Affected by preferAsyncLoad
    // - might include an option for whether this branch is async
    struct LoopIterationDescriptor {
      // Whether to accumulate in the SIMD matmul.
      var accumulateConditional: String = ""
      var addressSpace: MTLAddressSpace = .threadgroup
      var registerOffset: String = ""
    }
    
    // NOTE: Affected by preferAsyncLoad
    // - leading dimension changes
    func innerLoopTraversal(
      traversalStart: String,
      traversalEnd: String,
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
        // Load the RHS from threadgroup memory.
        // loop type: \(descriptor.addressSpace)
        ushort2 origin(c, d);
        simdgroup_matrix_storage<float> \(B);
        \(B).load(
          \(B)_src, \(leadingDimensionRHS(descriptor)),
          origin, \(!transposed(B)));
        
        // Issue one SIMD matmul instruction.
        \(C)_sram[c / 8].multiply(
          \(A)_sram[(\(descriptor.registerOffset) + d) / 8],
          \(B), \(descriptor.accumulateConditional));
      }
      
      """
    }
    
    // NOTE: Affected by preferAsyncLoad
    func innerLoopHead(
      headStart: UInt16,
      headEnd: UInt16,
      descriptor: LoopIterationDescriptor
    ) -> String {
      // TODO: Remove the if statements for the device codepath.
      """
      
      // iteration type: \(descriptor.addressSpace)
      #pragma clang loop unroll(full)
      for (ushort d = \(headStart); d < \(headEnd); d += 8) {
        // iteration type: \(descriptor.addressSpace)
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
        }
      }
      
      """
    }
    
    // NOTE: Affected by preferAsyncLoad
    func loopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      // TODO: Remove the if statements for the device codepath.
      """
      
      // Load the left-hand side.
      // iteration type: \(descriptor.addressSpace)
      \(allocateLHS(descriptor: descriptor))
      \(loadLHS(descriptor: descriptor))
      
      // Load the right-hand side.
      // iteration type: \(descriptor.addressSpace)
      \(loadRHS(descriptor: descriptor))
      \(declareRHSLocation(descriptor: descriptor))
      
      // iteration type: \(descriptor.addressSpace)
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
      
      """
    }
    
    func gatedLoopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      var descriptorDevice = descriptor
      var descriptorThreadgroup = descriptor
      descriptorDevice.addressSpace = .device
      descriptorThreadgroup.addressSpace = .threadgroup
      
      let blockDim = blockDimensions.traversal
      let condition = """
      (\(traversalOffset) == 0) &&
      (d_outer == 0)
      """
      
      return """
      
      if (\(condition)) {
        \(loopIteration(descriptor: descriptorDevice))
      } else {
        \(loopIteration(descriptor: descriptorThreadgroup))
      }
      
      """
    }
    
    // Outer loop over the head dimension.
    var outerIterationDesc = LoopIterationDescriptor()
    
    if cached(A) {
      let loopEnd = paddedHeadDimension
      let loopEndFloor = loopEnd - loopEnd % blockDimensions.head
      outerIterationDesc.accumulateConditional = "((d_outer > 0) || (d > 0))"
      outerIterationDesc.registerOffset = "d_outer"
      
      // Add the first iterations.
      var output = """
      
      \(allocateAccumulator())
      
      #pragma clang loop unroll(full)
      for (
        ushort d_outer = 0;
        d_outer < \(loopEndFloor);
        d_outer += \(blockDimensions.head)
      ) {
        // TODO: Add the gated loop iteration here, after debugging.
        \(loopIteration(descriptor: outerIterationDesc))
      }
      
      """
      
      // Add the last iteration, if unaligned.
      if loopEndFloor < loopEnd {
        output += """
        {
          ushort d_outer = \(loopEndFloor);
          \(loopIteration(descriptor: outerIterationDesc))
        }
        """
      }
      
      return output
    } else {
      outerIterationDesc.accumulateConditional = "true"
      outerIterationDesc.registerOffset = "0"
      
      // Add all of the iterations.
      return """
      
      \(allocateAccumulator())
      \(initializeAccumulator())
      
      #pragma clang loop unroll(disable)
      for (
        ushort d_outer = 0;
        d_outer < \(headDimension);
        d_outer += \(blockDimensions.head)
      ) {
        \(gatedLoopIteration(descriptor: outerIterationDesc))
      }
      
      """
    }
  }
}
