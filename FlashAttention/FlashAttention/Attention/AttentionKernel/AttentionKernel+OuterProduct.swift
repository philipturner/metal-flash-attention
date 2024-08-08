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
    
    func declareRHSLocation() -> String {
      """
      
      // Where the \(B) data will be read from.
      ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
      auto \(B)T_block = (threadgroup float*)(threadgroup_block);
      \(B)T_block = simdgroup_matrix_storage<float>::apply_offset(
        \(B)T_block, \(leadingBlockDimension(B)),
        \(B)_block_offset, \(!transposed(B)));
      
      """
    }
    
    func loadRHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
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
    
    // MARK: - Loop
    
    struct LoopIterationDescriptor {
      // Whether to accumulate in the SIMD matmul.
      var accumulateConditional: String = ""
      var registerOffset: String = ""
    }
    
    func innerLoopTraversal(
      traversalStart: String,
      traversalEnd: String,
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
        // Load the RHS from threadgroup memory.
        ushort2 origin(c, d);
        simdgroup_matrix_storage<float> \(B)T;
        \(B)T.load(
          \(B)T_block, \(leadingBlockDimension(B)),
          origin, \(!transposed(B)));
        
        // Issue one SIMD matmul instruction.
        \(C)_sram[c / 8].multiply(
          \(A)_sram[(\(descriptor.registerOffset) + d) / 8],
          \(B)T, \(descriptor.accumulateConditional));
      }
      
      """
    }
    
    func innerLoopHead(
      headStart: UInt16,
      headEnd: UInt16,
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = \(headStart); d < \(headEnd); d += 8) {
        \(innerLoopTraversal(
            traversalStart: "0",
            traversalEnd: paddedTraversalEdge,
            descriptor: iterationDesc))
        if (\(traversalOffset) + \(blockDimensions.traversal)
            < \(traversalDimension)) {
          \(innerLoopTraversal(
              traversalStart: paddedTraversalEdge,
              traversalEnd: "\(blockDimensions.traversal)",
              descriptor: iterationDesc))
        }
      }
      
      """
    }
    
    func loopIteration(
      descriptor iterationDesc: LoopIterationDescriptor
    ) -> String {
      """
      
      // Load the left-hand side.
      \(allocateLHS(descriptor: iterationDesc))
      \(loadLHS(descriptor: iterationDesc))
      
      // Load the right-hand side.
      \(loadRHS(descriptor: iterationDesc))
      \(declareRHSLocation())
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      \(innerLoopHead(
          headStart: 0,
          headEnd: paddedHeadEdge,
          descriptor: iterationDesc))
      if (d_outer + \(blockDimensions.head) < \(headDimension)) {
        \(innerLoopHead(
            headStart: paddedHeadEdge,
            headEnd: blockDimensions.head,
            descriptor: iterationDesc))
      }
      
      """
    }
    
    // Outer loop over the head dimension.
    var iterationDesc = LoopIterationDescriptor()
    
    if cached(A) {
      let loopEnd = paddedHeadDimension
      let loopEndFloor = loopEnd - loopEnd % blockDimensions.head
      iterationDesc.accumulateConditional = "((d_outer > 0) || (d > 0))"
      iterationDesc.registerOffset = "d_outer"
      
      // Add the first iterations.
      var output = """
      
      \(allocateAccumulator())
      
      #pragma clang loop unroll(full)
      for (
        ushort d_outer = 0;
        d_outer < \(loopEndFloor);
        d_outer += \(blockDimensions.head)
      ) {
        \(loopIteration(descriptor: iterationDesc))
      }
      
      """
      
      // Add the last iteration, if unaligned.
      if loopEndFloor < loopEnd {
        output += """
        {
          ushort d_outer = \(loopEndFloor);
          \(loopIteration(descriptor: iterationDesc))
        }
        """
      }
      
      return output
    } else {
      iterationDesc.accumulateConditional = "true"
      iterationDesc.registerOffset = "0"
      
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
        \(loopIteration(descriptor: iterationDesc))
      }
      
      """
    }
  }
}
