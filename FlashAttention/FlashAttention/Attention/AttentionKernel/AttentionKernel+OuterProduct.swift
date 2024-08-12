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
    
    // MARK: - Initialize
    
    func allocateAccumulator() -> String {
      """
      
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
    
    func allocateLHS() -> String {
      guard !cached(A) else {
        return ""
      }
      return """
      
      simdgroup_matrix_storage<float> \
      \(A)_sram[\(blockDimensions.head) / 8];
      
      """
    }
    
    // MARK: - Load LHS
    
    func declareLHSLocation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceLHS! {
      case .device:
        return """
        
        uint2 \(A)_src_offset(
          morton_offset.x + d_outer,
          \(clampedParallelizationThreadOffset));
        auto \(A)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(A), \(leadingDimension(A)), \(A)_src_offset, \(transposed(A)));
        
        """
      case .threadgroup:
        return """
        
        ushort2 \(A)_block_offset(
          morton_offset.x, 
          morton_offset.y + sidx * 8);
        auto \(A)_src = (threadgroup float*)(threadgroup_block);
        \(A)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(A)_src, \(leadingBlockDimension(A)),
          \(A)_block_offset, \(transposed(A)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        """
      }
    }
    
    func asyncLoadLHS() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(A)_offset(d_outer, \(parallelizationGroupOffset));
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
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
        ushort2 tile_src(D_src_dimension, R_dimension);
        ushort2 tile_dst(D_dst_dimension, R_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimension(A)), tile_dst,
          src, \(leadingDimension(A)), tile_src, \(transposed(A)));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    func loadLHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cached(A) else {
        return ""
      }
      
      switch descriptor.addressSpaceLHS! {
      case .device:
        return """
        
        \(declareLHSLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
          ushort2 origin(d, 0);
          \(A)_sram[d / 8].load(
            \(A)_src, \(leadingDimension(A)), origin, \(transposed(A)));
        }
        
        """
      case .threadgroup:
        return """
        
        \(asyncLoadLHS())
        \(declareLHSLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
          ushort2 origin(d, 0);
          \(A)_sram[d / 8].load(
            \(A)_src, \(leadingBlockDimension(A)), origin, \(transposed(A)));
        }
        
        """
      }
    }
    
    // MARK: - Load RHS
    
    func leadingDimensionRHS(
      _ descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceRHS! {
      case .device:
        return leadingDimension(B)
      case .threadgroup:
        return "\(leadingBlockDimension(B))"
      }
    }
    
    func declareRHSLocation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceRHS! {
      case .device:
        return """
        
        uint2 \(B)_src_offset(
          morton_offset.y + d_outer,
          morton_offset.x + \(traversalOffset));
        auto \(B)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(B), \(leadingDimension(B)), \(B)_src_offset, \(transposed(B)));
        
        """
      case .threadgroup:
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
    
    func loadRHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceRHS! {
      case .device:
        return declareRHSLocation(descriptor: descriptor)
      case .threadgroup:
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
        
        \(declareRHSLocation(descriptor: descriptor))
        
        """
      }
    }
    
    // MARK: - Inner Loop
    
    func innerLoopTraversal(
      traversalStart: String,
      traversalEnd: String,
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
        // Load the RHS from memory.
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
    
    func innerLoopHead(
      headStart: UInt16,
      headEnd: UInt16,
      descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpaceLHS! == .device ||
          descriptor.addressSpaceRHS! == .device {
        return """
        
        #pragma clang loop unroll(full)
        for (ushort d = \(headStart); d < \(headEnd); d += 8) {
          \(innerLoopTraversal(
              traversalStart: "0",
              traversalEnd: "\(blockDimensions.traversal)",
              descriptor: descriptor))
        }
        
        """
      } else {
        return """
        
        #pragma clang loop unroll(full)
        for (ushort d = \(headStart); d < \(headEnd); d += 8) {
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
    }
    
    // MARK: - Outer Loop
    
    struct LoopIterationDescriptor {
      // Whether to accumulate in the SIMD matmul.
      var accumulateConditional: String = ""
      var addressSpaceLHS: MTLAddressSpace?
      var addressSpaceRHS: MTLAddressSpace?
      var registerOffset: String = ""
    }
    
    func loopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpaceLHS! == .device ||
          descriptor.addressSpaceRHS! == .device {
        return """
        
        \(allocateLHS())
        \(loadLHS(descriptor: descriptor))
        \(loadRHS(descriptor: descriptor))
        \(innerLoopHead(
            headStart: 0,
            headEnd: blockDimensions.head,
            descriptor: descriptor))
        
        """
      } else {
        return """
        
        \(allocateLHS())
        \(loadLHS(descriptor: descriptor))
        \(loadRHS(descriptor: descriptor))
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
    }
    
    func gatedLoopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      var descriptorThreadgroup = descriptor
      descriptorThreadgroup.addressSpaceLHS = .threadgroup
      descriptorThreadgroup.addressSpaceRHS = .threadgroup
      if preferAsyncCache && preferAsyncLoad {
        return loopIteration(descriptor: descriptorThreadgroup)
      }
      
      var descriptorDevice = descriptor
      if preferAsyncCache {
        descriptorDevice.addressSpaceLHS = .threadgroup
      } else {
        descriptorDevice.addressSpaceLHS = .device
      }
      if preferAsyncLoad {
        descriptorDevice.addressSpaceRHS = .threadgroup
      } else {
        descriptorDevice.addressSpaceRHS = .device
      }
      
      let blockDim = blockDimensions.traversal
      let condition = """
      (
        (\(traversalDimension) % \(blockDim) == 0) ||
        (\(traversalOffset) + \(blockDim) <= \(traversalDimension))
      ) && (
        (\(headDimension) % \(blockDimensions.head) == 0) ||
        (d_outer + \(blockDimensions.head) <= \(headDimension))
      )
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
        \(gatedLoopIteration(descriptor: outerIterationDesc))
      }
      
      """
      
      // Add the last iteration, if unaligned.
      if loopEndFloor < loopEnd {
        outerIterationDesc.addressSpaceLHS = .threadgroup
        outerIterationDesc.addressSpaceRHS = .threadgroup
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
