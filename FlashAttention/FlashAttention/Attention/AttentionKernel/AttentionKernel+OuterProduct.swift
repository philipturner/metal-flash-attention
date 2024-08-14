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
    
    func allocateLHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      guard !cached(A) else {
        return ""
      }
      return """
      
      simdgroup_matrix_storage<float> \
      \(A)_sram[\(descriptor.registerSize) / 8];
      
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
        auto \(A)_src = simdgroup_matrix_storage<\(memoryName(A))>
        ::apply_offset(
          \(A), \(leadingDimension(A)),
          \(A)_src_offset, \(transposed(A)));
        
        """
      case .threadgroup:
        return """
        
        ushort2 \(A)_block_offset(
          morton_offset.x, 
          morton_offset.y + sidx * 8);
        auto \(A)_src = (threadgroup \(memoryName(A))*)(threadgroup_block);
        \(A)_src = simdgroup_matrix_storage<\(memoryName(A))>
        ::apply_offset(
          \(A)_src, \(leadingBlockDimension(A)),
          \(A)_block_offset, \(transposed(A)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        """
      }
    }
    
    func asyncLoadLHS(
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(A)_offset(d_outer, \(parallelizationGroupOffset));
        auto src = simdgroup_matrix_storage<\(memoryName(A))>
        ::apply_offset(
          \(A), \(leadingDimension(A)),
          \(A)_offset, \(transposed(A)));
        auto dst = (threadgroup \(memoryName(A))*)(threadgroup_block);
        
        ushort D_src_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort D_dst_dimension = \(descriptor.registerSize);
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
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          ushort2 \(A)_origin(d, 0);
          \(A)_sram[d / 8].\(loadFunction(A))(
            \(A)_src, \(leadingDimension(A)),
            \(A)_origin, \(transposed(A)));
        }
        
        """
      case .threadgroup:
        return """
        
        \(asyncLoadLHS(descriptor: descriptor))
        \(declareLHSLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          ushort2 \(A)_origin(d, 0);
          \(A)_sram[d / 8].\(loadFunction(A))(
            \(A)_src, \(leadingBlockDimension(A)),
            \(A)_origin, \(transposed(A)));
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
        auto \(B)_src = simdgroup_matrix_storage<\(memoryName(B))>
        ::apply_offset(
          \(B), \(leadingDimension(B)),
          \(B)_src_offset, \(transposed(B)));
        
        """
      case .threadgroup:
        return """
        
        ushort2 \(B)_block_offset(
          morton_offset.x,
          morton_offset.y);
        auto \(B)_src = (threadgroup \(memoryName(B))*)(threadgroup_block);
        \(B)_src = simdgroup_matrix_storage<\(memoryName(B))>
        ::apply_offset(
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
          auto src = simdgroup_matrix_storage<\(memoryName(B))>
          ::apply_offset(
            \(B), \(leadingDimension(B)),
            \(B)_offset, \(transposed(B)));
          auto dst = (threadgroup \(memoryName(B))*)(threadgroup_block);
          
          ushort D_src_dimension = min(
            ushort(\(blockDimensions.head)),
            ushort(\(headDimension) - d_outer));
          ushort D_dst_dimension = \(descriptor.registerSize);
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
        ushort2 \(B)_origin(c, d);
        simdgroup_matrix_storage<float> \(B);
        \(B).\(loadFunction(B))(
          \(B)_src, \(leadingDimensionRHS(descriptor)),
          \(B)_origin, \(!transposed(B)));
        
        // Issue one SIMD matmul instruction.
        \(C)_sram[c / 8].multiply(
          \(A)_sram[(\(descriptor.registerOffset) + d) / 8],
          \(B), \(descriptor.accumulateConditional));
      }
      
      """
    }
    
    func innerLoopHead(
      descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpaceLHS! == .device ||
          descriptor.addressSpaceRHS! == .device {
        return """
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
          \(innerLoopTraversal(
              traversalStart: "0",
              traversalEnd: "\(blockDimensions.traversal)",
              descriptor: descriptor))
        }
        
        """
      } else {
        return """
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
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
      var registerSize: UInt16 = .zero
    }
    
    func loopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      return """
      
      \(allocateLHS(descriptor: descriptor))
      \(loadLHS(descriptor: descriptor))
      \(loadRHS(descriptor: descriptor))
      \(innerLoopHead(descriptor: descriptor))
      
      """
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
        (\(headDimension) % 8 == 0) ||
        (d_outer + \(descriptor.registerSize) <= \(headDimension))
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
    
    // MARK: - Top Level Specification
    
    func loopEnd() -> UInt16 {
      paddedHeadDimension
    }
    
    func loopEndFloor() -> UInt16 {
      loopEnd() - loopEnd() % blockDimensions.head
    }
    
    func unrollStatement() -> String {
      if cached(A) {
        return "#pragma clang loop unroll(full)"
      } else {
        return "#pragma clang loop unroll(disable)"
      }
    }
    
    func initializeStatement() -> String {
      if cached(A) {
        // Zero-initialize during the multiply-accumulate loop.
        return ""
      } else {
        // Zero-initialize beforehand.
        return initializeAccumulator()
      }
    }
    
    func accumulateConditional() -> String {
      if cached(A) {
        return "((d_outer > 0) || (d > 0))"
      } else {
        // The accumulator is already initialized.
        return "true"
      }
    }
    
    func registerOffset() -> String {
      if cached(A) {
        return "d_outer"
      } else {
        return "0"
      }
    }
    
    func firstIterations() -> String {
      var descriptor = LoopIterationDescriptor()
      descriptor.accumulateConditional = accumulateConditional()
      descriptor.registerOffset = registerOffset()
      descriptor.registerSize = blockDimensions.head
      
      return """
      
      \(unrollStatement())
      for (
        ushort d_outer = 0;
        d_outer < \(loopEndFloor());
        d_outer += \(blockDimensions.head)
      ) {
        \(gatedLoopIteration(descriptor: descriptor))
      }
      
      """
    }
    
    func lastIteration() -> String {
      var descriptor = LoopIterationDescriptor()
      descriptor.accumulateConditional = accumulateConditional()
      descriptor.registerOffset = registerOffset()
      descriptor.registerSize = paddedHeadEdge
      
      return """
      
      if (\(loopEndFloor() < loopEnd())) {
        ushort d_outer = \(loopEndFloor());
        \(gatedLoopIteration(descriptor: descriptor))
      }
      
      """
    }
    
    // Collect all of the statements into one string.
    return """
    
    \(allocateAccumulator())
    \(initializeStatement())
    
    \(firstIterations())
    \(lastIteration())
    
    """
  }
}
