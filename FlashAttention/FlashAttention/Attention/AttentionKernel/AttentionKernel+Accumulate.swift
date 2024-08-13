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
    
    // MARK: - Initialize
    
    func allocateAccumulator() -> String {
      guard !cached(C) else {
        return ""
      }
      return """
      
      simdgroup_matrix_storage<float> \
      \(C)_sram[\(blockDimensions.head) / 8];
      
      """
    }
    
    func initializeAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        auto \(C) = \(C)_sram + (\(descriptor.registerOffset) + d) / 8;
        *\(C) = simdgroup_matrix_storage<float>(0);
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
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(descriptor.registerSize); d += 8) {
        auto \(C) = \(C)_sram + (\(descriptor.registerOffset) + d) / 8;
        *(\(C)->thread_elements()) *= \(scale);
      }
      
      """
    }
    
    // MARK: - Load/Store Accumulator
    
    func declareAccumulatorLocation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceLHS! {
      case .device:
        return """
        
        uint2 \(C)_src_offset(
          morton_offset.x + d_outer,
          \(clampedParallelizationThreadOffset));
        auto \(C)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(C), \(leadingDimension(C)), \(C)_src_offset, \(transposed(C)));
        
        """
      case .threadgroup:
        return """
        
        ushort2 \(C)_block_offset(
          morton_offset.x,
          morton_offset.y + sidx * 8);
        auto \(C)_src = (threadgroup float*)(threadgroup_block);
        \(C)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(C)_src, \(leadingBlockDimension(C)),
          \(C)_block_offset, \(transposed(C)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        """
      }
    }
    
    func asyncLoadAccumulator() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 C_offset(d_outer, \(parallelizationGroupOffset));
        auto src = simdgroup_matrix_storage<float>::apply_offset(
          \(C), \(leadingDimension(C)), C_offset, \(transposed(C)));
        auto dst = (threadgroup float*)(threadgroup_block);
      
        ushort D_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
        ushort2 tile(D_dimension, R_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingBlockDimension(C)), tile,
          src, \(leadingDimension(C)), tile, \(transposed(C)));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    func asyncStoreAccumulator() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(C)_offset(d_outer, \(parallelizationGroupOffset));
        auto src = (threadgroup float*)(threadgroup_block);
        auto dst = simdgroup_matrix_storage<float>::apply_offset(
          \(C), \(leadingDimension(C)), \(C)_offset, \(transposed(C)));
        
        ushort D_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
        ushort2 tile(D_dimension, R_dimension);
        
        simdgroup_event event;
        event.async_copy(
          dst, \(leadingDimension(C)), tile,
          src, \(leadingBlockDimension(C)), tile, \(transposed(C)));
        simdgroup_event::wait(1, &event);
      }
      
      """
    }
    
    func loadAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceLHS! {
      case .device:
        return """
        
        \(declareAccumulatorLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
          ushort2 origin(d, 0);
          \(C)_sram[d / 8].load(
            \(C)_src, \(leadingDimension(C)), origin, \(transposed(C)));
        }
        
        """
      case .threadgroup:
        return """
        
        \(asyncLoadAccumulator())
        \(declareAccumulatorLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
          ushort2 origin(d, 0);
          \(C)_sram[d / 8].load(
            \(C)_src, \(leadingBlockDimension(C)), origin, \(transposed(C)));
        }
        
        """
      }
    }
    
    func storeAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      switch descriptor.addressSpaceLHS! {
      case .device:
        return """
        
        \(declareAccumulatorLocation(descriptor: descriptor))
        
        if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
          #pragma clang loop unroll(full)
          for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
            ushort2 origin(d, 0);
            \(C)_sram[d / 8].store(
              \(C)_src, \(leadingDimension(C)), origin, \(transposed(C)));
          }
        }
        
        """
      case .threadgroup:
        return """
        
        \(declareAccumulatorLocation(descriptor: descriptor))
        
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
          ushort2 origin(d, 0);
          \(C)_sram[d / 8].store(
            \(C)_src, \(leadingBlockDimension(C)), origin, \(transposed(C)));
        }
        
        \(asyncStoreAccumulator())
        
        """
      }
    }
    
    func cacheAccumulator(
      descriptor: LoopIterationDescriptor,
      type: CachingOperationType
    ) -> String {
      guard !cached(C) else {
        return ""
      }
      
      if type == .load {
        return loadAccumulator(descriptor: descriptor)
      } else {
        return storeAccumulator(descriptor: descriptor)
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
          morton_offset.x + d_outer,
          morton_offset.y + \(traversalOffset));
        auto \(B)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(B), \(leadingDimension(B)), \(B)_src_offset, \(transposed(B)));
        
        """
      case .threadgroup:
        return """
        
        ushort2 \(B)_block_offset(morton_offset.x, morton_offset.y);
        auto \(B)_src = (threadgroup float*)(threadgroup_block);
        \(B)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(B)_src, \(leadingBlockDimension(B)),
          \(B)_block_offset, \(transposed(B)));
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
        
        \(declareRHSLocation(descriptor: descriptor))
        
        """
      }
    }
    
    // MARK: - Inner Loop
    
    func innerLoopHead(
      headStart: UInt16,
      headEnd: UInt16,
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = \(headStart); d < \(headEnd); d += 8) {
        // Load the RHS from memory.
        ushort2 origin(d, c);
        simdgroup_matrix_storage<float> \(B);
        \(B).load(
          \(B)_src, \(leadingDimensionRHS(descriptor)),
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
      if descriptor.addressSpaceLHS! == .device ||
          descriptor.addressSpaceRHS! == .device {
        return """
        
        // TODO: Allow this device path for all multiples of 8, instead of
        // just multiples of the block size.
        #pragma clang loop unroll(full)
        for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
          \(innerLoopHead(
              headStart: 0,
              headEnd: blockDimensions.head,
              descriptor: descriptor))
        }
        
        """
      } else {
        return """
        
        #pragma clang loop unroll(full)
        for (ushort c = \(traversalStart); c < \(traversalEnd); c += 8) {
          \(innerLoopHead(
              headStart: 0,
              headEnd: descriptor.registerSize,
              descriptor: descriptor))
        }
        
        """
      }
    }
    
    // MARK: - Outer Loop
    
    struct LoopIterationDescriptor {
      var addressSpaceLHS: MTLAddressSpace?
      var addressSpaceRHS: MTLAddressSpace?
      var registerOffset: String = ""
      var registerSize: UInt16 = .zero
    }
    
    func loopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      func multiplyAB() -> String {
        if descriptor.addressSpaceLHS! == .device ||
            descriptor.addressSpaceRHS! == .device {
          let blockDim = blockDimensions.traversal
          return """
          
          \(innerLoopTraversal(
              traversalStart: "0",
              traversalEnd: "\(blockDim)",
              descriptor: descriptor))
          if (
            (\(traversalDimension) % \(blockDim) == 0) &&
            (\(traversalOffset) + \(blockDim) == \(traversalDimension))
          ) {
             \(scaleAccumulator(
                 by: accumulateDesc.lastIterationScale,
                 descriptor: descriptor))
          }
          
          """
        } else {
          return """
          
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
          
          """
        }
      }
      
      return """
      
      \(allocateAccumulator())
      if (\(traversalOffset) == 0) {
        \(initializeAccumulator(descriptor: descriptor))
      } else {
        \(cacheAccumulator(
            descriptor: descriptor,
            type: .load))
        \(scaleAccumulator(
            by: accumulateDesc.everyIterationScale,
            descriptor: descriptor))
      }
      \(loadRHS(descriptor: descriptor))
      \(multiplyAB())
      \(cacheAccumulator(
          descriptor: descriptor,
          type: .store))
      
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
    
    // MARK: - Top Level Specification
    
    func loopEnd() -> UInt16 {
      paddedHeadDimension
    }
    
    func loopEndFloor() -> UInt16 {
      loopEnd() - loopEnd() % blockDimensions.head
    }
    
    func unrollStatement() -> String {
      if cached(C) {
        return "#pragma clang loop unroll(full)"
      } else {
        return "#pragma clang loop unroll(disable)"
      }
    }
    
    func registerOffset() -> String {
      if cached(C) {
        return "d_outer"
      } else {
        return "0"
      }
    }
    
    func firstIterations() -> String {
      var descriptor = LoopIterationDescriptor()
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
      descriptor.registerOffset = registerOffset()
      descriptor.registerSize = paddedHeadEdge
      descriptor.addressSpaceLHS = .threadgroup
      descriptor.addressSpaceRHS = .threadgroup
      
      return """
      
      if (\(loopEndFloor() < loopEnd())) {
        ushort d_outer = \(loopEndFloor());
        \(loopIteration(descriptor: descriptor))
      }
      
      """
    }
    
    // Collect all of the statements into one string.
    return """
    
    \(firstIterations())
    \(lastIteration())
    
    """
  }
}
