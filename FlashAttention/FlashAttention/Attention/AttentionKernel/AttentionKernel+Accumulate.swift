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
    
    func allocateAccumulator() -> String {
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
    
    func declareAccumulatorLocation() -> String {
      """
      
      // Where the \(C) data will be read from.
      ushort2 \(C)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
      auto \(C)_block = (threadgroup float*)(threadgroup_block);
      \(C)_block = simdgroup_matrix_storage<float>::apply_offset(
        \(C)_block, \(leadingBlockDimension(C)),
        \(C)_block_offset, \(transposed(C)));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      """
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
      """
      
      \(asyncLoadAccumulator())
      \(declareAccumulatorLocation())
      
      // Inner loop over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
        ushort2 origin(d, 0);
        \(C)_sram[d / 8].load(
          \(C)_block, \(leadingBlockDimension(C)),
          origin, \(transposed(C)));
      }
      
      """
    }
    
    func storeAccumulator(
      descriptor: LoopIterationDescriptor
    ) -> String {
      """
      
      \(declareAccumulatorLocation())
      
      // Inner loop over the head dimension.
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(blockDimensions.head); d += 8) {
        ushort2 origin(d, 0);
        \(C)_sram[d / 8].store(
          \(C)_block, \(leadingBlockDimension(C)), origin, \(transposed(C)));
      }
      
      \(asyncStoreAccumulator())
      
      """
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
    
    // MARK: - RHS
    
    func leadingDimensionRHS(
      _ descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpace == .device {
        return leadingDimension(B)
      } else {
        return "\(leadingBlockDimension(B))"
      }
    }
    
    func declareRHSLocation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpace == .device {
        return """
        
        uint2 \(B)_src_offset(
          morton_offset.x + d_outer,
          morton_offset.y + \(traversalOffset));
        auto \(B)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(B), \(leadingDimension(B)), \(B)_src_offset, \(transposed(B)));
        
        """
      } else {
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
      if descriptor.addressSpace == .device {
        return declareRHSLocation(descriptor: descriptor)
      } else {
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
      if descriptor.addressSpace == .device {
        return """
        
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
    }
    
    // MARK: - Outer Loop
    
    struct LoopIterationDescriptor {
      var addressSpace: MTLAddressSpace = .threadgroup
      var registerOffset: String = ""
      var registerSize: UInt16 = .zero
    }
    
    func loopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      func multiplyAB() -> String {
        if descriptor.addressSpace == .device {
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
      var descriptorDevice = descriptor
      var descriptorThreadgroup = descriptor
      descriptorDevice.addressSpace = .device
      descriptorThreadgroup.addressSpace = .threadgroup
      
      let blockDim = blockDimensions.traversal
      let condition = """
      \(!preferAsyncLoad) && (
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
        \(gatedLoopIteration(descriptor: outerIterationDesc))
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
        \(gatedLoopIteration(descriptor: outerIterationDesc))
      }

      """
    }
  }
}
