//
//  AttentionKernel+Caching.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/22/24.
//

// N x D
// parallelization x head

extension AttentionKernel {
  // Enumeration that encapsulates both loading and storing.
  enum CachingOperationType {
    case load
    case store
  }
  
  func cache(
    operand: AttentionOperand,
    type: CachingOperationType
  ) -> String {
    // MARK: - Operand
    
    func allocateOperand() -> String {
      if type == .load {
        return """
        
        simdgroup_matrix_storage<float> \
        \(operand)_sram[\(paddedHeadDimension / 8)];
        
        """
      } else {
        return ""
      }
    }
    
    func asyncAccessOperand() -> String {
      if type == .load {
        return """
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sidx == 0) {
          uint2 \(operand)_offset(d_outer, \(parallelizationGroupOffset));
          auto src = simdgroup_matrix_storage<float>::apply_offset(
            \(operand), \(leadingDimension(operand)),
            \(operand)_offset, \(transposed(operand)));
          auto dst = (threadgroup float*)(threadgroup_block);
          
          ushort D_src_dimension = min(
            ushort(\(blockDimensions.head)),
            ushort(\(headDimension) - d_outer));
          ushort D_dst_dimension = min(
            ushort(\(blockDimensions.head)),
            ushort(\(paddedHeadDimension) - d_outer));
          ushort R_dimension = min(
            uint(\(blockDimensions.parallelization)),
            uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
          ushort2 tile_src(D_src_dimension, R_dimension);
          ushort2 tile_dst(D_dst_dimension, R_dimension);
          
          simdgroup_event event;
          event.async_copy(
            dst, \(leadingBlockDimension(operand)), tile_dst,
            src, \(leadingDimension(operand)), tile_src,
            \(transposed(operand)));
          simdgroup_event::wait(1, &event);
        }
        
        """
      } else {
        return """
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sidx == 0) {
          uint2 \(operand)_offset(d_outer, \(parallelizationGroupOffset));
          auto src = (threadgroup float*)(threadgroup_block);
          auto dst = simdgroup_matrix_storage<float>::apply_offset(
            \(operand), \(leadingDimension(operand)),
            \(operand)_offset, \(transposed(operand)));
          
          ushort D_dimension = min(
            ushort(\(blockDimensions.head)),
            ushort(\(headDimension) - d_outer));
          ushort R_dimension = min(
            uint(\(blockDimensions.parallelization)),
            uint(\(parallelizationDimension) - \(parallelizationGroupOffset)));
          ushort2 tile(D_dimension, R_dimension);
          
          simdgroup_event event;
          event.async_copy(
            dst, \(leadingDimension(operand)), tile,
            src, \(leadingBlockDimension(operand)), tile,
            \(transposed(operand)));
          simdgroup_event::wait(1, &event);
        }
        
        """
      }
    }
    
    func leadingDimensionOperand(
      _ descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpace == .device {
        return leadingDimension(operand)
      } else {
        return "\(leadingBlockDimension(operand))"
      }
    }
    
    func declareOperandLocation(
      descriptor: LoopIterationDescriptor
    ) -> String {
      if descriptor.addressSpace == .device {
        return """
        
        uint2 \(operand)_src_offset(
          morton_offset.x + d_outer,
          \(clampedParallelizationThreadOffset));
        auto \(operand)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(operand), \(leadingDimension(operand)),
          \(operand)_src_offset, \(transposed(operand)));
        
        """
      } else {
        return """
        
        ushort2 \(operand)_block_offset(
          morton_offset.x, morton_offset.y + sidx * 8);
        auto \(operand)_src = (threadgroup float*)(threadgroup_block);
        \(operand)_src = simdgroup_matrix_storage<float>::apply_offset(
          \(operand)_src, \(leadingBlockDimension(operand)),
          \(operand)_block_offset, \(transposed(operand)));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        """
      }
    }
    
    // MARK: - Inner Loop
    
    func innerLoopHead(
      headStart: UInt16,
      headEnd: UInt16,
      descriptor: LoopIterationDescriptor
    ) -> String {
      if type == .load {
        return """
        
        #pragma clang loop unroll(full)
        for (ushort d = \(headStart); d < \(headEnd); d += 8) {
          ushort2 origin(d, 0);
          \(operand)_sram[(d_outer + d) / 8].load(
            \(operand)_src, \(leadingDimensionOperand(descriptor)),
            origin, \(transposed(operand)));
        }
        
        """
      } else {
        return """
        
        #pragma clang loop unroll(full)
        for (ushort d = \(headStart); d < \(headEnd); d += 8) {
          ushort2 origin(d, 0);
          \(operand)_sram[(d_outer + d) / 8].store(
            \(operand)_src, \(leadingDimensionOperand(descriptor)),
            origin, \(transposed(operand)));
        }
        
        """
      }
    }
    
    // MARK: - Outer Loop
    
    struct LoopIterationDescriptor {
      var addressSpace: MTLAddressSpace = .threadgroup
    }
    
    func loopIteration(
      descriptor: LoopIterationDescriptor
    ) -> String {
      func loadOperand() -> String {
        if type == .load {
          return asyncAccessOperand()
        } else {
          return ""
        }
      }
      
      func storeOperand() -> String {
        if type == .load {
          return ""
        } else {
          return asyncAccessOperand()
        }
      }
      
      if descriptor.addressSpace == .device {
        return """
        
        \(declareOperandLocation(descriptor: descriptor))
        \(innerLoopHead(
            headStart: 0,
            headEnd: blockDimensions.head,
            descriptor: descriptor))
        
        """
      } else {
        return """
        
        \(loadOperand())
        \(declareOperandLocation(descriptor: descriptor))
        if (d_outer + \(blockDimensions.head) <= \(headDimension)) {
          \(innerLoopHead(
              headStart: 0,
              headEnd: blockDimensions.head,
              descriptor: descriptor))
        } else {
          \(innerLoopHead(
              headStart: 0,
              headEnd: headDimension % blockDimensions.head,
              descriptor: descriptor))
        }
        \(storeOperand())
        
        """
      }
    }
    
    func gatedLoopIteration() -> String {
      var descriptorDevice = LoopIterationDescriptor()
      var descriptorThreadgroup = LoopIterationDescriptor()
      descriptorDevice.addressSpace = .device
      descriptorThreadgroup.addressSpace = .threadgroup
      
      let condition = """
      \(!preferAsyncCache) && (
        (\(headDimension) % \(blockDimensions.head) == 0) ||
        (d_outer + \(blockDimensions.head) <= \(headDimension))
      )
      """
      
      return """
      
      if (\(condition)) {
        if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
          \(loopIteration(descriptor: descriptorDevice))
        }
      } else {
        \(loopIteration(descriptor: descriptorThreadgroup))
      }
      
      """
    }
    
    return """
    
    \(allocateOperand())
    
    #pragma clang loop unroll(full)
    for (
      ushort d_outer = 0;
      d_outer < \(headDimension);
      d_outer += \(blockDimensions.head)
    ) {
      \(gatedLoopIteration())
    }
    
    """
  }
}

extension AttentionKernel {
  // Prepare the addresses and registers for the attention loop.
  func createSetup() -> String {
    // Allocate registers for the specified operand.
    func allocate(operand: AttentionOperand) -> String {
      """
      
      simdgroup_matrix_storage<float> \
      \(operand)_sram[\(paddedHeadDimension / 8)];
      
      """
    }
    
    // Initialize the output string.
    var output: String = ""
    
    switch type {
    case .forward:
      if cached(.Q) {
        output += cache(operand: .Q, type: .load)
      }
      if cached(.O) {
        output += allocate(operand: .O)
      }
      output += """
      
      float m = -numeric_limits<float>::max();
      float l = numeric_limits<float>::denorm_min();
      
      """
      
    case .backwardQuery:
      if cached(.Q){
        output += cache(operand: .Q, type: .load)
      }
      if cached(.dO) {
        output += cache(operand: .dO, type: .load)
      }
      if cached(.dQ) {
        output += allocate(operand: .dQ)
      }
      output += """
      
      float L_sram = L[\(clampedParallelizationThreadOffset)];
      \(computeD())
      
      """
      
    case .backwardKeyValue:
      if cached(.K) {
        output += cache(operand: .K, type: .load)
      }
      if cached(.V) {
        output += cache(operand: .V, type: .load)
      }
      if cached(.dK) {
        output += allocate(operand: .dK)
      }
      if cached(.dV) {
        output += allocate(operand: .dV)
      }
    }
    
    return output
  }
  
  // Store any cached outputs to memory.
  func createCleanup(type: AttentionKernelType) -> String {
    // Initialize the output string.
    var output: String = ""
    
    switch type {
    case .forward(let computeL):
      if cached(.O) {
        output += cache(operand: .O, type: .store)
      }
      if computeL {
        output += """
        
        if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
          // Premultiplied by M_LOG2E_F.
          float L_sram = m + fast::log2(l);
          L[\(clampedParallelizationThreadOffset)] = L_sram;
        }
        
        """
      }
      
    case .backwardQuery:
      if cached(.dQ) {
        output += cache(operand: .dQ, type: .store)
      }
      output += """
      
      if (\(unsafeParallelizationThreadOffset) < \(parallelizationDimension)) {
        D[\(clampedParallelizationThreadOffset)] = D_sram;
      }
      
      """
      
    case .backwardKeyValue:
      if cached(.dK) {
        output += cache(operand: .dK, type: .store)
      }
      if cached(.dV) {
        output += cache(operand: .dV, type: .store)
      }
    }
    
    return output
  }
}
