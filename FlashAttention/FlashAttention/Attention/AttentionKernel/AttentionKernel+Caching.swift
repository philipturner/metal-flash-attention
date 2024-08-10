//
//  AttentionKernel+Caching.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/22/24.
//

// N x D
// parallelization x head

extension AttentionKernel {
  // TODO: Start by splitting up load/store into modular chunks. Then,
  // de-duplicate the code and make them one function.
  
  func load(operand: AttentionOperand) -> String {
    func cacheOperand() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(operand)_offset(d_outer, \(parallelizationOffset));
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
          uint(\(parallelizationDimension) - \(parallelizationOffset)));
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
    }
    
    func declareOperandLocation() -> String {
      """
      
      ushort2 \(operand)_block_offset(
        morton_offset.x, morton_offset.y + sidx * 8);
      auto \(operand)_src = (threadgroup float*)(threadgroup_block);
      \(operand)_src = simdgroup_matrix_storage<float>::apply_offset(
        \(operand)_src, \(leadingBlockDimension(operand)),
        \(operand)_block_offset, \(transposed(operand)));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      """
    }
    
    func innerLoopHead(
      headStart: UInt16,
      headEnd: UInt16
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = \(headStart); d < \(headEnd); d += 8) {
        ushort2 origin(d, 0);
        \(operand)_sram[(d_outer + d) / 8].load(
          \(operand)_src, \(leadingBlockDimension(operand)),
          origin, \(transposed(operand)));
      }
      
      """
    }
    
    func loopIteration() -> String {
      return """
      
      \(declareOperandLocation())
      if (d_outer + \(blockDimensions.head) <= \(headDimension)) {
        \(innerLoopHead(
            headStart: 0,
            headEnd: blockDimensions.head))
      } else {
        \(innerLoopHead(
            headStart: 0,
            headEnd: headDimension % blockDimensions.head))
      }
      
      """
    }
    
    return """
    
    \(allocateCache(operand: operand))
    
    #pragma clang loop unroll(full)
    for (
      ushort d_outer = 0;
      d_outer < \(headDimension);
      d_outer += \(blockDimensions.head)
    ) {
      \(cacheOperand())
      \(loopIteration())
    }
    
    """
  }
  
  func store(operand: AttentionOperand) -> String {
    func cacheOperand() -> String {
      """
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 \(operand)_offset(d_outer, \(parallelizationOffset));
        auto src = (threadgroup float*)(threadgroup_block);
        auto dst = simdgroup_matrix_storage<float>::apply_offset(
          \(operand), \(leadingDimension(operand)),
          \(operand)_offset, \(transposed(operand)));
        
        ushort D_dimension = min(
          ushort(\(blockDimensions.head)),
          ushort(\(headDimension) - d_outer));
        ushort R_dimension = min(
          uint(\(blockDimensions.parallelization)),
          uint(\(parallelizationDimension) - \(parallelizationOffset)));
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
    
    func declareOperandLocation() -> String {
      """
      
      ushort2 \(operand)_block_offset(
        morton_offset.x, morton_offset.y + sidx * 8);
      auto \(operand)_src = (threadgroup float*)(threadgroup_block);
      \(operand)_src = simdgroup_matrix_storage<float>::apply_offset(
        \(operand)_src, \(leadingBlockDimension(operand)),
        \(operand)_block_offset, \(transposed(operand)));
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      """
    }
    
    func innerLoopHead(
      headStart: UInt16,
      headEnd: UInt16
    ) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = \(headStart); d < \(headEnd); d += 8) {
        ushort2 origin(d, 0);
        \(operand)_sram[(d_outer + d) / 8].store(
          \(operand)_src, \(leadingBlockDimension(operand)),
          origin, \(transposed(operand)));
      }
      
      """
    }
    
    func loopIteration() -> String {
      func loadOperand() -> String {
        return ""
      }
      
      func storeOperand() -> String {
        return cacheOperand()
      }
      
      return """
      
      \(loadOperand())
      \(declareOperandLocation())
      if (d_outer + \(blockDimensions.head) <= \(headDimension)) {
        \(innerLoopHead(
            headStart: 0,
            headEnd: blockDimensions.head))
      } else {
        \(innerLoopHead(
            headStart: 0,
            headEnd: headDimension % blockDimensions.head))
      }
      \(storeOperand())
      
      """
    }
    
    return """
    
    #pragma clang loop unroll(full)
    for (
      ushort d_outer = 0;
      d_outer < \(headDimension);
      d_outer += \(blockDimensions.head)
    ) {
      \(loopIteration())
    }
    
    """
  }
}

extension AttentionKernel {
  // Allocate registers for the specified operand.
  fileprivate func allocateCache(operand: AttentionOperand) -> String {
    """
    
    simdgroup_matrix_storage<float> \
    \(operand)_sram[\(paddedHeadDimension / 8)];
    
    """
  }
  
  // Prepare the addresses and registers for the attention loop.
  func createSetup() -> String {
    // Initialize the output string.
    var output: String = ""
    
    switch type {
    case .forward:
      if cached(.Q) {
        output += load(operand: .Q)
      }
      if cached(.O) {
        output += allocateCache(operand: .O)
      }
      output += """
      
      float m = -numeric_limits<float>::max();
      float l = numeric_limits<float>::denorm_min();
      
      """
      
    case .backwardQuery:
      if cached(.Q){
        output += load(operand: .Q)
      }
      if cached(.dO) {
        output += load(operand: .dO)
      }
      if cached(.dQ) {
        output += allocateCache(operand: .dQ)
      }
      output += """
      
      float L_sram = L[\(parallelizationThreadOffset)];
      \(computeD())
      
      // Store D[i] in memory sooner rather than later. That way, the compiler
      // has a chance to deallocate the parallelization offset.
      if (\(parallelizationOffset) + sidx * 8 + morton_offset.y < R) {
        D[\(parallelizationThreadOffset)] = D_sram;
      }
      
      """
      
    case .backwardKeyValue:
      if cached(.K) {
        output += load(operand: .K)
      }
      if cached(.V) {
        output += load(operand: .V)
      }
      if cached(.dK) {
        output += allocateCache(operand: .dK)
      }
      if cached(.dV) {
        output += allocateCache(operand: .dV)
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
        output += store(operand: .O)
      }
      if computeL {
        output += """
        
        if (\(parallelizationOffset) + sidx * 8 + morton_offset.y < R) {
          // Premultiplied by M_LOG2E_F.
          float L_sram = m + fast::log2(l);
          L[\(parallelizationThreadOffset)] = L_sram;
        }
        
        """
      }
      
    case .backwardQuery:
      if cached(.dQ) {
        output += store(operand: .dQ)
      }
      
    case .backwardKeyValue:
      if cached(.dK) {
        output += store(operand: .dK)
      }
      if cached(.dV) {
        output += store(operand: .dV)
      }
    }
    
    return output
  }
}
