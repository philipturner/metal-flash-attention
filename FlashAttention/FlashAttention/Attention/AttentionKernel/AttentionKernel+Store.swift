//
//  AttentionKernel+Store.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

extension AttentionKernel {
  func store(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let leadingBlockDimension = descriptor.leadingBlockDimension,
          let name = descriptor.name,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """

  {
    ushort2 threadgroup_origin = ushort2(0, sidx * 8) + morton_offset;
    auto dst = (threadgroup float*)(\(threadgroupAddress));
    dst = simdgroup_matrix_storage<float>::apply_offset(
      dst, \(leadingBlockDimension), threadgroup_origin, \(transposeState));
    
#pragma clang loop unroll(full)
    for (ushort d = 0; d < \(paddedD); d += 8) {
      ushort2 thread_origin(d, 0);
      \(name)_sram[d / 8].store(
        dst, \(leadingBlockDimension), thread_origin, \(transposeState));
    }
  }

"""
  }
  
  // Finish something during a pass that parallelizes over rows.
  func commitRows(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let index = descriptor.index,
          let leadingBlockDimension = descriptor.leadingBlockDimension,
          let leadingDimension = descriptor.leadingDimension,
          let name = descriptor.name,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """

  if (sidx == 0) {
    uint2 device_origin(0, \(index));
    auto src = (threadgroup float*)(\(threadgroupAddress));
    auto dst = simdgroup_matrix_storage<float>::apply_offset(
      \(name), \(leadingDimension), device_origin, \(transposeState));
   
    ushort R_tile_dimension = min(uint(R_group), R - \(index));
    ushort2 tile_src(D, R_tile_dimension);
    ushort2 tile_dst(D, R_tile_dimension);
    
    simdgroup_event event;
    event.async_copy(
      dst, \(leadingDimension), tile_dst,
      src, \(leadingBlockDimension), tile_src, \(transposeState));
    simdgroup_event::wait(1, &event);
  }

"""
  }
  
  // Finish something during a pass that parallelizes over columns.
  func commitColumns(descriptor: AttentionHBMAccessDescriptor) -> String {
    guard let index = descriptor.index,
          let leadingBlockDimension = descriptor.leadingBlockDimension,
          let leadingDimension = descriptor.leadingDimension,
          let name = descriptor.name,
          let threadgroupAddress = descriptor.threadgroupAddress,
          let transposeState = descriptor.transposeState else {
      fatalError("Descriptor was incomplete.")
    }
    
    return """

  if (sidx == 0) {
    uint2 device_origin(0, \(index));
    auto src = (threadgroup float*)(\(threadgroupAddress));
    auto dst = simdgroup_matrix_storage<float>::apply_offset(
      \(name), \(leadingDimension), device_origin, \(transposeState));
   
    ushort C_tile_dimension = min(uint(C_group), C - \(index));
    ushort2 tile_src(D, C_tile_dimension);
    ushort2 tile_dst(D, C_tile_dimension);
    
    simdgroup_event event;
    event.async_copy(
      dst, \(leadingDimension), tile_dst,
      src, \(leadingBlockDimension), tile_src, \(transposeState));
    simdgroup_event::wait(1, &event);
  }

"""
  }
}

extension AttentionKernel {
  func createCleanup(type: AttentionKernelType) -> String {
    // A threadgroup barrier, formatted to match the correct indentation.
    func threadgroupBarrier() -> String {
      return """

    threadgroup_barrier(mem_flags::mem_threadgroup);

  """
    }
    
    // Initialize the output string.
    var output: String = ""
    
    switch type {
    case .forward(let computeL):
      // O
      var accessDesc = AttentionHBMAccessDescriptor()
      accessDesc.index = "gid * R_group"
      accessDesc.leadingBlockDimension = leadingBlockDimensions.O
      accessDesc.leadingDimension = leadingDimensions.O
      accessDesc.name = "O"
      accessDesc.threadgroupAddress = "threadgroup_block"
      accessDesc.transposeState = transposeState.O
      
      output += threadgroupBarrier()
      output += store(descriptor: accessDesc)
      output += threadgroupBarrier()
      output += commitRows(descriptor: accessDesc)
      
      // L[i]
      if computeL {
        output += computeLTerm()
        output += """

  L_terms[linear_array_slot] = L_term;

"""
      }
      
    case .backwardQuery(let computeDerivativeQ):
      // dQ
      if computeDerivativeQ {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.index = "gid * R_group"
        accessDesc.leadingBlockDimension = leadingBlockDimensions.Q
        accessDesc.leadingDimension = leadingDimensions.Q
        accessDesc.name = "dQ"
        accessDesc.threadgroupAddress = "threadgroup_block"
        accessDesc.transposeState = transposeState.Q
        
        output += threadgroupBarrier()
        output += store(descriptor: accessDesc)
        output += threadgroupBarrier()
        output += commitRows(descriptor: accessDesc)
      }
      
      // D[i]
      output += """

  D_terms[linear_array_slot] = D_term;

"""
      
    case .backwardKeyValue(let computeDerivativeK):
      // dK
      if computeDerivativeK {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.index = "gid * C_group"
        accessDesc.leadingBlockDimension = leadingBlockDimensions.K
        accessDesc.leadingDimension = leadingDimensions.K
        accessDesc.name = "dK"
        accessDesc.threadgroupAddress = "threadgroup_block"
        accessDesc.transposeState = transposeState.K
        
        
        output += threadgroupBarrier()
        output += store(descriptor: accessDesc)
        output += threadgroupBarrier()
        output += commitColumns(descriptor: accessDesc)
      }
      
      // dV
      do {
        var accessDesc = AttentionHBMAccessDescriptor()
        accessDesc.index = "gid * C_group"
        accessDesc.leadingBlockDimension = leadingBlockDimensions.V
        accessDesc.leadingDimension = leadingDimensions.V
        accessDesc.name = "dV"
        accessDesc.threadgroupAddress = "threadgroup_block"
        accessDesc.transposeState = transposeState.V
        
        output += threadgroupBarrier()
        output += store(descriptor: accessDesc)
        output += threadgroupBarrier()
        output += commitColumns(descriptor: accessDesc)
      }
      
    }
    
    return output
  }
}
