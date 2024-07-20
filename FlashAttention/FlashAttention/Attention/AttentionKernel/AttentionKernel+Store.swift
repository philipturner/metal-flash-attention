//
//  AttentionKernel+Store.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/19/24.
//

struct AttentionHBMAccessDescriptor {
  var index: String?
  var leadingBlockDimension: UInt16?
  var leadingDimension: String?
  var name: String?
  var threadgroupAddress: String?
  var transposeState: Bool?
}

struct AttentionHBMAccessDescriptor2 {
  /// Name of the output, destination of a 32 x D block.
  var O: String?
  
  var transposeO: Bool?
  var leadingDimensionO: String?
  
  var matrixDimension: String?
  var matrixOffset: String?
}

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
  
  func store2(descriptor: AttentionHBMAccessDescriptor2) -> String {
    guard let O = descriptor.O,
          let transposeO = descriptor.transposeO,
          let leadingDimensionO = descriptor.leadingDimensionO,
          let matrixDimension = descriptor.matrixDimension,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    // 32 x 64 allocation in threadgroup memory
    // leading dimension = transposeO ? 32 : 64
    let leadingBlockDimensionO = transposeO ? UInt16(32) : UInt16(64)
    
    let loopBodyStoreO = """

ushort2 origin(d, 0);
\(O)_sram[(d_outer + d) / 8].store(
  \(O)_block, \(leadingBlockDimensionO), origin, \(transposeO));

"""
    
    return """

{
  // Find where the \(O) data will be written to.
  ushort2 \(O)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(O)_block = (threadgroup float*)(threadgroup_block);
  \(O)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(O)_block, \(leadingBlockDimensionO),
    \(O)_block_offset, \(transposeO));
  
  // Outer loop over D.
#pragma clang loop unroll(full)
  for (ushort d = 0; d < D; d += 64) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Iterate over the head dimension.
    ushort d_outer = d;
    if (\(paddedD) - d_outer >= 64) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < 64; d += 8) {
        \(loopBodyStoreO)
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(paddedD) % 64; d += 8) {
        \(loopBodyStoreO)
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      uint2 \(O)_offset(d, \(matrixOffset));
      auto src = (threadgroup float*)(threadgroup_block);
      auto dst = simdgroup_matrix_storage<float>::apply_offset(
        \(O), \(leadingDimensionO), \(O)_offset, \(transposeO));
     
      ushort D_dimension = min(ushort(64), ushort(D - d));
      ushort RC_dimension = min(
        uint(32), \(matrixDimension) - \(matrixOffset));
      ushort2 tile_src(D_dimension, RC_dimension);
      ushort2 tile_dst(D_dimension, RC_dimension);
      
      simdgroup_event event;
      event.async_copy(
        dst, \(leadingDimensionO), tile_dst,
        src, \(leadingBlockDimensionO), tile_src, \(transposeO));
      simdgroup_event::wait(1, &event);
    }
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
      var accessDesc = AttentionHBMAccessDescriptor2()
      accessDesc.O = "O"
      accessDesc.transposeO = transposeState.O
      accessDesc.leadingDimensionO = leadingDimensions.O
      accessDesc.matrixDimension = "R"
      accessDesc.matrixOffset = "gid * R_group"
      
      output += store2(descriptor: accessDesc)
      
      // L[i]
      if computeL {
        output += """

    if (linear_array_slot < R) {
      \(computeLTerm())
      L_terms[linear_array_slot] = L_term;
    }

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

  if (linear_array_slot < R) {
    D_terms[linear_array_slot] = D_term;
  }

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
