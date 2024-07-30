//
//  AttentionKernel+Caching.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/22/24.
//

// N x D
// parallelization x head

extension AttentionKernel {
  func load(name: String) -> String {
    func loadBlock(registerSize: UInt16) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(registerSize); d += 8) {
        ushort2 origin(d, 0);
        \(name)_sram[(d_outer + d) / 8].load(
          \(name)_block, \(leadingBlockDimension(name)),
          origin, \(transposed(name)));
      }
      
      """
    }
    
    return """

\(allocateOperand(name: name))
{
  // Find where the \(name) data will be read from.
  ushort2 \(name)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(name)_block = (threadgroup float*)(threadgroup_block);
  \(name)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(name)_block, \(leadingBlockDimension(name)),
    \(name)_block_offset, \(transposed(name)));
  
  // Outer loop over the head dimension.
#pragma clang loop unroll(full)
  for (
    ushort d_outer = 0;
    d_outer < \(headDimension);
    d_outer += \(blockDimensions.head)
  ) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      uint2 \(name)_offset(d_outer, \(parallelizationOffset));
      auto src = simdgroup_matrix_storage<float>::apply_offset(
        \(name), \(leadingDimension(name)),
        \(name)_offset, \(transposed(name)));
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
        dst, \(leadingBlockDimension(name)), tile_dst,
        src, \(leadingDimension(name)), tile_src, \(transposed(name)));
      simdgroup_event::wait(1, &event);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Inner loop over the head dimension.
    if (d_outer + \(blockDimensions.head) <= \(headDimension)) {
      \(loadBlock(registerSize: blockDimensions.head))
    } else {
      \(loadBlock(registerSize: headDimension % blockDimensions.head))
    }
  }
}

"""
  }
  
  func store(name: String) -> String {
    func storeBlock(registerSize: UInt16) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(registerSize); d += 8) {
        ushort2 origin(d, 0);
        \(name)_sram[(d_outer + d) / 8].store(
          \(name)_block, \(leadingBlockDimension(name)),
          origin, \(transposed(name)));
      }
      
      """
    }
    
    return """

{
  // Find where the \(name) data will be written to.
  ushort2 \(name)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(name)_block = (threadgroup float*)(threadgroup_block);
  \(name)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(name)_block, \(leadingBlockDimension(name)),
    \(name)_block_offset, \(transposed(name)));
  
    // Outer loop over the head dimension.
#pragma clang loop unroll(full)
  for (
    ushort d_outer = 0;
    d_outer < \(headDimension);
    d_outer += \(blockDimensions.head)
  ) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Inner loop over the head dimension.
    if (d_outer + \(blockDimensions.head) <= \(headDimension)) {
      \(storeBlock(registerSize: blockDimensions.head))
    } else {
      \(storeBlock(registerSize: headDimension % blockDimensions.head))
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      uint2 \(name)_offset(d_outer, \(parallelizationOffset));
      auto src = (threadgroup float*)(threadgroup_block);
      auto dst = simdgroup_matrix_storage<float>::apply_offset(
        \(name), \(leadingDimension(name)),
        \(name)_offset, \(transposed(name)));
     
      ushort D_dimension = min(
        ushort(\(blockDimensions.head)),
        ushort(\(headDimension) - d_outer));
      ushort R_dimension = min(
        uint(\(blockDimensions.parallelization)),
        uint(\(parallelizationDimension) - \(parallelizationOffset)));
      ushort2 tile(D_dimension, M_dimension);
      
      simdgroup_event event;
      event.async_copy(
        dst, \(leadingDimension(name)), tile,
        src, \(leadingBlockDimension(name)), tile, \(transposed(name)));
      simdgroup_event::wait(1, &event);
    }
  }
}

"""
  }
}

extension AttentionKernel {
  // Allocate registers for the specified operand.
  fileprivate func allocateOperand(name: String) -> String {
    """
    
    simdgroup_matrix_storage<float> \(name)_sram[\(paddedHeadDimension / 8)];
    
    """
  }
  
  // The thread's index along the parallelization dimension.
  fileprivate var parallelizationIndex: String {
    "gid * \(blockDimensions.parallelization) + sidx * 8 + morton_offset.y"
  }
  
  // Prepare the addresses and registers for the attention loop.
  func createSetup() -> String {
    // Initialize the output string.
    var output: String = ""
    
    switch type {
    case .forward:
      if cachedInputs.Q {
        output += load(name: "Q")
      }
      if cachedOutputs.O {
        output += allocateOperand(name: "O")
      }
      output += """

      float m = -numeric_limits<float>::max();
      float l = numeric_limits<float>::denorm_min();

      """
      
    case .backwardQuery(let computeDerivativeQ):
      if cachedInputs.Q {
        output += load(name: "Q")
      }
      if cachedInputs.dO {
        output += load(name: "dO")
      }
      if computeDerivativeQ, cachedOutputs.dQ {
        output += load(name: "dQ")
      }
      if computeDerivativeQ {
        output += """
        
        float L_term = L_terms[\(parallelizationIndex)];
        
        """
      }
      
      // One could optimize this by recycling dO if it's already cached.
      output += computeDTerm()
      
    case .backwardKeyValue(let computeDerivativeK):
      if cachedInputs.K {
        output += load(name: "K")
      }
      if cachedInputs.V {
        output += load(name: "V")
      }
      if computeDerivativeK, cachedOutputs.dK {
        output += allocateOperand(name: "dK")
      }
      if cachedOutputs.dV {
        output += allocateOperand(name: "dV")
      }
    }
    
    return output
  }
}

extension AttentionKernel {
  // Store any cached outputs to memory.
  func createCleanup(type: AttentionKernelType) -> String {
    // Initialize the output string.
    var output: String = ""
    
    switch type {
    case .forward(let computeL):
      // O
      if cachedOutputs.O {
        output += store(name: "O")
      }
      
      // L[i]
      if computeL {
        output += """
        
        if (\(parallelizationIndex) < R) {
          // Premultiplied by M_LOG2E_F.
          float L_term = m + fast::log2(l);
          L_terms[\(parallelizationIndex)] = L_term;
        }
        
        """
      }
    
    case .backwardQuery(let computeDerivativeQ):
      if computeDerivativeQ, cachedOutputs.dQ {
        output += store(name: "dQ")
      }
      output += """
      
      if (\(parallelizationIndex) < R) {
        D_terms[\(parallelizationIndex)] = D_term;
      }
      
      """
      
    case .backwardKeyValue(let computeDerivativeK):
      if computeDerivativeK, cachedOutputs.dK {
        output += store(name: "dK")
      }
      if cachedOutputs.dV {
        output += store(name: "dV")
      }
    }
    
    return output
  }
}