//
//  AttentionKernel+Caching.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/22/24.
//

// N x D
// parallelization x head

extension AttentionKernel {
  func load(operand: AttentionOperand) -> String {
    func loadBlock(registerSize: UInt16) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(registerSize); d += 8) {
        ushort2 origin(d, 0);
        \(operand)_sram[(d_outer + d) / 8].load(
          \(operand)_block, \(leadingBlockDimension(operand)),
          origin, \(transposed(operand)));
      }
      
      """
    }
    
    return """

\(allocateCache(operand: operand))
{
  // Where the \(operand) data will be read from.
  ushort2 \(operand)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(operand)_block = (threadgroup float*)(threadgroup_block);
  \(operand)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(operand)_block, \(leadingBlockDimension(operand)),
    \(operand)_block_offset, \(transposed(operand)));
  
  // Outer loop over the head dimension.
#pragma clang loop unroll(full)
  for (
    ushort d_outer = 0;
    d_outer < \(headDimension);
    d_outer += \(blockDimensions.head)
  ) {
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
        src, \(leadingDimension(operand)), tile_src, \(transposed(operand)));
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
  
  func store(operand: AttentionOperand) -> String {
    func storeBlock(registerSize: UInt16) -> String {
      """
      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < \(registerSize); d += 8) {
        ushort2 origin(d, 0);
        \(operand)_sram[(d_outer + d) / 8].store(
          \(operand)_block, \(leadingBlockDimension(operand)),
          origin, \(transposed(operand)));
      }
      
      """
    }
    
    return """

{
  // Where the \(operand) data will be written to.
  ushort2 \(operand)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(operand)_block = (threadgroup float*)(threadgroup_block);
  \(operand)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(operand)_block, \(leadingBlockDimension(operand)),
    \(operand)_block_offset, \(transposed(operand)));
  
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
        src, \(leadingBlockDimension(operand)), tile, \(transposed(operand)));
      simdgroup_event::wait(1, &event);
    }
  }
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
      if cachedInputs.Q {
        output += load(operand: .Q)
      }
      if cachedOutputs.O {
        output += allocateCache(operand: .O)
      }
      output += """

      float m = -numeric_limits<float>::max();
      float l = numeric_limits<float>::denorm_min();

      """
      
    case .backwardQuery(let computeDerivativeQ):
      if cachedInputs.Q {
        output += load(operand: .Q)
      }
      if cachedInputs.dO {
        output += load(operand: .dO)
      }
      if computeDerivativeQ, cachedOutputs.dQ {
        output += allocateCache(operand: .dQ)
      }
      if computeDerivativeQ {
        output += """
        
        float L_term = L_terms[\(parallelizationThreadOffset)];
        
        """
      }
      
      output += """
      
      float D_term;
      \(computeDTerm())
      if (\(parallelizationThreadOffset) < R) {
        D_terms[\(parallelizationThreadOffset)] = D_term;
      }
      
      """
      
    case .backwardKeyValue(let computeDerivativeK):
      if cachedInputs.K {
        output += load(operand: .K)
      }
      if cachedInputs.V {
        output += load(operand: .V)
      }
      if computeDerivativeK, cachedOutputs.dK {
        output += allocateCache(operand: .dK)
      }
      if cachedOutputs.dV {
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
      if cachedOutputs.O {
        output += store(operand: .O)
      }
      if computeL {
        output += """
        
        if (\(parallelizationThreadOffset) < R) {
          // Premultiplied by M_LOG2E_F.
          float L_term = m + fast::log2(l);
          L_terms[\(parallelizationThreadOffset)] = L_term;
        }
        
        """
      }
    
    case .backwardQuery(let computeDerivativeQ):
      if computeDerivativeQ, cachedOutputs.dQ {
        output += store(operand: .dQ)
      }
      
    case .backwardKeyValue(let computeDerivativeK):
      if computeDerivativeK, cachedOutputs.dK {
        output += store(operand: .dK)
      }
      if cachedOutputs.dV {
        output += store(operand: .dV)
      }
    }
    
    return output
  }
}
