//
//  AttentionKernel+LoadStore.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/22/24.
//

// Loading and storing operands that are cached in registers.

// SIMD Matrix Storage
//   register allocation per thread
//    (scalar footprint) * (2 * D / 8) bytes
// -> (scalar footprint) * (D / 4) bytes
//
// Forward
//   cache Q, O, m, l
//     FP32: 8 * (D / 4) + 8 bytes
//     FP16: 6 * (D / 4) + 8 bytes
//
// Backward Query (true)
//   cache dQ, dO, L[i], D[i]
//     FP32: 8 * (D / 4) + 8 bytes
//     FP16: 6 * (D / 4) + 8 bytes
//   cache Q
//     FP32: 12 * (D / 4) + 8 bytes
//     FP16:  8 * (D / 4) + 8 bytes
//
// Backward Key-Value (true)
//   cache dK, dV
//     FP32: 8 * (D / 4) bytes
//     FP16: 8 * (D / 4) bytes
//   cache K, V
//     FP32: 16 * (D / 4) bytes
//     FP16: 12 * (D / 4) bytes
//
// Backward Key-Value (false)
//   cache dV
//     FP32: 4 * (D / 4) bytes
//     FP16: 4 * (D / 4) bytes
//   cache K, V
//     FP32: 12 * (D / 4) bytes
//     FP16:  8 * (D / 4) bytes

struct AttentionLoadStoreDescriptor {
  /// Name of operand (N x D).
  var name: String?
  
  var transposeState: Bool?
  var leadingDimension: String?
  
  var matrixDimension: String?
  var matrixOffset: String?
}

extension AttentionKernel {
  // The size of a cached operand, along the head dimension.
  func paddedD() -> UInt16 {
    (headDimension + 8 - 1) / 8 * 8
  }
  
  func load(descriptor: AttentionLoadStoreDescriptor) -> String {
    guard let name = descriptor.name,
          let transposeState = descriptor.transposeState,
          let leadingDimension = descriptor.leadingDimension,
          let matrixDimension = descriptor.matrixDimension,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    let leadingBlockDimension = transposeState ?
    blockDimensions.R : blockDimensions.D
    
    let loopBody = """
    
    ushort2 origin(d, 0);
    \(name)_sram[(d_outer + d) / 8].load(
      \(name)_block, \(leadingBlockDimension), origin, \(transposeState));
    
    """
    
    func allocateLHS(name: String) -> String {
      """
      
      simdgroup_matrix_storage<float> \(name)_sram[\(paddedD() / 8)];
      
      """
    }
    
    return """

\(allocateLHS(name: name))
{
  // Find where the \(name) data will be read from.
  ushort2 \(name)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(name)_block = (threadgroup float*)(threadgroup_block);
  \(name)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(name)_block, \(leadingBlockDimension),
    \(name)_block_offset, \(transposeState));
  
  // Outer loop over D.
#pragma clang loop unroll(full)
  for (
    ushort d_outer = 0;
    d_outer < \(headDimension);
    d_outer += \(blockDimensions.D)
  ) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      uint2 \(name)_offset(d_outer, \(matrixOffset));
      auto src = simdgroup_matrix_storage<float>::apply_offset(
        \(name), \(leadingDimension), \(name)_offset, \(transposeState));
      auto dst = (threadgroup float*)(threadgroup_block);
     
      ushort D_src_dimension = min(
        ushort(\(blockDimensions.D)), ushort(\(headDimension) - d_outer));
      ushort D_dst_dimension = min(
        ushort(\(blockDimensions.D)), ushort(\(paddedD) - d_outer));
      ushort M_dimension = min(
        uint(32), \(matrixDimension) - \(matrixOffset));
      ushort2 tile_src(D_src_dimension, M_dimension);
      ushort2 tile_dst(D_dst_dimension, M_dimension);
      
      simdgroup_event event;
      event.async_copy(
        dst, \(leadingBlockDimension), tile_dst,
        src, \(leadingDimension), tile_src, \(transposeState));
      simdgroup_event::wait(1, &event);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterate over the head dimension.
    if (\(headDimension) - d_outer >= \(blockDimensions.D)) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(blockDimensions.D); d += 8) {
        \(loopBody)
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(headDimension % blockDimensions.D); d += 8) {
        \(loopBody)
      }
    }
  }
}

"""
  }
  
  func store(descriptor: AttentionLoadStoreDescriptor) -> String {
    guard let name = descriptor.name,
          let transposeState = descriptor.transposeState,
          let leadingDimension = descriptor.leadingDimension,
          let matrixDimension = descriptor.matrixDimension,
          let matrixOffset = descriptor.matrixOffset else {
      fatalError("Descriptor was incomplete.")
    }
    
    let leadingBlockDimension = transposeState ? 32 : blockDimensions.D
    let loopBody = """
    
    ushort2 origin(d, 0);
    \(name)_sram[(d_outer + d) / 8].store(
      \(name)_block, \(leadingBlockDimension), origin, \(transposeState));
    
    """
    
    return """

{
  // Find where the \(name) data will be written to.
  ushort2 \(name)_block_offset(morton_offset.x, morton_offset.y + sidx * 8);
  auto \(name)_block = (threadgroup float*)(threadgroup_block);
  \(name)_block = simdgroup_matrix_storage<float>::apply_offset(
    \(name)_block, \(leadingBlockDimension),
    \(name)_block_offset, \(transposeState));
  
  // Outer loop over D.
#pragma clang loop unroll(full)
  for (
    ushort d_outer = 0;
    d_outer < \(headDimension);
    d_outer += \(blockDimensions.D)
  ) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Iterate over the head dimension.
    if (\(headDimension) - d_outer >= \(blockDimensions.D)) {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(blockDimensions.D); d += 8) {
        \(loopBody)
      }
    } else {
#pragma clang loop unroll(full)
      for (ushort d = 0; d < \(headDimension % blockDimensions.D); d += 8) {
        \(loopBody)
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      uint2 \(name)_offset(d_outer, \(matrixOffset));
      auto src = (threadgroup float*)(threadgroup_block);
      auto dst = simdgroup_matrix_storage<float>::apply_offset(
        \(name), \(leadingDimension), \(name)_offset, \(transposeState));
     
      ushort D_dimension = min(
        ushort(\(blockDimensions.D)), ushort(\(headDimension) - d_outer));
      ushort M_dimension = min(
        uint(32), \(matrixDimension) - \(matrixOffset));
      ushort2 tile(D_dimension, M_dimension);
      
      simdgroup_event event;
      event.async_copy(
        dst, \(leadingDimension), tile,
        src, \(leadingBlockDimension), tile, \(transposeState));
      simdgroup_event::wait(1, &event);
    }
  }
}

"""
  }
}

extension AttentionKernel {
  // The thread's index along the parallelization dimension.
  fileprivate func parallelizationIndex() -> String {
    "gid * \(blockDimensions.parallelization) + sidx * 8 + morton_offset.y"
  }
  
  func createSetup(type: AttentionKernelType) -> String {
    func allocateAccumulator(name: String) -> String {
      let paddedD = (headDimension + 8 - 1) / 8 * 8
      return """
      
      simdgroup_matrix_storage<float> \(name)_sram[\(paddedD / 8)];
      
      """
    }
    
    // Initialize the output string.
    var output: String = ""
    
    switch type {
    case .forward:
      if cachedInputs.Q {
        var accessDesc = AttentionLoadStoreDescriptor()
        accessDesc.name = "Q"
        accessDesc.transposeState = transposeState.Q
        accessDesc.leadingDimension = leadingDimensions.Q
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += load(descriptor: accessDesc)
      }
      
      if cachedOutputs.O {
        output += allocateAccumulator(name: "O")
      }
      
      output += """

  float m = -numeric_limits<float>::max();
  float l = numeric_limits<float>::denorm_min();

"""
      
    case .backwardQuery(let computeDerivativeQ):
      if cachedInputs.Q {
        var accessDesc = AttentionLoadStoreDescriptor()
        accessDesc.name = "Q"
        accessDesc.transposeState = transposeState.Q
        accessDesc.leadingDimension = leadingDimensions.Q
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += load(descriptor: accessDesc)
      }
      
      if cachedInputs.dO {
        var accessDesc = AttentionLoadStoreDescriptor()
        accessDesc.name = "dO"
        accessDesc.transposeState = transposeState.O
        accessDesc.leadingDimension = leadingDimensions.O
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += load(descriptor: accessDesc)
      }
      
      if computeDerivativeQ {
        if cachedOutputs.dQ {
          output += allocateAccumulator(name: "dQ")
        }
        
        output += """

  float L_term = L_terms[\(parallelizationIndex())];

"""
      }
      
      output += computeDTerm()
      
    case .backwardKeyValue(let computeDerivativeK):
      if cachedInputs.K {
        var accessDesc = AttentionLoadStoreDescriptor()
        accessDesc.name = "K"
        accessDesc.transposeState = transposeState.K
        accessDesc.leadingDimension = leadingDimensions.K
        accessDesc.matrixDimension = "C"
        accessDesc.matrixOffset = "gid * 32"
        output += load(descriptor: accessDesc)
      }
      
      if cachedInputs.V {
        var accessDesc = AttentionLoadStoreDescriptor()
        accessDesc.name = "V"
        accessDesc.transposeState = transposeState.V
        accessDesc.leadingDimension = leadingDimensions.V
        accessDesc.matrixDimension = "C"
        accessDesc.matrixOffset = "gid * 32"
        output += load(descriptor: accessDesc)
      }
      
      if computeDerivativeK, cachedOutputs.dK {
        output += allocateAccumulator(name: "dK")
      }
      
      if cachedOutputs.dV {
        output += allocateAccumulator(name: "dV")
      }
    }
    
    return output
  }
}

extension AttentionKernel {
  func createCleanup(type: AttentionKernelType) -> String {
    // Initialize the output string.
    var output: String = ""
    
    switch type {
    case .forward(let computeL):
      // O
      if cachedOutputs.O {
        var accessDesc = AttentionLoadStoreDescriptor()
        accessDesc.name = "O"
        accessDesc.transposeState = transposeState.O
        accessDesc.leadingDimension = leadingDimensions.O
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += store(descriptor: accessDesc)
      }
      
      // L[i]
      if computeL {
        output += """

    if (linear_array_slot < R) {
      // Premultiplied by M_LOG2E_F.
      float L_term = m + fast::log2(l);
      L_terms[\(parallelizationIndex())] = L_term;
    }

"""
      }
    
    case .backwardQuery(let computeDerivativeQ):
      // dQ
      if computeDerivativeQ, cachedOutputs.dQ {
        var accessDesc = AttentionLoadStoreDescriptor()
        accessDesc.name = "dQ"
        accessDesc.transposeState = transposeState.Q
        accessDesc.leadingDimension = leadingDimensions.Q
        accessDesc.matrixDimension = "R"
        accessDesc.matrixOffset = "gid * 32"
        output += store(descriptor: accessDesc)
      }
      
      // D[i]
      output += """
  
  if (\(parallelizationIndex()) < R) {
    D_terms[\(parallelizationIndex())] = D_term;
  }

"""
      
    case .backwardKeyValue(let computeDerivativeK):
      // dK
      if computeDerivativeK, cachedOutputs.dK {
        var accessDesc = AttentionLoadStoreDescriptor()
        accessDesc.name = "dK"
        accessDesc.transposeState = transposeState.K
        accessDesc.leadingDimension = leadingDimensions.K
        accessDesc.matrixDimension = "C"
        accessDesc.matrixOffset = "gid * 32"
        output += store(descriptor: accessDesc)
      }
      
      // dV
      if cachedOutputs.dV {
        var accessDesc = AttentionLoadStoreDescriptor()
        accessDesc.name = "dV"
        accessDesc.transposeState = transposeState.V
        accessDesc.leadingDimension = leadingDimensions.V
        accessDesc.matrixDimension = "C"
        accessDesc.matrixOffset = "gid * 32"
        output += store(descriptor: accessDesc)
      }
    }
    
    return output
  }
}
